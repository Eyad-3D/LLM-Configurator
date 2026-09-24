"""Conservative memory model and evidence-aware ranking; no invented quality/speed scores.

Speed evidence is ranked strictly: a local measurement or tuning run is the only thing
that can say a configuration "runs well". Interpolated, community and estimated speeds are
shown with their label and can only say "likely", never "verified".
"""
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import math

from .domain import GIB, KV_BYTES_PER_ELEMENT, Requirements, Variant
from .launch import runtime_gpu_layers
from .learning import fit_efficiency
from .speed import active_fraction, estimate, gpu_blocks, kv_bytes, moved_expert_layers, output_bytes

MAX_AGE_DAYS = 30          # local measurements and tune results older than this are ignored
SLOW_FRACTION = 0.5        # verified speed at or above half the target "runs slowly"; below that "too slow"
COMFORT_FACTOR = 1.5       # verified speed this far above target is "comfortably above"
EXTRAPOLATE_LIMIT = 4      # never stretch one test to a context more than 4x longer
EXTRAPOLATE_MARGIN = 0.9   # single-point scaling is shaded down 10% to stay conservative
FULL_DEPTH_SLACK = 1024    # a test counts for a context when it ran with at least context-1024 tokens in memory
GPU_BACKENDS = {"cuda", "metal", "rocm", "vulkan"}
KV_TEXT = {"f16": "full precision (f16)",
           "q8_0": "8-bit compression (q8_0): about half the memory of full precision, usually with no noticeable quality loss",
           "q4_0": "4-bit compression (q4_0): about a quarter of the memory of full precision, with a small quality cost"}


def allocations(variant, context, users, gpu_layers, kv_cache_type="f16", n_cpu_moe=0, unified=False, ubatch=None):
    """Estimated bytes per memory pool.

    Returns {"ram", "vram", "kv_total", "kv_vram", "weights_ram", "weights_vram", "expert_ram",
    "total", "unified"}. With unified=True (Apple) "ram" is the whole shared pool and "vram" is
    the part the graphics chip uses *inside* that pool, so it must not be added to "ram" again.
    """
    total_layers = variant.layers
    if not 0 <= gpu_layers <= total_layers:
        raise ValueError("GPU layer allocation out of range")
    if kv_cache_type not in KV_BYTES_PER_ELEMENT:
        raise ValueError("kv_cache_type must be f16, q8_0 or q4_0")
    if type(n_cpu_moe) is not int or not 0 <= n_cpu_moe <= total_layers:
        raise ValueError("n_cpu_moe must be between 0 and the number of layers")
    # K + V per user as llama-server allocates them; no assumed prefix sharing.
    kv = kv_bytes(variant, context, users, kv_cache_type, ubatch)
    # Whole-file placement is approximate; 10% placement/metadata margin protects the estimate.
    weights = variant.size_bytes * 1.10
    # What llama.cpp really offloads: the last `blocks` transformer blocks plus, for any GPU run,
    # the output layer, which can be several blocks' worth of bytes (large vocabularies).
    blocks, output_on_gpu = gpu_blocks(variant, gpu_layers)
    fraction = blocks / total_layers
    if blocks == total_layers or not output_on_gpu:
        weights_vram = weights * fraction
    else:
        output = output_bytes(variant) * 1.10
        weights_vram = (weights - output) * fraction + output
    # --n-cpu-moe keeps the expert tensors of the affected GPU blocks in RAM.
    expert_ram = weights * variant.expert_fraction * moved_expert_layers(variant, gpu_layers, n_cpu_moe) / total_layers
    weights_vram -= expert_ram
    weights_ram = weights - weights_vram
    buffer = (0.75 + users * 0.20) * GIB
    # A layer's KV lives on that layer's device, even when its experts stay in RAM.
    ram = weights_ram + kv * (1 - fraction) + buffer
    vram = 0
    if gpu_layers:
        vram = weights_vram + kv * fraction + buffer  # the GPU keeps its own compute buffer, separate from the CPU's
        if unified:
            ram += vram  # one physical pool: GPU work lives inside RAM, no staging copy
        else:
            ram += 0.5 * GIB  # staging; model files are memory-mapped, not duplicated in full
    ram, vram = math.ceil(ram), math.ceil(vram)
    return {"ram": ram, "vram": vram, "kv_total": kv, "kv_vram": math.ceil(kv * fraction),
            "weights_ram": math.ceil(weights_ram), "weights_vram": math.ceil(weights_vram), "expert_ram": math.ceil(expert_ram),
            "total": ram if unified else ram + vram, "unified": bool(unified)}


def _fits(variant, context, users, layers, budget, gpu_budget, kv, moe, unified):
    memory = allocations(variant, context, users, layers, kv, moe, unified)
    return memory if memory["ram"] <= budget and memory["vram"] <= gpu_budget else None


def _boundary(low, high, ok, largest):
    """Binary search over a monotone predicate: largest x with ok(x) (True..False) or smallest (False..True)."""
    best = None
    while low <= high:
        middle = (low + high) // 2
        if ok(middle):
            best = middle
            low, high = (middle + 1, high) if largest else (low, middle - 1)
        else:
            low, high = (low, middle - 1) if largest else (middle + 1, high)
    return best


def placements(variant, context, users, budget, gpu_budget, has_gpu, kv_cache_type="f16", unified=False):
    """CPU-only, full GPU, the largest fitting layer split and (MoE) the fewest experts-on-CPU layers.

    Returns [(gpu_layers, n_cpu_moe, memory)]. VRAM grows and RAM shrinks with each GPU layer,
    so binary search finds the same split an exhaustive scan would, in O(log layers).
    """
    total = variant.layers
    args = (variant, context, users)
    found = []
    memory = _fits(*args, 0, budget, gpu_budget, kv_cache_type, 0, unified)
    if memory:
        found.append((0, 0, memory))
    if not has_gpu:
        return found
    vram = lambda layers, moe=0: allocations(*args, layers, kv_cache_type, moe, unified)["vram"] <= gpu_budget
    split = _boundary(1, total - 1, vram, largest=True)
    if split:
        memory = _fits(*args, split, budget, gpu_budget, kv_cache_type, 0, unified)
        if memory:
            found.append((split, 0, memory))
    memory = _fits(*args, total, budget, gpu_budget, kv_cache_type, 0, unified)
    if memory:
        found.append((total, 0, memory))
    elif variant.moe:
        # Keep attention and shared weights on the GPU; park only as many expert layers in RAM as needed.
        moe = _boundary(1, total, lambda k: vram(total, k), largest=False)
        memory = moe and _fits(*args, total, budget, gpu_budget, kv_cache_type, moe, unified)
        if memory:
            found.append((total, moe, memory))
    return found


def maximum_context(variant, users, layers, ram_budget, vram_budget, kv_cache_type="f16", n_cpu_moe=0, unified=False):
    # Memory grows with context, so searching whole 256-token steps gives the same rounded answer.
    def fits(steps):
        use = allocations(variant, steps * 256, users, layers, kv_cache_type, n_cpu_moe, unified)
        return use["ram"] <= ram_budget and use["vram"] <= vram_budget
    return (_boundary(1, variant.max_context // 256, fits, largest=True) or 0) * 256


def _recent(record, key="timestamp"):
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record[key])).total_seconds()
        return 0 <= age < MAX_AGE_DAYS * 86400
    except (KeyError, TypeError, ValueError):
        return False


def _speed(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _full_depth(record, context):
    """Was the speed measured with (nearly) `context` tokens already in memory?

    Speed tests run at depth context-640 and v0.3 benchmarks (no `depth`) at context-128; tune
    trials run at a short depth, which says little about a long conversation.
    """
    depth = record.get("depth")
    return depth is None or (type(depth) is int and depth + FULL_DEPTH_SLACK >= context)


def _settings(record):
    """Launch settings a measurement ran with; v0.3 records without settings ran f16 with no expert offload."""
    settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
    kv = settings.get("cache_type_k") or "f16"
    return {"cache_type_k": kv, "cache_type_v": settings.get("cache_type_v") or kv, "n_cpu_moe": settings.get("n_cpu_moe") or 0,
            "batch": settings.get("batch"), "ubatch": settings.get("ubatch"), "flash_attn": settings.get("flash_attn")}


def _same_placement(record, variant, hardware, layers, gpu_uuid, threads, kv_cache_type, n_cpu_moe, launch=None):
    """Same file, machine, placement and settings. With `launch`, the tuning knobs must match it too."""
    fingerprint = hardware.get("fingerprint")
    if not (fingerprint and record.get("variant_id") == variant.id and variant.sha256 and record.get("sha256") == variant.sha256
            and record.get("fingerprint") == fingerprint and record.get("gpu_layers") == layers
            and record.get("gpu_uuid") == gpu_uuid and record.get("threads") == threads and record.get("users") == 1):
        return False
    ran = _settings(record)
    if (ran["cache_type_k"], ran["cache_type_v"], ran["n_cpu_moe"]) != (kv_cache_type, kv_cache_type, n_cpu_moe):
        return False
    if launch is None:
        return True
    # Batch sizes and flash attention change speed; a v0.3 record (no settings) ran the defaults.
    return (ran["batch"] == launch.get("batch") and ran["ubatch"] == launch.get("ubatch")
            and ran["flash_attn"] in (None, launch.get("flash_attn", "auto")))


def matching_speed(records, variant, hardware, context, layers, gpu_uuid, threads, kv_cache_type="f16", n_cpu_moe=0, launch=None):
    """Newest recent full-depth test of exactly this configuration; a speed test beats other kinds."""
    found = None
    for record in reversed(records):
        if (record.get("context") != context or not _recent(record) or not _speed(record.get("tps"))
                or not _full_depth(record, context)
                or not _same_placement(record, variant, hardware, layers, gpu_uuid, threads, kv_cache_type, n_cpu_moe, launch)):
            continue
        if record.get("kind") == "speed_test":
            return record
        found = found or record
    return found


def interpolated_speed(records, variant, hardware, context, layers, gpu_uuid, threads, kv_cache_type="f16", n_cpu_moe=0, launch=None):
    """Speed at `context` from tests of the same variant, machine and placement at other contexts.

    Between two tests: linear in context. Only shorter tests: scale by the bytes read per token
    (weights + conversation memory) and shade down 10%, up to 4x the tested length. Only longer
    tests: the nearest one, because a shorter conversation is not slower. Never "measured".
    """
    points = {}
    for record in records:
        other = record.get("context")
        if (type(other) is int and other > 0 and other != context and _recent(record) and _speed(record.get("tps"))
                and _full_depth(record, other)
                and _same_placement(record, variant, hardware, layers, gpu_uuid, threads, kv_cache_type, n_cpu_moe, launch)):
            points[other] = record  # later records win
    below = max((c for c in points if c < context), default=None)
    above = min((c for c in points if c > context), default=None)
    if below and above:
        low, high = points[below]["tps"], points[above]["tps"]
        tps = low + (high - low) * (context - below) / (above - below)
        used, method = [below, above], f"between tests at {below:,} and {above:,} tokens"
    elif below and context <= below * EXTRAPOLATE_LIMIT:
        weights = variant.size_bytes * active_fraction(variant)
        ratio = (weights + kv_bytes(variant, below, 1, kv_cache_type)) / (weights + kv_bytes(variant, context, 1, kv_cache_type))
        tps = points[below]["tps"] * ratio * EXTRAPOLATE_MARGIN
        used, method = [below], f"scaled down from a test at {below:,} tokens"
    elif above:
        tps = points[above]["tps"]
        used, method = [above], f"taken from a test at a longer {above:,} tokens"
    else:
        return None
    return {"tps": round(tps, 2), "contexts": used, "method": method,
            "measurement_ids": [points[c].get("id") for c in used if points[c].get("id")]}


def matching_tuned(tuned, variant, hardware, context, layers, kv_cache_type="f16", n_cpu_moe=0, gpu_uuid=None):
    """Most recent tune started from this exact context and placement (FIXUPS pinned tuned record).

    The record's top-level context/gpu_layers/n_cpu_moe/kv_cache_type describe the candidate the
    tune started from (older records only have `best`). `same_placement` is False when the tuner
    moved layers, experts or the cache type: then its speed belongs to another placement. `verified`
    is True only when the tuner measured at (nearly) the full context.
    """
    fingerprint = hardware.get("fingerprint")
    several_gpus = len(hardware.get("gpus") or []) > 1
    for record in reversed(list(tuned or ())):
        if not isinstance(record, dict) or not isinstance(record.get("best"), dict):
            continue
        best, result = record["best"], record.get("best_result") or {}
        start = lambda key, fallback=None: record[key] if record.get(key) is not None else best.get(fallback or key)
        best_placement = {"gpu_layers": best.get("gpu_layers"), "n_cpu_moe": best.get("n_cpu_moe") or 0,
                          "cache_type_k": best.get("cache_type_k") or "f16"}
        if (not fingerprint or record.get("variant_id") != variant.id or record.get("fingerprint") != fingerprint
                or (variant.sha256 and record.get("sha256") not in (None, variant.sha256))
                or start("context") != context or start("gpu_layers") != layers
                or (start("n_cpu_moe") or 0) != n_cpu_moe or (start("kv_cache_type", "cache_type_k") or "f16") != kv_cache_type
                or (best.get("parallel") or 1) != 1 or (record.get("users") or 1) != 1
                or (layers and best.get("gpu_uuid") not in (None, gpu_uuid))
                or (layers and several_gpus and not best.get("gpu_uuid"))
                or not _recent(record) or not _speed(result.get("tps"))):
            continue
        depth = (record.get("settings") or {}).get("depth") if isinstance(record.get("settings"), dict) else None
        return {"tps": result["tps"], "pp_tps": result.get("pp_tps"), "improvement": record.get("improvement"),
                "timestamp": record["timestamp"], "stopped": record.get("stopped"), "depth": depth,
                "verified": type(depth) is int and depth + FULL_DEPTH_SLACK >= context,
                "same_placement": best_placement == {"gpu_layers": layers, "n_cpu_moe": n_cpu_moe, "cache_type_k": kv_cache_type},
                "best_placement": best_placement,
                "settings": {k: best[k] for k in ["threads", "batch", "ubatch", "flash_attn"] if best.get(k) is not None}}
    return None


def _community_for(records, variant):
    """Community rows that can describe this model file (community._same_model), found once per variant."""
    sha, repo, quant = (variant.sha256 or "").lower(), variant.repo, (variant.quant or "").upper()
    found = []
    for record in records:
        theirs = record.get("variant") if isinstance(record, dict) else None
        if not isinstance(theirs, dict):
            found.append(record)  # not a community row; let community.evidence decide
        elif sha and theirs.get("sha256"):
            if theirs["sha256"] == sha:
                found.append(record)
        elif theirs.get("repo") and theirs.get("repo") == repo and theirs.get("quant") == quant:
            found.append(record)
    return found


def _community_settings_match(record, kv_cache_type, n_cpu_moe):
    """Other people's runs count only with the same notepad format, expert offload and a full-depth test."""
    settings = record.get("settings") if isinstance(record, dict) else None
    if not isinstance(settings, dict) or "placement" not in settings:
        return True  # not a community row
    context = record.get("context")
    return ((settings.get("cache_type_k") or "f16") == kv_cache_type and (settings.get("n_cpu_moe") or 0) == n_cpu_moe
            and (type(context) is not int or _full_depth(record, context)))


def community_speed(records, variant, hardware, config, evidence=None):
    """community.evidence(...) with a defensive check; None when absent, malformed or not installed."""
    if not records:
        return None
    if evidence is None:
        try:
            from .community import evidence
        except ImportError:
            return None
    try:
        found = evidence(records, variant, hardware, config)
    except (KeyError, TypeError, ValueError):
        return None
    if not isinstance(found, dict) or not _speed(found.get("median_tps")) or type(found.get("n")) is not int or found["n"] < 1:
        return None
    return {k: found.get(k) for k in ["median_tps", "n", "similar", "range"]}


def _tps(value, target=None):
    """Plain number; one decimal when rounding would hide which side of the target it is on."""
    if value < 0.1:
        return "under 0.1"
    if target and value != target and round(value) == round(target):
        return f"{value:.1f}"
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def verdict(evidence, target, tps=None, interpolated=None, community=None, speed_estimate=None, users=1, tuned=None):
    """(verdict, one plain sentence). Only measured or tuned speed at this length can be runs_well/runs_slowly/too_slow."""
    goal = f"{target:g}"
    if evidence in {"measured", "tuned"} and tps is not None:
        where = "Tuned and tested on this computer" if evidence == "tuned" else "Tested on this computer"
        head = f"{where}: about {_tps(tps, target)} tokens per second"
        if not target:
            return "runs_well", f"{head}."
        if tps >= target:
            if tps >= target * COMFORT_FACTOR:
                return "runs_well", f"{head}, comfortably above your target of {goal}."
            return "runs_well", f"{head}, which meets your target of {goal}."
        if tps >= target * SLOW_FRACTION:
            return "runs_slowly", f"{head}, a bit below your target of {goal}."
        return "too_slow", f"{head}, well below your target of {goal}."
    likely = lambda value: "likely fast enough" if value >= target else f"possibly slower than your target of {goal}"
    if evidence == "tuned" and tuned:
        return "unknown", (f"Tuned with a short test: about {_tps(tuned['tps'], target)} tokens per second near the start "
                           f"of a conversation. Run a speed test to check it at this length.")
    if evidence == "interpolated":
        return "unknown", (f"Not tested at this length yet; tests at other lengths suggest about {_tps(interpolated['tps'], target)} "
                           f"tokens per second, {likely(interpolated['tps'])}.")
    if evidence == "community":
        people = "similar computer reports" if community["n"] == 1 else "similar computers report"
        return "unknown", (f"Not tested on this computer yet; {community['n']} {people} about "
                           f"{_tps(community['median_tps'], target)} tokens per second, {likely(community['median_tps'])}.")
    if evidence == "estimated":
        e = speed_estimate
        outlook = {"likely_meets": "likely fast enough", "borderline": f"might reach your target of {goal}",
                   "likely_below": f"probably slower than your target of {goal}"}[e["target_status"]]
        low, high = _tps(e["low_tps"]), _tps(e["high_tps"])
        span = f"under 0.1" if high == "under 0.1" else f"{low}–{high}"
        return "unknown", f"Not tested yet; a rough estimate says {span} tokens per second, {outlook}."
    if users != 1:
        return "unknown", "Not tested yet: speed with several people at once needs a speed test with that many users."
    return "unknown", "Not tested yet. Run a speed test to find out how fast it is."


def _launch(context, users, layers, total, threads, kv, moe, gpu, tuned, hide=None):
    """Launch-config fragment for launch.from_candidate (no model_path).

    `hide` is a GPU backend seen on this computer: CPU-only candidates name it so that
    launch adds `-dev none`, even when that GPU's free memory is unknown.
    """
    config = {"context": context, "parallel": users, "gpu_layers": layers, "total_layers": total, "threads": threads,
              "cache_type_k": kv, "cache_type_v": kv, "n_cpu_moe": moe,
              "flash_attn": "on" if kv != "f16" else "auto"}  # llama.cpp needs flash attention for a compressed V cache
    backend = gpu.get("backend") if gpu else hide if not layers else None
    if backend in GPU_BACKENDS:
        config["gpu_backend"] = backend  # also lets CPU-only runs hide the GPU (-dev none)
    if gpu and layers and isinstance(gpu.get("uuid"), str) and gpu["uuid"]:
        config["gpu_uuid"] = gpu["uuid"]
    if tuned:
        settings = dict(tuned["settings"])
        if kv != "f16" and settings.get("flash_attn") == "off":
            settings.pop("flash_attn")
        config.update(settings)
    return config


def _evidence(variant, hardware, records, tunes, crowd_rows, calibration, efficiency, req, context, layers, moe, gpu,
              gpu_uuid, threads, unified, hide, community_evidence, use_default_community):
    """Launch fragment and speed evidence for one placement (identical in every memory scenario).

    Order: a speed test of exactly this launch > a tune verified at this length > interpolation
    > a short tune > community > estimate. Tune settings are merged into the launch only when the
    tune kept this placement, and a test must match the merged launch to count as "measured".
    """
    kv, single = req.kv_cache_type, req.users == 1
    tune = matching_tuned(tunes, variant, hardware, context, layers, kv, moe, gpu_uuid) if single else None
    applied = tune if tune and tune["same_placement"] else None
    launch = _launch(context, req.users, layers, variant.layers, threads, kv, moe, gpu, applied, hide)
    args = (variant, hardware, context, layers, gpu_uuid, launch["threads"], kv, moe, launch)
    measurement = matching_speed(records, *args) if single else None
    verified_tune = applied if applied and applied["verified"] else None
    tps = measurement["tps"] if measurement else verified_tune["tps"] if verified_tune else None
    interpolated = crowd = None
    if tps is None and single:
        interpolated = interpolated_speed(records, *args)
        rows = [r for r in crowd_rows if _community_settings_match(r, kv, moe)]
        # community.evidence needs two matching rows; skip the call when there cannot be two.
        if len(rows) >= 2 or not use_default_community:
            crowd = community_speed(rows, variant, hardware, launch, community_evidence)
    speed_estimate = estimate(variant, hardware, calibration, context, layers, req.users, gpu, req.min_tps,
                              kv_cache_type=kv, n_cpu_moe=moe, unified=unified, efficiency=efficiency)
    evidence = ("measured" if measurement else "tuned" if verified_tune else "interpolated" if interpolated
                else "tuned" if applied else "community" if crowd else "estimated" if speed_estimate.get("available") else "none")
    return launch, measurement, tune, tps, interpolated, crowd, speed_estimate, evidence


def recommend(variants, hardware, requirements, measurements=(), calibration=None, community=(), tuned=(),
              community_evidence=None):
    req = requirements
    kv = req.kv_cache_type
    measurements = list(measurements or ())
    community = community.get("records", []) if isinstance(community, dict) else list(community or ())
    notes_extra = []
    ram_budget = max(0, hardware["ram_available"] - req.reserve_gib * GIB)
    selected = set(req.reclaim_pids)
    reclaim = sum(p.get("reclaimable") or 0 for p in hardware.get("processes", []) if p["pid"] in selected)
    scenarios = [("now", ram_budget)]
    if reclaim:
        scenarios.append(("after_closing", min(hardware["ram_total"] - req.reserve_gib * GIB, ram_budget + reclaim)))
    gpu = next((g for g in hardware["gpus"] if g.get("index") == req.gpu_index), None)
    # Any GPU llama.cpp could use, even one whose free memory is unknown: CPU-only runs must hide it.
    hide = next((g.get("backend") for g in [gpu, *hardware["gpus"], *(hardware.get("other_gpus") or [])]
                 if isinstance(g, dict) and g.get("backend") in GPU_BACKENDS), None)
    if gpu is not None and not _speed(gpu.get("available")):
        # Free memory unknown (not zero): never guess, fall back to CPU-only placements.
        notes_extra.append(f"{gpu.get('name') or 'The selected graphics chip'} does not report its free memory, so only CPU options are shown.")
        gpu = None
    unified = bool(gpu and gpu.get("unified"))
    gpu_cap = max(0, gpu["available"] - req.gpu_reserve_gib * GIB) if gpu else 0
    if unified:
        # Apple: the GPU working-set limit is a slice of the same RAM; after closing apps it can grow to the wired limit.
        limit = gpu.get("total") if _speed(gpu.get("total")) else gpu["available"]
        gpu_budgets = {"now": lambda budget: min(gpu_cap, budget),
                       "after_closing": lambda budget: min(max(gpu_cap, limit - req.gpu_reserve_gib * GIB), budget)}
        notes_extra.append("This computer's graphics chip shares main memory, so every byte is counted once, in one pool.")
    else:
        gpu_budgets = {"now": lambda budget: gpu_cap, "after_closing": lambda budget: gpu_cap}
    threads = hardware.get("cores") or hardware.get("threads") or 1
    efficiency = fit_efficiency(measurements, hardware, calibration, variants) if calibration else None
    by_variant, tunes_by_variant = defaultdict(list), defaultdict(list)
    for record in measurements:
        by_variant[record.get("variant_id")].append(record)
    for record in tuned or ():
        if isinstance(record, dict):
            tunes_by_variant[record.get("variant_id")].append(record)
    use_default_community = community_evidence is None
    results, rejected = [], {"context": 0, "memory": 0, "speed": 0}
    compressible, max_context_cache = [], {}
    metric = req.workload if req.workload != "documents" else "general"
    for variant in variants:
        if req.context > variant.max_context:
            rejected["context"] += 1
            continue
        records = by_variant.get(variant.id, [])
        tunes = tunes_by_variant.get(variant.id, [])
        crowd_rows = _community_for(community, variant) if community else []
        evidence_cache = {}  # the two memory scenarios share the same speed evidence
        score = variant.scores.get(metric) if req.include_rankings else None
        quality = "Base-model reference; this quantisation has not been evaluated" if score is not None else "No mapped workload benchmark"
        contexts = sorted({req.context, *[n for n in [8192, 16384, 32768, 65536] if req.context < n <= variant.max_context]})
        fitted_any = False
        for scenario, budget in scenarios:
            gpu_budget = gpu_budgets[scenario](budget)
            for context in contexts:
                fits = placements(variant, context, req.users, budget, gpu_budget, gpu is not None, kv, unified)
                if not fits:
                    rejected["memory"] += 1
                    continue
                fitted_any = True
                for layers, moe, memory in fits:
                    gpu_uuid = gpu["uuid"] if layers else None
                    key = (context, layers, moe)
                    if key not in evidence_cache:
                        evidence_cache[key] = _evidence(variant, hardware, records, tunes, crowd_rows, calibration, efficiency,
                                                        req, context, layers, moe, gpu, gpu_uuid, threads, unified, hide,
                                                        community_evidence, use_default_community)
                    launch, measurement, tune, tps, interpolated, crowd, speed_estimate, evidence = evidence_cache[key]
                    launch = dict(launch)
                    meets_speed = tps >= req.min_tps if tps is not None else None
                    if req.strict_speed and meets_speed is not True:
                        rejected["speed"] += 1
                        continue
                    badge, sentence = verdict(evidence, req.min_tps, tps, interpolated, crowd, speed_estimate, req.users, tune)
                    mode = "cpu" if layers == 0 else "gpu" if layers == variant.layers and not moe else "split"
                    key = (variant.id, scenario, layers, moe)
                    if key not in max_context_cache:
                        max_context_cache[key] = maximum_context(variant, req.users, layers, budget, gpu_budget, kv, moe, unified)
                    suffix = (f"|kv:{kv}" if kv != "f16" else "") + (f"|moe:{moe}" if moe else "")
                    placement = {"cpu": "runs on the processor only", "gpu": "runs fully on the graphics chip",
                                 "split": "split between the graphics chip and the processor"}[mode]
                    if moe:
                        placement = "runs on the graphics chip with some expert weights kept in main memory"
                    results.append({"id": f"{variant.id}|{scenario}|{context}|{layers}{suffix}", "variant_id": variant.id,
                                    "name": variant.name, "quant": variant.quant, "filename": variant.filename,
                                    "repo": variant.repo, "base_repo": variant.base_repo, "revision": variant.revision, "demo": variant.demo,
                                    "scenario": scenario, "mode": mode, "gpu_layers": layers, "total_layers": variant.layers,
                                    "gpu_index": req.gpu_index if layers else None, "context": context, "users": req.users,
                                    "runtime_gpu_layers": runtime_gpu_layers(layers, variant.layers),
                                    "kv_cache_type": kv, "n_cpu_moe": moe, "unified_memory": unified, "launch": launch,
                                    "memory_max_context": max_context_cache[key],
                                    "ram_bytes": memory["ram"], "vram_bytes": memory["vram"], "kv_bytes": memory["kv_total"],
                                    "memory_total_bytes": memory["total"],
                                    "ram_headroom_bytes": int(budget - memory["ram"]), "vram_headroom_bytes": int(gpu_budget - memory["vram"]),
                                    "quality_score": score, "quality_evidence": quality, "quality_metric": metric,
                                    "score_source": variant.score_source if req.include_rankings else None, "score_version": variant.score_version if req.include_rankings else None,
                                    "score_settings": variant.score_settings if req.include_rankings else None, "tps": tps, "speed_meets_target": meets_speed,
                                    "speed_evidence": {"measured": "Local synthetic generation benchmark; same context and configuration",
                                                       "tuned": "Local tuning run; same context and placement" if tps is not None
                                                       else "Local tuning run with a short test; not verified at this length",
                                                       "interpolated": "Interpolated from local tests at other context lengths; not verified at this length",
                                                       "community": "Community results from similar computers; not verified on this one"}.get(
                                                           evidence, "Rough estimate only; run a speed test to check it"
                                                           if evidence == "estimated" else "Unverified; run a speed test to check it"),
                                    "evidence": evidence, "verdict": badge, "verdict_text": sentence,
                                    "speed_interpolated": interpolated, "community": crowd, "tuned": tune,
                                    "speed_estimate": speed_estimate,
                                    "benchmark": measurement, "threads": launch["threads"], "metadata_date": variant.fetched_at,
                                    "file_bytes": variant.size_bytes,
                                    "explanation": f"Estimated memory fit at {context:,} tokens per user; {placement}. " +
                                    ("Quality ordering uses base-model evidence. " if score is not None else "Quality ranking unavailable. ") + sentence})
        if not fitted_any and kv == "f16":
            smaller = next((t for t in ["q8_0", "q4_0"] if any(placements(variant, req.context, req.users, b, gpu_budgets[s](b), gpu is not None, t, unified)
                                                            for s, b in scenarios)), None)
            if smaller:
                compressible.append((variant.name, smaller))
    # Scores from different index versions must never share a numerical ordering.
    versions = {r["score_version"] for r in results if r["quality_score"] is not None}
    comparable = len(versions) <= 1 and None not in versions
    comparison = add_quality_comparison(results, comparable)
    comparable = comparison["comparable"]
    def order(result):
        quality = result["quality_score"] if comparable and result["quality_score"] is not None else -1
        speed = next((v for v in [result["tps"], (result["speed_interpolated"] or {}).get("tps"),
                                  (result["community"] or {}).get("median_tps"), result["speed_estimate"].get("low_tps")] if v is not None), -1)
        verified = (result["speed_meets_target"] is True, result["speed_meets_target"] is not False)
        # A verified speed always outranks a guess: an estimate can never jump ahead of a real test.
        primary = ((quality, speed) if req.priority == "quality" else (verified[0], speed, quality) if req.priority == "speed"
                   else (*verified, quality))
        # Balanced: among equally good options at the asked length, the faster one (by its best evidence) first.
        return (*primary, result["scenario"] == "now", -abs(result["context"] - req.context), speed,
                -result["memory_total_bytes"])
    results.sort(key=order, reverse=True)
    # Show three distinct models first; fill remaining places with variant trade-offs.
    shortlist, seen = [], set()
    for result in results:
        if result["base_repo"] not in seen:
            shortlist.append(result["id"])
            seen.add(result["base_repo"])
        if len(shortlist) == 3:
            break
    configurations = {(r["base_repo"], r["quant"], r["mode"]) for r in results if r["id"] in shortlist}
    for result in results:
        if len(shortlist) == 3:
            break
        key = (result["base_repo"], result["quant"], result["mode"])
        if key not in configurations:
            shortlist.append(result["id"])
            configurations.add(key)
    notes = ["Memory fit is estimated, not a guarantee; rescan after freeing resources.",
             "Hardware-calibrated speed ranges are low-confidence estimates; only model benchmarks verify speed. Concurrent speed estimates are unavailable.",
             "Context includes prompt, conversation, reasoning and generated output. The KV cache (the model's short-term "
             f"notepad for this conversation) uses {KV_TEXT[kv]}.",
             "Memory maximum is not a validated usable-context or speed guarantee.",
             "Document ordering uses general intelligence as a proxy, not a long-context evaluation."]
    if compressible:
        names = ", ".join(sorted({name for name, _ in compressible})[:3])
        notes.append(f"{names} would fit with compressed conversation memory (KV cache q8_0 or q4_0: a smaller notepad "
                     "with a small quality cost). Turn on compressed notes to see it.")
    if any(r["n_cpu_moe"] for r in results):
        notes.append("\"Experts on CPU\" options keep part of a mixture-of-experts model in main memory "
                     "(llama.cpp --n-cpu-moe): slower than all-GPU, usually much faster than CPU only.")
    if efficiency:
        notes.append(f"Speed estimates are {efficiency['label']} on this computer.")
    notes.extend(notes_extra)
    return {"requirements": asdict(req), "hardware": hardware, "candidates": results, "shortlist": shortlist,
            "rejected": rejected, "quality_comparable": comparable, "quality_comparison": comparison, "reclaim_estimate_bytes": reclaim,
            "speed_adjustment": {k: efficiency[k] for k in ["factor", "spread", "n", "label"]} if efficiency else None,
            "notes": notes}


def add_quality_comparison(results, comparable):
    """Competition ranks over distinct eligible base models, never over quant/context copies."""
    models = {}
    for result in results:
        models.setdefault(result["base_repo"], set())
        if result["quality_score"] is not None:
            models[result["base_repo"]].add(result["quality_score"])
    # Conflicting reference scores for one model cannot form a fair comparison.
    comparable = comparable and all(len(scores) <= 1 for scores in models.values())
    rated = {model: next(iter(scores)) for model, scores in models.items() if len(scores) == 1}
    for result in results:
        score = result["quality_score"]
        rank = None
        behind = None
        beaten = None
        if comparable and score is not None:
            rank = 1 + sum(other > score for other in rated.values())
            behind = round(max(rated.values()) - score, 4)
            beaten = sum(other < score for other in rated.values())
        result["quality_comparison"] = {"rank": rank, "rated_models": len(rated), "eligible_models": len(models),
            "points_behind_best": behind, "models_below": beaten,
            "tied": sum(other == score for other in rated.values()) > 1 if rank else False,
            "reason": "incomparable" if not comparable else "missing" if score is None else "ranked"}
    return {"rated_models": len(rated), "eligible_models": len(models), "comparable": comparable,
            "scope": "Distinct models in eligible configurations, including speed-unverified options when allowed. Quantisations share a base-model reference rank."}
