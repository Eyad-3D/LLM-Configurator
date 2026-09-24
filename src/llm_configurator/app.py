"""Application orchestration shared by CLI and local interface.

Long operations are job functions `fn(progress, cancel)` for `jobs.JobManager`. Model files
come from the app's models folder, discovered model folders or CLI arguments, and llama.cpp
binaries from `runtime_install.binary`, never from browser input. Modules built by other
v0.4 workstreams are imported inside functions so this file loads without them.
"""
import inspect
from pathlib import Path
import secrets

from .catalogue import apply_scores, definitions, demo_variants
from .domain import Requirements, Variant, check_cancel, now
from .engine import recommend
from .hardware import scan
from .launch import from_candidate
from .storage import models_dir

WORKLOADS = ("general", "coding", "agentic", "documents")
TEST_KINDS = ("smoke", "speed", "full")
GOALS = ("generation", "balanced", "prompt")
# Launch keys a stored tune may change; context, users and the model file always come from the candidate.
TUNABLE = ("gpu_layers", "threads", "batch", "ubatch", "flash_attn", "cache_type_k", "cache_type_v", "n_cpu_moe")


def variants(store, demo=False):
    """Catalogue variants plus the user's own GGUFs (source "local"), which can be sized and run but never downloaded."""
    if demo:
        return demo_variants()
    result = [Variant(**v) for v in store.get("variants", [])]
    known = {v.id for v in result}
    local = []
    for record in store.get("local_variants", []):  # registered by `local --add` before v0.4's fix-up round
        try:
            local.append(Variant(**record["variant"]))
        except (KeyError, TypeError, ValueError):
            continue
    try:
        from . import discover
        local += getattr(discover, "local_variants", lambda store: [])(store)
    except ImportError:
        pass
    for variant in local:
        if variant.id not in known:
            result.append(variant)
            known.add(variant.id)
    return result


def find_variant(store, variant_id, demo=False):
    variant = next((v for v in variants(store, demo) if v.id == variant_id), None)
    if not variant:
        raise ValueError("Model ID not found. Run 'llm-config models' after refreshing metadata to see the exact IDs.")
    return variant


def _call(function, *args, **kwargs):
    """Call a module built by another workstream, leaving out keyword options its version does not take yet."""
    try:
        accepted = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return function(*args, **kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
        return function(*args, **kwargs)
    return function(*args, **{k: v for k, v in kwargs.items() if k in accepted})


def engine_extras(store, function=recommend):
    """community/tuned evidence for engines that accept it; older engines keep working."""
    try:
        accepted = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    extras = {"community": (store.get("community") or {}).get("records") or [], "tuned": store.get("tuned", [])}
    return {k: v for k, v in extras.items() if k in accepted}


def evaluate(store, payload, demo=False):
    requirements = Requirements(**payload)
    models = variants(store, demo)
    hardware = scan(include_processes=bool(requirements.reclaim_pids), process_ids=set(requirements.reclaim_pids))
    report = recommend(models, hardware, requirements, store.get("measurements", []), store.get("calibration"),
                       **({} if demo else engine_extras(store)))
    report["demo"] = demo
    report["catalogue_status"] = store.get("refresh_status")
    if not models:
        report["notes"].insert(0, "No catalogue data yet. Refresh model metadata or enable the fictional demo.")
    return report


def map_benchmark(store, base_repo, slug):
    entries = definitions(store)
    entry = next((e for e in entries if e["base_repo"] == base_repo), None)
    if not entry:
        raise ValueError("Model is not in the configured catalogue")
    cache = store.get("scores")
    if slug and (not cache or not any(i.get("slug") == slug for i in cache["data"])):
        raise ValueError("Unknown benchmark slug; refresh Artificial Analysis data first")
    from . import catalogue
    entry = catalogue.set_slug(store, base_repo, slug)  # locked, atomic write of the user's catalogue copy
    models = [Variant(**v) for v in store.get("variants", [])]
    for variant in models:
        if variant.base_repo == base_repo:
            variant.scores = {}
            variant.score_source = variant.score_version = variant.score_settings = None
            apply_scores(variant, entry, cache)
    store.put("variants", [v.to_dict() for v in models])
    return {"base_repo": base_repo, "aa_slug": slug}


# ---- Files and binaries -------------------------------------------------------------

def binary(store, name):
    from . import runtime_install
    return runtime_install.binary(store, name)


def local_model(store, variant, verify=True, progress=None, cancel=None):
    """A complete local copy of every file of this variant, or None. Never downloads.

    With verify=True a file is only returned when its SHA-256 matches the catalogue, so a
    same-name, same-size file with other contents is never mistaken for the model.
    """
    for record in store.get("local_variants", []):
        if record.get("variant", {}).get("id") == variant.id and Path(record.get("path", "")).is_file():
            return Path(record["path"])
    if variant.demo:
        return None
    from . import discover
    # discover also looks in the models folder, where downloads save every part flat (Q4_K_M/x.gguf -> x.gguf).
    found = _call(discover.find_for_variant, store, variant, verify=verify, progress=progress, cancel=cancel)
    return Path(found) if found else None


def require_model(store, variant, progress=None, cancel=None):
    path = local_model(store, variant, progress=progress, cancel=cancel)
    if not path:
        if variant.source == "local":
            raise ValueError(f"The file for {variant.name} {variant.quant} is no longer where it was found. "
                             "Run: llm-config local --scan")
        raise ValueError(f"{variant.name} {variant.quant} is not on this computer yet. Download it first.")
    return path


def download_plan(store, variant):
    from . import downloads
    plan = downloads.plan(variant, models_dir(store))
    plan.pop("directory", None)
    return {**plan, "local_copy": local_model(store, variant, verify=False) is not None}


def download_job(store, variant):
    """A model the user brought from their own disk is only ever reused, never fetched from the internet."""
    def run(progress, cancel):
        existing = local_model(store, variant, progress=_stage(progress, "verify"), cancel=cancel)
        if existing:
            return {"reused": True, "bytes": 0, "filename": existing.name, "variant_id": variant.id}
        if variant.source == "local":
            raise ValueError("This model came from a file on your computer, and the file is no longer there. "
                             "Put it back, or scan for models again.")
        from . import downloads
        path = downloads.download_variant(variant, models_dir(store), progress=progress, cancel=cancel)
        remember_download(store, variant, models_dir(store))
        return {"reused": False, "bytes": variant.size_bytes, "filename": Path(path).name, "variant_id": variant.id}
    return run


def remember_download(store, variant, directory):
    """downloads just checked every file's SHA-256; note it so the first test does not read the model again."""
    from . import discover
    remember = getattr(discover, "remember_hash", None)
    for f in variant.all_files():
        if f.get("sha256") and remember:
            try:
                remember(store, Path(directory) / Path(f["filename"]).name, f["sha256"])
            except (ValueError, OSError):
                pass  # only a speed-up: the file is simply checked again before its first use


def remove_download(store, variant):
    """Frees only this variant's files inside the app's models folder, never discovered copies elsewhere."""
    from . import downloads
    return downloads.remove_variant(variant, models_dir(store))


# ---- Candidates and launch settings -------------------------------------------------

def placement(candidate):
    layers, total = candidate.get("gpu_layers") or 0, candidate.get("total_layers")
    return candidate.get("mode") or ("cpu" if not layers else "gpu" if layers == total else "split")


def installed_backend(store):
    """The installed llama.cpp build's backend (cuda, vulkan, metal, cpu…), or None when unknown.

    It decides the GPU settings: a Vulkan build on an NVIDIA card must not get CUDA's. The last detect() result
    is reused while its llama-server still exists, like runtime_install.binary, so no process starts per launch."""
    try:
        cached = store.get("runtime") or {}
        server = (cached.get("binaries") or {}).get("llama-server") if cached.get("installed") else None
        if server and Path(server[0]).is_file():
            return cached.get("backend")
        from . import runtime_install
        return (runtime_install.detect(store) or {}).get("backend")
    except Exception:  # noqa: BLE001 - no runtime yet, or an unreadable one: fall back to the hardware's backend
        return None


def _gpu_uuid(candidate, hardware):
    if not candidate.get("gpu_layers") or candidate.get("gpu_index") is None:
        return None
    gpu = next((g for g in (hardware or {}).get("gpus") or [] if isinstance(g, dict) and g.get("index") == candidate["gpu_index"]), None)
    return gpu.get("uuid") if gpu else None


def _fits_now(store, variant, candidate, hardware, layers, moe):
    """True when the engine lists this placement as fitting in memory now, at the candidate's context."""
    kv = candidate.get("kv_cache_type") or "f16"
    requirements = Requirements(context=candidate["context"], min_tps=0, include_rankings=False, kv_cache_type=kv,
                                gpu_index=candidate.get("gpu_index") or 0)
    try:
        report = recommend([variant], hardware, requirements, store.get("measurements", []), store.get("calibration"))
    except Exception:  # noqa: BLE001 - unreadable hardware: treat the moved placement as not fitting
        return False
    return any(c["context"] == candidate["context"] and c["scenario"] == "now" and c["gpu_layers"] == layers
               and (c.get("n_cpu_moe") or 0) == moe for c in report["candidates"])


def tuned_changes(store, candidate, hardware, variant=None):
    """Launch settings from the saved tune that started from this candidate (engine.matching_tuned).

    The tune is matched on the candidate's own model, context, GPU layers, expert offload and notepad
    format, so a tune of another placement is never applied. When the tuner moved layers or experts, its
    settings belong to that new placement: they are used only together with it, and only if it still fits in
    memory now. Raises ValueError when there is no usable tune.
    """
    from .engine import matching_tuned
    if (candidate.get("users") or 1) != 1:
        raise ValueError("Tunes are made for one user at a time, so there is no saved tune for several users.")
    variant = variant or find_variant(store, candidate["variant_id"])
    kv, moe = candidate.get("kv_cache_type") or "f16", candidate.get("n_cpu_moe") or 0
    tune = matching_tuned(store.get("tuned", []), variant, hardware or {}, candidate["context"], candidate["gpu_layers"],
                          kv, moe, _gpu_uuid(candidate, hardware))
    if not tune:
        raise ValueError("No saved tune for this model, context, placement and notepad format yet. Run a tune first.")
    changes = dict(tune["settings"])
    if kv != "f16" and changes.get("flash_attn") == "off":
        changes.pop("flash_attn")  # llama.cpp needs flash attention for a compressed notepad
    if tune["same_placement"]:
        return changes
    moved, on_gpu = tune["best_placement"], bool(candidate["gpu_layers"])
    if moved["cache_type_k"] == kv and bool(moved["gpu_layers"]) == on_gpu and 0 <= (moved["gpu_layers"] or 0) <= variant.layers \
            and _fits_now(store, variant, candidate, hardware or {}, moved["gpu_layers"] or 0, moved["n_cpu_moe"]):
        return {**changes, "gpu_layers": moved["gpu_layers"] or 0, "n_cpu_moe": moved["n_cpu_moe"]}
    raise ValueError("The saved tune moved the model to a different split between the graphics chip and the processor, "
                     "and that no longer fits in free memory. Run a new tune.")


def launch_config(store, candidate, hardware, model_path=None, tuned=False, runtime_backend=None, **overrides):
    """Launch settings for a candidate. tuned=True adds the saved tune's settings (see tuned_changes).

    runtime_backend: the installed build's backend when the caller already knows it; otherwise it is looked up."""
    changes = tuned_changes(store, candidate, hardware) if tuned else {}
    backend = runtime_backend or installed_backend(store)
    return from_candidate(candidate, model_path, hardware, runtime_backend=backend, **{**changes, **overrides})


def candidate_for(store, variant, hardware, context=None, gpu_layers=None, kv_cache_type="f16"):
    """CLI: the engine's best-fitting candidate for one model, or a hand-built one for explicit --gpu-layers."""
    context = context or min(8192, variant.max_context)
    if type(context) is not int or not 256 <= context <= variant.max_context:
        raise ValueError(f"Context must be between 256 and {variant.max_context:,} tokens for this model")
    if gpu_layers is not None and not 0 <= gpu_layers <= variant.layers:
        raise ValueError(f"GPU layers must be between 0 and {variant.layers} for this model")
    requirements = Requirements(context=context, min_tps=0, include_rankings=False, kv_cache_type=kv_cache_type)
    report = recommend([variant], hardware, requirements, store.get("measurements", []), store.get("calibration"),
                       **engine_extras(store))
    matches = [c for c in report["candidates"] if c["context"] == context and c["scenario"] == "now"
               and c.get("kv_cache_type", kv_cache_type) == kv_cache_type]
    # Engines without KV compression support still describe placements; the notepad format is a launch setting.
    matches = [{"kv_cache_type": kv_cache_type, **c} for c in matches]
    if gpu_layers is None:
        if not matches:
            raise ValueError(f"{variant.name} {variant.quant} does not fit in free memory at {context:,} tokens. "
                             "Try a smaller --context, or choose --gpu-layers yourself.")
        return max(matches, key=lambda c: (c["gpu_layers"], -(c.get("n_cpu_moe") or 0)))
    chosen = next((c for c in matches if c["gpu_layers"] == gpu_layers), None)
    if chosen:
        return chosen
    gpu = next((g for g in hardware.get("gpus", []) if g.get("index") == requirements.gpu_index), None)
    if gpu_layers and not gpu:
        raise ValueError("No supported GPU was found. Use --gpu-layers 0 to run on the CPU.")
    return {"id": f"{variant.id}|manual|{context}|{gpu_layers}", "variant_id": variant.id, "name": variant.name,
            "quant": variant.quant, "filename": variant.filename, "context": context, "users": 1,
            "gpu_layers": gpu_layers, "total_layers": variant.layers, "gpu_index": requirements.gpu_index if gpu_layers else None,
            "threads": hardware.get("cores") or hardware.get("threads"), "kv_cache_type": kv_cache_type, "n_cpu_moe": 0,
            "mode": "cpu" if not gpu_layers else "gpu" if gpu_layers == variant.layers else "split", "scenario": "manual"}


def _stage(progress, step):
    """Forward progress with a `step` label so multi-part jobs read clearly."""
    return (lambda value: progress({**value, "step": step})) if progress else None


# ---- Test, tune, quality ------------------------------------------------------------

def test_job(store, variant, candidate, hardware, kind="full", tuned=False, min_tps=None):
    """min_tps: the speed the user asked for; below it the verdict is "works, but slowly"."""
    if kind not in TEST_KINDS:
        raise ValueError("Test kind must be smoke, speed or full")
    def run(progress, cancel):
        from . import testing
        path = require_model(store, variant, _stage(progress, "verify"), cancel)
        config = launch_config(store, candidate, hardware, path, tuned)
        server = binary(store, "llama-server")
        bench = binary(store, "llama-bench") if kind != "smoke" else None
        # Both naming styles so the testing module can read either.
        commands = {"server": server, "bench": bench, "llama-server": server, "llama-bench": bench}
        return _call(testing.run_tests, store, variant, config, commands, kind=kind, hardware=hardware, progress=progress,
                     cancel=cancel, min_tps=min_tps or None)
    return run


def tune_job(store, variant, candidate, hardware, budget_seconds=300, goal="generation"):
    if type(budget_seconds) is not int or not 60 <= budget_seconds <= 1800:
        raise ValueError("The time budget must be between 60 and 1800 seconds")
    if goal not in GOALS:
        raise ValueError("Goal must be generation, balanced or prompt")
    def run(progress, cancel):
        from . import tuner
        path = require_model(store, variant, _stage(progress, "verify"), cancel)
        base = launch_config(store, candidate, hardware, path)
        result = _call(tuner.tune, binary(store, "llama-bench"), variant, base, hardware, budget_seconds=budget_seconds,
                       goal=goal, progress=progress, cancel=cancel, allow_kv_compression=False)
        # Paths stay on this computer: the page and the store get the tuned settings, not the model's folder.
        best = {k: v for k, v in (result.get("best") or {}).items() if k != "model_path"}
        changes = {k: best[k] for k in TUNABLE if k in best and best[k] != base.get(k)}
        saved = result.get("stopped") != "cancelled"
        if saved:
            record, measurement = tune_records(result, best, variant, candidate, hardware, goal, budget_seconds)
            store.append("tuned", record)
            if measurement:
                store.append("measurements", measurement)
        return {**result, "best": best, "changes": changes, "saved": saved, "record_id": record["id"] if saved else None}
    return run


def tune_records(result, best, variant, candidate, hardware, goal=None, budget_seconds=None):
    """The pinned tuned record (FIXUPS.md) and, when the tune measured a speed, a §2.3 kind="tune" measurement.

    Tuned record: {**tune_result, variant_id, sha256, fingerprint, context, gpu_layers, n_cpu_moe, kv_cache_type,
    timestamp}. context, gpu_layers, n_cpu_moe and kv_cache_type describe the candidate the tune started from, as
    engine.matching_tuned reads them; where the tuner ended up is in `best`. The measurement says at which depth
    (tokens already in memory) it was taken, so the engine counts it as "tested" only when that was the full length.
    """
    settings = result.get("settings") if isinstance(result.get("settings"), dict) else {}
    depth = settings.get("depth") if type(settings.get("depth")) is int else result.get("depth")
    if type(depth) is int:
        # engine.matching_tuned reads the depth from `settings`; tuners that only report it at the top level
        # measured best_result at that depth.
        settings = {**settings, "depth": depth}
    record = {**result, "settings": settings, "best": best, "id": secrets.token_hex(6), "variant_id": variant.id, "sha256": variant.sha256,
              "fingerprint": hardware.get("fingerprint"), "context": candidate["context"],
              "gpu_layers": candidate["gpu_layers"], "n_cpu_moe": candidate.get("n_cpu_moe") or 0,
              "kv_cache_type": candidate.get("kv_cache_type") or "f16", "depth": depth,
              "runtime": result.get("runtime"), "runtime_build": result.get("runtime_build"), "timestamp": now(),
              "goal": goal or result.get("goal"), "users": candidate.get("users") or 1, "placement": placement(candidate),
              "budget_seconds": budget_seconds}
    speed = result.get("best_result") or {}
    if type(depth) is not int:  # the engine reads a record without depth as a full-length test
        return record, None
    if not isinstance(speed.get("tps"), (int, float)) or isinstance(speed.get("tps"), bool) or not speed["tps"] > 0:
        return record, None
    layers = best.get("gpu_layers", candidate["gpu_layers"]) or 0
    measurement = {"id": secrets.token_hex(6), "kind": "tune", "variant_id": variant.id, "sha256": variant.sha256,
                   "fingerprint": hardware.get("fingerprint"), "timestamp": record["timestamp"],
                   "context": candidate["context"], "users": best.get("parallel") or 1, "gpu_layers": layers,
                   "gpu_uuid": best.get("gpu_uuid") if layers else None,
                   "threads": result.get("threads") or best.get("threads"), "tps": speed["tps"], "pp_tps": speed.get("pp_tps"),
                   "depth": depth, "runtime": result.get("runtime"), "runtime_build": result.get("runtime_build"),
                   "settings": {k: best.get(k) for k in ["flash_attn", "cache_type_k", "cache_type_v", "batch", "ubatch", "n_cpu_moe"]},
                   "raw": {"tuned_record": record["id"]},
                   "note": "Measured on this computer with llama-bench while tuning, with the given depth of conversation "
                           "already in memory. Other programs running can change speed."}
    return record, measurement


def start_server(store, variant, candidate, hardware, tuned=False, progress=None, cancel=None):
    from . import llama_server
    path = require_model(store, variant, progress, cancel)
    config = launch_config(store, candidate, hardware, path, tuned)
    server = llama_server.LlamaServer(binary(store, "llama-server"), config)
    server.start(progress=progress, cancel=cancel)
    return server, config


def _quality_record(kind, variant, candidate, **values):
    return {"kind": kind, "variant_id": variant.id, "name": variant.name, "quant": variant.quant,
            "context": candidate.get("context"), "placement": placement(candidate), "timestamp": now(), **values}


def quiz_job(store, variant, candidate, hardware, workload="general", include_needle=False):
    if workload not in WORKLOADS:
        raise ValueError("Workload must be general, coding, agentic or documents")
    def run(progress, cancel):
        from . import evals
        server, config = start_server(store, variant, candidate, hardware, progress=_stage(progress, "start"), cancel=cancel)
        try:
            quiz = evals.run_quiz(server.chat, workload, progress=_stage(progress, "quiz"), cancel=cancel)
            needle = None
            if include_needle:
                check_cancel(cancel)
                try:
                    needle = evals.needle_test(server.chat, server.tokenize, config["context"], progress=_stage(progress, "needle"), cancel=cancel)
                except ValueError as error:  # keep the finished quiz; say why the recall test did not run
                    needle = {"kind": "needle", "error": str(error), "note": f"The long-document test could not run: {error}"}
            # Pinned shape (FIXUPS.md): one record per quiz run, the needle result inside it.
            store.append("quality_results", _quality_record(
                "quiz", variant, candidate, candidate_id=candidate.get("id"), workload=workload,
                **{k: quiz.get(k) for k in ["score", "ci_low", "ci_high", "correct", "total"]}, needle=needle, result=quiz))
            return {"quiz": quiz, "needle": needle}
        finally:
            server.stop()
    return run


class _Chat:
    """One model's chat session for a blind comparison: callable, and a context manager that stops the server."""
    def __init__(self, open_server):
        self._open, self._server = open_server, None

    def _live(self):
        if self._server is None:
            self._server = self._open()
        return self._server

    def __call__(self, messages, max_tokens=512, **options):
        return self._live().chat(messages, max_tokens=max_tokens, **options)

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._server is not None:
            self._server.stop()
            self._server = None

    stop = close


def compare_job(store, entries, hardware, prompts, max_tokens=512):
    """entries: [(label, variant, candidate)]. Models run one after another because of memory.

    The comparison is keyed by candidate id with the labels as friendly names, so evals.reveal returns
    candidate ids in `mapping`/`tallies` and the names in `labels` (FIXUPS.md reveal shape)."""
    def run(progress, cancel):
        from . import evals
        for _, variant, _ in entries:
            require_model(store, variant, _stage(progress, "verify"), cancel)  # fail before loading anything
        sessions = []
        def factory(variant, candidate):
            def make():
                session = _Chat(lambda: start_server(store, variant, candidate, hardware, cancel=cancel)[0])
                sessions.append(session)
                return session
            return make
        runners, names = {}, {}
        for label, variant, candidate in entries:
            key = candidate.get("id") or label
            key = key if key not in runners else f"{key}#{len(runners) + 1}"  # the same candidate twice still gets two slots
            runners[key], names[key] = factory(variant, candidate), label
        try:
            comparison_id = _call(evals.run_comparison, store, prompts, runners, max_tokens=max_tokens, progress=progress,
                                  cancel=cancel, names=names)
        finally:
            for session in sessions:
                session.close()
        return {"comparison_id": comparison_id}
    return run


def _without_paths(value):
    """Results kept in the store reach the browser page; folder names stay on this computer."""
    if isinstance(value, dict):
        return {k: _without_paths(v) for k, v in value.items() if k != "path"}
    if isinstance(value, list):
        return [_without_paths(v) for v in value]
    return value


def _check_config(store, variant, hardware, path):
    """Placement for llama-perplexity: as many GPU layers as fit at its short context, else the CPU.

    None (llama.cpp's own default) when the hardware cannot be read."""
    context = min(512, variant.max_context)
    try:
        try:
            candidate = candidate_for(store, variant, hardware, context=context)
        except ValueError:
            candidate = candidate_for(store, variant, hardware, context=context, gpu_layers=0)
        return launch_config(store, candidate, hardware, path)
    except (ValueError, KeyError, TypeError):
        return None


def quant_check_job(store, reference, others, hardware=None):
    """Compression check of `others` against `reference` with llama-perplexity (quantcheck.kl_check).

    Results are keyed by quant label; `variant_ids` maps each label back to its model ID. Nothing stored
    or returned holds a folder name. Temporary files go to the app's data folder, not the system's
    temporary folder (on Linux often held in memory)."""
    def run(progress, cancel):
        from . import quantcheck
        reference_path = require_model(store, reference, _stage(progress, "verify"), cancel)
        labels, paths = {}, {}
        for variant in others:
            path = require_model(store, variant, _stage(progress, "verify"), cancel)
            label = variant.quant or quantcheck.quant_label(path)
            label = label if label not in labels and label != (reference.quant or quantcheck.quant_label(reference_path)) else variant.id
            labels[label], paths[label] = variant, str(path)
        reference_label = reference.quant or quantcheck.quant_label(reference_path)
        config = _check_config(store, reference, hardware or scan(False), reference_path)
        result = _without_paths(_call(quantcheck.kl_check, binary(store, "llama-perplexity"), str(reference_path), paths,
                                      config=config, progress=progress, cancel=cancel, reference_label=reference_label,
                                      work_dir=str(store.directory / "tmp")))
        variant_ids = {label: v.id for label, v in labels.items()}
        by_variant = {variant_ids[label]: value for label, value in (result.get("results") or {}).items() if label in labels}
        # Pinned shape (FIXUPS.md): {kind, variant_id: <reference>, result, timestamp}; the rest are extra keys.
        store.append("quality_results", {"kind": "quant_check", "variant_id": reference.id, "result": result, "timestamp": now(),
                                         "reference_variant_id": reference.id, "name": reference.name,
                                         "reference_quant": reference_label, "variant_ids": variant_ids,
                                         "results": by_variant, "notes": result.get("notes", [])})
        return {**result, "reference_quant": reference_label, "reference_variant_id": reference.id, "variant_ids": variant_ids}
    return run


# ---- Export, runtime, local files -----------------------------------------------------

def export_config(store, variant, candidate, hardware, fmt, platform="posix", tuned=False, port=None):
    """port: the port of the server already running (it may have moved off the default when that was taken)."""
    from . import export
    if fmt not in {f["id"] for f in export.formats()}:
        raise ValueError("Unknown export format")
    if platform not in {"posix", "windows"}:
        raise ValueError("Platform must be posix or windows")
    path = local_model(store, variant, verify=False)
    config = launch_config(store, candidate, hardware, path or models_dir(store) / Path(variant.all_files()[0]["filename"]).name, tuned,
                           **({"port": port} if port is not None else {}))
    try:
        command = binary(store, "llama-server")
    except ValueError:
        command = None
    result = export.export(config, variant, fmt, platform=platform, server_command=command)
    if not path:
        result.setdefault("notes", []).insert(0, "The model is not downloaded yet. The file path shown is where it will be saved after you download it.")
    return result


def runtime_install_job(store, allow_unverified=False):
    def run(progress, cancel):
        from . import runtime_install
        return runtime_install.install(store, scan(False), progress=progress, cancel=cancel, allow_unverified=allow_unverified)
    return run


def page_file(record):
    """A local file record without its folder, for the browser page."""
    path = Path(record.get("path") or "")
    return {"filename": path.name, "size_bytes": record.get("size_bytes"), "sha256": record.get("sha256"),
            "source": record.get("source"), "variant_id": record.get("variant_id"), "gguf": record.get("gguf"),
            "mtime": record.get("mtime"), "verified": bool(record.get("verified")),
            "match_note": record.get("match_note"), "local_variant_id": record.get("local_variant_id")}


def page_location(location):
    """Folder labels relative to the home folder, never absolute paths."""
    path, home = Path(location.get("path") or ""), Path.home()
    try:
        label = "~/" + path.relative_to(home).as_posix()
    except ValueError:
        label = path.name or "folder"
    return {"source": location.get("source"), "exists": bool(location.get("exists")), "label": label}


def scan_job(store, extra_dirs=()):
    def run(progress, cancel):
        from . import discover
        return discover.scan(store, extra_dirs=tuple(extra_dirs), progress=progress, cancel=cancel)
    return run


def add_local_job(store, path):
    """Register any GGUF on disk so it can be tested, tuned and served like a catalogue model.

    Any part of a split model selects the whole model (llama.cpp loads the first part). The file is
    hashed first, so a catalogue copy is recognised exactly and a local model gets a stable ID. Files
    without a .gguf name (Ollama blobs) are accepted when their contents are GGUF.
    """
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise ValueError(f"{path.name} was not found. Check the path and try again.")
    def run(progress, cancel):
        from . import discover, gguf
        parts = gguf.shard_paths(path)
        for part in parts:
            discover.hash_cached(store, part, progress=progress, cancel=cancel)
        record = discover.add_file(store, parts[0])
        return {**record, "model_id": record.get("variant_id") or record.get("local_variant_id")}
    return run

