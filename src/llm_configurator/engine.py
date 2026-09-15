"""Conservative memory model and evidence-aware ranking; no invented quality/speed scores."""
from dataclasses import asdict
from datetime import datetime, timezone
import math

from .domain import GIB, Requirements, Variant
from .speed import estimate


def allocations(variant, context, users, gpu_layers):
    if not 0 <= gpu_layers <= variant.layers:
        raise ValueError("GPU layer allocation out of range")
    # Full-attention FP16 K + V, per sequence. No assumed prefix sharing.
    kv = 2 * variant.layers * variant.kv_heads * variant.head_dim * 2 * context * users
    fraction = gpu_layers / variant.layers
    # Whole-file placement is approximate; 10% placement/metadata margin protects the estimate.
    weights = variant.size_bytes * 1.10
    buffer = (0.75 + users * 0.20) * GIB
    ram = weights * (1 - fraction) + kv * (1 - fraction) + buffer
    vram = 0
    if gpu_layers:
        vram = weights * fraction + kv * fraction + buffer
        ram += 0.5 * GIB  # staging; model files are memory-mapped, not duplicated in full
    return {"ram": math.ceil(ram), "vram": math.ceil(vram), "kv_total": kv}


def maximum_context(variant, users, layers, ram_budget, vram_budget):
    low, high, best = 256, variant.max_context, 0
    while low <= high:
        context = (low + high) // 2
        use = allocations(variant, context, users, layers)
        if use["ram"] <= ram_budget and use["vram"] <= vram_budget:
            best, low = context, context + 1
        else:
            high = context - 1
    return best // 256 * 256


def matching_speed(records, variant, hardware, context, layers, gpu_uuid, threads):
    for record in reversed(records):
        if (record.get("variant_id") != variant.id or not variant.sha256 or record.get("sha256") != variant.sha256
                or record.get("fingerprint") != hardware["fingerprint"] or record.get("context") != context
                or record.get("gpu_layers") != layers or record.get("gpu_uuid") != gpu_uuid
                or record.get("threads") != threads or record.get("users") != 1):
            continue
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["timestamp"])).total_seconds()
            if 0 <= age < 30 * 86400 and math.isfinite(record["tps"]) and record["tps"] > 0:
                return record
        except (KeyError, ValueError, TypeError):
            continue
    return None


def recommend(variants, hardware, requirements, measurements=(), calibration=None):
    req = requirements
    ram_budget = max(0, hardware["ram_available"] - req.reserve_gib * GIB)
    selected = set(req.reclaim_pids)
    reclaim = sum(p.get("reclaimable") or 0 for p in hardware.get("processes", []) if p["pid"] in selected)
    scenarios = [("now", ram_budget)]
    if reclaim:
        scenarios.append(("after_closing", min(hardware["ram_total"] - req.reserve_gib * GIB, ram_budget + reclaim)))
    gpu = next((g for g in hardware["gpus"] if g["index"] == req.gpu_index), None)
    gpu_budget = max(0, gpu["available"] - req.gpu_reserve_gib * GIB) if gpu else 0
    threads = hardware.get("cores") or hardware.get("threads") or 1
    results, rejected = [], {"context": 0, "memory": 0, "speed": 0}
    for variant in variants:
        if req.context > variant.max_context:
            rejected["context"] += 1
            continue
        contexts = sorted({req.context, *[n for n in [8192, 16384, 32768, 65536] if req.context < n <= variant.max_context]})
        for scenario, budget in scenarios:
            for context in contexts:
                # Enumerate each layer, then retain CPU, maximal fitting split, and full GPU candidates.
                fits = []
                for layers in range(variant.layers + 1) if gpu else [0]:
                    memory = allocations(variant, context, req.users, layers)
                    if memory["ram"] <= budget and memory["vram"] <= gpu_budget:
                        fits.append((layers, memory))
                if not fits:
                    rejected["memory"] += 1
                    continue
                choices = {0, variant.layers, max((l for l, _ in fits if l < variant.layers), default=0)}
                for layers, memory in fits:
                    if layers not in choices:
                        continue
                    gpu_uuid = gpu["uuid"] if layers else None
                    measurement = matching_speed(measurements, variant, hardware, context, layers, gpu_uuid, threads) if req.users == 1 else None
                    tps = measurement["tps"] if measurement else None
                    meets_speed = tps >= req.min_tps if tps is not None else None
                    if meets_speed is False or (req.strict_speed and meets_speed is not True):
                        rejected["speed"] += 1
                        continue
                    metric = req.workload if req.workload != "documents" else "general"
                    score = variant.scores.get(metric) if req.include_rankings else None
                    quality = "Base-model reference; this quantisation has not been evaluated" if score is not None else "No mapped workload benchmark"
                    mode = "cpu" if layers == 0 else "gpu" if layers == variant.layers else "split"
                    results.append({"id": f"{variant.id}|{scenario}|{context}|{layers}", "variant_id": variant.id,
                                    "name": variant.name, "quant": variant.quant, "filename": variant.filename,
                                    "repo": variant.repo, "base_repo": variant.base_repo, "revision": variant.revision, "demo": variant.demo,
                                    "scenario": scenario, "mode": mode, "gpu_layers": layers, "total_layers": variant.layers,
                                    "gpu_index": req.gpu_index if layers else None, "context": context, "users": req.users,
                                    "runtime_gpu_layers": layers + 1 if layers == variant.layers else layers,
                                    "memory_max_context": maximum_context(variant, req.users, layers, budget, gpu_budget),
                                    "ram_bytes": memory["ram"], "vram_bytes": memory["vram"], "kv_bytes": memory["kv_total"],
                                    "ram_headroom_bytes": int(budget - memory["ram"]), "vram_headroom_bytes": int(gpu_budget - memory["vram"]),
                                    "quality_score": score, "quality_evidence": quality, "quality_metric": metric,
                                    "score_source": variant.score_source if req.include_rankings else None, "score_version": variant.score_version if req.include_rankings else None,
                                    "score_settings": variant.score_settings if req.include_rankings else None, "tps": tps, "speed_meets_target": meets_speed,
                                    "speed_evidence": "Local synthetic generation benchmark; same context and configuration" if measurement else "Unverified — benchmark this configuration",
                                    "speed_estimate": estimate(variant, hardware, calibration, context, layers, req.users, gpu, req.min_tps),
                                    "benchmark": measurement, "threads": threads, "metadata_date": variant.fetched_at,
                                    "file_bytes": variant.size_bytes,
                                    "explanation": f"Estimated memory fit at {context:,} tokens per user; {mode} execution. " +
                                    ("Quality ordering uses base-model evidence. " if score is not None else "Quality ranking unavailable. ") +
                                    ("Speed target met in a local synthetic test." if measurement else "Speed target is not yet verified.")})
    # Scores from different index versions must never share a numerical ordering.
    versions = {r["score_version"] for r in results if r["quality_score"] is not None}
    comparable = len(versions) <= 1 and None not in versions
    comparison = add_quality_comparison(results, comparable)
    comparable = comparison["comparable"]
    def order(result):
        quality = result["quality_score"] if comparable and result["quality_score"] is not None else -1
        speed = result["tps"] if result["tps"] is not None else result["speed_estimate"].get("low_tps", -1)
        primary = (quality, speed) if req.priority == "quality" else (speed, quality) if req.priority == "speed" else (result["speed_meets_target"] is True, quality)
        return (*primary, result["scenario"] == "now", -abs(result["context"] - req.context),
                -(result["ram_bytes"] + result["vram_bytes"]))
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
    return {"requirements": asdict(req), "hardware": hardware, "candidates": results, "shortlist": shortlist,
            "rejected": rejected, "quality_comparable": comparable, "quality_comparison": comparison, "reclaim_estimate_bytes": reclaim,
            "notes": ["Memory fit is estimated, not a guarantee; rescan after freeing resources.",
                      "Hardware-calibrated speed ranges are low-confidence estimates; only model benchmarks verify speed. Concurrent speed estimates are unavailable.",
                      "Context includes prompt, conversation, reasoning and generated output. KV cache uses FP16.",
                      "Memory maximum is not a validated usable-context or speed guarantee.",
                      "Document ordering uses general intelligence as a proxy, not a long-context evaluation."]}


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
