"""Conservative memory model and evidence-aware ranking; no invented quality/speed scores."""
from dataclasses import asdict
from datetime import datetime, timezone
import math

from .domain import GIB, Requirements, Variant


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


def recommend(variants, hardware, requirements, measurements=()):
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
                    score = variant.scores.get(metric)
                    quality = "Base-model reference; this quantisation has not been evaluated" if score is not None else "No mapped workload benchmark"
                    mode = "cpu" if layers == 0 else "gpu" if layers == variant.layers else "split"
                    results.append({"id": f"{variant.id}|{scenario}|{context}|{layers}", "variant_id": variant.id,
                                    "name": variant.name, "quant": variant.quant, "filename": variant.filename,
                                    "repo": variant.repo, "revision": variant.revision, "demo": variant.demo,
                                    "scenario": scenario, "mode": mode, "gpu_layers": layers, "total_layers": variant.layers,
                                    "gpu_index": req.gpu_index if layers else None, "context": context, "users": req.users,
                                    "runtime_gpu_layers": layers + 1 if layers == variant.layers else layers,
                                    "memory_max_context": maximum_context(variant, req.users, layers, budget, gpu_budget),
                                    "ram_bytes": memory["ram"], "vram_bytes": memory["vram"], "kv_bytes": memory["kv_total"],
                                    "ram_headroom_bytes": int(budget - memory["ram"]), "vram_headroom_bytes": int(gpu_budget - memory["vram"]),
                                    "quality_score": score, "quality_evidence": quality, "quality_metric": metric,
                                    "score_source": variant.score_source, "score_version": variant.score_version,
                                    "score_settings": variant.score_settings, "tps": tps, "speed_meets_target": meets_speed,
                                    "speed_evidence": "Local synthetic generation benchmark; same context and configuration" if measurement else "Unverified — benchmark this configuration",
                                    "benchmark": measurement, "threads": threads, "metadata_date": variant.fetched_at,
                                    "file_bytes": variant.size_bytes,
                                    "explanation": f"Estimated memory fit at {context:,} tokens per user; {mode} execution. " +
                                    ("Quality ordering uses base-model evidence. " if score is not None else "Quality ranking unavailable. ") +
                                    ("Speed target met in a local synthetic test." if measurement else "Speed target is not yet verified.")})
    # Scores from different index versions must never share a numerical ordering.
    versions = {r["score_version"] for r in results if r["quality_score"] is not None}
    comparable = len(versions) <= 1 and None not in versions
    results.sort(key=lambda r: (r["speed_meets_target"] is True,
                               r["quality_score"] if comparable and r["quality_score"] is not None else -1,
                               r["scenario"] == "now", -abs(r["context"] - req.context),
                               -(r["ram_bytes"] + r["vram_bytes"])), reverse=True)
    # Diverse shortlist; all feasible configurations remain inspectable.
    shortlist, seen = [], set()
    for result in results:
        key = (result["name"], result["quant"], result["mode"])
        if key not in seen:
            shortlist.append(result["id"])
            seen.add(key)
        if len(shortlist) == 5:
            break
    return {"requirements": asdict(req), "hardware": hardware, "candidates": results, "shortlist": shortlist,
            "rejected": rejected, "quality_comparable": comparable, "reclaim_estimate_bytes": reclaim,
            "notes": ["Memory fit is estimated, not a guarantee; rescan after freeing resources.",
                      "Generation speed is unknown until locally tested. Concurrency speed testing is not supported in v0.1.",
                      "Context includes prompt, conversation, reasoning and generated output. KV cache uses FP16.",
                      "Memory maximum is not a validated usable-context or speed guarantee.",
                      "Document ordering uses general intelligence as a proxy, not a long-context evaluation."]}
