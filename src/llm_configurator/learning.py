"""Learn how far this machine's real speed tests sit from the calibration estimate.

Only local measurements on this exact hardware count, at least two are required, and the
result only narrows *estimates*; it never turns an estimate into a verified speed.
"""
from datetime import datetime, timezone
import math
import statistics

from .speed import estimate

MIN_MEASUREMENTS = 2
MAX_AGE_DAYS = 90
MIN_SPREAD = 0.10  # never claim better than about ±10% from a handful of tests


def _usable(record, hardware):
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["timestamp"])).total_seconds()
        tps = record["tps"]
        return (record.get("fingerprint") == hardware.get("fingerprint") and (record.get("users") or 1) == 1
                and 0 <= age < MAX_AGE_DAYS * 86400 and isinstance(tps, (int, float)) and not isinstance(tps, bool)
                and math.isfinite(tps) and tps > 0)
    except (KeyError, TypeError, ValueError):
        return False


def _fit(ratios):
    logs = [math.log(r) for r in ratios]
    centre = statistics.median(logs)
    # Median absolute deviation (in log space) resists one odd test; scale 1.5 ≈ a broad band.
    mad = statistics.median(abs(x - centre) for x in logs)
    spread = max(MIN_SPREAD, 1.5 * mad)
    return {"factor": math.exp(centre), "spread": round(spread, 4), "n": len(ratios)}


def ratios(measurements, hardware, calibration, variants=()):
    """(mode, measured / estimate-midpoint) for every usable measurement with an estimate."""
    by_id = {v.id: v for v in variants}
    gpus = {g.get("uuid"): g for g in hardware.get("gpus", [])}
    result = []
    for record in measurements or ():
        variant = by_id.get(record.get("variant_id"))
        if variant is None or not _usable(record, hardware):
            continue
        if variant.sha256 and record.get("sha256") and record["sha256"] != variant.sha256:
            continue
        settings = record.get("settings") or {}
        layers = record.get("gpu_layers") or 0
        gpu = gpus.get(record.get("gpu_uuid"))
        try:
            kv = settings.get("cache_type_k") or "f16"
            moe = settings.get("n_cpu_moe") or 0
            unified = bool(gpu and gpu.get("unified"))
            guess = estimate(variant, hardware, calibration, record["context"], layers, 1, gpu, 0,
                             kv_cache_type=kv, n_cpu_moe=moe, unified=unified)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        # The unrounded midpoint: rounded ranges of very slow models could be 0.0.
        if not guess.get("available") or not guess.get("mid_tps", 0) > 0:
            continue
        moved = moe and variant.moe and layers
        mode = "cpu" if layers == 0 else "gpu" if layers >= variant.layers and not moved else "split"
        result.append((mode, record["tps"] / guess["mid_tps"]))
    return result


def fit_efficiency(measurements, hardware, calibration, variants=()):
    """Efficiency multipliers for `speed.estimate`, or None with fewer than two usable tests.

    Returns {"factor", "spread", "n", "label", "by_mode": {"cpu"|"gpu"|"split": {"factor", "spread", "n"}}}.
    `factor` scales the estimate's midpoint; the new range is midpoint·factor·e^(±spread).
    A mode-specific fit is used when that placement has two or more tests of its own.
    """
    found = ratios(measurements, hardware, calibration, variants)
    if len(found) < MIN_MEASUREMENTS:
        return None
    overall = _fit([r for _, r in found])
    by_mode = {}
    for mode in {m for m, _ in found}:
        values = [r for m, r in found if m == mode]
        if len(values) >= MIN_MEASUREMENTS:
            by_mode[mode] = _fit(values)
    return {**overall, "label": f"adjusted from {overall['n']} local measurements", "by_mode": by_mode}
