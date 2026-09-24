"""Explicit downloads (via `downloads`) and local llama-bench execution. Never invoked by a scan."""
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .domain import GIB, now
from .downloads import DownloadRedirect, digest, download_variant  # noqa: F401 (re-exported for compatibility)
from .engine import allocations
from .hardware import scan


def download(variant, directory, progress=None, cancel=None, token=None):
    """Kept for existing callers; resumable, verified downloads live in `downloads`."""
    return download_variant(variant, directory, progress=progress, cancel=cancel, token=token)


def bench(variant, model_path, executable, context, layers, gpu_index=0, timeout=600):
    if variant.demo or not variant.sha256:
        raise ValueError("Cannot benchmark demo models or variants without a published SHA256")
    if not 256 <= context <= variant.max_context or not 0 <= layers <= variant.layers:
        raise ValueError("Context or GPU layer count exceeds model limits")
    if digest(model_path) != variant.sha256:
        raise ValueError("Local GGUF SHA256 does not match the selected variant")
    binary = shutil.which(executable) or (str(Path(executable).resolve()) if Path(executable).is_file() else None)
    if not binary:
        raise ValueError("llama-bench was not found; install llama.cpp or provide --executable")
    hardware = scan(False)
    gpu = next((g for g in hardware["gpus"] if g["index"] == gpu_index), None)
    if layers and not gpu:
        raise ValueError("Selected NVIDIA GPU is unavailable")
    memory = allocations(variant, context, 1, layers)
    if memory["ram"] > hardware["ram_available"] - 2 * GIB or (layers and memory["vram"] > gpu["available"] - 0.5 * GIB):
        raise ValueError("Current resources do not meet the conservative benchmark memory check; free resources or reduce context")
    threads = hardware.get("cores") or hardware["threads"] or 1
    env = os.environ.copy()
    if layers:
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    # Test 128 generated tokens near the requested context capacity, not an empty cache.
    runtime_layers = layers + 1 if layers == variant.layers else layers
    command = [binary, "-m", str(Path(model_path).resolve()), "-p", "0", "-n", "128", "-d", str(context - 128),
               "-ngl", str(runtime_layers), "-t", str(threads), "-ctk", "f16", "-ctv", "f16", "-r", "3", "-o", "json",
               "-dev", "CUDA0" if layers else "none"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=env, check=False)
    if completed.returncode:
        raise ValueError(f"llama-bench failed ({completed.returncode}): {completed.stderr[-1200:]}")
    try:
        rows = json.loads(completed.stdout)
        row = next(r for r in rows if r.get("n_prompt") == 0 and r.get("n_gen") == 128 and r.get("n_depth") == context - 128)
        tps = float(row["avg_ts"])
        if not math.isfinite(tps) or tps <= 0:
            raise ValueError("Invalid speed")
        if row.get("n_gpu_layers") != runtime_layers or row.get("n_threads") != threads or row.get("type_k") != "f16" or row.get("type_v") != "f16":
            raise ValueError("Runtime settings differ from requested settings")
    except (ValueError, TypeError, KeyError, StopIteration):
        raise ValueError("Unrecognised llama-bench output or settings; use a build supporting JSON output and --n-depth") from None
    return {"variant_id": variant.id, "sha256": variant.sha256, "fingerprint": hardware["fingerprint"], "timestamp": now(),
            "context": context, "users": 1, "gpu_layers": layers, "gpu_uuid": gpu["uuid"] if layers else None,
            "threads": threads, "tps": tps, "runtime_build": row.get("build_commit"), "raw": row,
            "note": "Synthetic generation benchmark; not TTFT, quality or concurrent throughput. Load changes can affect speed."}
