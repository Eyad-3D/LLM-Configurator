"""Explicit downloads (via `downloads`) and local llama-bench execution. Never invoked by a scan."""
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess

from .domain import GIB, now
from .downloads import DownloadRedirect, digest, download_variant  # noqa: F401 (re-exported for compatibility)
from .engine import allocations
from .hardware import scan
from .runtime_install import list_devices, pick_device


def download(variant, directory, progress=None, cancel=None, token=None):
    """Kept for existing callers; resumable, verified downloads live in `downloads`."""
    return download_variant(variant, directory, progress=progress, cancel=cancel, token=token)


def _command(executable):
    """argv prefix for llama-bench: a list is used as given (tests, managed installs), a string is looked up."""
    if isinstance(executable, (list, tuple)) and executable:
        return [str(part) for part in executable]
    binary = shutil.which(executable) or (str(Path(executable).resolve()) if Path(executable).is_file() else None)
    if not binary:
        raise ValueError("llama-bench was not found; install llama.cpp or provide --executable")
    return [binary]


def gpu_placement(command, gpu, env=None):
    """How to point llama.cpp at one scanned GPU: (extra args, env, devices). NVIDIA is pinned by UUID through
    CUDA_VISIBLE_DEVICES (then it is the only CUDA device). The `-dev` name always comes from the build's own
    `--list-devices`, so a Vulkan, ROCm or Metal build never gets a made-up name like CUDA0 (llama.cpp exits on an
    unknown name). When the build cannot list devices (older builds), `-dev` is left out: llama.cpp's default
    `auto` then uses the visible GPU(s)."""
    env = dict(os.environ if env is None else env)
    backend = gpu.get("backend") or "cuda"    # scans before v0.4 only listed NVIDIA cards
    if backend == "cuda" and str(gpu.get("uuid") or "").startswith("GPU-"):
        env["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
    devices = list_devices(command, env)
    if devices is None:
        return [], env, None
    name = gpu.get("name") or "the selected GPU"
    if not devices:
        raise ValueError(f"This llama.cpp build sees no graphics card it can use, so it cannot put layers on {name}. "
                         "It may be a CPU-only build, or the graphics driver may be missing. Install the build that "
                         "matches your graphics card, or benchmark with 0 GPU layers.")
    device = pick_device(devices, dict(gpu, backend=backend))
    if not device:
        seen = ", ".join(f"{d['name']} ({d['description']})" for d in devices)
        raise ValueError(f"This llama.cpp build cannot tell which of its graphics devices is {name} (it sees: {seen}). "
                         "Benchmark with 0 GPU layers, or pick the GPU with a llama.cpp build for that card only.")
    return ["-dev", device], env, devices


def bench(variant, model_path, executable, context, layers, gpu_index=0, timeout=600):
    """Measure generation speed near a full context with llama-bench. `executable` is a path/name or an
    argv prefix (`command: str | list[str]`)."""
    if variant.demo or not variant.sha256:
        raise ValueError("Cannot benchmark demo models or variants without a published SHA256")
    if not 256 <= context <= variant.max_context or not 0 <= layers <= variant.layers:
        raise ValueError("Context or GPU layer count exceeds model limits")
    if digest(model_path) != variant.sha256:
        raise ValueError("Local GGUF SHA256 does not match the selected variant")
    command = _command(executable)
    hardware = scan(False)
    gpu = next((g for g in hardware["gpus"] if g["index"] == gpu_index), None)
    if layers and not gpu:
        raise ValueError("The selected GPU is unavailable")
    memory = allocations(variant, context, 1, layers, unified=bool(layers and gpu.get("unified")))
    vram_free = gpu.get("available") if layers else None
    if memory["ram"] > hardware["ram_available"] - 2 * GIB or (vram_free is not None and memory["vram"] > vram_free - 0.5 * GIB):
        raise ValueError("Current resources do not meet the conservative benchmark memory check; free resources or reduce context")
    threads = hardware.get("cores") or hardware["threads"] or 1
    if layers:
        device_args, env, devices = gpu_placement(command, gpu)
    else:
        # `none` is the one device name every build accepts: nothing is offloaded.
        device_args, env, devices = ["-dev", "none"], os.environ.copy(), None
    # Test 128 generated tokens near the requested context capacity, not an empty cache.
    runtime_layers = layers + 1 if layers == variant.layers else layers
    args = command + ["-m", str(Path(model_path).resolve()), "-p", "0", "-n", "128", "-d", str(context - 128),
                      "-ngl", str(runtime_layers), "-t", str(threads), "-ctk", "f16", "-ctv", "f16", "-r", "3", "-o", "json"]
    args += device_args
    completed = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, check=False,
                               errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
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
    # llama-bench has no --version; its rows carry the build. "b<N>" matches runtime_install.detect()["version"].
    number = row.get("build_number")
    version = f"b{number}" if isinstance(number, int) and not isinstance(number, bool) else row.get("build_commit")
    chosen = device_args[1] if len(device_args) == 2 else None
    backend = next((d["backend"] for d in devices or [] if d["name"] == chosen), None) if layers else "cpu"
    return {"variant_id": variant.id, "sha256": variant.sha256, "fingerprint": hardware["fingerprint"], "timestamp": now(),
            "context": context, "users": 1, "gpu_layers": layers, "gpu_uuid": gpu["uuid"] if layers else None,
            "threads": threads, "tps": tps, "runtime_build": version, "raw": row,
            "kind": "bench", "id": secrets.token_hex(6), "depth": context - 128,
            "settings": {"flash_attn": "auto", "cache_type_k": "f16", "cache_type_v": "f16", "batch": row.get("n_batch"),
                         "ubatch": row.get("n_ubatch"), "n_cpu_moe": 0},
            "runtime": {"version": version, "backend": backend},
            "note": "Synthetic generation benchmark; not TTFT, quality or concurrent throughput. Load changes can affect speed."}
