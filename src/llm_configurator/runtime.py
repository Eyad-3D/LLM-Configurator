"""Explicit downloads and local llama-bench execution. Never invoked by a scan."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .domain import GIB, now
from .engine import allocations
from .hardware import scan


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


class DownloadRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("Model downloads require HTTPS redirects")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def download(variant, directory):
    if variant.demo or not variant.sha256:
        raise ValueError("A real model with a published SHA256 is required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(variant.filename).name
    if target.exists():
        if digest(target) == variant.sha256:
            return target
        raise ValueError("Destination already exists with a different hash; choose a different directory")
    if shutil.disk_usage(directory).free < variant.size_bytes + GIB:
        raise ValueError("Insufficient disk space for the model plus 1 GiB headroom")
    url = f"https://huggingface.co/{variant.repo}/resolve/{quote(variant.revision)}/{quote(variant.filename)}"
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    partial = target.with_suffix(target.suffix + ".part")
    try:
        with partial.open("xb") as output, build_opener(DownloadRedirect()).open(Request(url, headers=headers), timeout=60) as response:
            written = 0
            while chunk := response.read(4 * 1024**2):
                written += len(chunk)
                if written > variant.size_bytes:
                    raise ValueError("Download exceeded catalogue file size")
                output.write(chunk)
        if written != variant.size_bytes or digest(partial) != variant.sha256:
            raise ValueError("Downloaded model does not match the pinned size and SHA256")
        partial.rename(target)
    except FileExistsError:
        raise ValueError("A partial download already exists; inspect/remove it before retrying") from None
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return target


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
