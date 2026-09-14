"""Read-only hardware and process inspection. Unsupported GPU telemetry stays unknown."""
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess

import psutil

from .domain import now


def nvidia_gpus():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], ["NVIDIA telemetry unavailable. Only CPU configurations can be estimated; AMD, Intel and Apple GPU support is not implemented."]
    try:
        result = subprocess.run(
            [executable, "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=True,
        )
        gpus = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            index, uuid, name, total, free, utilization, driver = row
            gpus.append({"index": int(index), "uuid": uuid, "name": name, "total": int(float(total) * 1024**2),
                         "available": int(float(free) * 1024**2), "utilization": float(utilization) if utilization.isdigit() else None,
                         "driver": driver, "backend": "cuda"})
        return gpus, []
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], ["Could not read NVIDIA memory. GPU capacity is unknown, not zero."]


def scan(include_processes=True):
    memory = psutil.virtual_memory()
    gpus, warnings = nvidia_gpus()
    processes = []
    if include_processes:
        for proc in psutil.process_iter(["pid", "name", "memory_info", "username", "create_time"]):
            try:
                info = proc.info
                rss = info["memory_info"].rss
                if rss < 32 * 1024**2 or info["pid"] == os.getpid():
                    continue
                # USS excludes shared pages; deliberately do not substitute RSS when unavailable.
                try:
                    uss = getattr(proc.memory_full_info(), "uss", None)
                except (psutil.Error, OSError):
                    uss = None
                processes.append({"pid": info["pid"], "name": info["name"] or "Unknown",
                                  "rss": rss, "reclaimable": int(uss * 0.75) if uss is not None else None,
                                  "created": info["create_time"]})
            except (psutil.Error, OSError, AttributeError):
                continue
        processes.sort(key=lambda item: item["rss"], reverse=True)
    identity = {"cpu": platform.processor() or platform.machine(), "cores": psutil.cpu_count(),
                "ram": memory.total, "os": platform.platform(),
                "gpus": [{k: gpu[k] for k in ["uuid", "name", "driver"]} for gpu in gpus]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"timestamp": now(), "fingerprint": fingerprint, "cpu": identity["cpu"], "os": identity["os"],
            "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.15), "ram_total": memory.total,
            "ram_available": memory.available, "swap_used": psutil.swap_memory().used,
            "disk_free": shutil.disk_usage(os.path.expanduser("~")).free,
            "gpus": gpus, "processes": processes[:40], "warnings": warnings}
