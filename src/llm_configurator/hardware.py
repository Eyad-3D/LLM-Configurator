"""Read-only hardware and process inspection. Unknown GPU telemetry stays unknown, never zero or guessed.

GPUs whose free memory can be read (NVIDIA via nvidia-smi, AMD via sysfs, Apple Silicon unified memory) go
in `gpus` with integer byte counts. GPUs we can name but whose free memory we cannot read go in `other_gpus`
so nothing that does arithmetic on `gpus[i]["available"]` ever sees a made-up number.
"""
import csv
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys

import psutil

from .domain import GIB, now

MIB = 1024**2
TIMEOUT = 2
PCI_VENDORS = {"0x10de": "nvidia", "0x1002": "amd", "0x8086": "intel"}
VENDOR_NAMES = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}
DRM_ROOT = "/sys/class/drm"
WINDOWS_GPU_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
WINDOWS_CPU_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
_cache = {}


def _cached(key, fn):
    """Static facts (names, chip, features) do not change while the app runs; read them once per process."""
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def clear_cache():
    _cache.clear()


def run(argv, timeout=TIMEOUT):
    """Run a read-only tool. Returns stdout or None; never raises."""
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False, **kwargs)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 or result.stdout else None


def nvidia_gpus():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return [], []
    try:
        kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        result = subprocess.run(
            [executable, "--query-gpu=index,uuid,name,memory.total,memory.free,utilization.gpu,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=True, **kwargs,
        )
        gpus = []
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            index, uuid, name, total, free, utilization, driver = row
            gpus.append({"index": int(index), "uuid": uuid, "name": name, "total": int(float(total) * 1024**2),
                         "available": int(float(free) * 1024**2), "utilization": float(utilization) if utilization.isdigit() else None,
                         "driver": driver, "backend": "cuda", "vendor": "nvidia", "unified": False})
        return gpus, []
    except (OSError, ValueError, subprocess.SubprocessError):
        return [], ["Could not read NVIDIA memory. GPU capacity is unknown, not zero."]


# ---------- small readers ----------

def _read(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _int(text):
    try:
        return int(str(text).strip())
    except (TypeError, ValueError):
        return None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "unknown"


def sysctl(names):
    """Read macOS sysctl values as {name: text}. Unknown keys are simply absent."""
    executable = shutil.which("sysctl") or "/usr/sbin/sysctl"
    output = run([executable, *names])
    values = {}
    for line in (output or "").splitlines():
        name, sep, value = line.partition(":")
        if sep and name.strip() in names:
            values[name.strip()] = value.strip()
    return values


def _flag(values, key):
    return None if key not in values else values[key] == "1"


# ---------- CPU name and features ----------

def _linux_cpu(cpuinfo_path="/proc/cpuinfo"):
    text = _read(cpuinfo_path) or ""
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() not in fields:
            fields[key.strip()] = value.strip()
    name = fields.get("model name") or fields.get("Model") or fields.get("Hardware") or fields.get("cpu model")
    if "flags" in fields:
        flags = set(fields["flags"].split())
        features = {"avx": "avx" in flags, "avx2": "avx2" in flags, "avx512": "avx512f" in flags,
                    "fma": "fma" in flags, "f16c": "f16c" in flags, "neon": False,
                    "dotprod": None, "i8mm": None, "sve": None}
    elif "Features" in fields:
        flags = set(fields["Features"].split())
        features = {"avx": False, "avx2": False, "avx512": False, "fma": None, "f16c": None,
                    "neon": "asimd" in flags or "neon" in flags, "dotprod": "asimddp" in flags,
                    "i8mm": "i8mm" in flags, "sve": "sve" in flags}
    else:
        features = None
    return name, features


def _mac_static():
    keys = ["machdep.cpu.brand_string", "hw.memsize", "hw.optional.arm64", "hw.perflevel0.physicalcpu",
            "hw.perflevel1.physicalcpu", "hw.optional.avx1_0", "hw.optional.avx2_0", "hw.optional.avx512f",
            "hw.optional.fma", "hw.optional.arm.FEAT_DotProd", "hw.optional.arm.FEAT_I8MM", "sysctl.proc_translated"]
    values = sysctl(keys)
    arm = values.get("hw.optional.arm64") == "1"
    features = None
    if values:
        if arm:
            features = {"avx": False, "avx2": False, "avx512": False, "fma": None, "f16c": None, "neon": True,
                        "dotprod": _flag(values, "hw.optional.arm.FEAT_DotProd"),
                        "i8mm": _flag(values, "hw.optional.arm.FEAT_I8MM"), "sve": False}
        else:
            features = {"avx": _flag(values, "hw.optional.avx1_0"), "avx2": _flag(values, "hw.optional.avx2_0"),
                        "avx512": bool(_flag(values, "hw.optional.avx512f")), "fma": _flag(values, "hw.optional.fma"),
                        "f16c": None, "neon": False, "dotprod": None, "i8mm": None, "sve": None}
    return {"name": values.get("machdep.cpu.brand_string"), "arm64": arm, "features": features,
            "memsize": _int(values.get("hw.memsize")), "translated": values.get("sysctl.proc_translated") == "1",
            "performance_cores": _int(values.get("hw.perflevel0.physicalcpu")),
            "efficiency_cores": _int(values.get("hw.perflevel1.physicalcpu"))}


def _windows_features():
    try:
        import ctypes
        present = ctypes.windll.kernel32.IsProcessorFeaturePresent
    except (ImportError, AttributeError, OSError):
        return None
    # Documented PF_* constants from winnt.h.
    check = lambda code: bool(present(code))
    arm = platform.machine().lower() in ("arm64", "aarch64")
    return {"avx": check(39), "avx2": check(40), "avx512": check(41), "fma": None, "f16c": None,
            "neon": arm and check(29), "dotprod": check(43) if arm else None, "i8mm": None, "sve": None}


def _windows_registry_reader():
    """Default registry reader: read(path) -> {"values": {...}, "subkeys": [...]} or None."""
    try:
        import winreg
    except ImportError:
        return None

    def read(path):
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                count_keys, count_values, _ = winreg.QueryInfoKey(key)
                subkeys = [winreg.EnumKey(key, i) for i in range(count_keys)]
                values = {}
                for i in range(count_values):
                    name, value, _ = winreg.EnumValue(key, i)
                    values[name] = value
                return {"values": values, "subkeys": subkeys}
        except OSError:
            return None
    return read


def cpu_info(system=None, registry=None, cpuinfo_path="/proc/cpuinfo"):
    """Friendly CPU name and instruction-set features ({feature: bool | None}); unknown stays None."""
    system = system or platform.system()
    if system == "Darwin":
        mac = _cached("mac_static", _mac_static)
        return mac["name"], mac["features"]
    if system == "Windows":
        registry = registry or _windows_registry_reader()
        key = registry(WINDOWS_CPU_KEY) if registry else None
        name = ((key or {}).get("values", {}).get("ProcessorNameString") or "").strip() or None
        return name, _windows_features()
    return _linux_cpu(cpuinfo_path)


# ---------- Apple Silicon ----------

def mac_gpu_limit(ram_total, wired_limit_mb):
    """Bytes macOS lets the GPU keep resident.

    An explicit `sysctl iogpu.wired_limit_mb` (> 0) wins. 0 or missing means macOS's default, which Metal reports
    as `recommendedMaxWorkingSetSize`: roughly two thirds of RAM on smaller Macs and about three quarters on
    larger ones. Reports vary by chip and macOS version (65-78%), so we use the cautious end: 2/3 up to 36 GiB,
    3/4 above. The override is capped at installed RAM.
    """
    if wired_limit_mb and wired_limit_mb > 0:
        return min(ram_total, wired_limit_mb * MIB), "iogpu.wired_limit_mb"
    share = 0.75 if ram_total > 36 * GIB else 2 / 3
    return int(ram_total * share), "estimated_default"


def _mac_displays():
    """GPU names/cores from system_profiler (slow: cached per process by the caller)."""
    output = run(["/usr/sbin/system_profiler", "SPDisplaysDataType", "-json"], timeout=5)
    try:
        return json.loads(output).get("SPDisplaysDataType") or [] if output else None
    except (ValueError, AttributeError):
        return None


def _parse_size(text):
    match = re.match(r"\s*([\d.]+)\s*(MB|GB)", text or "", re.I)
    if not match:
        return None
    return int(float(match.group(1)) * (GIB if match.group(2).upper() == "GB" else MIB))


def apple_gpus(ram_total, ram_available):
    """Returns (gpus, other_gpus, warnings, unified)."""
    mac = _cached("mac_static", _mac_static)
    warnings = []
    displays = _cached("mac_displays", _mac_displays)
    if mac["arm64"]:
        chip = (mac["name"] or "Apple Silicon").strip()
        wired = _int(sysctl(["iogpu.wired_limit_mb"]).get("iogpu.wired_limit_mb"))
        limit, source = mac_gpu_limit(ram_total, wired)
        cores = next((_int(d.get("sppci_cores")) for d in displays or [] if "apple" in str(d.get("sppci_model", "")).lower()), None)
        if mac["translated"]:
            warnings.append("Python is running under Rosetta (Intel emulation). Use a native Apple Silicon Python "
                            "and llama.cpp build, or speed will be much lower.")
        # Same memory pool as system RAM: the GPU can never have more free than the system has free.
        gpu = {"index": 0, "uuid": f"apple-{_slug(chip.replace('Apple', ''))}", "name": chip, "total": limit,
               "available": max(0, min(limit, ram_available)), "utilization": None, "driver": None,
               "backend": "metal", "vendor": "apple", "unified": True, "gpu_cores": cores, "limit_source": source,
               "performance_cores": mac["performance_cores"], "efficiency_cores": mac["efficiency_cores"]}
        return [gpu], [], warnings, True
    if displays is None:
        warnings.append("Could not list this Mac's graphics chips.")
    others = []
    for item in displays or []:
        name = item.get("sppci_model")
        if not name:
            continue
        vendor = next((v for v in ("amd", "intel", "nvidia") if v in (str(item.get("spdisplays_vendor", "")) + name).lower()), "unknown")
        others.append({"name": name, "vendor": vendor, "backend": "metal",
                       "total": _parse_size(item.get("spdisplays_vram") or item.get("spdisplays_vram_shared")),
                       "available": None, "unified": vendor == "intel",
                       "reason": "Free graphics memory cannot be read on Intel Macs."})
    return [], others, warnings, False


# ---------- Linux DRM (AMD / Intel) ----------

def _rocm_names():
    """Marketing names from rocm-smi or amd-smi, in the tools' own device order, and whether ROCm is installed."""
    rocm = bool(shutil.which("amd-smi") or shutil.which("rocm-smi"))
    tool = shutil.which("amd-smi")
    if tool:
        try:
            data = json.loads(run([tool, "static", "--asic", "--json"], timeout=4) or "")
            items = data if isinstance(data, list) else data.get("gpu_data", [])
            names = [((item.get("asic") or {}).get("market_name") or "").strip() for item in items]
            if names and all(names):
                return names, rocm
        except (ValueError, AttributeError, TypeError):
            pass
    tool = shutil.which("rocm-smi")
    if tool:
        try:
            data = json.loads(run([tool, "--showproductname", "--json"], timeout=4) or "")
            cards = sorted((k for k in data if k.startswith("card")), key=lambda k: _int(k[4:]) or 0)
            names = [(data[k].get("Card Series") or data[k].get("Card series") or "").strip() for k in cards]
            if names and all(names):
                return names, rocm
        except (ValueError, AttributeError, TypeError):
            pass
    return [], rocm


def drm_cards(root=DRM_ROOT):
    """GPUs visible in /sys/class/drm, one per PCI device, with vendor, driver and memory counters."""
    cards, seen = [], set()
    try:
        entries = sorted((e for e in os.listdir(root) if re.fullmatch(r"card\d+", e)), key=lambda e: int(e[4:]))
    except OSError:
        return []
    for entry in entries:
        device = os.path.join(root, entry, "device")
        vendor = PCI_VENDORS.get((_read(os.path.join(device, "vendor")) or "").lower())
        if not vendor:
            continue
        real = os.path.realpath(device)
        if real in seen:
            continue
        seen.add(real)
        driver = os.path.basename(os.path.realpath(os.path.join(device, "driver"))) if os.path.exists(os.path.join(device, "driver")) else None
        cards.append({"card": entry, "vendor": vendor, "driver": driver, "pci": os.path.basename(real),
                      "device_id": _read(os.path.join(device, "device")),
                      "product_name": _read(os.path.join(device, "product_name")) or None,
                      "unique_id": _read(os.path.join(device, "unique_id")) or None,
                      "vram_total": _int(_read(os.path.join(device, "mem_info_vram_total"))),
                      "vram_used": _int(_read(os.path.join(device, "mem_info_vram_used"))),
                      "gtt_total": _int(_read(os.path.join(device, "mem_info_gtt_total"))),
                      "gtt_used": _int(_read(os.path.join(device, "mem_info_gtt_used")))})
    return cards


def linux_gpus(start_index, have_nvidia, root=DRM_ROOT):
    """AMD cards with sysfs memory counters go in `gpus`; everything else we can see goes in `other_gpus`."""
    cards = drm_cards(root)
    gpus, others, warnings = [], [], []
    amd = [c for c in cards if c["vendor"] == "amd"]
    names, rocm = _cached("rocm_names", _rocm_names) if amd else ([], False)
    if len(names) != len(amd):
        names = []  # Tool order only maps safely onto sysfs order when the counts agree.
    for position, card in enumerate(amd):
        name = (names[position] if names else None) or card["product_name"] or f"AMD Radeon GPU ({card['device_id'] or card['pci']})"
        total, used = card["vram_total"], card["vram_used"]
        if total and used is not None and total >= 2 * GIB:
            gpus.append({"index": start_index + len(gpus), "uuid": f"amdgpu-{card['unique_id'] or card['pci']}",
                         "name": name, "total": total, "available": max(0, total - used), "utilization": None,
                         "driver": card["driver"], "backend": "rocm" if rocm else "vulkan", "vendor": "amd",
                         "unified": False, "pci": card["pci"], "gtt_total": card["gtt_total"]})
        else:
            # Small or missing VRAM usually means an integrated APU sharing system RAM; its GPU budget is not readable.
            others.append({"name": name, "vendor": "amd", "backend": "rocm" if rocm else "vulkan", "total": total,
                           "available": None, "unified": total is not None and total < 2 * GIB, "pci": card["pci"],
                           "driver": card["driver"],
                           "reason": "Integrated or unreadable AMD graphics: dedicated memory is too small or unknown."})
    for card in cards:
        if card["vendor"] == "intel":
            name = card["product_name"] or f"Intel graphics ({card['device_id'] or card['pci']})"
            others.append({"name": name, "vendor": "intel", "backend": "vulkan", "total": None, "available": None,
                           "unified": None, "pci": card["pci"], "driver": card["driver"],
                           "reason": "Intel drivers do not report free graphics memory in a way we can read."})
        elif card["vendor"] == "nvidia" and not have_nvidia:
            others.append({"name": f"NVIDIA GPU ({card['device_id'] or card['pci']})", "vendor": "nvidia", "backend": "cuda",
                           "total": None, "available": None, "unified": False, "pci": card["pci"], "driver": card["driver"],
                           "reason": "nvidia-smi is missing, so free graphics memory cannot be read."})
            warnings.append("An NVIDIA GPU is present but nvidia-smi was not found. Install or repair the NVIDIA driver "
                            "so its memory can be read.")
    return gpus, others, warnings


# ---------- Windows ----------

def windows_gpus(have_nvidia, registry=None):
    """Display adapters from the registry. Free memory is never exposed there, so all go in `other_gpus`.

    `HardwareInformation.qwMemorySize` is the 64-bit dedicated memory size; WMI's AdapterRAM caps at 4 GiB.
    """
    registry = registry or _windows_registry_reader()
    root = registry(WINDOWS_GPU_CLASS) if registry else None
    if not root:
        return [], ["Could not list graphics adapters from the Windows registry."]
    others, warnings, seen = [], [], set()
    for sub in root.get("subkeys", []):
        if not re.fullmatch(r"\d{4}", sub):
            continue
        values = (registry(WINDOWS_GPU_CLASS + "\\" + sub) or {}).get("values", {})
        match = re.search(r"ven_([0-9a-f]{4})", str(values.get("MatchingDeviceId", "")).lower())
        if not match or not str(values.get("MatchingDeviceId", "")).lower().startswith("pci\\"):
            continue  # virtual/remote display adapters
        vendor = PCI_VENDORS.get("0x" + match.group(1), "unknown")
        name = values.get("DriverDesc") or "Unknown graphics adapter"
        if (vendor == "nvidia" and have_nvidia) or (name, values.get("MatchingDeviceId")) in seen:
            continue
        seen.add((name, values.get("MatchingDeviceId")))
        size = values.get("HardwareInformation.qwMemorySize")
        if isinstance(size, (bytes, bytearray)):
            size = int.from_bytes(size[:8], "little")
        total = size if isinstance(size, int) and size > 0 else None
        reason = ("nvidia-smi is missing, so free graphics memory cannot be read." if vendor == "nvidia"
                  else "Windows does not report free graphics memory for this adapter.")
        others.append({"name": name, "vendor": vendor, "backend": "cuda" if vendor == "nvidia" else "vulkan",
                       "total": total, "available": None, "unified": vendor == "intel" and (total or 0) < 2 * GIB,
                       "driver": values.get("DriverVersion"), "reason": reason})
        if vendor == "nvidia":
            warnings.append("An NVIDIA GPU is present but nvidia-smi was not found. Install or repair the NVIDIA driver "
                            "so its memory can be read.")
    return others, warnings


# ---------- scan ----------

def gpu_scan(system, ram_total, ram_available, registry=None, drm_root=DRM_ROOT):
    """Returns (gpus, other_gpus, warnings, unified_memory)."""
    gpus, warnings = nvidia_gpus()
    others, unified = [], False
    try:
        if system == "Darwin":
            apple, others, extra, unified = apple_gpus(ram_total, ram_available)
            gpus += apple
        elif system == "Linux":
            found, others, extra = linux_gpus(len(gpus), bool(gpus), drm_root)
            gpus += found
        elif system == "Windows":
            others, extra = _cached("windows_gpus", lambda: windows_gpus(bool(gpus), registry))
        else:
            extra = []
    except Exception:  # Detection must never break the app; the NVIDIA result above still stands.
        extra = ["Could not check for non-NVIDIA graphics. Only the GPUs listed are known."]
    warnings += extra
    if not gpus:
        if others:
            warnings.append("Graphics found, but their free memory cannot be read, so estimates use the CPU and system RAM only.")
        else:
            warnings.append("No GPU with readable memory was found. Estimates use the CPU and system RAM only.")
    return gpus, others, warnings, unified


def scan(include_processes=True, process_ids=None):
    memory = psutil.virtual_memory()
    system = platform.system()
    gpus, other_gpus, warnings, unified = gpu_scan(system, memory.total, memory.available)
    try:
        cpu_name, cpu_features = _cached(("cpu", system), lambda: cpu_info(system))
    except Exception:
        cpu_name, cpu_features = None, None
    processes = []
    if include_processes:
        for proc in psutil.process_iter(["pid"]):
            try:
                if process_ids is not None and proc.pid not in process_ids:
                    continue
                info = proc.as_dict(attrs=["pid", "name", "memory_info", "create_time"])
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
                "gpus": [{k: gpu.get(k) for k in ["uuid", "name", "driver"]} for gpu in gpus]}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
    return {"timestamp": now(), "fingerprint": fingerprint, "cpu": identity["cpu"], "os": identity["os"],
            "cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=0.15), "ram_total": memory.total,
            "ram_available": memory.available, "swap_used": psutil.swap_memory().used,
            "disk_free": shutil.disk_usage(os.path.expanduser("~")).free,
            "gpus": gpus, "other_gpus": other_gpus, "processes": processes[:40], "warnings": warnings,
            "cpu_name": cpu_name, "cpu_features": cpu_features, "unified_memory": unified,
            "platform": {"system": system, "machine": platform.machine(), "release": platform.release()}}
