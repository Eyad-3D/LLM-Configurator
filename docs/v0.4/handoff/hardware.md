# Handoff: `hardware` workstream

## What I built

`hardware.scan()` now finds the hardware most local-LLM users have, not just NVIDIA:

- **Apple Silicon Macs**: one Metal GPU entry that shares system RAM (unified memory).
- **Intel Macs**: graphics chips listed from `system_profiler` (free memory unknown).
- **AMD on Linux**: memory read straight from sysfs files (no tools needed); nicer names from `amd-smi`/`rocm-smi` if installed.
- **Intel GPUs on Linux (i915/xe) and Windows AMD/Intel adapters**: listed by name, free memory left unknown.
- **Friendly CPU name** and **CPU features** (AVX2, AVX-512, NEON, ...) on Windows, macOS and Linux.
- NVIDIA behaviour is unchanged (same keys and values, plus the new `vendor`/`unified` keys the contract asks for).

## Public API as built

`scan(include_processes=True, process_ids=None) -> dict`. Every v0.3 key is kept with the same meaning. New keys:

| Key | Shape |
|---|---|
| `cpu_name` | `str \| None` — e.g. `"Apple M2 Pro"`, `"AMD Ryzen 9 7950X 16-Core Processor"` |
| `cpu_features` | `{"avx","avx2","avx512","fma","f16c","neon","dotprod","i8mm","sve": bool \| None}` or `None` when nothing could be read |
| `unified_memory` | `bool` — `True` only on Apple Silicon |
| `platform` | `{"system", "machine", "release"}` |
| `other_gpus` | list of GPUs we can see but whose **free memory is unknown** (see Deviations) |

Every entry in `gpus` has: `index, uuid, name, total (int), available (int), utilization, driver, backend, vendor, unified`.

- NVIDIA: as before + `vendor: "nvidia"`, `unified: False`.
- Apple Silicon: `{"index": 0, "uuid": "apple-m2-pro", "name": "Apple M2 Pro", "backend": "metal", "vendor": "apple", "unified": True, "total": <GPU wired limit>, "available": min(limit, ram_available), "driver": None, "utilization": None, "gpu_cores": int|None, "limit_source": "iogpu.wired_limit_mb"|"estimated_default", "performance_cores", "efficiency_cores"}`.
- AMD (Linux, dedicated VRAM ≥ 2 GiB): `uuid: "amdgpu-<unique_id or PCI address>"`, `total = mem_info_vram_total`, `available = total - mem_info_vram_used`, `backend: "rocm"` if `amd-smi`/`rocm-smi` is installed, else `"vulkan"`, plus `pci`, `gtt_total`. Indexes continue after NVIDIA ones (NVIDIA 0, AMD 1, ...).

Each `other_gpus` entry: `{"name", "vendor", "backend", "total": int|None, "available": None, "unified": bool|None, "reason": str, ...}`.

Helpers (usable by tests and other modules): `cpu_info(system=None, registry=None, cpuinfo_path=...) -> (name, features)`, `mac_gpu_limit(ram_total, wired_limit_mb) -> (bytes, source)`, `drm_cards(root)`, `linux_gpus(start_index, have_nvidia, root)`, `windows_gpus(have_nvidia, registry)`, `gpu_scan(system, ram_total, ram_available, registry=None, drm_root=None)`, `clear_cache()`.

The Windows registry reader is injectable: `registry(path) -> {"values": {...}, "subkeys": [...]} | None` (HKLM paths).

## Deviations from the contract and why

1. **GPUs with unknown free memory go into a new `other_gpus` key, not `gpus`.** `engine.recommend`, `runtime.py` and `launch.from_candidate` do arithmetic on `gpu["available"]`; a `None` there would crash. Putting them in `other_gpus` means old code cannot break, and nothing invents a number. This still satisfies "list them with unknown free memory".
2. **AMD APUs** (dedicated VRAM under 2 GiB, i.e. a small carve-out from system RAM) also go into `other_gpus` with `unified: True`. Their real GPU budget (GTT) is shared with system RAM and not cleanly readable.
3. The old warning "AMD, Intel and Apple GPU support is not implemented" is gone. CPU-only systems now get: "No GPU with readable memory was found. Estimates use the CPU and system RAM only."

## Apple Silicon memory rule (and why)

- If `sysctl iogpu.wired_limit_mb` is set above 0, that is the GPU limit (capped at installed RAM). `0` means "use macOS's default".
- Otherwise the default is what Metal reports as `recommendedMaxWorkingSetSize`. Public reports put it at roughly 65–78% of RAM depending on chip and macOS version. I use the cautious end: **2/3 of RAM up to 36 GiB, 3/4 above**, and label it `limit_source: "estimated_default"`.
- `available = min(limit, ram_available)`: it is the same memory as system RAM, so the GPU can never have more free than the system does.
- `wired_limit_mb` is re-read on every scan (one fast `sysctl` call) because users change it with `sudo sysctl`. The chip name, core counts and `system_profiler` output are cached per process.

Sources: [Apple: recommendedMaxWorkingSetSize](https://developer.apple.com/documentation/metal/mtldevice/recommendedmaxworkingsetsize), [llama.cpp discussion #2182](https://github.com/ggml-org/llama.cpp/discussions/2182), [devnote: override macOS Metal VRAM cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap), [SudoAll write-up (measured 78% on a 32 GiB M2 Max)](https://sudoall.com/apple-silicon-hidden-vram-local-llms/).

## Known gaps / unverified assumptions

- The exact macOS default share differs by chip; ours may be a little low (safe direction). Only a native Metal call would give the exact number, and that needs a new dependency.
- The older macOS 13 key `debug.iogpu.wired_limit` is not read.
- `system_profiler SPDisplaysDataType -json` field names (`sppci_model`, `sppci_cores`, `spdisplays_vram`, `spdisplays_vram_shared`, `spdisplays_vendor`) are from memory of real output, not verified on a Mac in this session. The first call takes ~0.5–1 s; it is cached.
- sysctl keys assumed: `machdep.cpu.brand_string`, `hw.memsize`, `hw.optional.arm64`, `hw.perflevel0/1.physicalcpu`, `hw.optional.arm.FEAT_DotProd`, `hw.optional.arm.FEAT_I8MM`, `hw.optional.avx1_0`, `hw.optional.avx2_0`, `hw.optional.avx512f`, `hw.optional.fma`, `sysctl.proc_translated` (Rosetta). Missing keys are just skipped.
- `rocm-smi --showproductname --json` → `{"card0": {"Card Series": ...}}` and `amd-smi static --asic --json` → `[{"asic": {"market_name": ...}}]` (newer amd-smi may wrap in `{"gpu_data": [...]}`, also handled). Tool names are only matched to sysfs cards when the counts agree; otherwise a generic name with the PCI device ID is used.
- Intel Arc memory is not read on Linux or Windows (xe/i915 have no stable free-memory file). They appear in `other_gpus`.
- Windows: `HardwareInformation.qwMemorySize` is dedicated memory only, and free memory is never available, so Windows AMD/Intel GPUs stay in `other_gpus`. CPU features use `IsProcessorFeaturePresent` (PF_AVX=39, PF_AVX2=40, PF_AVX512F=41, PF_ARM_V8=29, PF_ARM_V82_DP=43). Not run on a real Windows machine here.
- AMD `backend` is a best guess of which llama.cpp build fits: `rocm` if ROCm tools exist, else `vulkan`.
- If an AMD card's name changes (e.g. `rocm-smi` installed later), its fingerprint changes and old calibration for that machine is ignored. That is safe, just slightly wasteful.

## Requests

- **engine**: the Apple GPU's `available` is the same memory as `ram_available`. Don't add them together. When `gpu["unified"]` is true, count model bytes against one shared pool. Ignore `other_gpus` for memory maths (their `available` is always `None`); it is for display only.
- **engine / downloads (`runtime.py`)**: GPU entries are no longer only CUDA. `runtime.py` sets `CUDA_VISIBLE_DEVICES` and `-dev CUDA0`; please branch on `gpu["backend"]` (`metal` needs neither; `rocm` uses `HIP_VISIBLE_DEVICES`; `vulkan` uses `GGML_VK_VISIBLE_DEVICES` / `-dev Vulkan0`). Also the error text "Selected NVIDIA GPU is unavailable" should say "GPU".
- **ui-run**: `app.js` says "No supported NVIDIA telemetry" when `gpus` is empty. Suggested: show `hw.cpu_name || hw.cpu`, and when `gpus` is empty but `other_gpus` isn't, say "<name> found — free memory can't be read, so estimates use RAM". For `gpu.unified`, label it "Shared with system RAM" instead of "GPU memory". The GPU selector's empty option text "GPU telemetry unavailable" can stay.
- **lead / launch**: `launch.from_candidate` already uses `gpu.get("backend") or "cuda"`, which now yields `metal`/`rocm`/`vulkan` correctly.
- **calibration_worker**: GPU probes are CUDA-only; on other GPUs the calibration stays CPU-only, which is honest but could be extended later.

## How I tested

`tests/test_hardware.py` (25 tests), no network, no real tools:

- `subprocess.run`, `shutil.which`, `platform.system/machine` and `psutil.virtual_memory` are mocked; sysfs is a temp directory with real symlinks like `/sys/class/drm/cardN/device`; the Windows registry uses the injectable reader.
- Cases: M-series Mac (default limit, `wired_limit_mb` override, low free RAM, caching, Rosetta warning, sysctl failing), Intel Mac, Linux AMD sysfs-only, AMD with `rocm-smi`, AMD with `amd-smi`, NVIDIA + AMD indexing, AMD APU, Linux Intel Arc (xe), NVIDIA card without `nvidia-smi`, Windows NVIDIA + Intel iGPU (and `CREATE_NO_WINDOW`), Windows without NVIDIA tools, unreadable registry, CPU-only, `/proc/cpuinfo` for x86 and ARM, broken `nvidia-smi`, unexpected errors, timeouts.
- **Fingerprint**: tests copy the v0.3 algorithm verbatim and assert the new fingerprint is identical for NVIDIA-only, NVIDIA + Intel iGPU (Linux and Windows) and CPU-only machines.
- Every subprocess call is checked to use a list and a timeout ≤ 5 s, never `shell=True`.
- Full suite: `python3 -m unittest discover -s tests` → 97 tests OK. A real `scan()` on this Linux container runs in ~0.16 s (almost all of it the existing 0.15 s CPU sample).

## llama.cpp facts assumed (for the integration harness)

- Backend names used for `backend`: `cuda`, `metal`, `rocm` (HIP build), `vulkan`.
- Apple builds of llama.cpp use Metal by default and are limited by the GPU wired-memory limit above.
- Vulkan device selection via `GGML_VK_VISIBLE_DEVICES` and `-dev Vulkan0`; ROCm via `HIP_VISIBLE_DEVICES` (only mentioned in Requests, not used by this module).
