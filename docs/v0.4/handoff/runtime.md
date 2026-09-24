# Handoff: `runtime` workstream

Branch `claude/v04-runtime`. Files: `src/llm_configurator/runtime_install.py`, `tests/test_runtime_install.py`, this file.

## What I built

`runtime_install.py` makes llama.cpp "just there":

- **detect** finds an existing llama.cpp. It checks the CLI-configured folder, then the app-managed install, then `PATH` plus the usual Homebrew folders. It runs `llama-server --version` (15 s timeout) and reads the version, build number and backend.
- **choose_asset** picks the right official build for this computer from a GitHub release. It reads the asset file names (tolerant of naming drift) and explains the choice in plain words in `reason`.
- **install** downloads that build (plus the CUDA runtime files when needed) and checks its SHA256 while downloading. Downloads can resume. It unpacks safely into a hidden staging folder, checks that `llama-server --version` works, writes `manifest.json`, and then moves the folder into place in one step. If a graphics-card build won't start (for example, Vulkan system files are missing), it automatically installs the CPU build instead and says why.
- **install_archive**, **use_directory** and **binary** work as the contract describes.

## Public API as built

```python
detect(store) -> {"installed": bool, "source": "configured"|"managed"|"path"|None, "directory": str|None,
                  "version": str|None,   # "b5000" for numeric builds, else the raw string (e.g. "0.5.0")
                  "build": int|None, "backend": "cuda"|"vulkan"|"metal"|"rocm"|"cpu"|"unknown"|None,
                  "binaries": {"llama-server"|"llama-bench"|"llama-perplexity"|"llama-cli": [path]|None},
                  "warnings": [str],
                  # extra keys when installed: "commit": str|None, "tag": str|None
                 }
    # Also caches the result under store key "runtime" (plus "checked_at").

choose_asset(release, hardware, system=None, machine=None, allow_unverified=False, backend=None) -> {
    "name", "url", "size", "sha256" (hex or None only if allow_unverified), "backend", "reason",
    "companions": [{"name", "url", "size", "sha256"}],   # CUDA runtime (cudart) archive
    "tag", "build", "cuda": "13.4"|None, "unverified": bool}
    # system/machine default to platform.system()/platform.machine() (defaults are None in the
    # signature and resolved at call time, so tests can patch platform).

install(store, hardware, progress=None, cancel=None, release=None, allow_unverified=False) -> detect() result
    # plus "reason" (why this build), and fallback/unverified notes in "warnings".
install_archive(store, archive_path, progress=None, cancel=None) -> detect() result
use_directory(store, directory) -> detect() result   # writes settings.runtime_dir (merged into existing settings)
binary(store, name) -> list[str]                     # raises ValueError(NOT_INSTALLED) exactly as contracted

# Helpers other workstreams may find useful:
fetch_release() -> release dict      parse_version(text) -> {"version","build","commit","backend"}
classify(asset_name) -> dict|None     max_cuda_for_driver(driver, system) -> (major, minor)|None
safe_extract(archive, target, cancel=None)   find_bin_dir(directory) -> Path|None
```

Progress events use stages `release`, `download` (done/total in bytes across all files), `extract` and `check`.

Managed layout:

- `runtime_dir(store)/<tag>-<backend>/` holds the archive as unpacked plus `manifest.json`. The manifest keys are `tag, build, backend, cuda, source (release|archive), asset, sha256, companions, reason, version, bin_dir (relative), installed_at`.
- `runtime_dir/current.json` holds `{"directory": "<tag>-<backend>"}`.
- Partial downloads are kept in `runtime_dir/downloads/<sha16>-<name>.part`.

## Deviations from the contract, and why

1. **`releases/latest` is not enough.** Checked on 2026-09-24: `releases/latest` now returns **`v0.5.0`**, a normal release whose only asset is `nightly-tag.txt`. The ready-made builds ship as **prereleases** tagged `b<N>` (for example `b11158`, which has 48 assets). So `fetch_release()` uses `latest` only if it has builds. Otherwise it walks `releases?per_page=10` and takes the first non-draft release that has builds.
2. **Linux + NVIDIA uses CUDA, not Vulkan.** Official `ubuntu-cuda-12.8/13.4` builds and matching `cudart-…` runtime archives now exist, and CUDA is NVIDIA's own fast route. It is only chosen when the driver supports that CUDA version *and* the cudart companion exists. Otherwise it falls back to Vulkan, then CPU. Other GPUs on Linux use Vulkan, as the contract says.
3. **Automatic CPU fallback.** If the GPU build installs but `--version` fails, the CPU build is installed instead, and a warning explains why.
4. **Extra optional kwargs:** `choose_asset(..., allow_unverified=False, backend=None)`. Extra keys: `companions`, `tag`, `build`, `cuda`, `unverified`, and `commit`/`tag`/`reason` on the detect/install results.
5. **ROCm, SYCL, OpenVINO, OpenCL-Adreno and Snapdragon builds are never auto-picked.** They need vendor libraries on the system, or they target niche hardware. Users with those can still run `install-archive` or `use`.

## Asset choice rules and the driver table

- **Windows:** NVIDIA with a known driver gets the newest `win-cuda-<ver>-<arch>.zip` the driver can run, plus `cudart-llama-bin-win-cuda-<ver>-<arch>.zip` (these names have no build number). Otherwise Vulkan if any non-Apple GPU is listed, otherwise CPU. `win-cpu-arm64` is used on ARM laptops.
- **macOS:** `macos-arm64` (Metal) or `macos-x64` (treated as the CPU backend). Prefers `.tar.gz` (it keeps permissions and links) and falls back to `.zip`.
- **Linux:** see deviation 2. Also covers `ubuntu-arm64` and `ubuntu-vulkan-arm64`.
- The driver check is in `CUDA_DRIVERS`. It lists the oldest driver per CUDA version for Linux and Windows (12.0 → 13.0), taken from NVIDIA's *CUDA Toolkit Release Notes*, table "CUDA Toolkit and Corresponding Driver Versions". Versions not in the table (for example 13.4) are accepted under NVIDIA's **minor-version compatibility** rule: a driver ≥ the first driver of that major version (`CUDA_MAJOR_DRIVERS`: 12.x ≥ 525.60.13 / 528.33, 13.x ≥ 580.65.06 / 580.88) runs any build of that major version. The reason string says which rule applied. **I wrote these from memory. Someone should check them against the current NVIDIA release notes.**

## Known gaps

- **Not verified against real archives:** the folder layout inside the zip and tar.gz files, and whether Linux tarballs contain relative `.so` symlinks. `find_bin_dir` handles the root, `bin/`, `build/bin/` and any folder up to 4 levels deep. The cudart merge moves every library file from the archive folder that has the most libraries into the folder that holds `llama-server`.
- **Old driver + new GPU:** the CUDA 12.4 Windows build may lack native code for RTX 50-series (Blackwell) cards and rely on driver JIT (translating generic GPU code on first run). This isn't detected.
- **Detection is optimistic:** if a CUDA build's `--version` works but the GPU backend silently fails to load, detect still reports `cuda` (taken from the manifest). The testing workstream's smoke test will catch it.
- **Windows extra folders:** replacing an existing install on Windows can fail if llama-server is running. The user gets a plain "close it and try again" message.
- **macOS quarantine:** files downloaded by Python's urllib are *not* quarantined, and Python's extraction does not copy extended attributes. So a normal install needs no `xattr` step. For a `use DIR` build that came from a browser download, `detect()` checks `xattr -p com.apple.quarantine` and adds a warning with the exact `xattr -dr com.apple.quarantine "<dir>"` command. It never strips the attribute itself.
- **No `GITHUB_TOKEN` support.** The unauthenticated API allows 60 requests per hour per IP address, and an install uses 1–2.

## Requests

- **api / cli:**
  - `runtime install` should pass `hardware.scan(False)` and print `result["reason"]` and `warnings`.
  - `runtime use DIR` calls `use_directory`, which writes `settings.runtime_dir` itself (the contract says only the CLI writes settings, and this function is CLI-only). If you'd rather do the write yourself, tell the lead and I'll drop it.
- **hardware:** GPU dicts should keep `driver` (NVIDIA driver version string) and `vendor` (`"nvidia"`, `"amd"`, `"intel"`, `"apple"`). NVIDIA is also recognised by `backend == "cuda"` or "NVIDIA" in the name. Apple GPUs (`backend == "metal"`) are ignored when choosing a Vulkan build.
- **ui-run:** show `reason` next to the Install button result. It is written for users.

## How I tested

`python3 -m unittest tests.test_runtime_install` runs 55 tests with no network access. The full suite (`python3 -m unittest discover -s tests`) is green.

- **Fixtures:**
  - A fake release built from the **real 35 asset names of b11158**.
  - In-memory zip and tar.gz archives containing `/bin/sh` scripts that print `version: 5000 (abc1234)` to stderr.
  - A fake `_open` that serves blobs, honours or ignores `Range`, and returns 206/416.
- **Covered:**
  - Asset choice for Windows (NVIDIA new, old, unknown or too-old driver; AMD; CPU; ARM64; missing cudart), macOS (arm64/x64, zip vs tar), and Linux (CPU, Vulkan, arm64, CUDA by driver).
  - Missing digests on assets and companions, unofficial URLs, oversized assets.
  - `latest`→prerelease fallback.
  - Bad hash, oversized stream, cancel then resume with `Range`, server ignoring Range, 416 when already complete, HTTPS-only redirects.
  - Rejecting traversal attacks (`..`, absolute, `C:`, backslashes), zip symlinks, escaping tar symlinks and hard links, device files, and zip bombs. Allowing safe `.so` symlink chains and exec bits.
  - End-to-end installs (CPU, CUDA with cudart merge, macOS), GPU→CPU fallback, reinstall, cancel, and unverified installs.
  - `install_archive`, `use_directory`, detect precedence and broken-folder warnings, the `current.json` pointer, `binary()` using the cache.
  - One test that runs a real subprocess, skipped on Windows. The other tests mock `_run_version`.

## llama.cpp facts assumed (harness, please confirm)

1. `llama-server --version` exits 0 and prints `version: <build> (<commit>)` then `built with <compiler> for <triple>`, mostly on **stderr**. With dynamic backends it may first print `load_backend: loaded <X> backend from …` and `ggml_cuda_init: found N CUDA devices` / `ggml_vulkan: Found N Vulkan devices`.
2. Homebrew / Apple builds print no backend line; `arm64-apple-darwin` in the "built with" line is taken to mean Metal.
3. Library file names include `ggml-cuda`, `ggml-vulkan`, `ggml-hip` or `ggml-metal`. These are used as a backend hint when the output has none.
4. Asset naming (verified on b11158, 2026-09-24): `llama-b<N>-bin-<os>-<backend>[-<ver>]-<arch>.<zip|tar.gz>` with os `win|macos|ubuntu`, and companions `cudart-llama-bin-win-cuda-<ver>-<arch>.zip` / `cudart-llama-b<N>-bin-ubuntu-cuda-<ver>-<arch>.tar.gz`. Every asset had a `digest: "sha256:…"`.
5. `macos-x64` builds have Metal disabled (so the backend is recorded as `cpu`). Unverified.
6. Release archives unpack with `llama-server` and its libraries in one folder (the root, `build/bin/` or `llama-b<N>/`). Linux shared libraries resolve through `$ORIGIN`, so no `LD_LIBRARY_PATH` is needed. Unverified.
