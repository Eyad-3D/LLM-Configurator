# Fix-up: runtime-downloads

Branch `claude/v04-fix-runtime-downloads`, based on `076bab8`.
Files changed: `runtime_install.py`, `downloads.py`, `runtime.py`, `tests/test_runtime_install.py`, `tests/test_downloads.py`, this file.

## Fixed

### llama.cpp mismatches (`docs/v0.4/llama-cpp-facts.md`)

- **2. New `--version` format.** `parse_version` reads `version: 0.1.0-dev (build N, commit H)` and still reads the older `version: N (H)`.
  - With the new format, `version` is `"b<N>"`, the same as for numeric builds. The semver never changes between builds (`0.1.0-dev` on every nightly), so the build number is the only version that tells builds apart. The raw semver is kept in `semver`.
  - `"b<N>"` also passes `community._VERSION`. The raw `0.1.0-dev` would not.
- **3. `llama-bench` has no `--version`.** Nothing in this area runs it. `runtime.bench` now takes the build from the bench rows (`build_number` becomes `runtime_build`/`runtime.version` = `"b<N>"`, falling back to `build_commit`).
- **5. Backend showed "unknown" on CPU builds.** Added `list_devices(argv, env=None)`, which runs `llama-server --list-devices`, plus `parse_devices` and `device_backend`.
  - `detect()` now takes the backend from what the build can actually use: `(none)` means `cpu`; `CUDA*` cuda, `ROCm*`/`HIP*` rocm, `Vulkan*` vulkan, `Metal`/`MTL*` metal. CPU-side devices (BLAS, …) are ignored.
  - If a GPU build lists no devices (for example a broken driver), detect reports `cpu` and adds a plain warning.
  - If the build cannot list devices (older builds), it falls back to the manifest, then the version hints, then the library file names.
  - detect gains the key `devices: [{name, description, backend}]`. The `archive` install uses the same logic for archives whose name doesn't say.
- **6. `runtime.bench` hardcoded `-dev CUDA0`.** GPU placement is now backend-aware (`gpu_placement`, `runtime_install.pick_device`):
  - NVIDIA is pinned by UUID with `CUDA_VISIBLE_DEVICES`, and the device list is taken with that pin applied.
  - The `-dev` name always comes from the build's own `--list-devices`. It is matched by backend, then by card name when there are several. Vulkan and ROCm get no environment variables because their indexes don't map to our scan.
  - A build that sees no GPU, or can't tell which listed device is the card, is refused with a plain message **before** llama-bench runs. It never guesses a name.
  - A build that cannot list devices gets no `-dev` at all (llama.cpp's `auto`). `layers == 0` keeps `-dev none`, the one name every build accepts.
  - The error message now says "The selected GPU is unavailable", not "NVIDIA".
- **The real-runtime test `test_runtime_install_detects_configured_directory` passes**, and so do all real tests for this area (below).

### Requests in `docs/v0.4/handoff/*.md`

- **hardware (backend branching):** done as above. A GPU dict without `backend` (the pre-v0.4 scan) is treated as CUDA, the same as `launch.from_candidate`. `available: None` skips the VRAM check instead of crashing. `unified` is passed to `engine.allocations`.
- **harness 2 and 5; server "parse both formats":** done.
- **catalogue (shard files inside quant folders):** downloads saves every file under its base name in one folder (`Q4_K_M/x-00001-of-00003.gguf` becomes `x-00001-of-00003.gguf`), and llama.cpp loads split files from the first shard next to the others. I kept flattening because app.py and export already rely on it. discover does not (see Requests).
- **api / seam findings:**
  - `bench` accepts `command: str | list[str]` (ground rule 8).
  - Bench records now carry the §2.3 keys `kind: "bench"`, `id`, `depth`, `settings` and `runtime`, so CLI bench results can be shared.
  - Install progress is throttled (5 per second) and carries `bytes_per_second`/`eta_seconds`, like model downloads.
  - `install()` warns when a `runtime use` folder still wins over the fresh install.

### Adversarial review (4 subagents plus a final diff review)

- **Security:**
  - The CUDA runtime merge (`_merge_companion`) could move a relative symlink to a different depth, where it then pointed outside the install. It now moves plain files only.
  - Runtime archives are re-hashed from disk before use.
  - Unverified (`--allow-unverified`) leftovers are never reused.
  - Damaged archives (`TarError`, `BadZipFile`, file/folder clashes) give a plain `ValueError`.
  - Local model file names containing `:` (Windows drive letters or alternate data streams) and `x` / `x.part` clashes are refused.
- **Resume:**
  - **downloads, real bug:** a failed flush at close (disk full) was treated as a network blip, and a **short file could be renamed as verified**. Writes and close are now errors, and the size on disk is checked before the hash.
  - **downloads, real bug:** when the server ignored Range and kept dropping, the retry budget reset every time, so it **retried forever**. There is now a high-water mark.
  - A shorter 206 range (some CDNs cap them) now keeps the partial and asks for the rest.
  - runtime_install: a complete `.part` is verified without any network call. Re-hashing a resumed `.part` checks for cancel.
- **Asset names:** checked against the real b11159 and b11149 release pages (35 assets each, fetched on 2026-09-24).
  - Every CPU, CUDA, Vulkan, macOS and cudart name is classified correctly.
  - The picks for every OS/arch/GPU/driver combination tried were right.
  - `releases/latest` is `v0.5.0` with no binaries; `fetch_release` already falls back to the b<N> prereleases.
  - Digests are published. The largest asset is 567 MB.
- **Seams:** every call signature, return key and exception type matches. The rest went to Requests.
- **Final review of this diff** found 3 more bugs, all now fixed:
  - `pick_device` took the only listed device even when it was another backend's card (an Intel iGPU pick on a CUDA build went to the NVIDIA card). A name match is now required.
  - Name matching was plain substring matching (an A100 matched an A10; an empty description matched everything). It is now whole words only.
  - A failing `close()` could hide a cancel or a too-large reply. The original reason is now kept.
  - Also: `runtime.backend` now comes from the picked device, and `version: … (build 0, commit unknown)` from builds without git is parsed.

## Rejected, with reasons

- **CUDA 12.8 on Linux with older NVIDIA drivers (535/550/560) through NVIDIA's minor-version compatibility.** Those drivers still get Vulkan. Minor-version compatibility excludes PTX JIT and newer-driver features. Vulkan works well on NVIDIA, and a CUDA build that fails to start would only fall back to CPU.
- **No `CUDA_DRIVERS` rows for 13.1–13.4.** NVIDIA now publishes a minimum driver branch, not exact versions. 13.x is covered by the major-version rule (≥ 580.65.06 / 580.88), which the 13.4 release notes confirm.
- **ROCm assets (`rocm-10.0`) stay unpicked.** `classify` returns backend `None` for them. ROCm needs system libraries; users can run `install-archive` or `use`.
- **Re-hashing model downloads from disk** (a second writer in another process). A second full pass over tens of GB costs 10–30 s. Inside the app `exclusive="download:<variant>"` already prevents two writers. The flush failure, which was the real bug, is now closed by the size check. Runtime archives (under 1 GB) do get re-hashed.
- **A lock file for downloads across processes.** Stale locks after a crash are worse than the rare CLI+GUI race.
- **Keeping the relative folder of shard files** (`Q4_K_M/…`). app.py, export and discover would have to change together, and flattening already works with llama.cpp.
- **Removing paths from `detect()` warnings.** They are the user's own paths and help them act (the `xattr -dr` command). Hiding `directory`/`binaries` from the page needs a contract decision (Requests).

## Requests

- **discover:** `discover.py:426` looks for catalogue shards at `models_dir / f["filename"]` (with the quant folder), but downloads saves `models_dir / Path(f["filename"]).name`. Use `.name`, as app.py:118 and export already do, or downloaded split models are never found.
- **testing:** `testing.py:447/455` records `runtime_build = build_commit` and `runtime.version = str(build_number)`. Please use `"b<N>"` from `build_number` (falling back to `build_commit`) to match `runtime.bench` and `detect()["version"]`. `community._VERSION` rejects a bare commit hash.
- **testing / tuner / quantcheck / lead (`launch.py`):** don't hardcode a `-dev` name.
  - `runtime_install.detect(store)["devices"]` (no extra process) together with `runtime_install.pick_device(devices, gpu)` gives the right name, or `None` when it can't be told apart. In that case omit `-dev`, or refuse.
  - Only `none` is always valid. `launch.server_env` pins CUDA only when `gpu_backend == "cuda"`, which is correct. Set `config["device"]` from `pick_device` for the other backends.
- **api-cli:** `bench` should default to `app.binary(store, "llama-bench")` when `--executable` isn't given; `runtime.bench` accepts the argv list.
- **app (api-cli):** `app.py:115-118` falls back to `downloads.plan()` (size-only) even when `verify=True`, so `download_job` can report `reused: True` for an unhashed same-size file. Please pass `progress`/`cancel` to `discover.find_for_variant` too.
- **ui:** `runtime_install` job progress sends `total: null` for the `release`, `extract` and `check` stages. Please don't show "0 B" when `total` is null.
- **lead:** `/api/runtime` returns `detect()` raw (contract §5), which includes absolute `directory`/`binaries` paths and the new `devices` list (names only). Decide whether the page should get a path-free view.
- **lead:** all models share one folder. Two repos with the same file name clash: download refuses with "choose a different folder", which the web page cannot do. Per-repo subfolders would need downloads, app, discover and export to change together.
- **server-fake:** the 2 remaining suite failures are yours (`--draft-max` accepted, `llama-bench --version` answered). The fake also has no `--list-devices`. Real output is `Available devices:\n  (none)\n`, exit 0, on stdout, for both llama-server and llama-bench.

## Evidence

- **Real llama.cpp:** `4df29be`, built with `scripts/build_llama_cpp.sh`; tiny models from `scripts/make_tiny_models.py`.
  - `llama-server --list-devices` and `llama-bench --list-devices` both print `Available devices:` / `  (none)` and exit 0.
  - `LLM_CONFIG_REAL_RUNTIME=… python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real`: 32 tests, 30 pass. The 2 failures are the fake's (server-fake). `test_runtime_install_detects_configured_directory` and `test_runtime_bench_with_real_llama_bench` pass.
- **`use_directory` on the real bin dir:** `installed, configured, version b1, build 1, commit 4df29be, backend cpu, devices [], no warnings` in 0.05 s.
- **`install_archive`** with a zip and a tar.gz made from the real binaries: both install and detect as `b1`/`cpu`. The zip goes to `b1-cpu/` (backend from the name); the tar.gz to `local-cpu/` (backend from `--list-devices`).
- **`runtime.bench` with the real llama-bench:**
  - CPU run with `-dev none`: about 1600 t/s, `runtime {"version": "b1", "backend": "cpu"}`, depth 384, 0.5 s. The argv-list form works too.
  - GPU layers requested on this CPU-only build (NVIDIA or AMD in a mocked scan): refused before running with "This llama.cpp build sees no graphics card it can use…". No invalid `-dev` name was sent.
- **Unit tests:** `python3 -m unittest discover -s tests`, with the real runtime enabled: 632 tests, the only failures being the same 2 fake tests (they also fail on `076bab8`). New tests fail on the old code: 4 in `test_downloads` (retry loop, short 206, flush failure, names) and 5 in `test_runtime_install` (companion link, disk re-hash, offline complete partial, unverified reuse, damaged archive). Also new: parse/device/detect tests, throttled-progress and install-warning tests, and 10 bench device tests.
- **Not verified (no GPU here):** real `--list-devices` output for CUDA, Vulkan, ROCm and Metal builds. The format comes from `common/arg.cpp` (`  %s: %s (%zu MiB, %zu MiB free)`); the Metal name may be `Metal` or `MTL0`, and both are handled.
