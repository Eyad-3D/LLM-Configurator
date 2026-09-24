# Round 3: r3-misc

Branch `claude/v04-r3-misc`, based on `3f6d208` (`claude/keen-volta-7mnrm5`).
Files changed: `catalogue.py`, `runtime_install.py`, `downloads.py`, `runtime.py`, `tests/test_catalogue.py`,
`tests/test_runtime_install.py`, `tests/test_downloads.py`, this file. `quantcheck.py`, `discover.py` and `gguf.py` needed no change.

## Fixed

- **`catalogue.get_json` leaked secrets on redirects (security).** urllib copies every request header to the redirect
  target, so `Authorization: Bearer $HF_TOKEN` and the Artificial Analysis `x-api-key` went to whatever host a redirect named.
  - Added `_MetadataRedirect`, like `downloads.DownloadRedirect`. When the scheme, host or port changes, it drops `Authorization` and `x-api-key`.
    It also refuses an HTTPS request that redirects to plain HTTP (`ValueError`, which callers already expect).
  - The network call now goes through `catalogue._open`.
  - Tests: `RedirectHeaderTests` uses a real local HTTP server. The cross-host test **fails on the old code**, which sent
    `'authorization': 'Bearer hf_secret', 'x-api-key': 'aa_secret'` to the second host. A same-host redirect keeps both headers.
- **`runtime_install.install()` was shadowed by an older `settings.runtime_dir`.** The fresh install now wins.
  - `install()` and `install_archive()` clear `settings.runtime_dir` after a successful install. Other settings are kept.
  - A warning says so: "The app now uses this new llama.cpp instead of the folder you chose earlier with 'llm-config runtime use'…". It contains no path.
  - The old "keeps using the folder you chose" note is gone. The CLI's own clearing in `cli.py` is now a harmless no-op, and api-cli can drop it.
- **Byte progress carries `"unit": "bytes"`**: the runtime download (`runtime_install._stream`) and every model-download event (`downloads._Progress.emit`, all stages).
  The `release`/`extract`/`check` stages have `total: None` and no unit.
- **`runtime.bench` goes through `launch`.**
  - The environment comes from `launch.server_env`, so inherited `LLAMA_ARG_*` are dropped, `LLAMA_ARG_CORS_ORIGINS=localhost` is set, and NVIDIA is pinned by UUID.
    The GPU device listing runs with the same environment.
  - `-ngl` comes from `launch.runtime_gpu_layers`. Before, a partial offload of g blocks sent `-ngl g`, which is one block short under the lead's
    verified rule (g+1 for every g > 0). The row check compares against the same value.
  - `settings.flash_attn` is read from the row (`-1/0/1` → `auto/off/on`), not hardcoded.
  - The record was checked to have `kind="bench"`, `id`, `settings`, `depth` and `runtime` (it already had them).
- **`runtime_install._gpu_kinds`: rule documented.** See Rejected / decided for the Vulkan choice. The `_gpu_kinds` docstring now states the rule for callers:
  whenever the installed build has a GPU backend, a CPU-only run must pass `-dev none` (`gpu_layers=0` with `gpu_backend` set).
  `engine._launch` already does this from any GPU it sees, `other_gpus` included (`hide`). `runtime.bench` and `quantcheck._common_args` do it too.

## Already done (checked, no change)

- `quantcheck.estimate_logits_bytes` already uses a 20-byte header, and `_vocab_size` already reads `gguf.read_metadata(...)["summary"]["vocab_size"]`.
  Verified against real files (Evidence).
- `catalogue.refresh()` progress already has `stage/done/total/message` (api-cli request).
- `discover` looks for downloaded shards at `models_dir / Path(filename).name` (runtime-downloads request, `discover.py:463`).
- `runtime_install.parse_version` reads the new `--version` format (server-fake request). Verified on the real build below.

## Rejected / decided

- **Picking Vulkan when GPUs appear only in `other_gpus`: not done, on purpose.** Those are Windows AMD/Intel adapters and NVIDIA without nvidia-smi, all with `available: None`.
  The engine never places layers on a GPU whose free memory it can't read, so a Vulkan build would give no speed-up.
  It would only add a larger download and a driver that can fail to start.
  - The CPU build stays, and its reason now says why: "… Note: AMD Radeon RX 7900 XTX was found, but its free memory cannot be read, so the app would never place a model on it."
  - Once `hardware.py` can read such a card, the card moves to `gpus` and Vulkan is chosen automatically. That path already exists and is tested.
- **`downloads` calling `discover.remember_hash` itself**: left to `app.download_job` (assigned to r3-app-cli). Doing both would double-write `hash_cache`.

## Requests

- **lead (`launch.from_candidate`)**: a CPU-only candidate has `gpu_index=None`, so `gpu_backend` stays `None` unless the engine's `launch` dict sets it.
  Suggestion: when `gpu` is None and `runtime_backend` is a GPU backend, set `gpu_backend=runtime_backend`, so `-dev none` also follows from the installed build.
  Today it depends only on engine's `hide`, which covers the known cases.
- **lead (`launch.server_env`)**: pin `CUDA_VISIBLE_DEVICES` only for uuids starting with `GPU-`, as `runtime.gpu_placement` does.
  A placeholder uuid such as `[N/A]` would otherwise hide every CUDA device. (From the independent review.)
- **api-cli (`cli.py` `runtime install`)**: the manual `runtime_dir` clearing after `install` can go. `install()` does it and returns the note.
  `install-archive` gets the same behaviour for free.
- **ui**: download progress now has `unit: "bytes"`. Format `done/total` as sizes only when it is present.

## Evidence

- An independent review by a subagent found no serious issue. It confirmed the header removal, including on 307/308 and on chained redirects;
  that settings are cleared only after a successful install; and that the `-ngl` row check lines up.
  One nit was applied: the redirect response is closed before refusing. Known, accepted limit: `bench` now also stops at `launch`'s 1,048,576-token context cap, with a plain message.

- `python3 -m unittest discover -s tests`: **817 tests OK** (26 skipped).
- Real llama.cpp (`scripts/build_llama_cpp.sh`, llama-cpp-python 0.3.35 sdist, `version: 0.1.0-dev (build 1, commit 4df29be)`) and the tiny models:
  `tests.integration.test_real_runtime` + `test_fake_matches_real`: **43 tests OK**.
- `quantcheck.estimate_logits_bytes` vs the real `--kl-divergence-base` file (`llama-perplexity --chunks 3`, vocab from `gguf.read_metadata` = 422):

  | model | context | estimate | real file |
  |---|---|---|---|
  | tiny-llama-F16 | 128 | 162,584 | 162,584 |
  | tiny-llama-F16 | 256 | 327,704 | 327,704 |
  | tiny-qwen3moe-F16 | 128 | 162,584 | 162,584 |
  | tiny-qwen3moe-F16 | 256 | 327,704 | 327,704 |

- `runtime_install.detect` on the real bin dir: `installed: True, source: configured, version: b1, build: 1, backend: cpu, devices: [], warnings: []`.
  "b1" is right: this build reports build number 1.
- `runtime.bench` on real llama-bench, run with `LLAMA_ARG_N_GPU_LAYERS=99 LLAMA_ARG_CTX_SIZE=64` in the parent environment:
  - argv: `-m … -p 0 -n 128 -d 384 -ngl 0 -t 4 -ctk f16 -ctv f16 -r 3 -o json -dev none`.
  - The child's only `LLAMA_ARG_*` variable is `LLAMA_ARG_CORS_ORIGINS=localhost`.
  - Record: `kind: bench`, `id: 1b4a4d131dc6`, `depth: 384`,
    `settings: {flash_attn: auto, cache_type_k: f16, cache_type_v: f16, batch: 2048, ubatch: 512, n_cpu_moe: 0}`, `runtime: {version: b1, backend: cpu}`, `tps ≈ 849`.
