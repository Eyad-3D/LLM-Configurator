# Fix-up: `server-fake`

Branch `claude/v04-fix-server-fake`. Files changed: `src/llm_configurator/llama_server.py`, `tests/fixtures/fake_llama.py`,
`tests/test_llama_server.py`, `tests/integration/test_fake_matches_real.py`, this file.

Every "real" statement below was checked against the real llama.cpp 4df29be built in this container
(`scripts/build_llama_cpp.sh`, CPU only) or against the captured samples in `tests/integration/samples/`.

## Review findings (`docs/v0.4/review/server-fake.md`)

| # | Verdict | What changed |
|---|---|---|
| 1 GPU reason on CPU-only builds | **Fixed** | `no usable GPU` removed from the failure patterns (it is only a warning). GPU causes (`failed to initialize CUDA`, `vk::`, Metal…) are now a last guess, checked **after** the specific causes, the argument errors and the -9/137 exit code. `invalid device` keeps its own graphics-card entry. Test: every `error-server-*.txt` sample and `server-log.txt` give the right reason (`RealLogTests`). The fake prints the three real warning lines (byte-identical to `error-server-bad-flag.txt`) whenever it is CPU-only (`FAKE_LLAMA_GPU=none` or `FAIL=backend`) and `-ngl` is given, whatever its value. |
| 2 `chat_template_kwargs` | **Fixed** | `chat(..., chat_template_kwargs=None)`; `enable_thinking` is merged in and wins. Seam test: `testing.smoke_test` with a reasoning fake now passes. |
| 3 CORS / API key | **Fixed (CORS), rejected (API key)** | Servers get `LLAMA_ARG_CORS_ORIGINS=localhost`. Verified on the real server: `Origin: https://evil.example` gets no `Access-Control-Allow-Origin`, `http://localhost:3000` and `http://127.0.0.1:*` do, and the "CORS is set to allow all origins" warning is gone. A per-server API key was **not** added: the registry server is meant for other apps (they'd need the key), and with CORS closed a web page can no longer read answers or `/props`. |
| 4 VRAM 0 instead of None | **Fixed** | `[N/A]` rows make `_vram_by_pid` return None; when none of our PIDs is listed, `peak_vram_bytes` stays None. |
| 5 Server outlives the app on SIGTERM/SIGHUP | **Fixed here, needs a call** | New `install_exit_handlers()` (SIGTERM/SIGHUP/SIGBREAK → `sys.exit`, so atexit stops servers; the handler ignores a second signal). Windows: every server joins a Job Object with KILL_ON_JOB_CLOSE, so it dies with the app even on Task Manager kill. Test: parent killed with SIGTERM and SIGHUP → server gone. **Request to api-cli below** (nothing calls it yet). |
| 6 Other server on a fixed port looks ready | **Fixed** | A fixed port is checked with a bind before spawning (`port_is_free`, SO_REUSEADDR on POSIX like llama-server, so TIME_WAIT isn't "busy"); ready also requires our process to be alive. Test: a second server on a busy port fails without spawning. `host="localhost"` is mapped to `127.0.0.1` so `::1` can't be answered by another program (review agent finding). |
| 7 `--draft-max` accepted by the fake | **Fixed** | `--draft`, `--draft-n`, `--draft-max`, `--draft-min`, `--draft-n-min`, `--spec-ngram-size-n` exit 1 with the real text; `--spec-draft-model`, `--spec-draft-n-max/min`, `--spec-type` (validated) accepted. With `-md` + `--spec-type draft-simple` timings gain `draft_n`/`draft_n_accepted` like real. New reason: "no longer accepts one of the settings". |
| 8 Fake bench prints detailed causes | **Fixed** | Real (checked): without `-v`, one line `llama_bench: error: failed to load model '<path as given>'` (or `failed to create context with model`), stdout `[`. The fake does the same; `-v` adds the detail lines. Consequence for tuner/testing: see Requests. |
| 9 Old `--version` default | **Fixed** | Default is now the real format `version: 0.1.0-dev (build N, commit HASH)` / `built with GNU 13.3.0 for Linux x86_64` (`Darwin arm64`, `Windows AMD64` by platform). `FAKE_LLAMA_VERSION_STYLE=classic` gives `version: N (HASH)`. Default commit is now 7 hex chars (`fa4ec0d`) like real. |
| 10 `llama-bench --version` exit 0 | **Fixed** | Usage on stdout, `error: invalid parameter for argument: --version` on stderr, exit 1 (same as `version.txt`). |
| 11 Invented load fraction | **Fixed** | Loading progress has `total: None`, `done` = elapsed seconds, `elapsed_seconds`, and "(N s so far)" in the message. |
| 12 Registry stuck "starting" | **Fixed** | Any exception clears `starting`; non-ValueError errors are shown as a generic sentence (no paths). Log-file errors become a plain ValueError. |
| 13 User's `LLAMA_ARG_*` / `LLAMA_API_KEY` | **Fixed** | Filtered out of the child environment (`llama_server.server_env`). |
| 14 Other fake differences | **Fixed** | Stream: finish chunk without usage, then `choices: []` + usage + timings, then `[DONE]`. `/v1/models`: `models` list, `aliases`, `tags`, full `meta`. `/props`: `model_alias`, `model_ftype`, `modalities`, `chat_template_caps`, `default_generation_settings.params`. Logs: real `m.ss.mmm.uuu L comp func: msg` prefix, `llama_server: model loaded` / `listening on`, no "HTTP server is listening" or offload line, real CORS warning block, real load-failure tails. Bench: `load_mode` default `auto`, `-lm` accepted, `-mmp 0/1` → `none`/`mmap` (checked), `lazy_mode` removed. Perplexity: stdout starts `0.00 minutes`, `Final estimate` on the stderr `… - ETA ` line, `kl_divergence: computing over N chunks, n_ctx=…`, `saving all logits to …`. |

## Fake vs real (work list 2), all verified on the real binaries

- `-fa` for llama-bench: accepts `on off auto 1 0 -1 true false enabled disabled` (JSON `flash_attn` 1/0/-1); `yes` → usage + `error: invalid parameter for argument: -fa`, exit 1. llama-server accepts the same words. The fake matches.
- Unknown bench flag / `-dev CUDA0` on a CPU build: usage on stdout, `error: invalid device: CUDA0` + `error: invalid parameter for argument: -dev`, exit 1. Bad `-ctk` in bench: `error: invalid parameter for argument: -ctk`.
- `--list-devices` (server, bench, perplexity): stdout `Available devices:\n  (none)\n`, exit 0 on CPU-only. The fake prints that with `FAKE_LLAMA_GPU=none`, otherwise `  CUDA0: Fake GPU 24GB (24564 MiB, 23512 MiB free)` (GPU line format from llama.cpp source, not verifiable here).
- Quantized V cache without flash attention: server `llama_init_from_model: quantized V cache requires flash_attn to be enabled`; bench `llama_bench: error: failed to create context with model '…'`, and rows already printed stay (array unclosed). The old failure pattern didn't match the real text; fixed.
- `cache_prompt`: false → `cache_n` 0 and full `prompt_n` every time; true + identical prompt → `prompt_n` 1, `cache_n` = prompt − 1 (real sample 66/1 of 67). The fake matched already; now tested.
- llama-bench echoes the model path **as given** (relative stays relative), in rows and errors.

## Other fixes from the adversarial review (two subagents)

- `stop()` waited with psutil, which reaped our child and made `Popen` report exit code 0. Now it waits with `Popen` → the real code (−15 after SIGTERM on the fake; real llama-server exits 0 on SIGTERM).
- `_LIVE` was a WeakSet: a started server dropped without `stop()` was never stopped at exit. Now a locked set; `_stop_all` sets a shutdown flag so a job thread can't start a new server during exit, and catches `BaseException`.
- `request()`: a non-HTTP answer (`http.client.HTTPException`) or a non-object JSON body raised raw exceptions; `chat()` crashed on odd `choices`. All are plain ValueErrors now.
- Windows: `argtypes` set for the Job Object calls; the job uses Popen's own handle; temp-log delete retried (a concurrent `status()` read can block it).
- Failure reasons, checked against new real failures: a crash (`GGML_ASSERT`, `GGML_ABORT`, segfault, exit -6/-11/134/139, Windows 0xC0000005) was reported as "model file is missing" because gdb's backtrace prints "No such file or directory" — the missing-file rule is now anchored to `failed to open GGUF file '…'`, and crashes get "llama.cpp crashed with these settings". New: `(Permission denied)` → "not allowed to read the model file", a later split shard (`illegal split file idx … must be loaded with the first split`) → "choose the -00001-of-… file". Reasons name the file (basename only, never a path), so a missing **draft** model no longer reads like the main model is missing. `invalid device` now says the build may lack support for that graphics card instead of "update the driver".
- PeakMemory: a `[N/A]`/`[Insufficient Permissions]` row of another process no longer switches VRAM off (only our own rows count; our own `[N/A]` = unknown); an nvidia-smi time-out skips one sample instead of giving up. RAM also uses the OS's own high-water mark (`VmHWM` on Linux, `peak_wset` on Windows), so the model-load peak counts even though testing starts sampling after `/health` is ready. The process object is kept (no PID-reuse), and a monitor can be started again.
- Fake: a later split shard and a directory fail like real; `messages: []` is accepted and a non-list gets the real "Expected 'messages' to be an array"; streaming puts usage in a last `choices: []` chunk **only** with `stream_options.include_usage` (otherwise the finish chunk carries `timings`, checked on the real server); `-ot` accepted by bench; `-mmp` prints the real deprecation text without a newline; perplexity's final estimate equals its last chunk value.

## Rejected / not changed

- Per-server API key (finding 3): see above.
- Health check "any non-503 answer means another program": left as is. With the fixed-port pre-check and auto ports, another program can't be on our port; a very old llama.cpp without `/health` would otherwise fail instantly.
- PR_SET_PDEATHSIG on Linux: rejected, it fires when the *thread* that spawned the server ends (our job threads end right after start).
- The fake still accepts an **empty file** as a model (real: `failed to read magic`). The v0.4 server handoff told every workstream to use `touch`ed files with the fake, and other sessions may rely on it right now.
- Not changed (low value, or can't be verified here): the `-ts` / bad `--host` / directory-as-model wording, "memory ran out" for any SIGKILL (it says "most likely"), `-np` auto (4 slots) in the fake (the app always passes `-np`), an OOM for an enormous `-c` in the fake, Windows `private` vs working set and macOS `phys_footprint` for RAM, and a whole-GPU `memory.used` fallback for containers where our PID is not visible to nvidia-smi.
- The fake's GPU stays the default (speeds depend on it; the tuner tests need an optimum). CPU-only is `FAKE_LLAMA_GPU=none`.

## Requests

- **api-cli (`cli.py` main, `app.py` serve)**: call `llama_server.install_exit_handlers()` once at start-up, in the main thread. Without it a SIGTERM/SIGHUP (terminal closed, `kill`) leaves llama-server running on POSIX.
- **runtime-downloads (`runtime_install.parse_version`)**: the fake now prints the real default `version: 0.1.0-dev (build N, commit HASH)` / `built with GNU 13.3.0 for Linux x86_64|Darwin arm64|Windows AMD64`. The target is `<system> <cpu>` now (no `apple-darwin`). `test_real_runtime.test_runtime_install_detects_configured_directory` still fails on `build=None`. For the backend, `--list-devices` is reliable (format above); tests can use `FAKE_LLAMA_GPU=none` for CPU-only.
- **testing-tuner** (from the second review, reproduced on the real binary): `tuner._plain_failure` joins `stderr + stdout`; with `-o json` stdout is `"[\n"`, so every real load failure is shown as `llama-bench stopped with an error (1): [`. Read the last stderr line first. Also send `-lm none` instead of the deprecated `-mmp 0`: real llama-bench then prints `DEPRECATED: -mmp and --mmap are deprecated … instead.` **without a newline**, glued to the next stderr line (the fake now does the same).
- **testing-tuner**: real llama-bench **never** prints the out-of-memory cause without `-v` (just `llama_bench: error: failed to load model '…'` or `failed to create context with model '…'`), so `tuner.OOM_TEXT` / `testing` OOM hints never match real output. Options: treat those two lines as "probably not enough memory" when a smaller setting in the same run worked, or re-run the one failing combination with `-v` and read the tail. Also `-fa` 0/1 is fine on the real bench (checked). Minor: `ttft_s` uses wall-clock time; the fake doesn't sleep by default (`FAKE_LLAMA_SLEEP`), so fake runs show ~0.005 s while `timings.prompt_ms` says seconds.
- **api-http**: `ServerRegistry.status()["config"]["host"]` is now always `127.0.0.1`; `model_path` is still absolute (strip it, as before). A non-ValueError start failure now shows "The model server could not be started." in `error`.
- **lead (`launch.py`)**: `server_env` copies `LLAMA_ARG_*` from the user's environment; `llama_server` filters them now, but export scripts built from `server_env` would still inherit them. `--mlock`/`--no-mmap` still work but are deprecated (`--load-mode`).
- **quantcheck**: `estimate_logits_bytes` is 4 bytes low (the real header is `_logits_` + three int32 = 20 bytes, not 16; real 162584 vs estimate 162580). Otherwise nothing needed — `kl_check` gives the right numbers with the fake's new real layout (checked by hand) and the real integration test passes.
- **evals**: the needle request already worked (the fake answers "The secret code is X."); now covered by `SeamTests.test_evals_needle_test_passes_against_the_fake`.

## Test evidence

- `python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: **635 tests, OK** (24 skipped: the real-runtime tests without a build).
- `tests/test_llama_server.py`: 65 tests (new: `RealLogTests` — every real error sample plus the new real failures; `SeamTests` — pid/stop idempotence, template kwargs, testing smoke test, evals needle test, env filtering, CORS, busy fixed port, progress total, registry after OSError, CPU-only reasons, draft timings, exit handlers on SIGTERM/SIGHUP, real exit code, odd replies; PeakMemory rows, peak before sampling, restart).
- `tests/integration/test_fake_matches_real.py`: 19 tests (was 8; the 2 failing ones pass).
- Real runtime: `LLM_CONFIG_REAL_RUNTIME=<bin> LLM_CONFIG_TINY_MODELS=/opt/llama-work/models python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real` → 43 tests, 1 failure: `test_runtime_install_detects_configured_directory` (runtime-downloads, see Requests). All `llama_server`, smoke, speed, tuner, quantcheck and raw-tool tests pass.
- Manual seams against the fake: `testing.speed_test`, `tuner.tune` (finds 8 threads and full offload), `quantcheck.kl_check` (mean KLD 0.03 for Q4_K_M).
