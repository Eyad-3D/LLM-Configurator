# Handoff: `server` workstream

Branch `claude/v04-server`. Files: `src/llm_configurator/llama_server.py`, `tests/fixtures/fake_llama.py`,
`tests/fixtures/__init__.py`, `tests/test_llama_server.py`.

## What I built

- **`llama_server.py`**: starts, talks to and stops llama-server safely.
  - Uses `launch.server_args` / `launch.server_env` only. It doesn't build arguments itself.
  - Output goes to a log file, never an unread pipe. There is no shell.
  - Windows uses `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`. POSIX uses a new session.
  - Stopping kills the whole process tree with psutil (terminate → wait → kill). This also happens on errors, time-outs and `Cancelled`.
  - An `atexit` hook stops any server still running when Python exits.
  - HTTP uses urllib with `ProxyHandler({})`, so a system proxy never sees loopback traffic.
  - llama.cpp stderr is mapped to plain reasons.
  - Also included: peak-memory sampling, and the user's one long-running server (`ServerRegistry`).
- **`tests/fixtures/fake_llama.py`**: a stdlib-only, cross-platform imitation of llama-server, llama-bench, llama-perplexity and llama-cli. Formats come from llama.cpp master (commit 6b790a9, checked 2026-09-24 against the source). Details are in "Fake behaviour" below.
- **`tests/fixtures/__init__.py`**:
  - `fake_command(mode)` returns `[sys.executable, <abs path>/fake_llama.py, "--as", mode]`.
  - `write_fake_gguf(path, architecture="llama", layers=32, experts=0, name=None, context_length=32768)` writes a tiny **valid GGUF v3 header** (metadata only, no tensors).
  - `FAKE_LLAMA` is the script path.

## Public API as built

```python
free_port() -> int
failure_from_log(text, returncode=None) -> str | None        # extra helper, used by failure_reason()
kill_tree(pid, timeout=10) -> None                           # extra helper
TIMING_KEYS = ["prompt_n","prompt_ms","prompt_per_second","predicted_n","predicted_ms","predicted_per_second"]

class LlamaServer(command, config, log_path=None)
    # command: str | list[str] argv prefix; config: launch config (normalized in __init__, model_path required)
    .start(timeout=300, progress=None, cancel=None) -> self
        # ValueError(plain reason + "(exit code N)") on early exit; ValueError on timeout; Cancelled on cancel.
        # Picks a free port if config["port"] is None and retries once with a new port if that port was taken.
        # progress: {"stage": "loading"|"ready", "done": seconds, "total": timeout, "message": str}
    .base_url -> "http://127.0.0.1:<port>" | None     .port   .pid   .started_at (ISO)   .load_seconds (float)
    .ready() -> bool (process alive and /health == 200)      .running() -> bool
    .chat(messages, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300,
          cache_prompt=False, enable_thinking=None, extra=None) -> dict
    .complete(prompt, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300, cache_prompt=False, extra=None) -> dict
        # both return {"text", "reasoning": str|None, "finish_reason": "stop"|"length"|None,
        #              "timings": {six TIMING_KEYS (None if missing) + any other numeric llama.cpp keys, e.g. cache_n},
        #              "usage": {...}, "seconds": wall-clock float}
    .tokenize(text, add_special=False, timeout=60) -> list[int]
    .request(path, body=None, timeout=300) -> dict     # raw JSON GET (body None) / POST, errors → plain ValueError
    .stop(timeout=10) -> exit code | None    # idempotent
    .log_tail(chars=4000) -> str     # still works after stop, even when the temp log was deleted
    .failure_reason() -> str | None
    .warnings() -> [str]             # e.g. "no usable GPU found" while the server still runs (CPU fallback)
    context manager: __enter__ starts (default timeout) if not started; __exit__ stops.

class PeakMemory(pid, gpu_backend=None, interval=0.25)
    .start() -> self    .sample()    .stop() -> {"peak_ram_bytes": int|None, "peak_vram_bytes": int|None, "samples": int}

class ServerRegistry(log_dir=None)      # log_dir optional: logs to <log_dir>/llama-server.log, else a temp file
    .start(command, config, progress=None, cancel=None, timeout=300) -> status   # stops any previous server first
    .stop() -> status
    .status() -> {"running", "starting": bool, "base_url", "openai_base_url" (<base>/v1), "pid", "config",
                  "started_at", "model" (alias or model file name), "log_tail" (2000 chars), "error": str|None}
```

The plain reasons returned by `failure_reason()` are:

| Cause | Reason |
|---|---|
| Out of memory, including CUDA/Vulkan/Metal allocation failures | "Not enough memory to load this model…" |
| Unknown architecture | "…does not know this model's design (arch)…" |
| Missing file | "The model file is missing…" |
| Damaged GGUF | "…looks damaged or incomplete…" |
| Port in use | "The network port is already used…" |
| Quantised V cache without flash attention | "Compressed notes (the KV cache…) need flash attention…" |
| GPU backend | "…could not use the graphics card…" |
| Unknown flag | "…does not understand one of the settings…" |
| Exit code -9 or 137 | "…most likely because memory ran out" |

HTTP error messages: a context overflow becomes "The text is longer than the model's context window…". A 503 becomes "still loading". Other errors become "The model server refused the request (code): message".

## Deviations and decisions

- `chat`/`complete` add the optional keywords `cache_prompt=False` (default off, so each timing re-reads the whole prompt), `enable_thinking` (sent as `chat_template_kwargs`) and `extra` (extra body keys). The result gains the keys `reasoning` and `seconds`. `timings` also passes through `cache_n` and the per-token values.
- `start()` returns `self`, so `LlamaServer(...).start()` chains.
- Status dicts have extra keys: `starting` and `error`.
- `ServerRegistry.start` **replaces** a running server instead of refusing, because there is one server per user. `status()` stays readable while a start is in progress (separate locks).
- `ServerRegistry.status()["config"]` contains the absolute `model_path`. **The api workstream must strip it before sending it to the browser** (contract rule 10).
- `complete()` maps llama.cpp's `stop_type` to `finish_reason` (`eos`/`word` → `stop`, `limit` → `length`). Usage there comes from `tokens_evaluated`/`tokens_predicted`.
- VRAM is sampled at most once a second, because nvidia-smi is slow to start. If nvidia-smi is missing or fails, VRAM is `None` for the rest of the run. If it prints `[N/A]` (Windows WDDM), the value is 0 for our pids. That can under-report, so treat 0 on Windows with caution.

## Known gaps

- The long-running server's log file is not rotated. llama-server logs a few lines per request.
- No streaming client. The fake does support `"stream": true` (SSE) for other users.
- `stop()` exit codes are unreliable after psutil reaps the process (often 0). Don't use them.
- Only NVIDIA VRAM is measured. ROCm/Vulkan give `None`, and on unified memory VRAM is counted as RAM, as the contract says.

## Requests

- **api**:
  - Strip `config.model_path` and `log_tail` paths from `/api/serve` responses, or show only `model`.
  - Keep one `ServerRegistry(log_dir=<app data>/logs)` per app and call `.stop()` on shutdown. The `atexit` hook is a backstop, not the plan.
- **testing / tuner / evals / quantcheck / runtime**: use `from fixtures import fake_command, write_fake_gguf`. Use a real file for `-m`: the fake refuses missing files like llama.cpp does. An empty file (`touch`) also works.
- **runtime**: the real `--version` goes to **stderr**, and master now prints `version: 0.5.0 (build N, commit HASH)` instead of the classic `version: N (HASH)`. Parse both. The fake prints the classic form by default and the new form with `FAKE_LLAMA_VERSION_STYLE=new`.
- **tuner**: llama-bench `flash_attn` in JSON is an **int** (-1 auto / 0 off / 1 on). `-r` is a single int, not a list. When one instance fails to load, llama-bench exits 1 with a JSON array left **unclosed**, so parse the complete rows defensively. `use_mmap` no longer exists; `load_mode` replaced it.
- **testing**: `/completion` no longer has `stopped_eos`/`stopped_word`/`stopped_limit`. Use `stop_type`.

## How I tested

- `tests/test_llama_server.py` has 45 tests. All of them spawn **real subprocesses** of the fake over loopback. Nothing touches the network.
  - Server start: the health wait including a real 503 body, and the progress events.
  - Chat and completion: arithmetic, the needle, stop words, max-token limits, token counts, the timings shape and key order, and prompt caching on and off.
  - Thinking on and off, and settings changing speed.
  - Tokenize, and the context overflow error.
  - Process-tree kill through a wrapper process. The context manager. Log file kept or removed.
  - Ports: explicit, auto-picked with retry, and busy.
  - Failure reasons: oom, the GPU-layer limit, arch, port, missing, corrupt, and an unknown architecture read from the GGUF header.
  - Backend warning, time-out, cancel during load, missing executable, proxy variables ignored.
  - Registry: start, restart, stop, crash status, failed-start status, and status while starting plus cancel.
  - PeakMemory: real RSS of a tree, mocked nvidia-smi summed over the tree, and `None` cases.
  - Fake bench: field list and order, cartesian order, ranges, jsonl/md, OOM leaving a partial array, knobs with a real optimum.
  - Fake perplexity: KLD round trip with ordering Q8_0 < Q4_K_M < Q2_K, the same model giving 0, too-short text, a bad base file.
  - Fake CLI and `--version`.
- The full suite passes: `python3 -m unittest discover -s tests` ran 117 tests OK. I ran the server tests three times in a row, all green, with no leftover processes.

## llama.cpp facts assumed

These were checked against llama.cpp master source at commit 6b790a9. The harness should confirm them on a real build.

- **`/health`**: 503 `{"error":{"code":503,"message":"Loading model","type":"unavailable_error"}}` while loading, then 200 `{"status":"ok"}`. llama-server binds the port **before** loading the model.
- **Chat response key order**: `choices, created, model, system_fingerprint ("b<build>-<commit>"), object, usage, id, timings`.
  - `timings` order: `cache_n, prompt_n, prompt_ms, prompt_per_token_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_token_ms, predicted_per_second`.
  - `usage` also has `prompt_tokens_details.cached_tokens`.
  - `message.reasoning_content` is present only when non-empty.
- **Request fields**: `cache_prompt` (default true on the server) and `chat_template_kwargs` are accepted. `max_tokens` is an alias of `n_predict`.
- **Errors**: `{"error":{"code","message","type"}}`. Context overflow is 400 `exceed_context_size_error` with "request (N tokens) exceeds the available context size (M tokens), try increasing it".
- **`/tokenize`**: `{"content", "add_special": false, "with_pieces": false}` returns `{"tokens": [...]}`, or `[{"id","piece"}]` with pieces.
- **llama-bench `-o json`**: field order as in `FIELDS` in the test file. Defaults are `-p 512 -n 128 -b 2048 -ub 512 -r 5 -fa auto -ngl -1`.
  - Nesting, outermost first: model, ngl, ncmoe, sm, mg, dev, b, ub, ctk, ctv, nkvo, fa, t, d. Then all pp tests, then all tg tests, then pg.
  - The ranges `a-b`, `a-b+s` and `a-b*s` are accepted.
  - The fake's `load_mode="mmap"` and `lazy_mode="none"` string values are **unverified guesses**.
- **llama-perplexity**:
  - The `[1]x.xxxx,[2]…` chunk values go to stdout. `Final estimate: PPL = %.4f +/- %.5f` goes to stderr.
  - It needs at least 2·n_ctx tokens.
  - The base file starts with `_logits_` + int32 n_ctx, n_vocab, n_chunk + tokens. The fake appends a JSON trailer instead of real logits, so fake base files are not real base files.
  - The KLD summary blocks use the exact labels and `%10.6f` formats reported by the source check.
- **`--version`**: stderr, `version: N (hash)` classic or `version: X.Y.Z (build N, commit hash)` new, then `built with <compiler> for <target>`.
- **Messages**:
  - Ready: `srv          main: model loaded` / `main: listening on http://host:port`. The old "server is listening" string no longer exists.
  - Failures: `cudaMalloc failed: out of memory`, `unable to allocate CUDA0 buffer`, `unknown model architecture: 'x'`, `gguf_init_from_reader: invalid magic characters`, `failed to open GGUF file '…' (No such file or directory)`, `couldn't bind HTTP server socket`, `invalid device: X`, `warning: no usable GPU found, --gpu-layers option will be ignored`, `V cache quantization requires flash_attn`.
- **Flags**: llama-server `-fa on|off|auto`. llama-bench `-fa` accepts on/off/auto and 0/1/-1 as comma lists.

## Fake behaviour (so others can rely on it)

Run it as `fake_command(mode) + [real flags]`. The mode can also come from argv[0] containing `llama-server|llama-bench|llama-perplexity|llama-cli`. With no mode it exits 2.

**Common to all modes**

- `--version` exits 0 with the version on stderr. Build `6512`, commit `fa4ec0de`. Override with `FAKE_LLAMA_BUILD` / `FAKE_LLAMA_COMMIT`. `FAKE_LLAMA_VERSION_STYLE=new` switches the format.
- Unknown flags exit 1 with `error: invalid argument: X` (bench: `error: invalid parameter for argument: X`).
- Bad cache types, `-fa` values and `-dev` names fail too. `-dev none` means CPU. `-dev CUDA0`/`Vulkan0`/`Metal`… are accepted.
- **Models**:
  - A missing file fails with the missing-file message.
  - A file with 1–3 bytes, or not starting with `GGUF`, fails as corrupt.
  - An **empty file is accepted** as a 32-layer llama.
  - A `write_fake_gguf` file is read for `general.architecture`, `<arch>.block_count`, `<arch>.expert_count` and `<arch>.context_length`.
  - Architectures outside the fake's known list (GGUF names such as `llama`, `qwen3`, `qwen3moe`, `gemma3`, `gpt-oss`, …) fail with "unknown model architecture".

**Speed model** (deterministic)

- tg (generation speed) = `FAKE_LLAMA_TPS` (default 40) × factors.
  - Threads: best at `FAKE_LLAMA_BEST_THREADS` (default 8). Below it the factor is `0.35 + 0.65·t/8`. Above it, −4% per extra thread.
  - Flash attention: off gives ×0.9. `auto` counts as on.
  - GPU share of layers g: `1/((1-g)/0.3 + g)`. `-ngl -1` or ≥ layers means all. CPU-only is 0.3×.
  - KV cache: q8_0 ×0.97, q4_0 ×0.93, a quantised V cache another ×0.98.
  - MoE `--n-cpu-moe k`: `1/(1+0.6·k/layers)`. It only applies when the GGUF has experts.
  - Depth d: `1/(1+d/16384)` with flash attention, `/6144` without.
  - Allocated context: `1/(1+ctx/1M)`.
- pp (prompt reading speed) = 12 × the base TPS, with the same thread factor. GPU `1/((1-g)/0.1+g)`, flash attention off ×0.85.
  - ubatch is best at 512: −15% per doubling or halving (floor 0.5).
  - batch is best at 2048: −5% per doubling or halving.
- `FAKE_LLAMA_SLEEP=<f>` really sleeps for that fraction of the reported compute time. By default it doesn't sleep.

**server**

- `/health` returns 503 during `FAKE_LLAMA_LOAD_SECONDS` (default 0.2), then 200.
- Endpoints: `/v1/chat/completions` and `/chat/completions`, `/completion` and `/completions`, `/tokenize`, `/detokenize`, `/v1/models` (the id is `--alias` or the model path), and `/props`. All use the real shapes above, and `"stream": true` works (SSE; chat ends with `data: [DONE]`).
- Context per slot is `-c / -np` (default `-c 4096`). Longer prompts get the real 400 error.
- It keeps a prompt cache for one slot. `cache_prompt` defaults to true, like the real server, and then reports `cache_n` and a smaller `prompt_n`.
- **Tokenizer**: regex pieces (words, punctuation, whitespace runs, `<|…|>` specials), each one token. The id is `3 + crc32(piece) % 151000`. The chat template is ChatML.
- **Answers**, first match wins:
  1. `FAKE_LLAMA_REPLY` (exact text, for testing reply checks).
  2. A needle `The (secret|magic|special|hidden) (code|number|word|password|key|city) is X` anywhere in the prompt returns that sentence with a period.
  3. Arithmetic in the last user message (`+ - * / ( )`, `x`, `×`) returns just the number, e.g. "What is 2+2?" → `4`.
  4. `capital of <France|Japan|Germany|Italy|Spain|Canada|Australia|Egypt>`.
  5. `reply with (only|just) the word X` → `X`, or a quoted text after reply/say/repeat.
  6. Otherwise one of five fixed plain sentences, chosen by a crc32 of the message.
- Replies are cut at `stop` strings (`stop_type` "word") and at `max_tokens`/`n_predict` (`finish_reason` "length").
- `FAKE_LLAMA_REASONING=1` adds `reasoning_content` and its tokens, unless `chat_template_kwargs.enable_thinking` is false.
- `FAKE_LLAMA_FAIL` and related knobs (all print realistic stderr and exit 1 after the load delay unless noted):
  - `oom`: out-of-memory messages (a CUDA variant when ngl ≠ 0, a CPU one otherwise).
  - `arch`: unknown architecture `'made-up-arch'`.
  - `port`: the bind error, printed immediately.
  - `corrupt`: bad magic.
  - `backend`: prints `ggml_cuda_init` failure and the "no usable GPU" warning, then **keeps running on CPU** speeds. A `-dev` other than `none` fails with "invalid device".
  - `FAKE_LLAMA_MAX_GPU_LAYERS=N`: more than N layers offloaded gives OOM.
  - `-ctv q8_0` with `-fa off` fails with the flash-attention message.
- It logs realistic llama.cpp lines, including `load_tensors: offloaded X/Y layers to GPU`, `main: model loaded` and `main: listening on http://host:port`.

**bench**

- Comma lists and ranges for `-p -n -d -ngl -t -b -ub -fa -ctk -ctv -ncmoe -sm -mg -nkvo -dev -mmp`. `-pg p,n` can be repeated. `-r` is a single int.
- Output formats: `-o md|json|jsonl|csv` (sql is accepted but prints like md).
- Rows are the cartesian product in real llama-bench order, with every real JSON field.
  - `samples_ts` holds r values within ±0.8% of the mean, and `stddev` is computed from them.
  - `n_depth` (d) slows both pp and tg rows: they are measured at depth d. A `-pg` row is measured at d + n_prompt/2.
- OOM (`FAKE_LLAMA_MAX_GPU_LAYERS` / `FAIL=oom`) prints the rows finished so far, leaves the array unclosed and exits 1, like the real tool.
- `backends` is `CUDA`, or `CPU` under `FAIL=backend`.

**perplexity**

- `-f FILE -c N` (default 512) needs ≥ 2N tokens. It prints the chunk list and the final estimate.
- The PPL is derived from the model filename: base 6–9 × (1 + quant KLD).
- With `--kl-divergence-base F` (and without `--kl-divergence`) it writes F: `_logits_` header + tokens + a JSON trailer.
- `--kl-divergence --kl-divergence-base F` doesn't need `-f`. It prints the per-chunk table and all three summary blocks.
- The mean KLD comes from the quant in the filename: IQ1 0.9, Q2_K 0.25, IQ2 0.3, Q3_K 0.09, IQ3 0.1, IQ4 0.035, Q4_0 0.05, Q4_K 0.03, MXFP4 0.03, Q5 0.012, Q6_K 0.005, Q8_0 0.0015, BF16/F16 0.0002, otherwise 0.02. It is 0 when the model is the one that wrote the base file.
- Same top p = `100 − 60·√KLD` (floor 50). RMS Δp = `12·√KLD`.

**cli**

- `-p PROMPT -n N` prints the prompt plus the answer (same answer rules) to stdout, and `llama_perf_context_print` lines to stderr. `--no-display-prompt` hides the prompt.
