# Handoff: `testing` workstream

## What I built

`src/llm_configurator/testing.py` — "Test this model on my computer":

- **Smoke test**: loads the model with `LlamaServer`, asks one fixed question ("What is 17 + 25? Reply with just the number."), and checks the reply: not empty, valid UTF-8, not stuck repeating, no raw chat-template markers, and the right answer (42). Hidden reasoning is turned off with `chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "low"}`, and any `<think>…</think>` is removed before checking. A `<think>` with no closing tag counts as an empty answer. Load time is recorded.
- **Speed test**:
  - llama-bench measures reading speed (pp) and writing speed (tg) at the user's conversation length. The settings the bench reports are checked against the ones requested, the same way `runtime.bench` does.
  - One llama-server run measures the first-word delay (a ~1500-token prompt with `max_tokens=1`) while `PeakMemory` samples. A short 64-token reply follows, so the peak includes a working KV cache.
  - Peak memory is compared with `engine.allocations`.
- **run_tests**: runs smoke, speed or both, saves the speed record with `store.append("measurements", …)`, and returns a plain verdict.
- Servers and bench processes are always stopped, including on `Cancelled`, timeouts and exceptions. llama-bench runs under `Popen` with a poll loop that checks cancel and a deadline, then kills the whole process tree via psutil.

## Public API as built

```python
smoke_test(server_command, config, progress=None, cancel=None, server_factory=None, timeout=300) -> {
    "ok": bool, "stage_failed": None|"start"|"reply", "load_seconds": float|None, "reply": str|None,
    "checks": [{"name", "ok", "detail"}],  # names: not_empty, valid_utf8, not_repeating, no_template_leak, correct_answer
    "message": str, "log_tail": str}

speed_test(bench_command, server_command, variant, config, hardware, progress=None, cancel=None,
           server_factory=None, peak_factory=None, allocations=None, repetitions=2,
           bench_timeout=1800, server_timeout=300) -> {
    "measurement": record (§2.3, kind="speed_test"),
    "summary": {"pp_tps", "tps", "ttft_s", "depth"},
    "memory": {"estimated_ram_bytes", "estimated_vram_bytes", "peak_ram_bytes", "peak_vram_bytes",
               "within_estimate": bool|None, "note": str}}
    # raises ValueError(plain) on bench failure / settings mismatch / server start failure; Cancelled on cancel

run_tests(store, variant, config, commands, kind="full", hardware=None, progress=None, cancel=None,
          min_tps=None, server_factory=None, peak_factory=None, allocations=None) -> {
    "smoke": dict|None, "speed": dict|None, "verdict": "works"|"works_slowly"|"failed",
    "verdict_text": str, "speed_error"?: str}
```

- `commands` is `{"llama-server": argv, "llama-bench": argv}`. The short names `"server"` and `"bench"` also work.
- Helpers you can use: `bench_plan(context)`, `bench_args(config, n_prompt, n_gen, depth, repetitions)`, `parse_bench_json(text)`, `check_bench_settings(row, config)`, `reply_checks(text)`, `strip_thinking(text)`, `estimate_memory(variant, config, hardware, allocations)`, `memory_verdict(...)`, `run_process(argv, timeout, env, cancel)`, `verdict(smoke, speed, speed_error, min_tps)`.

### Measurement record

It has every existing key plus `kind="speed_test"`, `pp_tps`, `ttft_s`, `depth`, `peak_ram_bytes`, `peak_vram_bytes`, `estimated_ram_bytes`, `estimated_vram_bytes`, `settings`, `runtime`, and `id` (`secrets.token_hex(6)`). Notes on individual fields:

- `context` is the per-slot context and `users` is `config["parallel"]`.
- `threads` is `config["threads"]`, or llama-bench's reported `n_threads` when the config left it unset.
- `gpu_uuid` is only set when `gpu_layers > 0`.
- `raw` is `{"bench": [pp_row, tg_row], "server_timings": {...}, "plan": {...}}`.
- `runtime` is `{"version": str(build_number) or build_commit, "backend": row "backends" or config gpu_backend}`.

A test checks that `engine.matching_speed` accepts the record.

## Rules and decisions

- **Depth rule**: `depth = context − n_prompt − n_gen`, with `n_prompt = 512` and `n_gen = 128`. This is the contract's `-d <context-640>`, so the measurement ends exactly at the configured context. When `context < 768`, the sizes shrink to `n_prompt = max(32, context//2)` and `n_gen = max(16, context//4)`, and depth is `max(0, …)`. So `depth + p + n ≤ context` always holds.
- **Repetitions**: 2 (`-r 2`), to keep the test short. It can be overridden.
- **First-word delay (`ttft_s`)**: the wall clock of a non-streaming `max_tokens=1` request. That includes the HTTP round trip, which is what the user feels. If the wall clock is unusable, it falls back to `timings.prompt_ms / 1000`. The raw `prompt_ms` is kept in `raw.server_timings`. The prompt is `min(1500, context − 256)` tokens of deterministic, varied filler text (about 0.75 words per token).
- **Peak memory**: sampling starts after the server reports ready, because the pid is only known after `start()`. It covers the prompt plus a short generation. Loading spikes before "ready" are not captured.
- **Memory verdict**:
  - A pool is only compared when both the estimate and the peak are known. If nothing can be compared, `within_estimate` is `None`.
  - When a pool is over, the note says so, e.g. "Real memory use was over the estimate: RAM by 25%."
  - VRAM is `None` on non-CUDA backends (per the `PeakMemory` contract), so only RAM is compared there.
- **`engine.allocations` kwargs**: the call passes only the kwargs its signature accepts (`kv_cache_type`, `n_cpu_moe`, `unified`), checked with `inspect.signature`. So it works with both the current 4-argument version and the v0.4 version. If K and V use different cache types, it passes `f16`, the conservative choice.
- **Verdict**:
  - `failed` when the smoke test fails, or when only a speed test was requested and it failed.
  - `works` with an explanatory sentence when the smoke test passed but the speed test failed (the model does run).
  - `works_slowly` when `min_tps` is given and the measured tg is below it.
  - If the speed test fails, nothing is saved.
- **llama-bench arguments**: built here by `bench_args`, not by `tuner.bench_args`, so this module doesn't depend on another workstream's code. They mirror `launch.server_args`: `-ngl` uses `runtime_gpu_layers`, `-dev none` follows the same rule, and `launch.server_env` pins CUDA. If the tuner's version lands, the lead may switch to it.

## Deviations from the contract

- These are extra optional keyword arguments, which the contract allows:
  - `server_factory`, `peak_factory` and `allocations` on all three functions, so tests can inject fakes.
  - `min_tps` on `run_tests`.
  - Timeouts on `smoke_test` and `speed_test`.
- `speed_test` raises `ValueError` on failure instead of returning an error dict. `run_tests` catches it and reports `speed_error`.

## Known gaps

- Peak memory misses load-time spikes that happen before "ready" (see above).
- llama-bench and llama-server each load the model once. That is two loads for a full test, so it takes longer on big models.
- The smoke question is English arithmetic. Tiny models (under 1B) may fail it even when they work. If that proves too strict, `correct_answer` could become advisory.
- `not_repeating` treats 16 or more identical characters in a row as stuck, which could flag a reply that legitimately draws a line of `=` signs.

## Requests

- **server**: `LlamaServer.chat` should accept an optional `chat_template_kwargs=None` and forward it in the request body. I call it with that kwarg and retry without it on `TypeError`, so it still works if you don't add it. Please also set `.pid` once `start()` returns, and make `stop()` safe to call twice or after a failed start.
- **api**: call `run_tests(store, variant, config, {"llama-server": binary(store, "llama-server"), "llama-bench": binary(store, "llama-bench")}, kind, hardware, progress, cancel, min_tps=requirements.min_tps)`.
- **engine**: `matching_speed` could prefer `kind="speed_test"` records. They include `pp_tps`, `ttft_s` and `depth` for richer verdicts.

## How I tested

`tests/test_testing.py` has 29 tests, and the full suite (101 tests) is green. The tests use:

- A small llama-bench stand-in written to a temp dir at test time. It is a real subprocess that echoes the requested flags back as llama-bench JSON rows.
- A `FakeServer` and a `FakePeak`.

Covered cases:

- success (thinking removed, load time recorded, server stopped)
- out of memory at load
- garbage (repeating) reply
- template leak
- unfinished thinking
- a server without `chat_template_kwargs`
- cancel before start
- cancel mid-bench (the process is killed quickly; the server is never started)
- bench timeout
- bench settings mismatch (refused, not saved)
- bench out of memory
- JSON parsing with log noise and jsonl
- depth clamping
- bench args mirroring the launch settings
- the full record shape, accepted by `engine.matching_speed`
- memory over the estimate (percentage shown), and unknown memory staying `None`
- new `allocations` kwargs passed only when accepted
- server stopped when chat raises
- `run_tests` verdicts: works, works_slowly, failed, smoke-only, speed failure not saved, missing runtime, cancel

## llama.cpp facts assumed (for the harness to confirm)

- **llama-bench flags**:
  - `-m -p -n -d/--n-depth -r -o json -ngl -t -b -ub -ctk -ctv -ncmoe -dev`
  - `-fa on|off`, per the current README (`-fa, --flash-attn <on|off|auto>`). Older builds took `0|1`, so I only pass `-fa` when the config says `on` or `off`.
- **llama-bench JSON**:
  - A list of rows with `n_prompt`, `n_gen`, `n_depth`, `avg_ts`, `n_gpu_layers`, `n_threads`, `n_batch`, `n_ubatch`, `type_k`, `type_v`, `flash_attn`, `build_commit`, `build_number` and `backends`.
  - `flash_attn` is compared loosely: -1/0/1 or bool/"on"/"off".
  - The pp row is `n_gen == 0` and the tg row is `n_prompt == 0`, both at the requested `n_depth`.
  - Logs go to stderr. The parser also tolerates stray lines and jsonl.
- **Bench output checks**:
  - `n_cpu_moe` is only checked if the row has that field (field name not verified).
  - An out-of-memory failure shows "out of memory" or "alloc" in stderr.
- **llama-server**:
  - `/v1/chat/completions` returns `timings.prompt_ms`.
  - `chat_template_kwargs.enable_thinking=false` disables Qwen3-style thinking. `reasoning_effort` is read by gpt-oss templates.
