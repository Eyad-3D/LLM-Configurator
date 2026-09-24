# Fix-up: `testing-tuner`

Branch `claude/v04-fix-testing-tuner`. Files: `src/llm_configurator/testing.py`, `src/llm_configurator/tuner.py`,
`tests/test_testing.py`, `tests/test_tuner.py`, this file.

How I reviewed: three read-only review agents, each with its own lens:
- seams: calls into `llama_server`, `launch`, `engine` and `community`, and what `app`, `cli` and `ui` read back;
- real llama.cpp: flags and JSON checked against `tests/integration/samples/`;
- budget, cancel and measurement honesty.

Then I ran everything against a real llama.cpp build (commit 4df29be, CPU, 4 cores) with the tiny dense and MoE models.

## Fixed

### testing.py

1. **Thinking was never switched off (high).**
   - The code called `server.chat(..., chat_template_kwargs=...)`, but the real `LlamaServer.chat` has no such keyword. The call raised `TypeError`, and a blanket `except TypeError` quietly retried without it.
   - Result: reasoning models (Qwen3 and similar) spent the whole 200-token smoke budget thinking into `reasoning_content`, so they "failed" with an empty reply.
   - Now the switch goes in through `extra={"chat_template_kwargs": {"enable_thinking": False, "reasoning_effort": "low"}}`, with `cache_prompt=False` sent explicitly. The `TypeError` fallback is gone, so real bugs are no longer retried silently.
   - If a reply holds only reasoning, the smoke test now says the model "spent its whole answer thinking".
   - The test fake had an invented signature, which is how this slipped through. A new test checks that the fake's `chat`, `tokenize`, `start`, `stop` and `log_tail` signatures, and the `PeakMemory.__init__` signature, match the real classes exactly.
2. **Measurement honesty for the first-word delay (TTFT).**
   - A tiny warm-up request now runs before the timed one.
   - The filler prompt is sized with the model's own `/tokenize` (mismatch 10). It shrinks until it fits `min(1500, context − 256)`, and falls back to the word estimate if tokenizing fails.
   - `cache_prompt: false` is always sent (mismatch 7); real `cache_n` is 0.
   - The record gains `ttft_prompt_tokens` (the server's own `prompt_n`).
   - The verdict now says "starts answering a 1,509-token new message after X s". It no longer puts the first-word delay next to "N tokens already in memory", which only the bench measured.
3. **Peak memory now includes loading.**
   - The sampler starts as soon as the server has a process id, while `start()` is still loading. Before, it started after "ready" and missed load-time peaks.
4. **§2.3 record and sharing seams.**
   - `runtime.version` is now `b<build_number>`, the llama.cpp release tag. The community `_VERSION` pattern only takes up to 4 plain digits, so builds from 10000 up would have been dropped.
   - `runtime.backend` is now lower-case (`cpu`, `cuda`, `vulkan`, `metal`, `rocm`) instead of the bench's `"CPU"` or `"CUDA,CPU"`. The community `BACKENDS` set would have thrown those away.
   - `settings` always includes `cache_type_k` and `n_cpu_moe` (engine request).
   - `id`, `kind`, `settings` and `runtime` are always present (community request).
   - A new test puts the record through `community.anonymize`.
5. **Bench behaviour.**
   - Depth is capped at `MAX_BENCH_DEPTH = 32768`. llama-bench re-reads the whole depth before every repetition, so a 128K context on CPU would take hours. `depth` in the record says what was used.
   - `mmap=False` now becomes `-lm none`, mirroring the server's `--no-mmap`. Before, it wasn't mirrored at all.
   - `-v` is now passed, so a failed load shows llama.cpp's real reason.
   - JSON parsing now shares `tuner.parse_rows`, which copes with an unclosed array and `samples_ts` lists.
   - Failure text no longer contains file paths.
6. **Progress units.** `server.start` reported progress as elapsed seconds out of its timeout. That is now wrapped so the test's own stage and step counts are kept.
7. Removed dead code in `verdict`. `hardware=None` no longer crashes at the end of `speed_test`.

### tuner.py

1. **Budget.**
   - `fit` could go negative. `[:-1]` still ran trials after the time reserved for the confirmation run was gone, so winners went unconfirmed. `fit` is now `max(0, …)`.
   - A zero-second run caused a `ZeroDivisionError`. The per-trial cost now has a floor of 0.01 s.
   - The baseline gets the full `timeout` (it can't be estimated yet). Before, it got 2 × budget and was then told "Try a smaller context". A baseline timeout now says the model is too slow to tune here.
2. **Fair comparisons.**
   - The current settings are re-measured in every step, even when they use a "llama.cpp default" (`threads=None`, `flash_attn=auto`). That costs one extra run.
   - Before, candidates were compared with the cold baseline number from minutes earlier.
   - **Drift:** the gap between two measurements of the same settings is tracked as `drift`. Every later gain, and the confirmation, must beat `max(3 %, stddev sum, drift)`.
   - On the real tiny models, llama-bench's in-process stddev said about 1 %, but the same settings moved about 20 % between processes. The tuner used to chase that noise for 3 passes. The final check then undid it, and the notes still claimed gains.
3. **Confirmation.**
   - The confirmation now needs the same noise margin over the baseline. It used to accept any gain above 0.
   - When it fails, the per-step "X made writing N % faster" notes are dropped, so they no longer contradict "did not hold up".
4. **Real llama-bench syntax.**
   - `-fa on/off` instead of `1/0`, as `--help` documents. Both work on the real build (see evidence below).
   - `-lm none` instead of the deprecated `-mmp 0`.
   - `-v`, so out-of-memory and other load errors are visible. Without it, real llama-bench prints only `failed to load model '<path>'`, and the OOM regex could never match.
5. **Failure text.** It now comes from stderr, not stdout, which is only `[`. It gives llama.cpp's cause (`error loading model: …`) with paths replaced by `<file>`. Paths used to leak into `trials[].error`.
6. **Process cleanup.**
   - `run_process` now uses temp files and runs in its own session or process group.
   - It kills the whole tree (psutil) in a `finally`, so cancel, timeout, `KeyboardInterrupt` and wrapper scripts can't leave llama-bench holding VRAM.
   - A terminal Ctrl+C no longer races the cancel flag.
7. **Result.**
   - New keys `runtime`, `runtime_build`, `threads` (the thread count llama-bench really used when unset), `depth` and `drift`, so the api can write the pinned "tune measurement".
   - The old `settings` key (bench sizes) is renamed to `bench`, because §2.3 `settings` means llama.cpp settings. Nobody read the old key.
   - `pp` and `tg` rows with an unusable `avg_ts` are skipped instead of hiding a good row.

## Rejected ideas and why

- **Send `-fa 1/0` for old builds.** `--help` documents `on|off|auto`, and the app installs a current build. The server also uses `on/off`. Keeping one spelling is better than guessing at old builds.
- **Run the tuner at the full `context − 640` depth.** It would multiply tuning time at long contexts, and the budget is 1–30 min. Tune records now carry `depth`. The engine should label or scale them (see Requests).
- **Estimate the first-word delay from `timings.prompt_ms`.** The wall clock of a non-streamed `max_tokens=1` request is what the user feels. `prompt_ms` stays in `raw.server_timings`.
- **Make `correct_answer` advisory.** The real tiny models answer 42. A model that can't add 17 + 25 is not "working".
- **Re-run failed loads with `-v`.** Always passing `-v` is simpler. It adds about 20–35 KB of stderr per load, which goes to a temp file.

## Requests

- **engine:**
  - `matching_tuned` treats `best_result.tps` as the speed at the full context. It was measured at `result["depth"]` (≤ 1024). Please label it, or scale it like `interpolated_speed` does.
  - `matching_speed` needs `record.threads == threads`. testing now stores the thread count llama-bench really used when the config leaves threads unset, so pass the same resolved number.
- **api (`app.tune_job`):**
  - Store `depth`, `runtime`, `runtime_build`, `n_cpu_moe` and `kv_cache_type` (`best.cache_type_k`) in the tuned record, as in the FIXUPS pinned shape.
  - Build the optional §2.3 `kind="tune"` measurement from `best_result`, `best` (settings), `threads`, `runtime`, `runtime_build` and `depth`.
  - Pass `min_tps=requirements.min_tps` to `run_tests`, so `works_slowly` can happen.
- **ui (`run.js`):** the tuner already ends `notes` with the stop reason, so drop the UI's own stop sentence (it shows twice).
- **server-fake:**
  - The fake llama-bench rejects `-lm none` (`invalid parameter for argument: -lm`). The real one accepts it and reports `"load_mode": "none"`. Please accept `-lm/--load-mode`.
  - Please also print a llama.cpp-style `error loading model: …` line on stderr for load failures, since `-v` is now always passed.

## Test evidence

- `python3 -m unittest tests.test_testing tests.test_tuner`: 83 tests, OK (was 64; new tests cover every item above).
- `python3 -m unittest discover -s tests`: 623 tests. The only 2 failures are the known server-fake ones (`test_bench_has_no_version_flag_like_real`, `test_server_rejects_removed_draft_max_like_real`).
- Real runtime (`LLM_CONFIG_REAL_RUNTIME=…/build/bin LLM_CONFIG_TINY_MODELS=/opt/llama-work/models python3 -m unittest tests.integration.test_real_runtime`): 24 tests.
  - `test_smoke_test`, `test_speed_test`, `test_tuner_small_budget` and `test_tuner_bench_args_accepted_by_real_bench` pass.
  - The only failure is `test_runtime_install_detects_configured_directory` (runtime-downloads area, known).
- Real llama-bench facts checked here:
  - `-lm none` works (row `"load_mode": "none"`).
  - `-fa on` with `-ctk/-ctv q8_0 -d 64` gives `"flash_attn": 1`.
  - Without `-v`, a corrupt file prints only `llama_bench: error: failed to load model '…'`. With `-v` it also prints `llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within the file bounds…`, and an unknown architecture prints `unknown model architecture: 'notarealarch'`.
  - A two-model run where the second model fails leaves the JSON array unclosed.
- Real runs of the modules (context 2048, CPU):

| Run | Result |
|---|---|
| smoke, tiny-llama-F16 | ok, reply `'42'`, load 0.1 s |
| speed, tiny-llama-F16 | pp 3050 t/s, tg 486 t/s at depth 1408. First word 0.41 s for 1509 prompt tokens (`cache_n` 0). Memory within estimate. `runtime {"version": "b1", "backend": "cpu"}`, threads 2 |
| speed, tiny-qwen3moe-F16 | pp 2285 t/s, tg 271 t/s at depth 1408, first word 0.47 s |
| tune, dense, default settings, 40 s balanced | converged in 6.8 s: "starting settings were already the fastest". Before the drift fix: 3 passes chasing ±20 % noise, then rejected at confirmation |
| tune, dense, `threads=1` start, 40 s | best threads 4, improvement 3.61×, confirmed, 21.5 s |
| tune, MoE, `gpu_layers=4 n_cpu_moe=2`, 30 s | `n_cpu_moe` 0, 2 and 4 swept in one `-ncmoe` comma list, converged in 9.4 s (within budget; 27.3 s before the drift fix, also within budget) |
