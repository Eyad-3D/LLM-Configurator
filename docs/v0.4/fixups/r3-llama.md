# Round 3: `r3-llama` (llama_server, testing, tuner, the fake llama.cpp)

Branch `claude/v04-r3-llama`, based on `claude/keen-volta-7mnrm5` (3f6d208). Every llama.cpp behaviour below was checked on a
**real** build: llama.cpp 4df29be from the PyPI `llama-cpp-python==0.3.35` sdist (`scripts/build_llama_cpp.sh`), CPU only,
with the tiny models from `scripts/make_tiny_models.py`. There is no GPU here, so GPU out-of-memory text is still unverified.

## Items

| Item | Result |
|---|---|
| Fake llama-bench accepts `-lm/--load-mode`, prints `error loading model: …` for load failures | **Already done** in the fix-up round. Now proven line by line against the real binary (`RealAndFakeSideBySideTests.test_bench_load_failures_read_the_same`, with and without `-v`, for a corrupt file, an unknown architecture and a missing file; `test_bench_load_mode_none_like_real`). |
| `tuner._plain_failure` reads stderr first; `-lm none` instead of `-mmp 0` | `-lm none` was already done. **Fixed** the rest: `bench_failure` now reads **only stderr** (stdout is `"[\n"` with `-o json`), splits off the `DEPRECATED: … instead.` text that real llama-bench glues to the next line, ignores lines that mention memory but carry on (`warning`, `falling back`), names a missing file by its file name only, and explains "quantized V cache requires flash_attn". |
| Honest out-of-memory detection | **Fixed.** "Ran out of memory" is said only when llama.cpp itself printed it (`-v` is always passed, so it does). With no cause at all the text is `NO_CAUSE` ("…gave no reason. They may need more memory than is free, or the file may be damaged."). Inside a tuner step it becomes `PROBABLY_MEMORY` ("Probably not enough memory: a lighter setting of the same model worked just before…") **only** when a setting with a lower `_risk` already ran in the same step. `testing._start_message` no longer turns any reason containing "memory" or "alloc" into "ran out of memory"; the killed-process guess ("most likely because memory ran out") is passed on as the guess it is. No verbose re-run was added: `-v` is already on every llama-bench call, so a re-run would print nothing new. |
| Filler prompt sizing (`server.tokenize`, trim to `context − 256`) | **Already fixed** in the fix-up round (`fitted_prompt`, commit 2481881). I reproduced the api sessions' failure on the pre-fix-up `testing.py` (both 2048 and 4096 fail with "The text is longer than the model's context window") and showed the current code passes at 256, 384, 512, 768, 1024, 2048, 4096 and 8192. Hardened it: the shrink loop now always makes progress (≥ 16 tokens per try, 8 tries), so it can no longer give up with a prompt that is still too long; `ttft_target(config)` holds the sizing rule (per user context, since `launch` passes `-c context×parallel`). |
| Tuner measures the final `best` at full depth and stores `settings.depth` | **Done.** See below. |
| `ServerRegistry` / exit handlers keep `LLAMA_ARG_CORS_ORIGINS=localhost`; optional `LLAMA_ARG_ENDPOINT_SLOTS=0` | CORS was already set. **Added** `LLAMA_ARG_ENDPOINT_SLOTS=0` in `llama_server.server_env` (every server the app starts, including the user's). Real llama-server then answers `/slots` with `501 {"error":{"code":501,"message":"This server does not support slots endpoint. Start it with `--slots`","type":"not_supported_error"}}`; chat is unaffected. Honest note: on this build `/slots` showed sampler settings and token counts, **not** prompt text, so this is a small hardening, not a leak fix. The app never reads `/slots`. The fake now serves `/slots` and honours the variable. |

### Tuner: full-length check

The search still runs at a short depth (`min(1024, context/4)` rounded to 256) to save time. After the search and the
confirmation run, the tuner measures the **start and the winner** again at `min(context − n_prompt − n_gen, 32768)` tokens
(= `context − 640` by default, the same rule and cap as the speed test), in one llama-bench run when they differ in one
setting, else two. Then:

- `baseline`, `best_result` and `improvement` are the full-length numbers (so the UI/CLI "before → after" stays like-for-like);
  the short-depth numbers move to `result["search"] = {"depth", "baseline", "best_result", "improvement"}`.
- `result["depth"]` and `result["settings"]["depth"]` say the depth `best_result` was measured at. `result["settings"]` also
  carries `flash_attn, cache_type_k, cache_type_v, batch, ubatch, n_cpu_moe` of `best` (the same keys the speed test stores).
  `result["bench"]` still describes the search runs.
- If the winner fails at full length, or is not faster than the start there by more than the noise margin (the same
  `max(min_gain, noise, drift)` rule as every other comparison), **the start is kept** and a note says why (with "Probably
  not enough memory…" only when the winner differs from the start just by more offload or bigger chunks, see `_lighter`).
- If the start fails at full length but the winner runs, the winner is kept, the short-test numbers stay, and the note gives
  the winner's full-length speed. If both fail, the note says the chosen context may not fit in memory.
- **Running out of time is not a verdict**: a timed-out full-length run keeps what the search found (short depth) with a
  "ran out of time" note, and no further full-length run is started. Each run gets its share of the time left.
- When the numbers are full-length, the per-step notes are labelled "In the shorter search test: …", since they are
  short-depth gains. "Your starting settings were already the fastest" appears only when the search found nothing;
  when a winner was later dropped, the first note says the start is kept and why.
- It runs only when the estimated cost fits the remaining budget (plus the usual one-trial allowance). Otherwise
  `settings.depth` stays the short depth and a note says "There was no time left to measure with your full conversation length…".
  The estimate is `(repetitions+1) × 1.5 × ((2 × depth + n_prompt) ÷ reading speed + n_gen ÷ writing speed)` per setting.
  On the real build it errs on the safe side (the check itself took 7.2 s at context 4096). With no reading speed
  (`n_prompt=0`) the check is skipped with its own note.
- `tune(..., verify_full_depth=False)` switches it off (used by a few tests).

The existing test that asserted `"settings" not in result` was pinned to the old reading; it now checks the new shape.

## Real-runtime evidence (llama.cpp 4df29be, CPU, 4 cores)

| Check | Result |
|---|---|
| Old `testing.py` (4b8e02d) speed test, tiny llama F16, context 2048 / 4096 | both **fail**: "The text is longer than the model's context window…" (reproduced) |
| Current `speed_test` at 256 / 384 / 512 / 768 / 1024 / 2048 / 4096 / 8192 | all OK; first-word prompt 128 / 170 / 261 / 519 / 739 / 1509 / 1509 / 1509 tokens |
| `llama-bench` corrupt / unknown arch / missing file, **no** `-v` | exit 1, stdout `[`, stderr only `llama_bench: error: failed to load model '…'` |
| same **with** `-v` | adds `llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within the file bounds…` / `unknown model architecture: 'notarealarch'` / `gguf_init_from_file: failed to open GGUF file '…' (No such file or directory)` |
| Real CPU out of memory (`ulimit -v 3000000`, `-d 0,2000000`) **no** `-v` | the `d=0` rows are printed, then only `llama_bench: error: failed to create context with model '…'` |
| same **with** `-v` | `ggml_aligned_malloc: insufficient memory (attempted to allocate 7813.00 MB)`, `…failed to allocate buffer of size 8192524288`, `alloc_tensor_range: failed to allocate CPU buffer…`, `llama_init_from_model: failed to initialize the context: failed to allocate buffer for kv cache` (context padded to 256 cells: 2,000,128 × 4 KiB) |
| llama-server `-c 2000000` under the same limit | prints the same four lines **without** any flag, then `exiting due to model loading error` |
| `LLAMA_ARG_ENDPOINT_SLOTS=0` | `/slots` → 501 (text above); unset → 200 with slot settings |
| `tune`, tiny llama, context 4096, 60 s | 21.7 s, converged; search at 1024: 241 → 649 t/s writing; full length 3456: 83 → 352 t/s (4.22×), threads 4, flash attention off |
| `tune`, tiny llama, context 8192, 12 s budget | 10.5 s, stopped on budget, depth 1024 with the "no time left" note |
| `tune`, tiny qwen3moe, context 2048, 40 s | 13.5 s, full length 1408: 205 → 667 t/s |
| `python3 -m unittest discover -s tests` | 847 tests OK (32 skipped: real-runtime tests without the env vars) |
| `LLM_CONFIG_REAL_RUNTIME=… LLM_CONFIG_TINY_MODELS=… python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real` | 52 tests OK |

New tests: `tests/test_tuner.py` `FullDepthTests` (14) and `FailureTextTests` (11); `tests/test_testing.py` (3: lumpy
tokenizer, target per context, start-failure wording); `tests/test_llama_server.py` `test_slots_monitoring_page_is_off`;
`tests/integration/test_fake_matches_real.py` `FakeMemoryAndSlotsTests` (3, normal suite, pinned to the captured real
text), `RealAndFakeSideBySideTests` (4, runs the real binary and the fake on the same command) and `RealRoundThreeTests`
(2: speed test at 1024/2048/4096, tuner full-length check). The fake gained `FAKE_LLAMA_MAX_CONTEXT` (context
out-of-memory with the real lines; llama-bench only with `-v`, llama-server always).

## Independent review

A separate reviewer read the diff and reproduced two real problems with scratch scripts. All are fixed, with tests:

1. A full-length run that ran out of time counted as "the winner failed", and good tuned settings were dropped. It is now
   treated as "not measured".
2. A failing start at full length hid the winner's result. Both cases are now reported.
3. The full-length comparison had no noise margin, so a winner 0.1 % faster was kept, next to notes saying "about the same".
4. "Your starting settings were already the fastest" appeared after a winner had been dropped.
5. `PROBABLY_MEMORY` could be claimed in the cache step, where the heavier f16 setting had run. `_risk` puts a
   compressed cache last, but that is the lighter one.
6. Full-length and confirm runs were counted in "N settings failed to run and were ignored".
7. The "no time left" note appeared when the real reason was a missing reading speed.
8. `std::bad_alloc` is now read as out of memory. A killed run (-9/137) now says "possibly because memory ran out".
   The last-line fallback skips warning and deprecation lines.

Outside my files (see Requests): `run.js` counting `full_depth` trials as tried settings, and a stale comment in `app.py`.

## Rejected

- **A separate verbose re-run on failure**: `-v` is already passed on every llama-bench call (fix-up round), so the cause is
  in the first run's stderr when llama.cpp gives one. A re-run would cost a model load and print the same text.
- **"Probably not enough memory" across different steps or processes of other models**: only a lighter setting of the same
  model in the same step counts as evidence; anything wider would be a guess.

## Requests

- **r3-app-cli (`app.tune_job`)**: the tune result now has `settings` (with `depth`), a top-level `depth`, and `search`.
  With `{**result, …}` they reach the tuned record, so `engine.matching_tuned` sees `settings.depth = context − 640` and
  marks the tune verified. The comment "No §2.3 tune measurement: the tuner measures at a shallow depth" no longer holds
  when `result["depth"] == result["settings"]["depth"]` is the full length; the optional `kind="tune"` measurement can use
  `best_result`, `result["settings"]`, `result["depth"]`, `threads`, `runtime`, `runtime_build`. Please update that
  comment in `app.py` (around line 306). When the check was
  skipped (`depth` is the short search depth) it should stay out of `measurements`, as today.
- **r3-engine-community**: nothing required. `best_result.tps` is now measured at `settings.depth` (full length when the
  budget allowed). Contexts above 33,408 are capped at depth 32,768 (like speed tests), so with `FULL_DEPTH_SLACK = 1024`
  they never count as verified; consider the same exception you make for speed tests, if any.
- **r3-ui (`run.js` ~line 1001)**: `tried` excludes `baseline` and `confirm` trials. Please also exclude
  `t.step === "full_depth"`: those one or two runs re-measure the start and the winner and are not settings tried.
  `baseline` and `best_result` are both full-length numbers now when `depth === settings.depth > search.depth`
  (like-for-like). The short-search numbers are in `result.search` if you want to show them.
- **lead (`launch.server_env`, `tests/integration/test_real_runtime.py`)**: export scripts could also set
  `LLAMA_ARG_ENDPOINT_SLOTS=0` like `llama_server.server_env`. `test_real_runtime.test_speed_test` still uses context 8192
  as a workaround for the old filler bug; 2048 works now (see `RealRoundThreeTests.test_speed_test_fits_short_contexts`).
- **docs (`llama-cpp-facts.md`)**: add the CPU out-of-memory text above (llama-bench hides it without `-v`, llama-server
  always prints it) and the `/slots` 501 answer.
