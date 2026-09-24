# Fix-up: `quantcheck` (compression-loss check)

Branch `claude/v04-fix-quantcheck`. Files changed: `src/llm_configurator/quantcheck.py`, `tests/test_quantcheck.py` (27 → 49 tests), this note. The corpus was checked and left unchanged.

## Fixed

1. **Cut-off output from llama-perplexity (llama-cpp-facts mismatch 4).**
   - `parse_kld_output` now returns two extra keys:
     - `complete`: True only when the closing report arrived in full. That means the `Mean KLD` line plus the final `Same top p` line; the early-2024 layout never prints `Same top p`, so it only needs `Mean KLD`.
     - `from_rows`: the keys that had to be filled from the last per-window row.
   - `kl_check` re-runs a candidate whose report was cut short. It keeps the saved reference predictions, so the reference is never run again. There are at most `MAX_ATTEMPTS = 3` runs. Only after that does it fall back to the best partial run.
   - The result then carries `partial`, a plain sentence such as "…cut short 3 times in a row, so some figures come from its last progress line, which covers all 12 windows but with fewer decimal places; missing: median, worst 1%". The same sentence goes into `plain` and `notes`.
   - Every candidate result now also has `attempts`.
   - A non-zero exit, or a known error (for example "inconsistent vocabulary"), is **not** retried.
   - An unknown silent failure (exit 0, no numbers at all) is retried and then reported as an error.
   - **Retries use half the CPU threads** (`_fewer_threads`). The root cause, from llama.cpp's `common/log.cpp`: the logger is deliberately never shut down (upstream issue 22142), so whatever is still queued at exit is lost. Busy compute threads starve the log thread. The numbers do not depend on the thread count.
2. **A last line cut off in the middle is ignored.** Found by the review: output ending in `Same top p: 1` used to count as complete, with `same_top_p = 1.0`. Now `_whole_lines` drops a final line that has no newline before parsing.
3. **A candidate timeout is now that candidate's `error`.** Before, it threw away the whole check. A reference timeout still raises. If a retry times out, the earlier partial result is kept.
4. **The progress bar never moves backwards** when a candidate is re-run.
5. **No absolute paths in results or messages.**
   - `reference.path`, `results[label].path` and `corpus.path` became `file` (the bare file name).
   - The temp-space note says "the system's temporary folder" or "the work folder".
   - Error tails from llama-perplexity have folder names removed, including the other parts of a split model.
   - "Model file not found" now shows only the file name.
   - Nothing in the api/cli/ui read the old `path` keys (checked with grep).
6. **Size of the predictions file.** The header is 20 bytes (`_logits_` plus 3 × int32), not 16. The formula now matches `perplexity.cpp` line for line.
   - The review confirmed it gives the recorded 216,772 bytes for the tiny model. That model's real vocabulary is **422** tokens, so the "464-token" figure in `llama-cpp-facts.md` is wrong (lead's file).
7. **Vocabulary size.** The code reads `summary["vocab_size"]` from `gguf.read_metadata` when the discover session adds it. Otherwise it uses scalar metadata keys, and otherwise the 262,144 assumption. When the size is unknown, the note now says "assumes a very large one" instead of "at most", because vocabularies above 262k exist.
8. **Only the settings the check uses are validated.** These are `gpu_layers`, `total_layers`, `gpu_uuid`, `gpu_backend`, `device`, `threads` and `n_cpu_moe`, passed through `launch.normalize`. Before, a server-only setting (for example a q8_0 V cache with flash attention off) could fail the check.
9. **Smaller fixes:**
   - A `work_dir` that doesn't exist is created. An unusable one gives a plain error with no path.
   - `corpus.chunks` and `corpus.tokens` follow the chunk count llama.cpp reports.
   - The reference PPL falls back to the last `[N]x` progress value when the `Final estimate` line is lost.
   - New plain reasons for:
     - a reference file saved with a different window size;
     - an unreadable predictions file;
     - an unknown argument.
   - `process.wait` in cleanup no longer hides the original error.
   - The source comment for the "different top word" bands now names the right README table: the "LLaMA 2 vs 3" table, not the scoreboard.

## Checked and left as is

- **Flags.** Every flag passed exists in `help-llama-perplexity.txt`: `-m -f -c -b --chunks --kl-divergence-base --kl-divergence -ngl -dev -t --n-cpu-moe`. No server-only flags are passed. The `launch` helpers still have the same names and signatures.
- **Thresholds.**
  - The subagent checked every cited KLD figure against the vendored `tools/perplexity/README.md`: Q8_0 0.001355, Q6_K 0.005452, Q5_K_M 0.010762, Q4_K_M 0.031273, Q3_K_M 0.101913, Q2_K 0.331–0.445, IQ1_M 1.393.
  - It also checked the "different top word" figures (2.3, 4.0, 8.1 and 28.9%).
  - The verdicts are labelled rough guidance in both the notes and the docstring.
- **Corpus.** The subagent spot-checked about 10 distinctive passages: 43,750 bytes, stories with invented names, explainers, a recipe, a letter, an FAQ, a chat, and Python/JS/SQL/JSON. It found no match with known texts.
- **Temp cleanup.** The folder is removed in a `finally` on success, error, timeout, cancel and unexpected exceptions, and also during retries. Tests cover each case.

## Rejected (with reasons)

- **More than 3 attempts:** each retry is a full pass over the model, which takes minutes on real models. With the half-threads retry, 3 attempts left 1 fallback in 40 even under heavy load, and 0 in 40 without load.
- **Fewer threads from the first run:** it would slow every check to protect against a problem that is rare on an idle machine (1–2 cut-offs in 40 runs).
- **`--poll 0`:** it helped less than fewer threads (14/40 against 22/40 cut off under load).
- **Checking the saved predictions file against its header size:** the real tool writes the file through a stream that is closed normally, and a candidate run already reports an unreadable file. The shared fake writes a smaller, non-real file, and a strict check would break it for no real gain.
- **Reading the token list length ourselves:** that parsing belongs to `gguf.py` (discover). See Requests.

## Requests

- **discover (`gguf.read_metadata`):** please add `summary["vocab_size"]`, the length of the `tokenizer.ggml.tokens` array, read from the array header without loading the array. quantcheck already reads it. Without it the size estimate assumes 262,144, which is about 2× too large for Llama 3.
- **api-cli (`app.quant_check_job`):**
  - The stored record should follow FIXUPS: `{"kind": "quant_check", "variant_id": <reference id>, "result": <kl_check result>, "timestamp"}`. Keep the extra keys if you like, but add `result`: today the reference PPL and corpus info are lost.
  - The job return `{**result, "reference": reference.quant}` replaces kl_check's `reference` dict with a string. Please use another key, for example `reference_quant`.
  - A variant with `quant=None` gets the label `None`. Please fall back to `quantcheck.quant_label(path)` or the variant id.
  - Please pass `work_dir=<app data folder>/tmp`. The default system temp folder can be RAM-backed tmpfs on Linux, and at the defaults the file is about 0.9–1.6 GB.
- **ui-quality (`static/quality.js` `renderResult`):**
  - The result keys are quant labels, and the job returns `variant_ids: {label: variant_id}`. Please look up names through that map rather than `seen.get(key)`.
  - Optionally show `partial` or `error` in their own line; they are also inside `plain` and `notes`.
- **lead (`docs/v0.4/llama-cpp-facts.md`):**
  - The tiny model's vocabulary is 422, not 464.
  - Mismatch 4's rate depends heavily on load. On this 4-core box it was 1–2 in 40 runs when idle and 22 in 40 with 3 busy cores. Cause: the leaked async logger (`common_log_main`, upstream issue 22142).

## Requests addressed to quantcheck (from `docs/v0.4/handoff/*.md`)

- **harness #4** (re-run when the summary is missing, then fall back to rows): done, see Fixed 1–2.
- **ui-quality:** `same_top_p` stays in percent and `mean_delta_p` in percentage points: unchanged.
- **server:** use the shared fake: added `test_shared_fake_llama_perplexity`, which runs `kl_check` against `fixtures.fake_command("perplexity")`.

## Evidence

- `python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: the only failures are the 2 known `test_fake_matches_real` ones (server-fake area: `--draft-max` and `llama-bench --version`), which FIXUPS.md already lists as known and which do not touch quantcheck (626 tests ran). `tests.test_quantcheck`: 49 tests, all OK.
- Real llama.cpp was built from the llama-cpp-python 0.3.35 sdist with `scripts/build_llama_cpp.sh`, plus the tiny models.
  - `tests.integration.test_real_runtime`: 24 tests, 23 pass, including `test_quantcheck_kl_check` and `test_quantcheck_parser_on_real_sample`.
  - The one failure is `test_runtime_install_detects_configured_directory` (`build` is None), which belongs to runtime-downloads.
- **Stress runs.** `kl_check` on tiny-llama Q8_0 (reference) against Q4_K_M and F16, with `context=128`, `chunks=3`; each run compares 2 files, so 20 runs = 40 comparisons.

  | Run | Load | Comparisons that needed a retry | Fell back after 3 tries |
  |---|---|---|---|
  | Before half-threads retries | 3 of 4 cores busy (25 runs) | 27 of 50 | 8 of 50 |
  | Before half-threads retries | 3 of 4 cores busy (20 runs) | 26 of 40 | 11 of 40 |
  | Final code | 3 of 4 cores busy (20 runs) | 19 of 40 | 1 of 40 |
  | Before half-threads retries | idle (20 runs) | 4 of 40 | 0 of 40 |

  - In every fallback, the result said so plainly. There were no errors and no wrong "complete" results.
- **Raw truncation rate** of `llama-perplexity --kl-divergence` (40 runs each):

  | Setting | Cut off |
  |---|---|
  | idle | 1/40 |
  | 3 cores busy | 22/40 |
  | 3 cores busy, `-t 1` | 2/40 |
  | 3 cores busy, `-t 3` | 8/40 |
  | 3 cores busy, `--poll 0` | 14/40 |
  | idle, `-t 1` | 0/40 |
