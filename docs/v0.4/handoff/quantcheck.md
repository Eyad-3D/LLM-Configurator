# Handoff: `quantcheck` (compression-loss check)

## What I built

- `src/llm_configurator/quantcheck.py`: measures how much quality a compressed model file loses compared with a reference file (like comparing a low-quality JPEG with the original photo). It uses llama.cpp's `llama-perplexity --kl-divergence` and explains the result in one plain sentence per file.
- `src/llm_configurator/data/quantcheck_corpus.txt`: 43.7 KB of **original** English sample text (stories with dialogue, explainers, a recipe, lists, an email, an FAQ, chat-style turns) plus about 8% code (Python, JavaScript, SQL, shell, JSON). Place names are invented. Nothing is copied.
- `tests/test_quantcheck.py`: 27 tests, no network.

How it runs:
1. It checks the inputs, picks a small text budget (about 6k tokens by default), and estimates the size of the temporary predictions file. If free disk space is less than that estimate plus 256 MB, it stops with a plain error before running anything.
2. The reference model reads the sample text and saves its predictions: `-m REF -f corpus --kl-divergence-base TMP/reference.kld --chunks N -c CTX -b CTX [common flags]`.
3. Each candidate runs **one at a time**: `-m CAND -f corpus --kl-divergence-base TMP/reference.kld --kl-divergence -c CTX -b CTX [common flags]`.
4. The temp folder (`llmc-quantcheck-*`) is always deleted in a `finally`: on success, error, timeout, cancel, or any unexpected exception.

## Public API as built

```python
parse_kld_output(text) -> dict
# Contract keys: mean_kld, median_kld, kld_99, same_top_p, ppl_base, ppl, mean_delta_p
# Extra keys: max_kld, kld_999, ppl_ratio, rms_delta_p, mean_kld_uncertainty,
#             same_top_p_uncertainty, final_ppl (reference run's "Final estimate"), chunks_done
# Units: KLD values raw; same_top_p, mean_delta_p, rms_delta_p are percent (0-100). Missing -> None; nan/inf -> None.

kl_check(perplexity_command, reference_path, candidates: dict[label -> path], corpus_path=None, context=512,
         chunks=None, config=None, progress=None, cancel=None,
         reference_label=None, vocab_size=None, work_dir=None, timeout=3600, run=None) -> dict
# {
#   "reference": {"label", "path", "ppl", "logits_bytes"},
#   "results": {label: {**parsed (minus final_ppl), "plain": str, "verdict": "negligible"|"small"|"moderate"|
#                       "large"|"severe"|None, "path": str, "error": str|None}},
#   "corpus": {"path", "bytes", "context", "chunks", "tokens"},
#   "notes": [plain str, ...],
#   "estimated_temp_bytes": int,
# }
# progress(dict): {"stage": "reference"|"candidate"|"done", "done", "total" (= models x chunks), "message", "label"}

interpret(parsed, reference_label="the reference") -> {"plain": str, "verdict": str|None}
quant_label(path) -> str                       # "Q4_K_M" from a filename, else the filename
estimate_logits_bytes(vocab_size, context, chunks) -> int
CORPUS_PATH                                    # packaged sample text
```

Extra optional keyword arguments:
- `reference_label`: the name used in sentences. It defaults to the quant name taken from the filename, for example "Q8_0".
- `vocab_size`: used for the disk estimate. If it is not given, the code tries `gguf.read_metadata` (imported inside the function). If that fails too, it assumes 262,144 (the largest common vocabulary) and says in the notes that the estimate is an upper bound.
- `work_dir`: where the temp folder goes. It defaults to the system temp dir.
- `timeout`: seconds per llama-perplexity run.
- `run`: an injectable runner, `run(argv, env=, timeout=, cancel=, on_line=) -> (returncode, output)`.

Example sentence: `Picks a different top word about 8% of the time compared with Q8_0 — usually hard to notice in chat.`

## Deviations from the contract and why

- The contract did not fix units, so `same_top_p` and `mean_delta_p` are **percent**, matching what llama.cpp prints. Early-2024 builds print "Same top" as a 0–1 fraction; the parser converts it to percent.
- A candidate that fails does **not** stop the whole check. Its entry gets `error` (a plain reason), `verdict: None` and a "Could not compare …" sentence, and the other candidates still run. A reference failure raises `ValueError`, because nothing can be compared without it. `Cancelled` propagates as usual.
- Extra return keys `estimated_temp_bytes` and `reference.logits_bytes`, and the extra progress key `label`. The contract allows extra keys.

## Launch config mapping (§2.2)

These keys are reused from `config` (the same flags are used for the reference and every candidate):

| config key | llama-perplexity flag |
|---|---|
| `gpu_layers` | `-ngl` via `launch.runtime_gpu_layers` (a full offload adds 1 for the output layer) |
| `device` | `-dev` (or `-dev none` for CPU-only on a GPU backend, same rule as `launch.server_args`) |
| `threads` | `-t` |
| `n_cpu_moe` | `--n-cpu-moe` |
| `gpu_uuid` | `CUDA_VISIBLE_DEVICES` via `launch.server_env` |

Left out on purpose:
- **Server-only flags:** host, port, alias, `--jinja`, `-np`, draft model.
- **`model_path`:** each run uses its own file.
- **`context`:** the check uses its own small window.
- **KV-cache type (`-ctk`/`-ctv`) and flash attention:** leaving them out means the check measures the weight compression alone.
- **`batch`/`ubatch`:** `-b` is always set equal to `-c`. llama-perplexity runs `batch / ctx` sequences at once, which multiplies memory use. It also asserts `n_batch < n_ctx || n_batch % n_ctx == 0`, and a server batch such as 1000 with ctx 512 would crash that assert.

## Verdict thresholds (rough guidance)

The anchors come from llama.cpp's own **LLaMA 3 8B scoreboard** in `tools/perplexity/README.md`, which measures each quant against FP16:

| Quant | Mean KLD | Same top p |
|---|---|---|
| Q8_0 | 0.0014 | 97.7% |
| Q6_K | 0.0055 | 96.0% |
| Q5_K_M | 0.011 | |
| Q4_K_M | 0.031 | 91.9% |
| Q3_K_M | 0.10 | |
| Q2_K | 0.33–0.45 | 71.1% |
| IQ1_S/M | 1.4–2.3 | |

Bands on mean KLD: `<0.01` negligible (Q8/Q6 level), `<0.05` small (Q5/Q4_K_M), `<0.15` moderate (Q4_0/Q3), `<0.5` large (Q2), else severe (IQ2_XXS/IQ1).

If only "Same top p" is known, bands on the share of positions where the top word differs: `<5%`, `<10%`, `<18%`, `<35%`, else severe.

The notes tell the user these verdicts are rough guidance, not a guarantee. The results are relative to the chosen reference: a Q8_0 reference makes a candidate look slightly better than an FP16 reference would. A short sample (about 3k scored tokens) has wider error bars than the README's full Wikitext run. The `mean_kld_uncertainty` and `same_top_p_uncertainty` values are returned so the UI can show them.

## Known gaps

- The disk estimate uses the real llama.cpp file layout, but the vocabulary size only comes from `gguf.read_metadata` if that reader exposes a `<arch>.vocab_size` or `tokenizer.ggml.vocab_size` scalar. Most GGUFs store only the token *array*, so the estimate often falls back to the 262k upper bound (about 1.6 GB at the defaults of ctx 512 × 12 chunks, compared with about 0.93 GB for a 152k vocabulary). This errs on the safe side.
- The same `config` is used for every model. If the reference (for example Q8_0) is much bigger than the candidate the config was tuned for, it may not fit on the GPU. The caller should pass a config suited to the largest file, or none. A per-model config would be a small addition if the API wants it.
- A text budget of about 6k tokens gives a quick, rough answer, not a publication-grade number.
- Peak memory is not measured here.

## Requests to other owners

- **discover (`gguf.read_metadata`)**: please add the vocabulary size to `summary` (for example `"vocab_size": len(tokenizer.ggml.tokens)`, read from the array header without loading it). I'd then read `summary["vocab_size"]` to make the disk estimate exact. Right now I only look at `metadata[...]` scalars.
- **server (`tests/fixtures/fake_llama.py --as perplexity`)**: please match the formats in my test fixtures (`KLD_CURRENT` and `REFERENCE_RUN` in `tests/test_quantcheck.py`). Base mode (without `--kl-divergence`) must **write the file** given by `--kl-divergence-base` (starting with the bytes `_logits_`) and print `[1]x,[2]y,…` and then `Final estimate: PPL = …`. KLD mode needs to print the per-chunk table and the three `======` blocks.
- **api**:
  - `POST /api/quality/quant-check` should run `kl_check` in a job with `exclusive="compute"`, passing `progress`/`cancel` through.
  - `candidates` should map the variant's quant label (or variant id) to the local path from `discover.find_for_variant`.
  - Store with `store.append("quality_results", {"kind": "quant_check", "variant_id": <reference id>, "result": ..., "timestamp": now()})`.
  - The perplexity command comes from `runtime_install` (a `llama-perplexity` binary next to `llama-server`).
- **ui-quality**: show `results[label]["plain"]`, and optionally the `verdict` as a badge. Show `notes`. Show the uncertainty values if space allows.

## How I tested

`python3 -m unittest discover -s tests`: 99 tests pass (27 new).

- **Parser.** It is tested against three layouts copied from the real format strings in llama.cpp's `perplexity.cpp`:
  - current builds (`LOG`, with 95%/90% lines);
  - mid-2024 builds (b2800–b3900, `printf`, which print the duplicate `99.0%` line that is really the 95% value, so only the first one is used);
  - early-2024 builds (b1960–b2700: `Average:`, `KLD_99 :`, and "Same top" only as a 0–1 fraction in the per-chunk rows).

  Other parser cases: the reference run's `Final estimate`, missing lines, empty input, ANSI colours, CRLF, `+/-` in place of `±`, a Δ mangled by a console code page (`Î”p`), `nan`, scientific notation, and percentile lines that must not be read as chunk rows.
- **End-to-end.** A fake `llama-perplexity` Python script, created in each test's temp dir, writes the logits file in base mode and prints realistic output. The tests check:
  - the two-step flow and the argv of each call;
  - the progress events;
  - verdicts and sentences;
  - that `work_dir` is empty afterwards.
- **Cleanup and failure cases:**
  - a reference failure (the real tool's "you need at least N tokens" case, which exits 0);
  - a candidate failure (broken model, reported per model);
  - cancel between models (injected runner);
  - cancel of a running process (the real subprocess is killed within seconds);
  - timeout;
  - an unexpected exception;
  - disk-space refusal (no process started);
  - the unknown-vocabulary fallback;
  - input validation.
- **Config mapping.** It is checked with an injected runner: server-only flags are absent, `-ngl` is N+1 for a full offload, `-dev none` is used for CPU-only on a GPU backend, and `CUDA_VISIBLE_DEVICES` is set.

## llama.cpp facts assumed (the harness should confirm)

I checked these against `ggml-org/llama.cpp` sources fetched on 2026-09-24: `tools/perplexity/perplexity.cpp` and `common/arg.cpp` on master, plus `examples/perplexity/perplexity.cpp` at tags b1960, b2050, b2300, b2450, b2700, b2800, b3500, b3700 and b4000.

1. Flags:
   - `--kl-divergence-base FNAME` (alias `--save-all-logits`) and `--kl-divergence` exist for the perplexity example only.
   - `--chunks N` exists for perplexity, imatrix and retrieval.
   - `-f`, `-c`, `-b`, `-t`, `-ngl`, `-dev` and `--n-cpu-moe` are common flags.
2. Base run (no `--kl-divergence`):
   - It prints `[i]ppl,` per chunk with no newline until the end, then `Final estimate: PPL = X +/- Y`.
   - The logits file starts with `_logits_`.
   - It **exits 0 even on failure**. For example, with fewer than `2*ctx` tokens it only logs "you need at least N tokens". So I check that the file exists and is non-trivial.
3. Logits file size: 8-byte magic + n_ctx + n_vocab + n_chunk (int32 each) + `n_chunk*n_ctx` int32 tokens + per chunk `(n_ctx-1-n_ctx/2)` rows × `(2*((n_vocab+1)/2)+4)` uint16. Only the second half of each window is scored and stored.
4. KLD run: the file's n_ctx must not exceed the current `-c`, so I use the same `-c`. A vocabulary mismatch logs "inconsistent vocabulary".
5. The summary block formats are exactly those in the test fixtures. `Same top p` and `Δp` are printed as percent.
6. Per-chunk rows:
   - since about b2800: `chunk, PPL ± u, ln ratio ± u, KLD ± u, Δp RMS ± u %, Same top p ± u %` (11 numbers);
   - before that: `chunk, PPL, ln ratio ± u, KLD ± u, same-top fraction ± u` (8 numbers).
7. `-b` larger than `-c` makes the tool run `b/c` sequences in parallel. It asserts `n_batch < n_ctx || n_batch % n_ctx == 0`.
8. I merge stdout and stderr, because the summary uses `LOG` (stdout) or `printf` (older builds) while progress uses `LOG_INF` (stderr).
