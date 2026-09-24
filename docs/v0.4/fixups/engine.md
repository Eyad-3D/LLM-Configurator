# Fix-up: `engine` (engine.py, speed.py, learning.py)

Branch `claude/v04-fix-engine`. Files: `engine.py`, `speed.py`, `learning.py`, `tests/test_engine.py`, `tests/test_speed.py`,
`tests/test_learning.py`, and this note. No other files touched.

Review method: three independent review agents (maths, seams, honesty + performance), then a real llama.cpp build
(`scripts/build_llama_cpp.sh`, CPU-only, llama-cpp-python 0.3.35 sources) run on the tiny models plus three mid-size
random models I generated (32k–65k vocabulary): a dense `llama`, a sliding-window `gemma3`, and a `qwen3moe` with 16 experts.

## Fixed

### Memory model (checked against real llama.cpp)

1. **Sliding-window cache with several users was under-counted.** `launch` passes `-c context×users -np users`. With an
   explicit `-np`, llama-server sets `kv_unified = false`, so each user gets their own cache. The cache also rounds up to
   256 cells. Real log lines:

   | Case | Cells logged |
   |---|---|
   | Gemma-like model, 2 users, 8k context | full layers 8192 cells, sliding layers 1024 cells, 2/2 seqs |
   | Same model with `-ub 1024` | sliding layers 1536 cells |
   | Context 5000 | 5120 cells |

   New formula (`speed.kv_bytes`), per K+V element size:
   `per_token_layer × users × ((L−S) × pad256(c) + S × pad256(min(pad256(c), window + ubatch)))`,
   with ubatch defaulting to 512. `allocations(..., ubatch=None)` passes it through.
   - Before: `S × min(c·u, window·u + 512)`. That missed 512·(u−1) tokens per sliding layer: 0.6 GiB at 4 users on a
     Gemma-3-27B-like model.
   - Evidence: `test_kv_matches_real_llama_cpp_logs` reproduces **every** logged size exactly:
     - dense model: f16, q8_0 and q4_0; 1 and 3 users; 4k and 16k
     - Gemma-like model: contexts 512/2k/8k/32k × 1/2/4 users; `-ub 1024`; q8_0; an unpadded 5000-token context

2. **A partial GPU split puts the output layer on the GPU.** In `llama-model.cpp` `load_tensors`,
   `i_gpu_start = n_layer + 1 − ngl`, and the output layer counts as layer `n_layer`. So `-ngl g` (what
   `launch.runtime_gpu_layers` passes for a split) offloads **g−1 blocks plus the output layer**.
   - The output layer (vocabulary × width) is often much bigger than one block: about 1.2 GB for Gemma-3-27B's tied
     262k vocabulary, against about 0.25 GB per block. Small splits were therefore under-counted in VRAM.
   - New helper `speed.gpu_blocks(variant, gpu_layers)` asks `launch.runtime_gpu_layers` how many layers really go to
     the GPU. That keeps it right if the lead changes that function (see Requests).
   - `allocations` now uses `(W − O)·blocks/L + O` for VRAM weights and the rest for RAM, where
     `O = speed.output_bytes = max(W/L, min(0.15·W, 2 GiB))` and W is file bytes × 1.10. The catalogue has no vocabulary
     size, so O is a bounded allowance: it covers Gemma-3-27B and Llama-70B, and is never less than one block.
   - Full offload and CPU-only placements are unchanged.
   - KV follows the blocks on the GPU. `moved_expert_layers` and the speed model use the real block range too. At
     `-ngl 40` on a 48-layer model, blocks 0–8 are on the CPU, so `--n-cpu-moe 10` moves only 1 GPU block.

3. **Confirmed unchanged by the real runs:**
   - The dense KV formula and the q8_0/q4_0 block sizes (34/32 and 18/32 bytes) match to the MiB.
   - `CPU_Mapped model buffer` equals the file's tensor data.
   - A 4-shard split loads with the same buffer as the single file, so `size_bytes` = total across shards is right.
   - `--n-cpu-moe` doesn't change memory on a CPU-only build (expected).
   - Flash attention `auto` resolves to enabled on the CPU.
   - Compute buffers were 25–68 MiB on these models, well inside the flat `0.75 + 0.2·users` GiB buffer.

### Seams

4. **Tested settings must match the launch.** `matching_speed` and `interpolated_speed` now need these to match the
   candidate's actual `launch`: threads (from a merged tune when there is one), `batch`, `ubatch`, `flash_attn` (a record
   without it counts as compatible) and `cache_type_v`.
   - Before, a tune's threads went into `launch` but tests were matched on core count. So a speed test of a tuned
     candidate never became "measured".
   - Also before, a default-settings test was labelled "same configuration" for a tuned launch.
   - Checked with the seam script: speed-test records built exactly as `testing.speed_test` writes them (via
     `app.launch_config`) match as `measured` in 290 of 290 cases (CPU, NVIDIA, Apple; f16 and q8_0), including after a tune.
5. **Only full-depth tests verify a context.** A record counts for context c only if `depth` is missing (v0.3/`runtime.bench`
   measured at c−128) or `depth + 1024 ≥ c`. `testing.speed_test` uses c−640, so its records count. Tune trials
   (`depth = min(1024, c/4)`) don't. `speed_test` records are preferred over other kinds (testing's Request).
6. **Tunes now match the pinned FIXUPS record shape.** The match uses the top-level `context`, `gpu_layers`, `n_cpu_moe`
   and `kv_cache_type` (the placement the tune started from), falling back to `best`.
   - If the tuner moved the placement (layers, experts or cache type), the tune is attached as `tuned` with
     `same_placement: false` and `best_placement`. Nothing is merged and no speed is claimed.
   - If the tune kept the placement, its threads/batch/ubatch/flash_attn go into `launch` as before.
   - Its speed is "verified" only when `settings.depth` (from `**tune_result`) is at full depth. Otherwise evidence
     `tuned`, verdict `unknown`, `tps` None, and the sentence "Tuned with a short test: about X tokens per second near the
     start of a conversation. Run a speed test to check it at this length."
   - On multi-GPU machines a tune without `best.gpu_uuid` is not credited to a GPU candidate.
   - New `tuned` keys: `depth`, `verified`, `same_placement`, `best_placement`. All old keys are kept.
7. **Community rows must also match** the cache type, `n_cpu_moe` and full depth. `community.evidence` ignored them, so a
   q4_0 candidate was shown f16 results. The engine pre-filters rows before calling `community.evidence`, and the
   injected `community_evidence` hook still gets every row that isn't a community row.
8. **CPU-only candidates always hide GPUs** (`-dev none`), including a selected GPU whose free memory is unknown and cards
   listed only in `other_gpus`. Before, a ROCm/Vulkan build could still use them.
9. `runtime_gpu_layers` in candidates now comes from `launch.runtime_gpu_layers`, one source of truth.

### Honesty and wording

10. Under the "speed" priority, a verified speed that meets the target always ranks above estimates and community
    numbers. Before, an estimate of 32 tok/s could outrank a real 25. "Balanced" now uses speed as a tie-breaker after
    quality and closeness to the requested length.
11. Clearer verdict sentences:
    - At target: "…about 15 tokens per second, which meets your target of 15."
    - Near the target the value shows one decimal ("14.9 … a bit below", never "15 … below 15").
    - Target 0 makes no comparison.
    - Tiny estimates say "under 0.1".
    - Commas replace dashes.
    - Estimated candidates' `speed_evidence` says "Rough estimate only; run a speed test to check it".
    - The explanation says "runs on the processor only" or "split between the graphics chip and the processor" instead
      of "cpu execution".
12. A missing `hardware["fingerprint"]` no longer raises and never matches anything.
13. `learning` ignores shallow tune trials (same depth rule) and records without a fingerprint.

### Performance

- Community rows are indexed once per report.
- Speed evidence (measurement, tune, interpolation, community, estimate) is computed once per (context, layers, experts)
  and shared by the "now" and "after closing" scenarios.
- Tunes are grouped by variant.

Timings with the reviewer's fixture (real catalogue shapes, 8 MoE, 3 sharded, 200 measurements, 30 tunes, 2,000
community rows):

| Case | Before | After |
|---|---|---|
| 40 models | 43–249 ms | 25–104 ms |
| 116 models | up to 835 ms | 59–224 ms |

## Rejected, with reasons

- **Count the CPU "repack" buffer as extra RAM.** Real Q4_K runs show `CPU_REPACK model buffer` at about 50% of the file,
  on top of `CPU_Mapped`. The mapped pages of repacked tensors are read once and are clean file cache the OS can drop, so
  the real need stays about one file. Counting both would hide models that run fine. Note for testing: peak RSS can read
  up to about 1.5× the file on CPU, so a "higher than estimated" memory note may appear there.
- **A context-dependent compute buffer for `flash_attn: off`.** Without flash attention, llama.cpp allocates about
  `n_ctx × ubatch × n_head × 4` bytes, and the catalogue has no `n_head`. `auto` resolves to on for CPU, CUDA and Metal,
  so this only matters when a tune picks `off` at long contexts. Left as a known gap.
- **Adopting a tuner-moved placement into the candidate.** The candidate's memory, id and placement would no longer
  describe what runs. It is shown as `tuned.best_placement` instead. `app.launch_config(tuned=True)` can apply it
  (see Requests).
- **Per-GPU output-layer sizes from the GGUF.** `Variant` has no vocabulary or tensor table (a domain.py change). The
  bounded allowance is used instead.
- **Tuning the 256-cell pad or sliding-window rules to tiny models.** All checks use the formulas from llama.cpp's source,
  and the mid-size models only confirm them.

## Not verifiable here

- No GPU in this container, so GPU placement (`-ngl`, `--n-cpu-moe` moving expert tensors, Metal unified memory) comes
  from reading `llama-model.cpp` and `common/arg.cpp`, not from logs. GPU compute-buffer sizes weren't measured.

## Requests

- **lead (`launch.py`):** make `runtime_gpu_layers(g, L)` return `g + 1` for every `g > 0`, not only at full offload.
  - Today a "split of g layers" really runs g−1 blocks plus the output layer, so the UI's "g of L layers on the GPU" is
    off by one.
  - The engine reads `launch.runtime_gpu_layers`, so its memory numbers follow automatically.
  - The tuner's `gpu_layers` step and recorded `gpu_layers` stay in transformer blocks.
- **api-cli (`app.py`):**
  - The tuned record should follow the pinned shape: spread `**result` so that `settings.depth`, `confirmed` and
    `seconds` are kept, and add top-level `n_cpu_moe` and `kv_cache_type`. Without `settings.depth` the engine treats
    every tune as a short test.
  - `launch_config(tuned=True)` should use the candidate's own `tuned` (or `engine.matching_tuned`) instead of
    `best_tune`, which matches by the `placement` label only. It can apply a 48-layer MoE tune to a 35-layer split
    candidate and crash `normalize` ("gpu_layers must be an integer between 0 and 36", reproduced).
  - With the engine already merging same-placement tune settings into `launch`, `tuned=True` only adds value when
    `tuned.same_placement` is false (apply `best_placement` after a memory check).
- **community:** `evidence()` should match `settings.cache_type_k`, `n_cpu_moe` and full depth itself. The engine does it
  now, but other callers don't. Also cache `hardware_class`/`_personal_words`, which were 89% of the old report time.
- **runtime-downloads (`runtime.bench`):** add `kind: "bench"`, `settings` and `id`, and use `launch`/`testing.bench_args`.
  Its records have no settings, so they are read as f16 and default batch, and never match a compressed-cache or tuned
  candidate.
- **ui:** `speedPresentation` shows `c.tps` as "Measured local generation speed" for `evidence: "tuned"` too. It should
  say "measured after tuning", and it could show `speed_interpolated.tps` and `community.median_tps` (today they appear
  only in `verdict_text`). A short tune has `tps: null`, `evidence: "tuned"` and `verdict: "unknown"`.
- **testing:** `memory_verdict` should expect a CPU peak RSS up to about 1.5× the file for repacked quants (see Rejected).
- **tuner:** consider measuring the final `best` at full depth (`-d context−640`) and storing `settings.depth` for it.
  Then a tune can verify speed at the chosen length.

## How I tested

- `python3 -m unittest discover -s tests`: 612 tests. The only failures are the 2 known fake-vs-real tests in
  `test_fake_matches_real`, which also fail on the base commit (server-fake owns them).
- `npm ci && npm test`: 56/56 pass.
- Real runtime: `tests.integration.test_real_runtime` gives 23/24. The failure is the known `runtime_install` build
  parsing.
- New tests:
  - KV sizes vs real logs
  - output layer on a split
  - llama.cpp's `--n-cpu-moe` block order
  - launch-settings matching
  - short-depth tunes and tests
  - `speed_test` preference
  - the pinned tuned shape and moved placements
  - community filtering with real `community.anonymize` rows
  - GPU hiding for unknown and `other_gpus` cards
  - speed-priority ordering
  - verdict wording
  - learning's depth filter
