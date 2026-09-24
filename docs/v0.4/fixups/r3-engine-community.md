# Round 3: `r3-engine-community`

Branch `claude/v04-r3-engine-community`. Files: `engine.py`, `community.py`, `export.py`, `tests/test_engine.py`,
`tests/test_speed.py`, `tests/test_community.py`, `tests/test_export.py`, `.github/ISSUE_TEMPLATE/community-results.yml`
and this note. `speed.py` and `learning.py` needed no change.

## Fixed

1. **Short tunes (`matching_tuned`).** Already labelled by the fix-up round: a tune measured at a short depth gives
   evidence `tuned`, `tps: null`, verdict `unknown`, and says "near the start of a conversation". New:
   - `tuned.scaled_tps` scales the tune's speed to the candidate's length with the same bytes-per-token rule as
     `interpolated_speed` (shaded 10%).
   - It is set only when the tune kept this placement, has a known depth above 0, and the stretch is at most 4×.
     Otherwise it is `null`.
   - The sentence adds "Scaled to this length that is roughly N, likely fast enough / possibly slower…".
   - The ranking uses it as a speed tie-break after interpolation. It never makes a result verified.
   - Depth is read from `settings.depth`, falling back to the pinned record's top-level `depth` (from `**tune_result`).
2. **Files found on disk (`source == "local"`, `sha256=None`)** now match their tests by `variant_id`, which is
   `gguf.local_id` (path-based). If both sides have a hash, the hashes must agree. Catalogue files still need a matching
   hash. Tunes already matched this way.
3. **Runtime backend.** A real bug. `launch.from_candidate(runtime_backend=)` sets `gpu_backend`, but then merges the
   candidate's `launch`, and the engine had put the **card's** backend there. So a Vulkan build on an NVIDIA card still got
   CUDA and `CUDA_VISIBLE_DEVICES`. Fixes:
   - `recommend(..., runtime_backend=None)`: when it is a GPU backend, it becomes `launch.gpu_backend`. It also decides
     which GPUs CPU-only runs hide (`-dev none`), even when the scan saw no GPU.
   - If it is `"cpu"`, a note says llama.cpp cannot use the graphics chip right now. `detect()` also returns `"cpu"` for a
     GPU build that found no device, so the note does not claim which one.
4. **`community_speed`** also catches `AttributeError`. `n_cpu_moe` was already passed (it is in the launch fragment).
5. **`community.evidence`** now matches these itself, the same way the engine does:
   - `cache_type_k` (a missing value counts as f16)
   - `n_cpu_moe` (already done)
   - full depth: `depth` missing, or `depth + 1024 ≥ context`

   The round-2 note rejected matching on cache type. The engine already filtered on it, so other callers now agree
   with the engine.
6. **Report-time hot spot.** `evidence` memoises `hardware_class` (`_CLASS_CACHE`, keyed by the hardware fields it
   reads). It is used for matching only. Sharing (`anonymize`) always scrubs names with the current host and user.
   `hardware_class` also looks up the host and user names once instead of once per name.
7. **Community `backend` follows the runtime.**
   - `hardware_class` uses the llama.cpp build's GPU backend (from llama-bench `backends`) over the card's own: a
     Vulkan build on an RTX card is shared as `vulkan`.
   - A processor-only build (`backends: "CPU"`/`"BLAS"`) ignores `-ngl`, so its record is shared as a CPU run:
     placement `cpu`, `gpu_layers` 0, no GPU name.
8. **Export.** New `export(..., runtime_backend=None)`: a GPU runtime backend replaces the config's `gpu_backend`. That
   sets the docker image and stops CUDA env for a Vulkan build. Configs from `from_candidate(runtime_backend=)` plus the
   engine fix above already carry it. The Vulkan docker note now adds that NVIDIA cards need the NVIDIA Container
   Toolkit, and that the CUDA image is the usual choice there.
9. **Long speed tests.** Found by the review. `testing.bench_plan` caps depth at 32,768 tokens, so a 64k test was
   never "measured", and before this change it was not used at all.
   - `interpolated_speed` now places each test at the tokens it really had in memory: the context at full depth, else
     `depth`.
   - A capped 64k test is scaled from 32k (evidence `interpolated`, not verified).
   - The same test also counts, unscaled, for shorter contexts.
   - Short tune trials still cannot stretch past 4×.
10. **Issue form text** lists what the JSON really holds: quant, layer count, SHA-256, a name only for catalogue models,
    month, llama.cpp version and backend.

## Checked, already done (no change)

- **iSWA sliding-window cache.** The request's formula `min(ctx, pad256(window·(n_seq if unified else 1) + ubatch))`
  is what `speed.kv_bytes` already does per user: `pad256(min(pad256(c), window + ubatch))`. Because `pad256(c)` is a
  multiple of 256, both formulas give the same number. Confirmed in the source and in real logs (Evidence).
  `launch` always passes `-np`, and `server.cpp` only sets `kv_unified = true` when `-np` is automatic, so the unified
  case never applies to our launches.
- **Resolved threads.** The engine always resolves `threads`: the tune's threads, else `cores`, else `threads`, else 1.
  It puts that number in `launch`. `testing.speed_test` stores `config["threads"]` when it is set, so the stored number
  is the launched one. A test (`test_resolved_threads_match_what_testing_records`) pins this.
- **hardware-catalogue:**
  - MXFP4 `expert_fraction` is already treated as a byte share.
  - `other_gpus` are only used to hide GPUs, never in memory maths.

## Rejected

- **Review: don't let the host build choose the docker image.** It is true that a container ships its own llama.cpp.
  But the config's `gpu_backend` already comes from the runtime (the lead's `from_candidate` plus fix 3), and
  `export` cannot see the card's vendor. The request asks for the runtime backend. The NVIDIA + Vulkan case gets a note
  instead.
- **Review: compare file size for local files.** Measurement records have no size field. If a file at the same path is
  replaced, the ids stay equal until it is hashed. Low risk; no change.

## Requests

- **r3-app-cli (`app.py`):** without this, fixes 3 and 8 do nothing in the app. Pass
  `runtime_install.detect(store)["backend"]`:
  - to `recommend(...)` in `evaluate` and `candidate_for`. The simplest way is to add `"runtime_backend"` to
    `engine_extras`, which already filters by the accepted parameters.
  - to `launch.from_candidate` in `launch_config`
  - to `export.export` in `export_config`
- **lead (`launch.py`):** in `from_candidate`, apply `runtime_backend` **after** `config.update(launch)`. Today the
  candidate's `launch["gpu_backend"]` overwrites it, so the parameter only works when the engine got the runtime
  backend too.
- **r3-llama (tuner):** the engine reads the full-depth check from `settings.depth`, falling back to top-level `depth`.
  If the final `best` is measured at full depth, put that depth in `settings.depth` (and `best_result` from that run);
  the trial depth can stay top-level.
- **testing (later):** community rows and local tests above 33.8k tokens can never be "measured" while
  `MAX_BENCH_DEPTH = 32768`. They now show as interpolated. Raising the cap (or a separate long run) would verify them.

## Evidence

- `python3 -m unittest discover -s tests`: **821 tests OK** (26 skipped). `npm test`: **71/71 pass**.
- Real llama.cpp (llama-cpp-python 0.3.35 sources, `scripts/build_llama_cpp.sh`):
  `tests.integration.test_real_runtime` + `test_fake_matches_real` = **43/43 OK**.
- **Sliding-window source** (`src/llama-kv-cache-iswa.cpp`):
  `size_swa = GGML_PAD(std::min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256)`, with
  `size_base = n_ctx_seq = GGML_PAD(n_ctx / n_seq_max, 256)` when not unified (`llama-context.cpp`).
- **Sliding-window logs.** I generated two random `gemma3` models: 6 layers (5 sliding, pattern 6), 4 KV heads × 64,
  window 128 (like gpt-oss) and window 1024 (like Gemma 3). Then I started real `llama-server -c ctx·np -np np -lv 4` and
  summed the `llama_kv_cache: size = … MiB` lines. **18 of 18 match `speed.kv_bytes`** to 0.01 MiB. They are now pinned
  in `test_speed.SlidingWindowRealLogTests`. Examples:

  | Window | Context × users | Other | Cells logged (full / sliding) | MiB logged = ours |
  |---|---|---|---|---|
  | 128 | 8192 × 2 | | 8192 / 768 (2/2 seqs) | 23.50 |
  | 128 | 8192 × 2 | `-ub 1024` | 8192 / 1280 | 28.50 |
  | 1024 | 8192 × 4 | | 8192 / 1536 (4/4 seqs) | 62.00 |
  | 1024 | 1000 × 2 | | 1024 / 1024 | 12.00 |
  | 128 | 300 × 1 | | 512 / 512 | 3.00 |
  | 1024 | 4096 × 3 | q8_0 | 4096 / 1536 | 18.33 |

  Note: this build only prints these lines with `-lv 4`.
- **Independent review** by a separate agent (maths, privacy of community output, honesty of labels, seams). It found:
  - no division-by-zero or sliding-window problems
  - `interpolated_speed` unchanged for full-depth tests
  - `_CLASS_CACHE` never on the sharing path, and its key covers every field `hardware_class` reads
  - no renamed or removed keys (`scaled_tps` is new)

  Its findings, and what happened to each:
  - app not passing the backend: a request above
  - unbounded `scaled_tps` stretch: fixed (4× limit)
  - "processor-only" wording: fixed
  - docker image: rejected, with a note added
  - local file size: rejected
  - capped long tests: fixed (item 9)
  - template wording: fixed
- Performance: `PerformanceTests.test_recommend_is_fast_for_a_large_catalogue` passes unchanged.
