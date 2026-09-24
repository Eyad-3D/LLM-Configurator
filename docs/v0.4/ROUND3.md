# v0.4 round 3: the last cross-area requests

The fix-up round is merged (HEAD after this file). Results: 809 Python tests, 71 UI tests and **43/43 real llama.cpp tests** pass. The fix-up sessions left requests for each other in `docs/v0.4/fixups/*.md` (each has a **Requests** section). This round closes them.

Each item below is a request from those notes. **First check whether it is already done** (some were fixed in parallel). If it's wrong, reject it with a reason. Implement the rest with tests. The pinned shapes in `docs/v0.4/FIXUPS.md` still apply.

## Changed by the lead since the fix-up round (don't undo)

- `launch.runtime_gpu_layers(g, L)` now returns **g+1 for every g > 0**. Verified in llama.cpp `llama-model.cpp` (`i_gpu_start = n_layer + 1 - n_gpu_layers`): `-ngl N` puts N−1 blocks plus the output layer on the GPU. Engine, speed, tuner, testing and export already call it.
- `launch.server_env` drops inherited `LLAMA_ARG_*` and sets `LLAMA_ARG_CORS_ORIGINS=localhost` (llama-server's default CORS `*` lets any website read its answers). Export scripts now export that variable too.
- `launch.from_candidate(..., runtime_backend=None)`: pass `runtime_install.detect(store)["backend"]`. A Vulkan build on an NVIDIA card must not get CUDA env/devices.
- `launch.normalize` rejects every control character in `alias`, `device`, `gpu_uuid` and `draft_model_path`.
- `app.download_job` reuses a user's own local file (job result `reused: true`) and refuses only when it has gone.

## Sessions and their items

### `r3-app-cli`: `app.py`, `cli.py`, `tests/test_cli.py`

Sources: requests to api-cli in `fixups/api-http.md`, `discover.md`, `engine.md`, `evals.md`, `quantcheck.md`, `hardware-catalogue.md`, `community-export.md`, `server-fake.md`, `testing-tuner.md`, `docs.md`.

- **Tuned record:** use the full pinned shape (spread `**result` so `settings.depth`, `confirmed` and `seconds` are kept). Add top-level `n_cpu_moe`, `kv_cache_type`, `depth`, `runtime` and `runtime_build`. Add the optional `kind="tune"` measurement, and return `saved`.
- **`launch_config(tuned=True)`:** use the candidate's own `tuned` / `engine.matching_tuned`, not `best_tune` by placement label. Today a 48-layer MoE tune applied to a 35-layer split crashes `normalize`. Pass `runtime_backend`.
- **`quiz_job`:** write one pinned record (nested `needle`, `candidate_id`, full `result`).
- **`compare_job`:** key runners by candidate id and pass `names=` so `reveal()` returns ids plus `labels`.
- **`quant_check_job`:**
  - Store the pinned record with `result`.
  - Don't overwrite kl_check's `reference`; use `reference_quant`.
  - Fall back to `quantcheck.quant_label(path)` for the label.
  - Pass `work_dir=<data dir>/tmp`.
  - Keep no paths in anything stored or returned.
  - Return `variant_ids: {label: variant_id}`.
  - Accept `hardware`.
- **`test_job`:** pass `min_tps` through to `run_tests`; add `test --min-tps`.
- **Local models:**
  - `local_model`/`require_model` pass `progress`/`cancel`.
  - `variants()` merges `discover.local_variants(store)`, deduped by id.
  - `add_local_job` delegates to `discover.add_file` (first shard via `gguf.shard_paths`).
  - `download_job` calls `discover.remember_hash` after verified downloads.
  - `page_file` keeps `match_note` and adds `local_variant_id`.
- **Catalogue:**
  - `map_benchmark` goes through `catalogue.set_slug`.
  - `refresh(cancel=)`.
  - `models add` then `refresh_entry`, with a `--config-repo` flag.
- **CLI:**
  - Call `llama_server.install_exit_handlers()` once at start-up (main thread).
  - Use `discover.locations(store, dirs)`.
  - Print `local_variant_id` when there's no catalogue id.
  - Print paths with `errors="replace"`.
  - `community share` prints `skipped`.
  - `.ps1` is written with `utf-8-sig` and `.sh` with mode 755.
- **`export_config(port=)`**.

### `r3-server`: `server.py`, `tests/test_server.py`, `tests/test_api_v04.py`

- `POST /api/quality/vote` accepts `slot: "tie"`.
- `/favicon.ico` returns 204 (no console error).
- `GET /api/export` → `{"formats": export.formats()}`.
- Pass `hardware` to `quant_check_job` and `min_tps` (latest report) to `test_job`.
- `/api/local-models`: use `discover.locations(store)` and include `local_variant_id` (never paths).
- Pass `refresh(cancel=)`; progress keys are `stage/done/total/message`.
- The scan job kind is `local_scan`.
- `community/share` passes `skipped` through.
- `serve` status never carries `model_path`.
- Update the test `Fakes` to the real module signatures.
- Pass `runtime_backend` wherever launch configs are built.

### `r3-llama`: `llama_server.py`, `testing.py`, `tuner.py`, `tests/fixtures/*`, `tests/test_llama_server.py`, `tests/test_testing.py`, `tests/test_tuner.py`, `tests/integration/test_fake_matches_real.py`

- The fake llama-bench accepts `-lm/--load-mode` and prints llama.cpp-style `error loading model: …` lines on stderr for load failures.
- `tuner._plain_failure`: read the last stderr line first (stdout is `"[\n"` with `-o json`). Use `-lm none` instead of the deprecated `-mmp 0`, if not done.
- Real llama-bench never prints the out-of-memory cause without `-v`. Make OOM detection honest: "probably not enough memory" only when a smaller setting in the same run worked, or when a verbose re-run says so.
- **Filler prompt sizing:** measure with `server.tokenize` and trim to `context − 256`. The api sessions saw "The text is longer than the model's context window" at 2048/4096 on the real tiny model.
- **Tuner:** measure the final `best` at full depth (`-d context−640`, clamped) and store `settings.depth`, so a tune verifies speed at the chosen length.
- `ServerRegistry` / exit handlers: keep `LLAMA_ARG_CORS_ORIGINS=localhost` (already set). Optionally set `LLAMA_ARG_ENDPOINT_SLOTS=0` for the user's server.

### `r3-engine-community`: `engine.py`, `speed.py`, `learning.py`, `community.py`, `export.py`, their tests, `.github/ISSUE_TEMPLATE/*`

- `matching_tuned`: label or scale `best_result.tps` by the depth it was measured at (`settings.depth`), like `interpolated_speed`.
- `matching_speed` compares the **resolved** thread count; testing now stores the threads llama-bench actually used.
- `source == "local"` variants (`sha256=None`) match measurements and tunes by `variant_id`.
- iSWA cache (Gemma 3 / gpt-oss): sliding layers hold about `min(ctx, pad256(window·(n_seq if unified else 1) + ubatch))` cells, not `window`.
- `community_speed`: catch `AttributeError`; pass `n_cpu_moe` in the evidence config. `community.evidence` matches `settings.cache_type_k`, `n_cpu_moe` and depth itself, and caches `hardware_class`/`_personal_words` (the report-time hot spot).
- Export: the docker image and community `backend` come from the **runtime** backend (`runtime_install.detect`) when known, not the hardware.

### `r3-ui`: `static/*`, `tests/ui-*.test.cjs`, `package.json`, `package-lock.json`

- Speed labels:
  - `evidence: "tuned"` reads "measured after tuning".
  - Show `speed_interpolated.tps` and `community.median_tps` with honest labels.
  - A short tune has `tps: null`, `verdict: "unknown"`.
- Tune panel: drop the UI's own stop sentence (the tuner's `notes` already say it).
- Compression check results: look up names through `variant_ids: {label: variant_id}`.
- Community share: show "N results were measured on different hardware and were left out" when `skipped > 0`. With `fits_in_url: false`, show copy-JSON plus paste instructions.
- Local models:
  - `f.filename`, `l.label`, `local_variant_id`.
  - Show "saved" only when `result.saved !== false`.
  - Hide "step x of y" when `done`/`total` are null.
  - The scan kind is `local_scan`.
  - Reveal uses ids plus `labels`.
  - `downloads/plan.local === true` hides the Download button.
- Hardware:
  - `cpu_name || cpu`.
  - List `other_gpus` (name plus reason) and `warnings`.
  - Label unified GPUs "Shared with system RAM".
  - Escape `gpu.name`.

### `r3-misc`: `runtime_install.py`, `downloads.py`, `runtime.py`, `catalogue.py`, `quantcheck.py`, `discover.py`, `gguf.py` and their tests

- `runtime_install.install()`: an HTTP-triggered install must not be shadowed by an older `settings.runtime_dir`. Clear it or make the fresh install win, and say so. Byte progress carries `"unit": "bytes"`.
- `runtime_install._gpu_kinds`: when choosing Vulkan from `other_gpus`, CPU-only runs need `-dev none`. Document the rule for callers.
- `catalogue.get_json`: strip `Authorization` (HF_TOKEN) on cross-host redirects, like `downloads.py`.
- `quantcheck.estimate_logits_bytes`: the header is 20 bytes (`_logits_` plus 3×int32), not 16. Use `gguf.read_metadata(...)["summary"]["vocab_size"]`, now available.
- `runtime.bench`: make sure records carry `kind`, `id`, `settings`, `depth` and `runtime` (runtime-downloads says done; verify). It goes through `launch` for the env (CORS, filtered `LLAMA_ARG_*`).

## Everyone

- Own only your files. Write needs elsewhere under **Requests** in `docs/v0.4/fixups/r3-<area>.md`.
- Real llama.cpp (build in the background, see FIXUPS.md) for anything touching llama.cpp behaviour. The final check is `tests.integration.test_real_runtime` + `test_fake_matches_real`.
- Done means: `python3 -m unittest discover -s tests` and `npm test` green, and the notes file written with fixed / rejected / requests / evidence. Commit and push to `claude/v04-<area>`.
