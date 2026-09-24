# Round 3 notes: r3-app-cli (`app.py`, `cli.py`, `tests/test_cli.py`)

Branch `claude/v04-r3-app-cli`, based on `claude/keen-volta-7mnrm5` (3f6d208). Every `app.py` signature stays
backward compatible: only keyword arguments were added, and `server.py` works unchanged.

## Fixed

| Item | What changed |
|---|---|
| Tuned record | `app.tune_records()` builds the pinned record: `{**tune_result, variant_id, sha256, fingerprint, context, gpu_layers, n_cpu_moe, kv_cache_type, timestamp}` plus `depth`, `runtime`, `runtime_build` (and the extra keys `id, goal, users, placement, budget_seconds`). `settings.depth`, `confirmed` and `seconds` survive because the whole result is spread. `depth` is `settings.depth` when the tuner sets it, else the tuner's `depth`. It is also written into `settings.depth`, because `engine.matching_tuned` reads only that, so today's tunes are labelled with the depth they were measured at. **`context/gpu_layers/n_cpu_moe/kv_cache_type` are the placement the tune started from**, because that is how `engine.matching_tuned` reads them (its docstring and `test_tunes_match_the_pinned_record_shape`). Before, `gpu_layers` came from `best`. Where the tuner ended up stays in `best`. |
| Tune measurement | A §2.3 record with `kind: "tune"`, `id`, `settings` (the six §2.3 keys from `best`), `depth`, `threads` (the count llama-bench really used), `runtime`, `runtime_build`, `tps`, `pp_tps`, `users`, `gpu_layers`, `gpu_uuid`, and `raw.tuned_record` pointing at the tune. It is written only when the depth is a known integer, because the engine reads a record without `depth` as a full-length test. A shallow one never shows as "tested" (engine `_full_depth`). |
| `saved` | `tune_job` returns `saved` (false for a cancelled tune, which is not stored) and `record_id` (None when not saved). |
| `launch_config(tuned=True)` | New `app.tuned_changes(store, candidate, hardware, variant=None)` uses `engine.matching_tuned` on the candidate's own model, context, GPU layers, expert offload, notepad format and GPU. A tune that started from another placement is never applied, so the "48-layer MoE tune on a 35-layer split" crash cannot happen. It applies the tune's threads, batch, ubatch and flash attention (flash attention "off" is dropped for a compressed notepad, as the engine does). If the tuner moved layers or experts, its settings belong to that new placement. They are used together with it only when the engine lists it as fitting in memory now, on the same GPU and on the same side (GPU or CPU only). Otherwise it raises "…no longer fits in free memory. Run a new tune.", because batch sizes found for another split can run out of memory. Candidates for several users get "Tunes are made for one user at a time". `best_tune` is removed; nothing else called it. |
| `runtime_backend` | `launch_config(..., runtime_backend=None)` is a new keyword; r3-server's branch already passes it. Without it, `app.installed_backend(store)` reads the cached `store["runtime"]` while its llama-server still exists, as `runtime_install.binary` does, and runs `detect()` only when there is no cache. On any error it gives None. The value goes into `launch.from_candidate(..., runtime_backend=)` for test, tune, quiz, compare, quant-check, serve, export and run. |
| `quiz_job` | One pinned record: `kind, variant_id, candidate_id, workload, score, ci_low, ci_high, needle (result or None), result (full run_quiz result), timestamp` (plus `name, quant, context, placement, correct, total`). The separate `needle` record is gone; the UI already reads only `kind == "quiz"`. |
| `compare_job` | Runners are keyed by candidate id, with `names={candidate_id: label}` passed to `evals.run_comparison`, so `reveal()` returns ids plus `labels`. The same candidate twice gets `id#N`. `server.by_candidate` still gives the same output: the ids pass through and `labels` maps id to name. |
| `quant_check_job` | Stores `{kind, variant_id: <reference>, result, timestamp}` plus extras. `variant_ids` is now `{label: variant_id}` everywhere, and `reference_quant` is new. kl_check's `reference` dict is no longer overwritten; the job returns `reference_quant` and `reference_variant_id` instead. A variant without `quant` is named by `quantcheck.quant_label(path)`, and so is the reference (`reference_label=`). `work_dir=<data dir>/tmp`. `hardware` is accepted, as before. `path` keys are removed from everything stored or returned, and kl_check already redacts folders in its messages. |
| `test_job` / `test --min-tps` | Already done in the fix-up round. Kept, and a test covers it. |
| Local models | `local_model` passes `progress`/`cancel` to `discover.find_for_variant`, and its own size-only models-folder fallback is removed, because discover now checks flat and nested copies itself. `variants()` merges `discover.local_variants`, deduped by id (already done). `add_local_job` delegates to `discover.add_file` with the first shard (already done). `download_job` and CLI `download` call `discover.remember_hash` for every verified file (`app.remember_download`). `page_file` keeps `match_note` and adds `local_variant_id`. |
| Catalogue | `map_benchmark` saves via `catalogue.set_slug` (locked, atomic; no fixed `catalogue.tmp`). `refresh` gets `cancel=` and its progress is passed through unchanged, because catalogue now sends `stage/done/total/message`. `models add --config-repo REPO` is passed to `add_entry`, then `refresh_entry(..., cancel=)`. |
| CLI | `main()` calls `llama_server.install_exit_handlers()` once, in the main thread. Tested for real: a SIGTERM to `llm-config run` stops its llama-server. `local` uses `discover.locations(store, dirs)` and prints `local_variant_id` (already done). Paths are printed through `cli.shown()`, which replaces characters the terminal cannot show. `community share` says how many results were left out and, when `fits_in_url` is false, tells the user to paste the text into the empty form. `.ps1` is written as `utf-8-sig` and `.sh` gets 0755 (already done; checked for real). New `export --port`. |
| Review fixes | An independent review of the diff found these, now fixed. A SIGTERM or SIGHUP during a job cancels the job and waits for it, like Ctrl+C, so llama-bench and llama-perplexity are not left running. `remember_download` also ignores `OSError`. Other variants whose quant label equals the reference's are named by their id. `export --port 0` is passed on, so launch rejects it plainly. The `--config-repo` help and the `community share` message ("could not be shared (measured on different hardware, or incomplete)") are in plainer words. |
| `export_config(port=)` | New keyword. The server can pass the port of the server that is already running. |

## Rejected / changed

- **`kv_cache_type` from `best.cache_type_k`** (testing-tuner request): not done. The engine reads the top-level
  value as the tune's starting notepad format. The app tunes with `allow_kv_compression=False`, so the two are
  the same anyway. `best.cache_type_k` stays inside `best`.
- **Keeping a separate `needle` record for old readers**: dropped, as the pinned shape asks. See the request to r3-server below.

## Requests

- **r3-server (`tests/test_api_v04.py`)**: two tests assert the old shapes this round was asked to change. With
  this branch they fail until updated:
  - `test_quiz_saves_quality_results_and_stops_server`: kinds are now `["quiz"]` (needle is inside the quiz record).
  - `test_quant_check_needs_local_files_and_saves_results`: `done["result"]["reference"]` is kl_check's dict now. Use `done["result"]["reference_quant"] == "Q8_0"`.
  - Optional: the `Fakes` could gain `discover.remember_hash`, `llama_server.install_exit_handlers` and the real
    `kl_check`/`run_comparison` keywords (`reference_label`, `work_dir`, `names`). app.py tolerates their absence
    (`_call` / `getattr`), and `tests/test_cli.py` adds some of them locally.
  - `/api/serve/start` and `/api/export` can pass `port=` to `app.export_config` for a server that moved off 8080.
- **r3-server (`server.py`)**: `use_tune` could call `app.tuned_changes(store, chosen, hardware, variant)` instead of a whole `launch_config`, because it only asks "is there a usable tune?".
- **r3-llama (`tuner.py`)**: `tune_records` pairs `best_result.tps` with `settings.depth`. When the final full-depth measurement lands, please put its speed in `best_result` (or the depth in `settings.depth` only when `best_result` came from that run). Otherwise a shallow speed would be labelled as full depth.
- **r3-ui (`quality.js:851`)**: a compression-check job returns `reference_quant` (a string). `reference` is now kl_check's dict. The page already prefers the selected reference, and it could read `out.reference_quant` next.
- **r3-engine-community (`export.py:199`)**: the note "CUDA_VISIBLE_DEVICES pins the model to the graphics card it was tested on" now shows on every export, even on CPU-only machines. Since the lead's change, `env` always holds `LLAMA_ARG_CORS_ORIGINS`. Show the note only when `"CUDA_VISIBLE_DEVICES" in env`.
- **r3-misc (`runtime_install`)**: a real source build still shows as `llama.cpp b1 (cpu backend)` (build 1, commit 4df29be). This is known.

## Evidence

- `python3 -m unittest discover -s tests`: 821 tests, 2 failures. Both are the r3-server tests above, which assert the old quiz and quant-check shapes. I also ran my app.py and cli.py on r3-server's pushed branch (4a0a37d): `test_server`, `test_api_v04` and `test_cli` give 108 tests with the same 2 failures, and nothing else.
- `npm test`: 71/71.
- Real llama.cpp (`scripts/build_llama_cpp.sh`, tiny models), CLI end to end on a fresh data folder:
  - `runtime use <bin>`, then `local --add` for tiny-llama Q8_0 and Q4_K_M, which gives `local:…` ids.
  - `recommend`: both listed. After `test`, Q4_K_M shows "Tested on this computer: about 1038 tokens per second".
  - `test --kind full --min-tps 5 --context 2048`: works; reading 5179 t/s, writing 1038 t/s, memory within the estimate.
  - `tune --budget 60 --context 2048`: stored record top-level `context 2048, gpu_layers 0, n_cpu_moe 0,
    kv_cache_type f16, depth 512, runtime {version b1, backend cpu}, runtime_build 4df29be, threads 4`, with
    `confirmed`, `seconds` and `goal` kept. There is a `kind: "tune"` measurement at depth 512, which the engine does not count as measured at 2048.
  - `test --kind smoke --tuned`: uses the tune.
  - `quiz --needle`: one `quiz` record with the needle nested. The tiny random model scores 0/53, which is expected.
  - `quant-check Q8_0 Q4_K_M`: "less than 1% different top word". The stored record has `variant_ids {Q4_K_M: …}`
    and `reference_quant Q8_0`, and holds no paths. Temporary files went to `<data>/tmp`.
  - `export`: `.sh` is 0755, and `.ps1` starts with the BOM `ef bb bf`. `--port 8099` shows in the address.
  - `run --port 8097`: `/v1/models` answers, and a request with a foreign Origin gets no CORS header. A SIGTERM to the CLI leaves no llama-server behind.
- `tests.integration.test_real_runtime` + `test_fake_matches_real`: 43 tests, OK.
