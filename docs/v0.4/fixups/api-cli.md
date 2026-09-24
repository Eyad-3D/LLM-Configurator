# Fix-up notes: api-cli (`cli.py`, `app.py`, `tests/test_cli.py`)

Branch `claude/v04-fix-api-cli`, based on `claude/keen-volta-7mnrm5`.

## Fixed

Review findings (`docs/v0.4/review/api-cli.md`). All 17 were verified; 17 were real and are fixed.

| # | Fix |
|---|---|
| 1 | `app.local_model(verify=True)` never returns an unverified file. The models-folder fallback (for shards that downloads saves flat) now hashes each file before trusting it. The size-only answer is kept only for `verify=False`, which is used for plans and export paths. A same-size file with the wrong contents now leads to a download, and `downloads` explains the clash. |
| 2 | `app.variants` merges `discover.local_variants(store)`, so scanned GGUFs that match no catalogue model can be recommended, tested, tuned and run. Their IDs are printed by `local`. `local --add` accepts files without a `.gguf` suffix (Ollama blobs), and discover checks the contents. |
| 3 | `local --add` takes any part of a split model and registers the first part, via `gguf.shard_paths` + `discover.add_file` (the discover request). Every part is hashed first, so the ID is stable and based on the SHA. |
| 4 | `export --output` writes bytes, never text mode, so there is no `\r\n` on Windows. `.ps1` is written as UTF-8 with BOM, and `.sh` (or anything starting `#!`) gets 0755. The "save as UTF-8 with BOM" note and the "Save this as / chmod +x" instructions are dropped because the CLI already did those steps. |
| 5 | `show_tune` prints writing and reading speed, each with its own ratio. It lists only settings that actually changed (from the new `changes` key in the tune result), in plain words and without `None`. |
| 6 | `main()` reconfigures stdout/stderr with `errors="replace"`, so redirected output on Windows code pages can't crash. `local` now uses ASCII markers. Also added: `llm-config models \| head` exits quietly instead of printing "Broken pipe". |
| 7 | `download --directory DIR` registers the finished file with `discover.add_file`. If that fails, it prints the `local --scan --dir` hint. |
| 8 | `quiz --needle` prints "found X of Y" and evals' `note`. |
| 9 | `app.test_job(..., min_tps=None)` passes the value on to `testing.run_tests`. The CLI adds `test --min-tps` (default 15, `0` = none). |
| 10 | The TTY progress line is cut to the terminal width minus one. |
| 11 | `loading`/`starting`/`scanning` are open-ended: the bar shows elapsed seconds, never a percentage. The label is the job's plain `message`, or the stage in words ("First word"), never a raw ID. Tune progress reads "7s of 1m 00s", and small files read as MB. |
| 12 | `bench --executable` defaults to `app.binary(store, "llama-bench")`. Bench measurements also get `id` and `kind: "bench"`, so `community share` can pick them. |
| 13 | `best_tune(..., users=None, kv_cache_type=None)` only matches tunes made for the same number of users and notepad format. `launch_config(tuned=True)` passes both, and old callers keep working. |
| 14 | `local --dir` paths are resolved before they are saved. `discover.locations(store, dirs)` is used in both listings. |
| 15 | `local_model` and `require_model` take `progress`/`cancel` and pass them to `find_for_variant`. Every job (test, tune, quiz, compare, quant-check, download) hashes with a "verify" progress step and can be cancelled. The CLI `download` does its reuse check as a job with a progress bar. |
| 16 | `download` without `--yes` and without a keyboard says "Add --yes …", and EOF counts as "no". |
| 17 | `models add` runs `catalogue.refresh_entry` and prints how many downloadable versions were found. If that fails, it says to run `llm-config refresh` later. `refresh` now runs as a job, so Ctrl+C stops it and it shows a count-based progress bar. |

Handoff requests (CLI/app side):

- **docs:** `recommend --kv` already existed on the merged base. CI's `recommend --demo --context 2048` passes (test `test_existing_ci_commands`).
- **downloads:** the command plans first (sizes, free space), then asks, then downloads with a progress bar. Ctrl+C prints "Paused. Run the same command again to resume." and exits 130. Own local models are never downloaded (`app.download_job` and the CLI refuse them).
- **runtime:** `runtime install` passes `scan(False)` and prints `reason` ("Why this build: …") and `warnings`. After a managed install, the CLI clears `settings.runtime_dir` (a leftover from `runtime use`), which would otherwise still take priority in `detect()`. `runtime use DIR` works (checked against real llama.cpp).
- **export:** see finding 4.
- **discover:** `local --add` → `discover.add_file`, `--dir` → `extra_dirs`, `local_variants` merged into `app.variants` (findings 2, 3, 14).
- **catalogue:** `models add|remove` and a cancellable refresh (finding 17).
- **engine:** `engine.recommend(..., community=(), tuned=())` accepts a list or a dict. `app.evaluate` passes `community.records` (a list) and the `tuned` list, and `matching_tuned` reads `best.*`, `best_result.tps`, `sha256`, `fingerprint` and `timestamp`, all of which the new tuned record has.
- **testing:** `min_tps` is passed (finding 9). `commands` has both naming styles.
- **tuner / pinned tuned record:** `store.append("tuned", {**tune_result, "best" (without model_path), variant_id, sha256, fingerprint, context, gpu_layers, n_cpu_moe, kv_cache_type, timestamp})`, plus the extra keys `id, users, placement, goal, budget_seconds`. `kv_cache_type` is the notepad format the user asked for, so `--tuned` finds the tune again even when the tuner compressed the cache to fit (that choice stays in `best.cache_type_k`). `allow_kv_compression=False`: with `--kv q8_0`, the tuner's cache step would otherwise try *un*-compressing. A cancelled tune is not stored.
- **evals / pinned quiz record:** `{"kind": "quiz", variant_id, candidate_id, workload, score, ci_low, ci_high, needle, result, timestamp}`, plus name/quant/context/placement/correct/total. A separate `kind: "needle"` record is still written for pre-fix-up readers (the existing api-http test expects it). If `needle_test` raises ValueError (for example, the context is too short), the quiz is kept and `needle` becomes `{"error", "note"}`.
- **quantcheck / pinned record:** `{"kind": "quant_check", "variant_id": <reference>, "result", "timestamp"}`, plus the older keys. `path` keys are removed from the stored and returned result (they reach `/api/quality/results`). `kl_check` gets `config=`, a launch config that fits the reference at context 512 (CPU if nothing fits, `None` if the hardware can't be read), so `-ngl/-t/-dev` are passed. `quant_check_job(store, reference, others, hardware=None)` stays backward compatible.

Found in my own review and the real-runtime run:

- **Ctrl+C left llama-bench running.** After an interrupted `Thread.join()`, a second `join()` returned at once, so the CLI exited while the job was still stopping, and `llama-bench` kept running as an orphan (seen with real llama.cpp on `tune` and `test`). `run_job` now polls the job state for up to 120 s. After the fix, tune/test Ctrl+C exits with 130 in about 0.15 s and leaves no processes behind. Test: `test_ctrl_c_waits_until_the_job_has_stopped`.
- No §2.3 `kind: "tune"` measurement is written. The tuner measures at a shallow depth, and `engine.matching_speed` would show that faster number as "tested on this computer" at the full context.
- `compare_job` hashes with progress/cancel.
- `run` flushes the address line so `llm-config run … | tee` shows it at once.
- `argparse` prog is `llm-config` (it used to print `__main__.py` when run with `python -m`).
- `app._call(fn, …)` leaves out keyword options that an older module version doesn't take. `engine_extras` already worked this way, and the shared test fakes in `test_api_v04.py` still have the old signatures.

## Rejected

None. All 17 findings reproduced as described.

## Requests to other areas

- **api-http (`server.py`, `tests/test_api_v04.py`):**
  - Pass `hardware` to `app.quant_check_job(store, ref, others, hardware)`. Without it the job scans hardware itself.
  - Pass `min_tps=<latest report's requirements.min_tps>` to `app.test_job`.
  - Update the `Fakes` in `test_api_v04.py` to the real signatures: `discover.find_for_variant(…, progress, cancel)`, `discover.locations(store=None, extra_dirs=())`, `discover.local_variants`, `discover.add_file`, `gguf.shard_paths`, and `tune(…, allow_kv_compression=False)`. `tests/test_cli.py` adds them locally in `match_real_signatures`.
  - Once your quiz test reads `quality_results[0]["needle"]`, the extra `kind: "needle"` record can go.
  - Serve `/api/local-models` with `local_variant_id` so the page can use scanned models.
- **testing-tuner:**
  - (1) `test --kind full --context 2048` against the real tiny llama fails the speed step: "The text is longer than the model's context window". The first-word-delay filler must fit a small context; at 8192 it works.
  - (2) testing passes `chat_template_kwargs=` to `LlamaServer.chat`, which does not accept it, so it retries without, and thinking models may reply empty. Use `enable_thinking=False` or `extra=` (coordinate with server-fake).
- **engine:**
  - Models found only by a scan have `sha256=None`, so `_same_placement` never counts their test results. For `source == "local"`, match on `variant_id` (the ID already includes a path or SHA key).
  - Consider `kind`/`depth` in `matching_speed` if tune measurements are ever wanted.
- **runtime-downloads:**
  - `install()` should clear or override `settings.runtime_dir` itself for the HTTP path (the CLI does it now).
  - Progress with byte counts should carry `"unit": "bytes"` (runtime download, community import). The CLI already treats `verifying/download/extract` stages as bytes.
  - The version shows as `0.1.0-dev (unknown backend)` for a real build (`build 1, commit 4df29be`). This is the known parse_version and `--list-devices` item.
- **catalogue:** give `refresh()` progress the standard `stage/done/total/message` keys. The CLI maps `models_done/models_total` for now.
- **server-fake:** the two known `test_fake_matches_real` failures (`llama-bench --version`, `--draft-max`) are still open.

## Test evidence

- `python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: 620 tests, 3 failures. The same 3 fail on the base branch without my changes (`test_bench_has_no_version_flag_like_real`, `test_server_rejects_removed_draft_max_like_real`, and with real runtime `test_runtime_install_detects_configured_directory`). All are known and outside this area. `tests/test_cli.py` has 36 tests, all passing; `test_api_v04` and `test_server` pass.
- With `LLM_CONFIG_REAL_RUNTIME` and `LLM_CONFIG_TINY_MODELS` set: the same 3 known failures, everything else passes.
- `npm ci && npm test`: 56 pass, 0 fail.
- `--help` exits 0 for all 31 commands and subcommands. No arguments exits 2 with usage. An unknown ID exits 1 with a plain error. `PYTHONIOENCODING=cp1252 llm-config local > file` exits 0.

End to end with real llama.cpp (build 1, commit 4df29be, CPU) and the tiny models from `scripts/make_tiny_models.py`, `--data-dir` in the scratchpad:

| Command | Result |
|---|---|
| `runtime use <bin>` / `runtime status` | exit 0; all four tools ready |
| `local --add tiny-llama-Q4_K_M.gguf`, `…Q8_0.gguf` | exit 0; "Registered as your own model … Model ID: local:1cea…:tiny-llama-Q4_K_M.gguf" |
| `local --add split/tiny-llama-Q8_0-00003-of-00004.gguf` | exit 0; registers part 00001 |
| `local --add unknown-arch.gguf` | exit 1; plain "cannot size yet" error |
| `recommend --context 2048 --min-tps 1` | exit 0; both local models listed with Model IDs |
| `test ID --kind full` (8192) | exit 0; smoke OK, reading 1855 t/s, writing 82 t/s, memory within estimate |
| `tune ID --budget 60` | exit 0 in 11 s; writing 447 → 842 t/s (1.88x), reading 0.74x, "flash attention = off" |
| `test ID --tuned --kind smoke` | exit 0 |
| `quiz ID --needle --context 4096` | exit 0; 0 of 53 (a random tiny model) and recall 0 of 3, both stored in the pinned shape |
| `quant-check Q8_ID Q4_ID` | exit 0; "practically the same"; stored result has no paths |
| `export ID --format llama-server --tuned --output start.sh` | 0755, `-fa off`; running `./start.sh` answered `/v1/chat/completions` |
| `export … --platform windows --output start.ps1` | starts with the BOM `EF BB BF` |
| `run ID --port 8093 --tuned` | printed the address; chat request OK; SIGINT → "Stopping the server…", exit 0, no leftover process |
| SIGINT during `tune` / `test` | exit 130 after about 0.15 s, no leftover llama-bench (before the fix, llama-bench was orphaned) |
