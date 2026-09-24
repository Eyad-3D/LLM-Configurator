# Fix-up notes: api-http (`server.py`)

Branch `claude/v04-fix-api-http`. Files changed: `src/llm_configurator/server.py`, `tests/test_api_v04.py` (plus this file). `tests/test_server.py` needed no change.

Job logic (test, tune, quiz, compare, quant check, download) lives in `app.py`, which belongs to **api-cli**. I did not edit it. Where the fix belongs in `app.py`, it is listed under Requests. `server.py` detects new `app.py` parameters, so both branches merge in either order.

## Fixed

Review findings are numbered as in `docs/v0.4/review/api-http.md` (R#). Handoff requests are named by their file.

| What | Addresses |
|---|---|
| `serve()` turns SIGTERM, SIGHUP and SIGBREAK into the Ctrl+C path. `server_close()` then stops the llama-server. Checked on the real build: after `kill -TERM <app>`, no llama-server is left. | R2 |
| Path scrubbing reworked. (1) Folder names may contain spaces (`Program Files`, `My Models`, `OneDrive - Contoso Ltd`). (2) A value that is just an absolute path becomes its file name. (3) More folders are replaced whole by `~`: the models folder, the runtime folder, configured runtime and models folders, the temp folder, the Python prefix, the package folder and the discovered model folders. Replacement matches whole folders only, so `/tmp` no longer eats the end of `/x/private/tmp`. (4) A long `log_tail` drops its first line, which may begin partway through a path. | R3, ui-run #6 |
| Job errors, progress and results are scrubbed inside the job wrapper, before `jobs.py` stores them. Paths never sit in job memory, and every route (and any future one) inherits this. | ui-run #6, "path-free job errors and results" |
| `/api/export`, `/api/serve/start` and `/api/test` use the saved tune when the page doesn't send `tuned`. An explicit `tuned: false` still wins. Detection goes through `app.launch_config(..., tuned=True)`, so it follows api-cli's matching rules. | R4, ui-run #5 |
| The served llama-server uses port 8080 when it is free, so the exported client setups work unchanged. A busy port falls back to a free one. The check ignores TIME_WAIT, as llama-server does. Checked on the real build: the server listened on 8080. | R5 |
| A tune job result gets `saved: false` when the tune was cancelled (the record is not stored). `run.js` can show the right sentence. | R7 (server half) |
| `/api/serve` status and the serve job result carry `candidate_id` and `variant_id` (null when idle). `config.*_path` values are file names. `IDLE` gains `starting` and `error`, so idle and running have the same keys. | R11, ui-run #4, FIXUPS pinned shape |
| `server_close()` waits up to 15 s in total for cancelled jobs, so llama-bench and llama-perplexity children are killed before the app exits. | R12 |
| `/api/export` keeps `instructions` and `notes` unscrubbed (merge commands for split models). | R13 |
| llama-server "loading" progress is sent as `done/total = null` plus `seconds`, so the tray no longer shows "step 37.5 of 300". Only LlamaServer's own messages are rewritten; evals' "loading" stage (a real count) is untouched. | R14 (server-side option) |
| Local models (`source == "local"`) are never downloaded or deleted. If the file is still there, `/api/downloads` returns `reused`. If it moved, the call returns 409 with a plain message instead of a Hugging Face 404. `/api/downloads/plan` still answers, with `local: true`, because run.js calls it for every candidate. `/api/downloads/remove` returns 409. | R15, discover "never downloaded" |
| `/api/local-models` lists `discover.locations(store)`, which includes the app's models folder. | R16, discover request |
| `ServerRegistry(log_dir=<data>/logs)`, stopped on shutdown. | server request |
| The latest report's `requirements.min_tps` is remembered and passed to `app.test_job(min_tps=…)` when that function accepts it (api-cli is adding it). | R10 (server half), testing request |
| Reveal returns the pinned shape: `mapping[i] = {"A": candidate_id, "B": …}` plus `slots`, `item`, `vote` and `winner` (candidate ids). `tallies`, `overall` and `speed` are keyed by candidate id, plus `labels: {candidate_id: friendly name}`. The label→id map is kept in memory per comparison. After a restart, labels are used as ids (quality.js already accepts that). | FIXUPS reveal shape, ui-quality request |
| Reveal while answers are still coming returns 409. Before, it set `revealed` and made the running compare job fail. | review agent (evals seam) |
| Vote accepts `"tie"`, as evals does. An unknown comparison id now returns 404 instead of 400 on GET, vote and reveal. | evals request |
| Job kind `scan` renamed `local_scan` (pinned kind list). | FIXUPS job kinds |
| `GET /api/export` → `{"formats": export.formats()}`. | ui-run #3 |
| `POST /api/catalogue {base_repo, gguf_repo}` checks the names, calls `catalogue.add_entry`, then runs a `catalogue_refresh` job with `refresh_entry` (it waits for `refresh_lock`). `POST /api/catalogue/remove {base_repo}` calls `remove_entry`. Both clear the cached candidates and return 409 in demo mode. | catalogue request (optional) |
| `/api/community/share` accepts 1–50 ids matching `[A-Za-z0-9_-]{1,64}` (the same as `share_payload`; the old hex-only rule rejected valid ids). `json` and `issue_url` are passed through unchanged, because they are exactly what the user posts. | community request |
| Token comparison uses `secrets.compare_digest`. `/api/credentials/save` checks that `remember` is a bool. `/api/map` checks that `base_repo` and `slug` are text (a missing key gave the error text `'base_repo'`). | own review |
| A corrupt user `catalogue.json` no longer makes `/api/state` fail. It returns `definitions: []` plus `definitions_error`. | review agent |

Response shapes are additive only. The one pinned exception is reveal's `mapping` entries, which are now keyed by candidate id.

## Rejected or not mine (with reasons)

- **R1 (CORS `*` on every llama-server): real, but not in my files.** Confirmed on the real build: a chat request with `Origin: https://evil.example` got `Access-Control-Allow-Origin: https://evil.example`. The fix belongs in `llama_server.py` (Request to server-fake below). `server.py` cannot set the child's environment.
- **R6 (`quality.js` reads `f.path`/`l.path`): real, UI-owned.** The API returns `filename`/`label` on purpose. See the Request to ui.
- **R7 (UI half), R14 (jobs.js half):** UI-owned. The server now sends `saved` and null steps, so the UI only needs to read them.
- **R8 (require_model hashing without progress/cancel), R9 (`discover.local_variants` not merged into `app.variants`, `local_variant_id` missing from `page_file`): real, in `app.py`.** See the Requests to api-cli. The api-cli review lists the same items (#2, #15).
- **R10 (app half):** `app.test_job` needs a `min_tps` parameter. The server already passes it once that exists.
- **Record shapes in `quality_results` and `tuned` (FIXUPS pinned): written in `app.py`,** not `server.py`. See the Requests to api-cli.
- **Removing `directory`/`binaries` from `/api/runtime`:** not done, because only additive changes are allowed. Their values are reduced to names (`bin`, `llama-server`).
- **Hiding blinded outputs while a comparison runs (the answer order leaks which model is which):** evals-owned. The server now refuses reveal while running. See the Request to evals.

## Requests

- **api-cli (`app.py`)**
  - `test_job(..., min_tps=None)` → `testing.run_tests(..., min_tps=min_tps)`. The CLI can add `test --min-tps`.
  - `tune_job`: store the pinned tuned record `{**tune_result, variant_id, sha256, fingerprint, context, gpu_layers, n_cpu_moe, kv_cache_type, timestamp}`. Optionally also store a `kind="tune"` measurement. Return `saved` yourself too (the server fills it in if it is missing).
  - `quiz_job`: store `{"kind": "quiz", variant_id, candidate_id, workload, score, ci_low, ci_high, "needle": result|None, "result": full run_quiz result, timestamp}`. Today it stores a separate `needle` record and no `candidate_id`/`result`.
  - `quant_check_job`: store `{"kind": "quant_check", "variant_id": <reference>, "result": …, timestamp}`. Drop `path` from `results[*]`, and drop `corpus.path` and `reference.path` from both the stored record and the job result. Key the job result's `results` by variant id, as quality.js looks them up that way.
  - `local_model`/`require_model(progress=, cancel=)` forwarded to `discover.find_for_variant` (R8).
  - Merge `discover.local_variants(store)` into `variants()`, and add `local_variant_id` to `page_file` (R9).
  - `download_job`: write `hash_cache` for the files it just verified, so the first test does not re-hash them (R8).
  - `export_config`: accept `port=` so the server can pass the running server's port when it had to fall back from 8080.
- **server-fake (`llama_server.py`)**: start every llama-server with `LLAMA_ARG_CORS_ORIGINS=localhost` in its environment (the env var is ignored by builds that lack it) (R1). Optionally set `LLAMA_ARG_ENDPOINT_SLOTS=0` for the user's server.
- **ui (`quality.js`, `run.js`, `jobs.js`)**
  - Local models: use `f.filename` and `l.label` (R6), and update the ui-quality fixtures to that shape.
  - Show "saved" only when `result.saved !== false` (R7).
  - Hide "step x of y" when `done`/`total` are null (the server now sends null for model loading) (R14).
  - The scan job's kind is now `local_scan`.
  - Reveal now uses candidate ids plus `labels`.
  - `/api/downloads/plan` has `local: true` for the user's own files, so the Download button can hide.
- **evals**
  - Optional `ids={label: candidate_id}` on `run_comparison`/`start_comparison`, saved on the record, so reveal survives an app restart.
  - While `state == "running"`, `blinded()` should not show which slots are already `ready`, because answer order gives the model away.
- **testing-tuner**: with the tiny real model, the speed test fails at both 2048 and 4096 context with "The text is longer than the model's context window". `filler_prompt` assumes about 0.75 words per token. Measure it with `server.tokenize` and trim it to fit `context - 256` (I saw this with the real build; real-size models may be fine).

## Test evidence

- `python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: 620 tests, the same 3 failures as the base branch, none in my files:
  - `test_adapters.test_aa_pagination_and_null_scores` (credentials/keyring)
  - `test_fake_matches_real` ×2 (server-fake)
- `python3 -m unittest tests.test_api_v04 tests.test_server`: OK. There are 15 new tests in `FixupTests`, `ServeProcessTests` (a real `serve` subprocess exits 0 on SIGTERM) and `DemoCatalogueTests`, plus the opt-in `RealRuntimeHttpTests`.
- `npm ci && npm test`: 56 pass, 0 fail.
- Real llama.cpp: I built it with `JOBS=4 bash scripts/build_llama_cpp.sh /opt/llama-work` (bin: `/opt/llama-work/llama_cpp_python-0.3.35/vendor/llama.cpp/build/bin`) and made the tiny models with `scripts/make_tiny_models.py`.
  - `LLM_CONFIG_REAL_RUNTIME=<bin> LLM_CONFIG_TINY_MODELS=/opt/llama-work/models python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real`: 32 tests, the 3 known failures (runtime_install build parsing, fake `--draft-max`, fake bench `--version`).
  - `tests.test_api_v04.RealRuntimeHttpTests` with the same variables: OK. It covers a smoke test, serve start, `/health` and serve stop, with a data folder named "My Data", and asserts that no folder reaches the page.
- End-to-end run of the real app, with both folders named with spaces:
  ```
  llm-config --data-dir "<tmp>/My Data" runtime use <bin>
  llm-config --data-dir "<tmp>/My Data" local --add "<tmp>/My Models/tiny-llama-{Q4_K_M,Q8_0,F16}.gguf"
  python3 -m llm_configurator --data-dir "<tmp>/My Data" serve --no-browser --port 18765
  ```
  I drove it over urllib, checking every response for the temp folder, the bin folder, "My Models" and "My Data":
  - `/api/runtime`, `/api/local-models`, and the scan job (`local_scan`)
  - `/api/recommend` (context 2048)
  - plan/download of a local model (the old 409 led to the fix above)
  - test smoke (passed, 42 answered)
  - test speed (fails in testing, see Requests)
  - tune with a 60 s budget (done, `saved: true`). The export afterwards carries the tuned `-fa off`.
  - export formats
  - quiz with needle, blind compare Q4 vs Q8 with vote A, vote tie, and reveal by candidate id
  - unknown comparison → 404
  - quant check F16 vs Q4_K_M (done, negligible)
  - quality results
  - serve start on 8080 with `candidate_id`
  - test while serving → 409
  - serve stop
  - SIGTERM with a server running, after which no llama-server was left

  The only folders in any response were the export script's own paths, which are allowed.
