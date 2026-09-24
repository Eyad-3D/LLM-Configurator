# Handoff: `api` workstream

Branch `claude/v04-api`. Files: `server.py`, `cli.py`, `app.py`, `tests/test_api_v04.py`, `tests/test_cli.py` (`tests/test_server.py` is unchanged and passes).

## What I built

- **`app.py`** holds the shared logic, so the HTTP server and the CLI run exactly the same code. Each long operation is a job function `fn(progress, cancel)` for `JobManager`. Other workstreams' modules are imported inside functions.
- **`server.py`** implements every §5 endpoint.
  - One `JobManager` per server.
  - The latest `/api/recommend` report is cached behind a lock for `candidate_id` lookups.
  - Demo mode refuses everything that would download, test, tune, check quality, export or serve (409).
  - The static allowlist now includes `run.js`, `jobs.js`, `quality.js` and `quality.css`. Missing files return 404.
  - Closing the server (`server_close`) cancels running jobs and stops the managed llama-server.
- **`cli.py`** implements every §6 command, with `--help` text.
  - Progress shows on stderr: a one-line bar on a terminal, and a plain line at each 10% step or stage change when piped.
  - Ctrl+C cancels the running job cleanly and exits with 130.
  - `--json` is available on test, tune, quiz, quant-check, local, runtime status and export.
  - Existing commands still work. `recommend` gained `--kv`; before this, the base branch crashed there because `Requirements` has `kv_cache_type` but the parser didn't.

## Public API as built

### HTTP (§5)

All endpoints keep the existing security rules:

- loopback `Host` only
- `X-Session-Token` on every API call
- `Origin` must be same-origin or absent on POST
- JSON body limited to 64 KiB
- CSP and `no-store`

Errors are `{"error": "plain message"}` with status 400, 404 or 409. Status 500 is returned only for unexpected bugs. Unknown body keys are rejected with 400.

| Endpoint | Notes |
|---|---|
| `GET /api/runtime` | `runtime_install.detect(store)`. Paths are reduced to file or folder names. |
| `POST /api/runtime/install` `{}` | 202 job, kind `runtime_install`, exclusive `download:runtime`. Returns the running job instead of starting a second one. |
| `GET /api/jobs`, `GET /api/jobs/<id>`, `POST /api/jobs/<id>/cancel` | The id must match `job-\d{1,9}`, otherwise 404. |
| `GET /api/local-models` | `{"files": [{filename, size_bytes, sha256, source, variant_id, gguf, mtime, verified}], "locations": [{source, exists, label}]}`. `label` is `~/…` relative to the home folder, otherwise just the folder name. |
| `POST /api/local-models/scan` `{}` | 202 job, kind `scan`. The result is the list of page-safe files. |
| `POST /api/downloads/plan` `{variant_id}` | `downloads.plan(...)` plus `local_copy` (bool). |
| `POST /api/downloads` `{variant_id}` | 202 job, exclusive `download:<id>`. Result: `{reused, bytes, filename, variant_id}`. A second request for the same model returns the running job. |
| `POST /api/downloads/remove` `{variant_id}` | `{freed_bytes}`. Returns 409 while any job uses the model or the managed server is serving it. Only the app's models folder is touched. |
| `POST /api/test` `{candidate_id, kind?, tuned?}` | `kind` is smoke, speed or full (default full). 202 job, exclusive `compute`. |
| `POST /api/tune` `{candidate_id, budget_seconds: int 60–1800, goal?}` | 202 job. The result's `best` has no `model_path`. |
| `POST /api/quality/quiz` `{candidate_id, workload?, include_needle?}` | `workload` defaults to the workload of the latest comparison. Result: `{quiz, needle}`. |
| `POST /api/quality/compare` `{candidate_ids: 2–3 unique, prompts: 1–5 non-empty ≤ 4000 chars}` | 202 job. Result: `{comparison_id}`. Labels are `"<name> <quant>"`, made unique if needed. |
| `GET /api/quality/compare/<id>`, `POST /api/quality/vote` `{comparison_id, item: int 0–99, slot: A/B/C}`, `POST /api/quality/reveal` `{comparison_id}` | `comparison_id` must match `[A-Za-z0-9_-]{1,64}`. |
| `POST /api/quality/quant-check` `{reference_variant_id, variant_ids: 1–3 unique}` | The reference must not be in the list. 202 job. |
| `GET /api/quality/results` | The last 200 records from `quality_results`. |
| `POST /api/export` `{candidate_id, format, platform?, tuned?}` | `format` must be one of `export.formats()` ids. `content` is returned unscrubbed (see Deviations). |
| `GET /api/serve`, `POST /api/serve/start` `{candidate_id, tuned?}`, `POST /api/serve/stop` `{}` | Status is idle until the first start. Stop also cancels a start that is still running. |
| `GET /api/community`, `POST /api/community/import` `{}`, `POST /api/community/share` `{measurement_ids: 1–20 hex strings}` | The import job result is `{source, fetched_at, rejected, count}`; the records themselves come from `GET /api/community` (last 5000). |

Other server behaviour:

- **Stale candidates.** An unknown or stale `candidate_id` returns 409 `"Compare again first. …"`. The cache is cleared when a refresh or `/api/map` finishes. A report computed across such a change is thrown away (generation counter).
- **One model at a time.** Compute jobs return 409 while the managed server runs. Each compute job checks this again when it gets the `compute` slot, and fails with a plain message if a server started in the meantime.
- **Job cap.** At most 12 queued or running jobs; beyond that, 409.
- **Path scrubbing.** Every JSON reply goes through `server.scrub`:
  - The home folder and the data folder are replaced first.
  - Absolute POSIX, Windows, UNC and `~/` paths in any string are reduced to their file name.
  - NaN and ±inf become `null`, and `Path` and other objects become strings.
  - These keys are passed through unchanged: `got`, `expected`, `text`, `prompt`, `prompts`, `reply`, `content` (model-written text) and `label` (already safe).

### `app.py` (shared by server and CLI)

```
variants(store, demo=False) -> [Variant]              # catalogue + registered local GGUFs ("local_variants")
find_variant(store, variant_id, demo=False) -> Variant  # ValueError if unknown
engine_extras(store, function=engine.recommend) -> {"community": [...], "tuned": [...]} filtered to accepted params
evaluate(store, payload, demo=False) -> report          # passes community/tuned when engine accepts them (not in demo)
binary(store, name) -> argv                             # runtime_install.binary
local_model(store, variant, verify=True) -> Path|None   # registered local file → discover.find_for_variant → complete files in models_dir (downloads.plan)
require_model(store, variant) -> Path                   # ValueError("… is not on this computer yet. Download it first.")
download_plan / download_job / remove_download
placement(candidate) -> "cpu"|"gpu"|"split"
best_tune(store, variant_id, context, placement, fingerprint=None) -> record|None   # highest best_result.tps
launch_config(store, candidate, hardware, model_path=None, tuned=False, **overrides) -> launch config
candidate_for(store, variant, hardware, context=None, gpu_layers=None, kv_cache_type="f16") -> candidate  # CLI
test_job, tune_job, quiz_job, compare_job, quant_check_job, runtime_install_job, scan_job, add_local_job -> fn(progress, cancel)
start_server(store, variant, candidate, hardware, tuned=False, progress=None, cancel=None) -> (LlamaServer, config)
export_config(store, variant, candidate, hardware, fmt, platform="posix", tuned=False) -> export result
page_file(record), page_location(location)              # page-safe local file records
```

Records written:

- **`tuned`** (`store.append`):
  - identification: `{id, variant_id, sha256, context, users, placement, gpu_layers, fingerprint, timestamp, goal, budget_seconds}`
  - results: `best` (a launch config without `model_path`), `baseline`, `best_result`, `improvement`, `stopped`, `notes`, `trials` (count)
  - A cancelled tune is not saved.
- **`quality_results`**:
  - `{kind: "quiz"|"needle"|"quant_check", variant_id, timestamp, …}`
  - quiz and needle records also carry `name, quant, context, placement`
  - quant_check records also carry `reference_variant_id, variant_ids, results: {variant_id: parsed}`
- **`local_variants`** (new key): `[{variant: Variant dict, path, added_at}]`, written by `llm-config local --add`.
- **`settings.models_dir`**, written by `llm-config settings --models-dir`.
- **Measurements:** `testing.run_tests` appends them itself, as the contract says. The API does not add a second record.

**`--tuned` and `tuned: true`** reuse the best tune for the same variant, context, placement and hardware fingerprint. They apply only the `TUNABLE` keys: `gpu_layers, threads, batch, ubatch, flash_attn, cache_type_k/v, n_cpu_moe`. Context, users and the model file always come from the candidate.

### CLI (§6)

- **Commands:** `runtime status|install [--allow-unverified]|install-archive PATH|use DIR`, `download VARIANT_ID [--directory] [--yes]`, `local [--scan] [--dir …] [--add PATH] [--json]`, `test`, `tune`, `quiz`, `export`, `run`, `quant-check`, `community import [--source]|share IDS…|status`, `settings [--models-dir]`, `models [add BASE GGUF | remove BASE]`.
- **Model options:** test, tune, quiz, export and run accept `--context`, `--gpu-layers`, `--kv` and `--json`.
- **Choosing the configuration:** the CLI runs the engine on that one model and takes the "now" candidate with the most GPU layers. With `--gpu-layers N` it builds the candidate by hand when the engine didn't list that placement.
- **Exit codes:** 0 on success, 1 on errors, 130 on Ctrl+C. `test` also returns 1 when the verdict is `failed`, and `run` returns 1 when the server dies.

## Deviations from the contract and why

1. **The export response keeps the real model path in `content`.** It is a script the user copies into a terminal, and a file name alone would not run. Other fields are scrubbed.
2. **Export "placeholder path".** When the model isn't downloaded yet, I use the path it *will* have in the models folder, not a fake placeholder, and add a note. This way the script works once the download finishes.
3. **Optional extra body fields:**
   - `tuned` (bool) on `/api/test`, `/api/export` and `/api/serve/start`, to use the saved tune.
   - Test `kind` defaults to `full`, tune `goal` to `generation`, export `platform` to `posix`.
4. **Extra 409s:**
   - Compute jobs are refused while the managed server runs (memory safety).
   - `downloads/remove` is refused while the model is in use.
   - More than 12 active jobs are refused.
   - A missing module (`ImportError`, before the merge) returns 409 "This feature is not installed…".
5. **New store key `local_variants`** for GGUFs registered with `local --add`. I kept it separate from `variants`, because `catalogue.refresh` rewrites that key. Please add it to §2.4.
6. **`map_benchmark`** now rewrites only catalogue variants (`store["variants"]`), so local variants never end up in the catalogue cache.

## Known gaps

- The UI can't register a custom local GGUF (by design: no paths from the browser). That is CLI-only (`local --add`).
- `/api/local-models/scan` has no `extra_dirs` (no paths from the browser). The CLI has `--dir`.
- Quant-check runs with `config=None`, so the perplexity tool picks its own GPU settings.
- The server doesn't auto-download before test, tune or serve. The job fails with "Download it first".
- CLI progress is polled every 0.2 s, so very fast jobs show only their final line.

## Requests to other owners

- **lead:** add `local_variants` to §2.4 (see Deviations 5).
- **ui-run / ui-quality:**
  - Every POST that starts work returns a job dict with 202; errors are always `{"error"}`.
  - `candidate_id` needs a fresh `/api/recommend` after a refresh or map change (409 "Compare again first").
  - The download result has `filename` (no path). Local model locations have `label`, not `path`.
  - Optional: send `tuned: true` to test, export or serve with the saved tune.
  - The static allowlist also serves `quality.css` and `run.js`.
- **engine:** `evaluate` passes `community=<store community.records>` and `tuned=<store tuned>` only when `recommend` accepts those keyword names (checked with `inspect.signature`). `candidate_for` reads the candidate keys `kv_cache_type` (defaults to the requested value when missing), `n_cpu_moe`, `gpu_layers`, `context`, `scenario` and `mode`.
- **testing:** see assumption 5 below about the `commands` argument.

### Assumptions about other modules' signatures (please verify at merge)

1. **`runtime_install`:**
   - `detect(store)` returns a JSON-safe dict.
   - `binary(store, name)` raises `ValueError` when llama.cpp is missing.
   - `install(store, hardware, progress=, cancel=, allow_unverified=)`, `install_archive(store, path, progress=, cancel=)`, `use_directory(store, directory)`.
2. **`downloads`:**
   - `plan(variant, directory)` has `files[].present` meaning "complete", plus `total_bytes`, `remaining_bytes`, `disk_free` and `enough_space`.
   - `download_variant(variant, directory, progress=, cancel=)` returns the first file's path. Files are saved as `directory / Path(filename).name`. I rely on that name only as a fallback, when discover hasn't indexed the file.
   - `remove_variant(variant, directory) -> int`.
   - Download progress may carry `bytes_per_second` and `eta_seconds`; the CLI shows them.
3. **`discover`:**
   - `find_for_variant(store, variant, verify=bool)`. The API uses `verify=False` for the plan/export checks, which must be fast. Jobs use `verify=True`.
   - `locations()`, `scan(store, extra_dirs=tuple, progress=, cancel=)` and `hash_cached(store, path, progress=, cancel=)`. Records have `path`.
4. **`gguf`:** `variant_from_file(path, sha256=...)` returns a `Variant`.
5. **`testing`:**
   - `run_tests(store, variant, config, commands, kind=, hardware=, progress=, cancel=)`.
   - **`commands` is a dict with both naming styles:** `{"server", "bench", "llama-server", "llama-bench"}`, each an argv prefix. `bench` is `None` for `kind="smoke"`.
   - The result has `verdict` and `verdict_text`, plus `smoke.ok`/`message` and `speed.summary`/`memory.note` for CLI output.
6. **`tuner`:** `tune(bench_command, variant, base_config, hardware, budget_seconds=, goal=, progress=, cancel=)`, with `best` as a launch config.
7. **`llama_server`:**
   - `LlamaServer(command, config)` with `.start(progress=, cancel=)`, `.stop()`, `.chat(messages, max_tokens=…)`, `.tokenize(text)`, `.ready()`, `.failure_reason()`.
   - `.base_url` has no `/v1`; the CLI appends it.
   - `ServerRegistry()` with `.start(command, config, progress=, cancel=)`, `.stop()`, `.status()`. `status()["running"]` is a bool.
8. **`evals`:**
   - `run_quiz(chat, workload, progress=, cancel=)` and `needle_test(chat, tokenize, context_tokens, progress=, cancel=)`.
   - `run_comparison(store, prompts, runners, max_tokens=, progress=, cancel=)`. **Each runner is a zero-argument factory.** It returns a session object that is callable as `chat(messages, max_tokens)`, can be used as `with factory() as chat:`, and has `.close()`/`.stop()`. The model loads lazily on the first call. Sessions still open when `run_comparison` returns are closed by the api, so either usage style is safe.
   - `blinded`, `vote(store, id, item, slot)`, `reveal`.
   - Comparison ids must match `[A-Za-z0-9_-]{1,64}`.
9. **`quantcheck`:**
   - `kl_check(command, reference_path, {label: path}, progress=, cancel=)`.
   - Labels are the quant names (a variant id if two share a quant). The API maps them back to variant ids.
10. **`export`:** `formats()` and `export(config, variant, fmt, platform=, server_command=)`. `server_command` is `None` when llama.cpp isn't installed.
11. **`community`:**
    - `import_records(store, source=None, progress=, cancel=)` saves to the store itself.
    - `share_payload(store, ids)`.
    - Measurement ids are lowercase hex (4–64 characters).
12. **`catalogue`:** `add_entry(store, base_repo, gguf_repo)` and `remove_entry(store, base_repo)`.

## How I tested

- `python3 -m unittest discover -s tests`: 116 tests, all passing (72 existing plus 44 new).
- **`tests/test_api_v04.py`** injects fake modules through `sys.modules` and the package attribute. It covers:
  - auth on every new route, and validation errors (35 bad bodies, none of which start work)
  - demo-mode 409s, unknown ids, and stale candidates (including the refresh race)
  - the full job lifecycle: download, cancel, duplicate requests, test, tune with a later `tuned` reuse, quiz, blind compare (at most one model loaded), quant-check, serve start/stop/shutdown
  - the job cap, NaN results, local models, community, runtime, and a missing-module 409
  - `scrub` unit tests, and "no absolute path in any job poll"
- **`tests/test_cli.py`** covers:
  - `--help` for every command, and the CI commands
  - progress output for a TTY and for pipes
  - Ctrl+C cancelling a download (exit 130, no finished file)
  - every new command with fake modules, and the exit codes
- A security-review subagent read the endpoints. Its findings were fixed and covered by tests: re-checking the running server when a job starts, path-scrub gaps with spaces, UNC paths and the home folder, a job-list crash on NaN, the stale report race, and busy checks on remove.

## llama.cpp facts assumed

- None directly. This workstream never builds llama.cpp arguments: it uses `launch.from_candidate` / `server_args` and the other modules.
- The CLI prints `<base_url>/v1` as the OpenAI-compatible address. This assumes llama-server serves the OpenAI API under `/v1`.
