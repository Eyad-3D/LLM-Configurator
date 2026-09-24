# Round 3: `r3-server`

Files: `src/llm_configurator/server.py`, `tests/test_api_v04.py` (`tests/test_server.py` unchanged). Branch `claude/v04-r3-server`.

## Items

| Item | Status |
|---|---|
| `POST /api/quality/vote` accepts `slot: "tie"` | Already done in the fix-up round (`choice(..., ("A","B","C","tie"))`), tested in `test_reveal_names_candidates_and_waits_for_answers`. Checked again on the real app. |
| `/favicon.ico` → 204 | **Fixed.** Answered before the token check (browsers send none), after the loopback Host check (a foreign Host still gets 403). No body, `Cache-Control: max-age=86400`. |
| `GET /api/export` → `{"formats": export.formats()}` | Already done; tested in `test_scan_job_kind_and_export_formats`. |
| `hardware` to `quant_check_job` | **Fixed.** Passes the latest comparison's hardware (`None` when there is none, so the app scans once itself, as before). |
| `min_tps` to `test_job` | Already done (signature check, from the latest report's `requirements.min_tps`); tested. |
| `/api/local-models`: `discover.locations(store)` + `local_variant_id` | `locations(store)` was already used. **Fixed:** a server-side `page_file` wrapper adds `local_variant_id` and `match_note` from the record when `app.page_file` does not (r3-app-cli adds them there; then the wrapper does nothing). Only string/None values are copied; `scrub` still runs over the reply. The scan job result uses the same wrapper. |
| `refresh(cancel=)`, progress `stage/done/total/message` | **Fixed.** `refresh` gets a `threading.Event` (through `app._call`, so an older `refresh` without `cancel` still works). New `POST /api/refresh/cancel` (empty body, same auth as every POST) sets it; shutdown (`server_close`) sets it too. A cancelled refresh ends with `result = {"warnings": [...], "cancelled": true}`. Progress is stored as given (it already has the four keys) and scrubbed when sent. The catalogue "add model" job passes its job `cancel` to `refresh_entry`. |
| Scan job kind `local_scan` | Already done; tested. |
| `community/share` passes `skipped` | Already passed through (the reply is `share_payload`'s dict). Now tested with `skipped`, `fits_in_url`, `records`. |
| Serve status never carries `model_path` | **Fixed.** `config` drops every `*_path` key (model and draft). The top-level `model` (file name or alias) still names what runs. The UI never read `config.model_path` (grep of `static/`). This removes a nested key; the ROUND3 item asks for exactly that, so it overrides the "only add keys" rule. |
| Test `Fakes` match the real signatures | **Fixed.** Every fake function and fake `LlamaServer`/`ServerRegistry` method now has the real parameter list. Added the missing `discover.local_variants/add_file/remember_hash`, `gguf.shard_paths`, `quantcheck.quant_label`, and the `LlamaServer` methods (`complete`, `request`, `running`, `warnings`, `log_tail`). New test `test_fakes_accept_every_argument_of_the_real_modules` compares the parameter names with the real modules, so the fakes can't drift again. |
| `runtime_backend` wherever launch configs are built | **Fixed for the server's own calls** (the saved-tune probe and serve start). The server passes `runtime_backend=` to `app.launch_config`. Today it goes through `**overrides` to `launch.from_candidate`. If r3-app-cli's `launch_config` finds the backend itself and forwards it, the duplicate keyword raises a `TypeError` naming `runtime_backend`, and the server calls again without it (tested). Serve start uses a fresh `runtime_install.detect`; the tune probe uses the cached `store["runtime"]`. Test, tune, quiz, compare and export configs are built inside `app` (r3-app-cli's item). |

## Also fixed

- `serve()` printed its address before it installed its SIGTERM/SIGHUP handlers. A stop right after start-up killed the process uncleanly (exit −15). This could leave a llama-server running with nobody owning it. `ServeProcessTests` failed three runs out of three on the base commit here. The handlers are now installed first.

## Independent review

A separate review agent read the diff for security and seams and found no security problem. It confirmed:
- `favicon` sits behind the Host check.
- `refresh/cancel` sits behind the full POST checks.
- `match_note` holds only fixed strings, and every response is still scrubbed.
- The `runtime_backend` fallback is correct for every `app.launch_config` variant.
- The signal reorder is safe.

Acted on:
- **Cancel race:** each refresh now gets its own cancel Event, so a late cancel of an old refresh can't stop the next one.
- **No `detect()` on every request:** the saved-tune probe now uses only the cached backend. Before, it ran a full `detect()` on each test, export or serve request when nothing was installed.
- **Stricter fake check:** the signature test now also compares required arguments.
- **Fake parameter name:** one fake used a different parameter name (`ids`), now fixed.

Left as is: test, tune, quiz, compare and export configs inside `app` still need the backend (r3-app-cli's item).

## Rejected

- None.

## Requests

- **r3-app-cli:** `app.page_file` should include `local_variant_id` and `match_note` itself (already on your list). The server fills them in meanwhile. If you give `launch_config` its own `runtime_backend` parameter, an explicit value from the caller should win over the detected one. The server passes the value it detected.
- **r3-ui (optional):** `POST /api/refresh/cancel` exists now if you want a Stop button for "Refresh model list"; `GET /api/refresh` then shows `result.cancelled: true`. Serve status `config` no longer has `model_path`; use the top-level `model`.
- **r3-misc (`catalogue.py`):** in `refresh`, add `check_cancel(cancel)` after the `while pending` loop. If a worker raises `Cancelled` in the same `wait()` round as the last futures, the loop ends without a check. The refresh then "finishes" with the warning "unexpected reply from Hugging Face (Cancelled)" instead of stopping. Found by the review; not reproduced on the real app, where the cancel was clean.
- **lead:** `/api/runtime` still returns `detect()` as it comes. `scrub` turns `directory` and `binaries` into bare names (`"bin"`, `"llama-server"`), checked on the real app. No path reaches the page.

## Evidence

- `python3 -m pip install -e .` (plus `pip install --ignore-installed cryptography` for the keyring panic), then `python3 -m unittest discover -s tests`: **819 tests OK, 26 skipped**.
- `npm ci && npm test`: **71 pass, 0 fail**.
- Real llama.cpp: `JOBS=4 bash scripts/build_llama_cpp.sh /opt/llama-work` (bin `/opt/llama-work/llama_cpp_python-0.3.35/vendor/llama.cpp/build/bin`), `scripts/make_tiny_models.py`. `LLM_CONFIG_REAL_RUNTIME=<bin> LLM_CONFIG_TINY_MODELS=/opt/llama-work/models python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real tests.test_api_v04.RealRuntimeHttpTests`: **44 tests OK** (43 + the HTTP end-to-end test).
- Real app, both folders with spaces (`…/My Data`, `…/My Models`), `runtime use <bin>`, `local --add` of tiny Q4_K_M and Q8_0, `python3 -m llm_configurator --data-dir "…/My Data" serve --no-browser --port 18765`. It was driven over urllib, and every response was checked for the temp folder, `/opt/llama-work`, `My Models`, `My Data` and `/tmp/`:
  ```
  favicon = (204, b'')            favicon_bad_host = 403
  export_formats = ['llama-server', 'ollama', 'docker-compose', 'openai-python', 'continue', 'open-webui', 'lmstudio']
  local_models = [('tiny-llama-Q4_K_M.gguf', 'local:5713aa7612896f87:tiny-llama-Q4_K_M.gguf', None), ('tiny-llama-Q8_0.gguf', 'local:cfe434857b9de824:…', None)]
  scan = ('local_scan', 'done', 2)
  runtime = {'installed': True, 'backend': 'cpu', 'directory': 'bin'}
  test (full) = ('done', None, 'works')
  serve_status = running, model 'tiny-llama-Q4_K_M.gguf', config keys without any *_path, candidate_id matches
  quant_check = ('done', None, {'Q4_K_M': 'local:5713aa7612896f87:tiny-llama-Q4_K_M.gguf'})
  compare = done; vote "tie" = 200; reveal = 200 with candidate ids + labels
  community share = 200 {'records': 1, 'skipped': 0, 'fits_in_url': True}
  refresh 202 → cancel 200 → running False, result.cancelled True, progress {stage, done, total, message, …}
  ```
  SIGTERM to the app left no llama-server running.
