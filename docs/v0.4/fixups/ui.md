# Fix-up notes: `ui`

Branch: `claude/v04-fix-ui`. Files: `src/llm_configurator/static/*`, `tests/ui-run.test.cjs`, `tests/ui-quality.test.cjs`, this file.

Three read-only audits compared every UI field with `server.py` and the module that produces it. Each audit covered one group of files: `run.js`/`jobs.js`, `quality.js`, and `app.js`/`wizard.js`/`selects.js`. I then clicked through the real app in Chromium with Playwright, in two runs: `serve --demo`, and a normal run with an empty data folder.

`npm test` now runs 71 tests, up from 56, and all of them pass.

## Fixed

### Job progress (`jobs.js`)

- **Tuning** progress counts seconds used of the time budget. It showed as "step 75 of 300"; it now reads "1 min of 5 min used".
- **Model loading** (serve) counts seconds against a 300 s time limit. It now shows "12 s so far" with no progress bar, because the limit is not how long loading will take.
- **Engine install** stages send `done: 0, total: null`, and the tray showed "Unpacking llama.cpp · 0 KiB". Bytes now appear only for the `download` and `verifying` stages, or when a speed is sent.
- Progress messages go through the path filter.

### Test step (`run.js`)

- Checks showed machine ids such as `not_empty`. They now show the server's sentence (`detail`), with plain fallback labels.

### Tune step (`run.js`)

- **Changed settings.** Changes were compared only against keys that `candidate.launch` has, and `launch` has no `batch`/`ubatch`, so batch changes were hidden. A missing value now counts as "default".
- **"Tried N settings"** no longer counts the baseline run or the final double-check run.
- **Notes.** The stop reason showed twice, because the tuner's notes already say why it stopped. "Already the fastest" also repeated the headline. Both are fixed.

### Use step (`run.js`)

- When a server stops by itself, its `error` is shown, with paths removed.
- A server that is still starting (`starting`) now says so.

### Sharing a result (new, `run.js`)

This follows the community handoff request. After a full test, a "Share this result (optional)" box appears.

- It calls `POST /api/community/share` with the test's `speed.measurement.id`.
- It shows the exact JSON first, with a Copy button.
- Only then does it show "Open GitHub to post it".
- When `fits_in_url` is false, it tells the person to copy first and paste the text into the page.
- The link is shown only when it is `https://github.com/…`, and it opens with `rel="noopener noreferrer"`.

### Demo mode (`run.js`)

- The long demo sentence now appears once at the top of the panel. Each step says "Not available in demo mode (see the note at the top)."
- The engine step and the Use step show the Demo state, so there is no Install button that can only fail.

### Hardware box (`app.js`)

These follow the hardware handoff request.

- The processor shows `cpu_name` (falling back to `cpu`) instead of "x86_64".
- An Apple GPU (`unified`) is labelled "Shared with system RAM, not extra memory".
- When `gpus` is empty but `other_gpus` is not, it says "<name> found — free memory can't be read, so estimates use RAM". The old text was "No supported NVIDIA telemetry".
- `hardware.warnings` are shown.
- The GPU box follows the GPU picked in Advanced settings.

### Cards and details (`app.js`)

These follow the engine handoff request.

- MoE candidates (`mode == "split"` and `n_cpu_moe > 0`) say "Experts on CPU".
- Details show `memory_total_bytes` as "Estimated memory in total". On unified machines, VRAM is labelled as part of RAM.
- The conversation memory row and the settings JSON use the real `kv_cache_type` instead of a hard-coded f16. The settings JSON also includes `n_cpu_moe`.
- Speeds with `interpolated` or `community` evidence show that number ("~18.3 tok/s, estimated from your tests at other lengths" or "Reported by 3 similar computers"). The badge says "Not tested yet · estimated from nearby tests".
- The old "Speed unverified" tag is hidden when the engine sends a verdict, because it repeated the badge.
- `verdict_text` is no longer repeated at the end of the explanation.
- Details dates are readable.
- The source link is shown only for `https://` links.

### Compressed conversation memory (new control, `index.html`, `app.js`, `wizard.js`)

- `kv_cache_type` was never sent, so the engine's hint "turn on compressed notes" pointed nowhere.
- There is now a control, "Fit longer conversations in less memory", with Off, Half size and Quarter size. Its help text uses a zip-folder comparison.
- The choice is sent with `/api/recommend` and shown in the answer summary.

### Community import (new, `app.js`, `index.html`)

- The results page has a "Speed results shared by other people" section.
- It shows the count and date from `GET /api/community`.
- "Get the latest shared results" runs `POST /api/community/import` as a job, then says "Press Recalculate to use them."

### Model files on this computer (`app.js`, `index.html`)

`LocalModels` was built but never mounted. It now opens from a section on the results page.

### Offline warnings (`app.js`, `wizard.js`)

- With no internet, the page listed about 40 lines like "…: Metadata unavailable (URLError)…".
- They now become one sentence: "Couldn't get the latest details for 39 models (is the internet connected?). Using what's already saved on this computer." At most three other notes are shown, then "and N more notes".

### Quality panel (`quality.js`)

- **Local files.** The server sends `filename` and folder `label`, never `path`. The list showed "Unnamed file" and "LM Studio: undefined"; both are fixed. `match_note` is shown when present.
- **Compression check, file names.** Result keys are quant labels such as "Q4_K_M", not variant ids. They are now mapped through `variant_ids`, so each heading shows the full file label.
- **Compression check, other fixes:**
  - `verdict` is shown as a plain badge: No real difference, Small, Noticeable, Large or Very large difference.
  - A per-file `error` is shown.
  - `same_top_p` is always treated as a percentage, as llama.cpp prints it. Before, 0.8 % would have been shown as 80 %.
- **Blind compare, "No preference".** It now sends `slot: "tie"`, which `evals.vote` supports. If the server refuses it with a 400, which is today's behaviour, it stays a no-preference on the page only, with no error shown.
- **Blind compare, reveal.** The reveal accepts the pinned `labels: {candidate_id: name}` for friendly names, and the older shapes still work. Answers with `ready: false` show "Still writing this answer…".
- **Quiz history.** When a model is no longer in the list, its row uses the name saved in the record instead of its raw id.
- **Network errors.** Browser network errors (`TypeError` with no status) get the plain "Couldn't reach the app" message.

### Security

- **HTML strings.** Every `innerHTML` template in `app.js` and `wizard.js` was re-read. Numbers that came straight from the server are now escaped too: process ids, ranks and rejection counts.
- **Dynamic content.** `run.js` and `quality.js` build all dynamic content with `textContent`.
- **CSP.** There are no inline scripts or handlers, and no `style="…"` in markup. The real CSP header (`script-src 'self'; style-src 'self'`) caused no console violations in Chromium.
- **XSS tests.** Hostile model names, quants, verdict text, explanations, dates, prompts, model answers, error text and share JSON all render as text. A `javascript:` score source and a non-GitHub share link are not linked.
- **Blind compare.** A new test checks `outerHTML`, including attributes, for candidate ids, variant ids and friendly names before reveal, including after a "tie" vote.

### Other

- **`api()` in `app.js`** no longer crashes on a response that isn't JSON. It keeps the status so callers can explain it.
- **Dark theme.** The quality tabs were painted as bright green buttons by the generic dark `button` rule. That rule now skips them.

## Field map (UI ↔ server)

| UI reads | Server field (producer) | Unit / note |
|---|---|---|
| hardware `cpu_name` ∥ `cpu`, `gpus[].{index,name,available,total,unified}`, `other_gpus[].name`, `warnings[]` | hardware.py | bytes |
| candidate `verdict` (runs_well/runs_slowly/too_slow/unknown), `verdict_text`, `evidence`, `mode`, `n_cpu_moe`, `kv_cache_type`, `memory_total_bytes`, `unified_memory`, `speed_interpolated.tps`, `community.{median_tps,n}` | engine.py | bytes, tokens/s |
| job `state`, `progress.{stage,done,total,message,bytes_per_second,eta_seconds}` | jobs.py and each job | `tune`/`loading`: seconds; `download`/`verifying`: bytes; else steps |
| test `verdict`, `verdict_text`, `smoke.checks[].{name,ok,detail}`, `speed.summary.{pp_tps,tps,ttft_s,depth}`, `speed.memory.*`, `speed.measurement.id` | testing.py | s, tokens/s, bytes |
| tune `best`, `baseline`/`best_result.{tps,pp_tps}`, `improvement` (ratio), `trials[].{status,step}`, `stopped`, `notes[]` | tuner.py | ratio, not % |
| export `filename, content, instructions[], notes[]` | export.py | |
| serve `running, starting, error, openai_base_url, model, candidate_id?, variant_id?` | llama_server.py, server.py | |
| share `json` (string), `issue_url`, `fits_in_url`, `records` (count) | community.share_payload | |
| community `records[]`, `fetched_at`; import result `count, rejected` | server.py, community.py | |
| quiz `score, ci_low, ci_high` | evals.py | 0–1 fractions |
| blinded `items[].outputs[].{slot,text,ready}` | evals.blinded | |
| reveal `mapping[i].{A,B,C}` (or `.slots`), `tallies`, `labels?` | evals.reveal, pinned | |
| quant check `results{quant: {same_top_p, mean_kld, median_kld, kld_99, ppl_base, ppl, mean_delta_p, plain, verdict, error}}`, `variant_ids{quant: id}`, `reference` (quant), `notes[]` | quantcheck.py, app.py | `same_top_p` in %; Δp in percentage points |
| local models `files[].{filename, size_bytes, source, variant_id, gguf.{name,quant,layers}, verified, match_note?}`, `locations[].{source,label,exists}` | app.page_file / page_location | no paths |

## Rejected, with reasons

- **The full-test progress bar restarts between the quick check and the speed test.** The test sends 0–2 of 2, then 0–3 of 3. Weighting stages in the UI would guess at durations. The text shows the stage, which is honest.
- **Speed-test measurement IDs for sharing from the card.** The card's `benchmark.id` exists only for measured candidates, and sharing belongs next to the test that produced the number. So sharing is offered only in the Test step.
- **`GET /api/export` for format labels.** This is not a UI fix. The UI still tries it and falls back to its built-in list.
- **Weighting the quiz `pct()` guess.** Evals always sends 0–1, so it stays as it is.

## Requests

- **api-http:**
  1. Accept `slot: "tie"` in `POST /api/quality/vote` (`server.py` allows only A/B/C). `evals.vote` supports it, and the UI already sends it.
  2. Serve `/favicon.ico`, for example as 204 or a small SVG. Today it returns 403 "Reload the application…", and that shows as a console error on every page load.
  3. Optional: add `GET /api/export` → `{formats: export.formats()}` so the tabs get the real labels and descriptions.
  4. Optional: keep `match_note` in `app.page_file`.
  5. The job kind for disk scans is still `"scan"`; FIXUPS pins `local_scan`. The UI doesn't depend on either.
- **engine:** the verdict is `unknown` for interpolated and community evidence. The UI now adds the evidence qualifier to the badge. If the engine later sends real verdicts for these, nothing needs to change in the UI.

## Screenshots (Playwright, Chromium; not committed)

I took screenshots at 360 px and 1280 px, in light and dark, against `serve --demo`. They covered:

- the welcome screen
- the results page
- the results page with the new sections open
- the details dialog
- the run panel
- each of the six steps, after pressing its first button
- the three quality tabs

What I checked:

- There is no horizontal overflow at 360 px.
- There are no CSP violations and no page errors.
- The only HTTP errors are expected: demo 409s, the optional `GET /api/export` 404, and the favicon 403 (Request 2).

What I fixed after looking:

- the repeated demo paragraph in every step
- the "Ready" label and Install button on demo steps
- the bright green dark-mode quality tabs
- the raw ISO date
- "· 1" at the end of the answer summary, which now reads "1 at once"
- the "Speed unverified" tag and verdict sentence repeated on cards

The non-demo run with an empty data folder had no internet access. It reached the empty results state. The wall of about 40 "Metadata unavailable (URLError)" lines is now one sentence.

## How I tested

- `npm ci && npm test`: 71 tests pass. 15 of them are new, and 2 old assertions were corrected: `same_top_p` units, and check ids that were treated as labels.
- `python3 -m pip install -e . && python3 -m unittest discover -s tests`: 604 tests ran. The 2 failures are the known fake-llama cases in `test_fake_matches_real` (owned by server-fake), which were already failing before my changes. The earlier `test_adapters` error was the keyring panic, and it went away after `pip install --ignore-installed cryptography`.
