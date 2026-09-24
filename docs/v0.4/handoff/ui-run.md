# Handoff: `ui-run`

Branch: `claude/v04-ui-run`

## What I built

- **"Get it running" button** on every recommendation card (`app.js`). It opens a step panel (`run.js`, a `<dialog>` in `index.html`) with six steps:
  1. **Get the engine**: shows version and backend (never the folder). An *Install the engine* button runs a job.
  2. **Download the model**: shows size and free disk space. Offers *Download*, *Pause* (keeps the partial file), *Cancel download* (cancels, then deletes the partial files), *Resume*, *Delete partial download*, *Use the copy on my disk* (when `local_copy`) and *Remove from disk*. Blocks with a plain reason when there isn't enough disk space.
  3. **Test it**: *Run the full test* (`kind: "full"`) or *Quick check only* (`kind: "smoke"`). Shows the verdict, the list of checks, load time, reading speed, first-word delay, writing speed, and memory used vs our estimate.
  4. **Tune it**: time budget of 1, 5 or 15 min (radio buttons), and a goal (faster writing / balanced / faster reading). Shows before → after for writing and reading speed, which settings changed (in plain names), how many settings were tried, why it stopped, and any notes.
  5. **Check quality**: calls `QualityPanel.mount(container, ctx)` when `window.QualityPanel` exists. Otherwise it shows a placeholder.
  6. **Use it**: export tabs (one per format) with a copy button that falls back when the clipboard is unavailable, plus a Mac/Linux ↔ Windows toggle (Windows is detected automatically). Also Start/Stop server, which shows the OpenAI address with a copy button.
- Each step header shows its state: `Checking… / Ready / Not installed / Not downloaded / Found on your disk / Not yet (+reason) / Working… / Paused / Done / Needs attention / Demo / Server running`.
- **Jobs tray** (`jobs.js`, `#jobs-tray` in `index.html`): a small fixed box with every running job. Each row has a progress bar, bytes/speed/time left for downloads or the stage for tests and tunes, and a Cancel button. Finished jobs leave the tray after about 6 s. On screens ≤ 600 px it starts folded.
- **Verdict badges** on cards: `Runs well` / `Runs slowly` / `Too slow` / `Not tested yet`. A badge gets a qualifier (`· estimate`, `· estimated from nearby tests`, `· based on other people’s results`) unless the evidence is `measured` or `tuned`. `verdict_text` is shown under the title. For older reports without `verdict`, the badge falls back to measured speed only (`tps` + `speed_meets_target`); otherwise it shows `Not tested yet`.
- **Dark theme** via `prefers-color-scheme: dark` (the app had no dark theme before). Added a reduced-motion rule, and fixed the stretched *Recalculate* button in the results nav.
- `index.html` now loads `quality.css`, `jobs.js`, `quality.js` and `run.js`. The footer says v0.4.
- After the panel closes, if a download, test or tune finished, it calls `refreshVisibleSpeeds()` (from `wizard.js`) so the cards pick up new verdicts.

## Public API as built

```js
window.Jobs = {
  watch(jobId, onUpdate) -> stop()   // polls GET /api/jobs/<id> every 700 ms (4 s when the tab is hidden)
                                     // until done/failed/cancelled. 404 -> onUpdate({state:"failed", error:"…no longer tracked…"}).
                                     // Network errors back off exponentially; 6 in a row -> failed ("Lost contact with the app").
  list() -> Promise<job[]>           // GET /api/jobs, refreshes the tray
  cancel(jobId) -> Promise<job>      // POST /api/jobs/<id>/cancel {}
  track(job) -> job                  // add a job (e.g. a 202 response) to the tray right away
  describe(job) -> {fraction: 0..1|null, text}   // plain progress text
  format: { bytes(n), duration(seconds) },
  config: { interval: 700, hiddenInterval: 4000, listInterval: 2000 },
}
window.RunPanel = { open(candidate, {candidates, workload, demo}), close() }
// app.js globals (classic script): verdictBadge(candidate) -> escaped HTML string, verdictOf(candidate) -> verdict id, openRunPanel(candidate)
```

`QualityPanel.mount(container, ctx)` gets `ctx = { api, candidate, candidates, workload, onJob }`:
- `api(path, options)` accepts both `api(path, body)` (the app.js convention: body means POST, no body means GET) and fetch-style `api(path, {method, body})`, where `body` may be an object or a JSON string. It returns parsed JSON. It throws `Error` with `.status` and the server's `error` text.
- `onJob(job)` calls `Jobs.track(job)`, so the job shows in the tray. ui-quality can then `Jobs.watch(job.id, …)` itself.
- It is mounted once per panel open, the first time the step is expanded.

### Endpoints and fields relied on (§5)

| Call | Fields read |
|---|---|
| `GET /api/runtime` | `installed`, `source` (managed/configured/path), `version`, `backend`, `warnings[]`. `directory` is **never** shown |
| `POST /api/runtime/install {}` | 202 job; `result` = detect() (`installed`) |
| `GET /api/jobs` | `{jobs: [...]}`: `id, kind, title, state, progress, subject` |
| `GET /api/jobs/<id>` | job: `state, progress{stage, done, total, message, bytes_per_second, eta_seconds}, result, error` |
| `POST /api/jobs/<id>/cancel {}` | job |
| `POST /api/downloads/plan {variant_id}` | `total_bytes, remaining_bytes, disk_free, enough_space, local_copy, files.length` |
| `POST /api/downloads {variant_id}` | 202 job; `result.reused`, `result.bytes` |
| `POST /api/downloads/remove {variant_id}` | `freed_bytes` |
| `POST /api/test {candidate_id, kind}` | 202 job; `result.verdict (works/works_slowly/failed)`, `verdict_text`, `smoke{ok, load_seconds, checks[{name, ok, detail}], message}`, `speed{summary{pp_tps, tps, ttft_s, depth}, memory{estimated_ram_bytes, estimated_vram_bytes, peak_ram_bytes, peak_vram_bytes, within_estimate, note}}` |
| `POST /api/tune {candidate_id, budget_seconds: 60/300/900, goal}` | 202 job; `result.best, baseline{tps, pp_tps}, best_result{tps, pp_tps}, improvement, trials[].status, stopped, notes[]` |
| `POST /api/export {candidate_id, format, platform}` | `filename, content, instructions[], notes[]` |
| `GET /api/export` *(optional, see Requests)* | `{formats: [{id, label, description}]}`; falls back to a built-in list on any error |
| `GET /api/serve` | `running, base_url, openai_base_url, model`, and optionally `candidate_id`/`variant_id` |
| `POST /api/serve/start {candidate_id}` | 202 job; `result` = status |
| `POST /api/serve/stop {}` | status |
| Candidate (§4.12) | `id, variant_id, name, quant, context, mode, verdict, verdict_text, evidence, kv_cache_type, n_cpu_moe, launch` (to diff tuned settings), `demo` |

Every error with status 409 in demo mode shows: "Not available in demo mode. The demo models are made up…". Other 409s (for example "Compare again first") show the server's message.

## Deviations and decisions

- **Export formats list**: §5 has no formats endpoint. The UI tries `GET /api/export` for `{formats}` and otherwise uses a built-in list with the §4.9 ids: `llama-server, ollama, lmstudio, open-webui, continue, openai-python, docker-compose`.
- **Pause vs cancel**: the jobs API only has cancel. *Pause* = cancel the job and keep the `.part` files; resuming starts the download job again. *Cancel download* = cancel, then `POST /api/downloads/remove`.
- **Export content may contain the model's path.** That's the point of a start script. It's the only place a path can appear, and it comes straight from `export.py`. Nothing else shows a path. The engine step shows version and backend, not `directory`.
- **Test/tune/serve/quality** show "Not yet: get the engine first / download the model first" until steps 1–2 are done. In demo mode they show the demo message instead.
- **Reattaching after a reload**: on open, running jobs from `GET /api/jobs` are reattached. Matching uses `kind` containing `runtime` / `download` / `serve`, and for downloads `subject.variant_id` or `subject.candidate_id`. Test and tune results are kept in page memory only, per candidate, while the page is open.
- **Server "running here" check**: compares `status.candidate_id / variant_id / model / config.alias` with the candidate's `id / variant_id / name / filename`.
- `package.json` test script is now `node --test "tests/ui-*.test.cjs"`, so `ui-quality.test.cjs` runs automatically after merge. Node 22 expands the glob itself.
- **ui-flow.test.cjs is unchanged.** Card markup changed: the details link now reads "View details ↗", and the button and badges were added. No existing assertion depended on those strings.

- **No paths in messages.** Job errors, 409 texts, runtime warnings and the running model name go through a `plain()` filter. It replaces anything that looks like a file path (`/…/…` or `C:\…`) with "a file in the app’s folder". Network failures show "The app didn’t respond. Is it still running?" instead of browser errors.
- **Repeat clicks:** each start action (install, download, test, tune, serve) ignores clicks while its first request is still on its way.
- **Screen readers:** the tray updates rows in place, so focus and clicks survive polling. It announces only "X started / finished / failed / stopped" through a hidden live region. The panel's progress text is not a live region; completions are announced once.

## Known gaps

- No way to see a finished test or tune result again after a page reload, except through the card verdict after *Compare again* (depends on engine using stored measurements and tunes).
- The tray is inert while the step panel is open (it's a modal dialog). The panel shows its own progress and Cancel buttons.
- The dark theme covers the main surfaces. A few rarely seen old elements (for example `.mapping` rows) still use light-mode colours.
- Screenshots were taken with a mocked API (Playwright request interception), not against a real `serve`, because `server.py` doesn't yet serve `run.js`/`jobs.js`.

## Requests

- **api**:
  1. Add `run.js`, `jobs.js`, `quality.js`, `quality.css` to the static allowlist in `server.py` (as JS/CSS MIME types), keeping the CSP.
  2. Please give jobs these `kind`s: `runtime_install`, `download`, `test`, `tune`, `serve`. Set `subject: {candidate_id, variant_id}` on download, test, tune and serve jobs. Give them human `title`s (e.g. "Download Qwen3 8B Q4_K_M"), because the tray shows them.
  3. Optional: `GET /api/export` → `{"formats": export.formats()}`.
  4. Have `GET /api/serve` status include `candidate_id` (or `variant_id`), so the panel can tell whether *this* model is the one running.
  5. `/api/export` and `/api/serve/start` should use the best stored tune for the candidate when one exists. The UI says "The best settings are saved on this computer for this model."
  6. Please make job `error` text path-free on the server too (`jobs.py` passes `str(OSError)`, which often quotes paths). The UI filters paths, but the server is the right place.
  7. Download progress: send `done`/`total` in bytes plus `bytes_per_second`, `eta_seconds` (as in §4.2). Runtime install progress: bytes too.
- **engine**: keep `candidate.launch` keys aligned with the launch-config names. The tune step diffs `best` against `launch` for `threads, batch, ubatch, flash_attn, cache_type_k, cache_type_v, n_cpu_moe, gpu_layers, context, parallel`.
- **ui-quality**: see the `ctx.api` notes above. Please don't rely on the panel being modal-free: the tray is covered while the step panel is open, so show progress inside your own panel.

## How I tested

- `npm test` runs `ui-flow.test.cjs` (18 existing tests, unchanged, passing) and the new `ui-run.test.cjs` (22 tests). The new tests use jsdom with a fake API and fake job store. They cover: badges and fallback, opening the panel, no paths shown, runtime install job and tray, download progress (bytes/speed/ETA), cancel removes partial files, pause and resume, local-copy reuse, the disk-space block, full test rendering, failed quick check, tune before/after and request body, export tabs + arrow keys + copy + platform toggle, clipboard fallback, serve start/stop + URL copy, demo 409, other 409 messages, tray list and cancel, `Jobs.watch` polling and 404 handling, keyboard navigation and Escape with focus return, QualityPanel ctx and placeholder, path-free job errors, ignoring repeat clicks, a download picked up again after a reload, and plain network errors.
- `python3 -m unittest discover -s tests`: all pass. I made no Python changes.
- An accessibility and bug-review subagent read the code. I fixed what it found: jobs picked up after a reload stuck on Working…, double clicks, paths in errors, tray rebuilds that caused live-region noise and lost focus, unstable ids that lost focus, `--muted` contrast (now #5f6b64) and some jargon.
- Playwright (Chromium) screenshots at 360 px and 1280 px, light and dark, with the real CSP header and a mocked API. I checked the results cards, the test result, the tune result, the export/serve step and a running download with the tray. There were no CSP or script errors; the only 404s were `quality.js`/`quality.css` (not on this branch) and the optional `GET /api/export`. Things fixed after looking: long memory text, wide speed numbers, the tray covering content on phones, a stale "Ready" next to a running download, and the stretched Recalculate button.

## llama.cpp facts assumed

None directly. The UI only shows what the API returns. It does assume tokens/s values are per-sequence generation and prompt speeds, as §4.5/§4.6 describe.
