# Handoff: `ui-quality`

Branch: `claude/v04-ui-quality`. Files: `src/llm_configurator/static/quality.js`, `src/llm_configurator/static/quality.css`, `tests/ui-quality.test.cjs`, this file.

## What I built

- **`window.QualityPanel.mount(container, ctx)`**: a panel with three tabs.
  - **Quick quiz**: pick a model (from `ctx.candidates`, default `ctx.candidate`) and a question type (default `ctx.workload`). There's an optional long-document recall test. It runs a job and shows "7 of 10 right · 70%", the likely range (`ci_low`–`ci_high`) as text and as a bar, and each question in a `<details>`. It also shows "Scores so far" (latest quiz per variant for the chosen workload, from `/api/quality/results`). When the top two ranges overlap it says **"Too close to call"**. It only names a clear winner when the ranges don't overlap.
  - **Try my prompts** (blind compare): pick 2–3 models. Type 1–5 prompts (each ≤ 4000 characters, counted in code points like Python's `len`, with a live counter; there's also a `maxlength`). Then run the job. The view then switches to a blind voting screen: the prompt, answers "Answer A/B/C" in the order the server gives, and buttons "A is best", "B is best", "C is best" and "No preference", plus "Skip voting on the rest". **Reveal** stays disabled until every prompt has a vote or a skip. After reveal, each answer shows "Written by <model>", then vote tallies and a "treat a one-vote lead as a tie" hint.
  - **Compression check**: the reference defaults to the **largest file of the same model** as `ctx.candidate`. If `/api/local-models` knows about on-disk copies, the largest on-disk one wins. The candidates list shows only other files of the same model (same `repo`, else `name`), 1–3 of them, with the current candidate pre-ticked. A warning covers disk space and time. Each result shows a plain sentence (the server's `plain`, or built from `same_top_p`). A "Show the numbers" `<details>` lists same top word %, mean/median/99% KL, perplexity reference → file, and mean Δp, each with a plain explanation. It also shows the server's `notes`.
- **`window.LocalModels.mount(container, ctx)`**: lists `/api/local-models` files. Each row shows the name (GGUF name or file name), size, quant, layers and the friendly source folder. It also shows a "Known model" or "Not in the catalogue" chip and how the file was checked (fingerprint vs name+size). A "Where we looked" `<details>` lists each folder, and a **Scan my disk** button runs the scan job and reloads the list.
- **Jobs**: each long job gets a live status line (`role=status`, `aria-live=polite`), a `<progress>` (with a value when `progress.total` is known, indeterminate otherwise) and a **Stop** button (`POST /api/jobs/<id>/cancel`). Failed jobs show `job.error`. Cancelled jobs show "Stopped before it finished". API errors (400/404/409, including demo mode and "Compare again first") show the server's plain `error` text. Network failures get a "check the app is still running" message.
- **Safety and honesty**:
  - All text goes in via `textContent` (a small `h()` DOM builder). There's no `innerHTML`, no inline handlers and no inline styles in markup (CSP-safe). The range bars set `element.style.left/width` through the CSSOM, which CSP allows even without `'unsafe-inline'`.
  - The blind view keeps only slot letters and answer text. No model id, name, variant or quant appears in its DOM text or attributes until reveal. A test checks the whole panel's `outerHTML`. The `/api/quality/vote` response (tallies, possibly keyed by model) is deliberately ignored before reveal.
- **Accessibility**:
  - WAI-ARIA tabs: `tablist`/`tab`/`tabpanel`, roving `tabindex`, and ←/→ (wrapping), Home and End. Selection follows focus.
  - Every control has a `<label>` or `<legend>`. Vote buttons use `aria-pressed` inside a labelled `role=group`. Remove buttons use `aria-label="Remove prompt N"`.
  - The vote announcements use a visually hidden live region, which becomes visible only for errors. Counters are linked with `aria-describedby`.
- **Styling** (`quality.css`, everything under `.qp`):
  - Colours come from the page's `--green`, `--line` and `--muted`. Surfaces are `color-mix()` tints of `currentColor`, and the textarea uses the `Field`/`FieldText` system colours, so the panel follows whichever light or dark theme the page sets.
  - Brighter good/bad colours apply under `prefers-color-scheme: dark` (unless `:root[data-theme="light"]`) and under `:root[data-theme="dark"]`.
  - At ≤ 480 px, tabs share the width and wrap their labels, and the number list stacks. `prefers-reduced-motion` turns off transitions and animations.
  - Checked with Playwright screenshots at 360 px and 1280 px, light and a simulated dark host theme. There's no horizontal overflow at 360 px.

## Public API as built

```js
window.QualityPanel.mount(container, ctx) -> { unmount(), select(tabKey /* "quiz"|"compare"|"compress" */) }
window.LocalModels.mount(container, ctx)  -> { unmount(), refresh() }
```

`ctx` (all optional except what the tab needs):

| key | used for |
|---|---|
| `api(path, options?) -> Promise<json>` | GET when `options` is undefined. POST as `{ method: "POST", body: <plain object> }`: **the host JSON-encodes `body`** and adds `X-Session-Token`. It must reject with an `Error` whose `.message` is the server's `{"error"}` text (`.status` optional), like `app.js`'s `api`. If `ctx.api` is missing, a built-in fetch wrapper does exactly this using `<meta name="session-token">`. |
| `candidate` | default model for the quiz and compare, and it picks the model group for compression. Fields used: `id, name, quant, mode, variant_id, repo, file_bytes, context` |
| `candidates` | the list offered in all three tabs (same fields) |
| `workload` | default quiz question type (`general`, `coding`, `agentic`, `documents`) |
| `onJob(job)` | called when a job starts (first job dict) and again when it ends (final job dict), so the jobs tray can show it. Errors thrown by it are ignored. |
| `pollMs` | extra: polling interval for `/api/jobs/<id>` (default 700 ms; tests use 1) |

Mounting again on the same container unmounts the previous instance first. `unmount()` stops all polling timers and removes the DOM. Jobs keep running on the server.

### Endpoints and fields relied on

| Call | Body sent | Fields read |
|---|---|---|
| `GET /api/jobs/<id>` | – | `id, state (queued/running/done/failed/cancelled), progress.{message, stage, done, total}, result, error` |
| `POST /api/jobs/<id>/cancel` | `{}` | – |
| `POST /api/quality/quiz` | `{candidate_id, workload, include_needle?: true}` | 202 job. `result` is either the `run_quiz` dict or `{quiz: <run_quiz>, needle: <needle_test>}`. From quiz: `correct, total, score, ci_low, ci_high, items[{id, ok, expected, got}], note`. From needle: `results` or `positions` `[{position, ok or found}]`, `context_tokens?`, `note?` |
| `GET /api/quality/results` | – | `results[]` where `kind == "quiz"`: `variant_id` (or `candidate_id`), `workload`, `score, ci_low, ci_high`, `timestamp`. Score fields may also sit under `entry.result` |
| `POST /api/quality/compare` | `{candidate_ids: [2–3], prompts: [1–5 trimmed strings]}` | 202 job, `result.comparison_id` |
| `GET /api/quality/compare/<id>` | – | `items[{prompt, outputs[{slot, text}]}]` |
| `POST /api/quality/vote` | `{comparison_id, item: <0-based index>, slot: "A"}` | nothing (ignored until reveal) |
| `POST /api/quality/reveal` | `{comparison_id}` | `tallies` (or `votes`): `{label: count}`. Mapping, any of: `mapping: [{A: label, …}]` per item, `mapping: [{slots: {…}}]`, `items: [{outputs: [{slot, label}]}]`, or one shared `mapping: {A: label}`. A label is turned into a display name by matching candidate `id`, then `variant_id`, then `name`, else shown as-is |
| `POST /api/quality/quant-check` | `{reference_variant_id, variant_ids: [1–3]}` | 202 job. `result.reference`, `result.results{variant_id: {same_top_p, mean_kld, median_kld, kld_99, ppl_base, ppl, mean_delta_p, plain}}`, `result.notes[]` |
| `GET /api/local-models` | – | `files[{path, size_bytes, source, variant_id, gguf.{name, quant, layers}, verified}]`, `locations[{source, path, exists}]` |
| `POST /api/local-models/scan` | `{}` | 202 job (result unused; the list is re-fetched) |

A POST that returns a non-job object (no `id`/`state`) is treated as an immediate result.

## Deviations from the contract, and why

- `ctx.api(path, options)`: the contract doesn't define `options`. I chose `{method: "POST", body: object}` (above). **`app.js`'s current `api(path, body)` can't be passed in directly.** Wrap it: `(path, o) => api(path, o && o.body)`, or leave `ctx.api` out and the built-in fetch wrapper takes over.
- `onJob(job)` is called at job start and job end (not on every poll). I poll `/api/jobs/<id>` myself and don't depend on `window.Jobs`.
- Extra `ctx.pollMs` key, and mount returns a handle (`unmount`, `select`/`refresh`).
- "No preference" and "Skip voting on the rest" are UI-only: they send no request, because `vote` has no skip value in the contract.

## Known gaps

- The reveal and needle result shapes aren't pinned in the contract. I accept several shapes (above). If `api`/`evals` choose another one, `slotLabel()` or `renderNeedle()` in `quality.js` needs a line.
- Unit assumptions:
  - `score`, `ci_low`, `ci_high` and `same_top_p` are treated as fractions when ≤ 1, otherwise as percentages.
  - `mean_delta_p` is shown as-is in **percentage points** (llama.cpp prints `Mean Δp: -0.4 %`).
- Quiz history is matched by `variant_id`. Two placements of the same file (CPU vs GPU) share one row.
- The compression tab offers only variants present in `ctx.candidates`, so with "Compare all configurations" off the user may see fewer quants. It marks on-disk files but doesn't block choosing files that aren't on disk; the server's 409/400 message is shown instead.
- No dark theme exists in `style.css` yet. The panel adapts on its own, but host `.secondary` buttons, inputs and `.tag` chips keep their light colours until `ui-run` adds dark tokens.

## Requests

- **ui-run** (`index.html`, `package.json`):
  - Add `<link rel="stylesheet" href="/quality.css" />` and `<script defer src="/quality.js"></script>`.
  - Change the npm test script to `node --test tests/ui-*.test.cjs` so `tests/ui-quality.test.cjs` runs in `npm test`. For now run it with `node --test tests/ui-quality.test.cjs`.
  - Pass `ctx.api` in the shape above, or leave it out.
  - If you add a dark theme, please also set `color-scheme: light dark` on `:root` so the textarea's `Field` colours follow it.
- **api**:
  - Serve `quality.js` and `quality.css`, with `text/javascript` and `text/css`.
  - Please make `/api/quality/reveal` return `{"mapping": [{"A": label, "B": label, …} per item], "tallies": {label: n}}` and use **candidate ids** as labels, so the UI can show friendly names.
  - Please keep the `/api/quality/vote` response free of anything that maps slots to models before reveal. The UI ignores it anyway.
  - Please have quiz results stored in `quality_results` carry `workload, score, ci_low, ci_high, variant_id` at the top level.
- **evals / quantcheck**: keep `score`/`ci_*` as 0–1 fractions, and keep `mean_delta_p` in percentage points as llama.cpp prints it. If `same_top_p` is converted, either unit works.

## How I tested

- `node --test tests/ui-quality.test.cjs`: 13 tests in jsdom with a mocked `ctx.api` and job sequences. They cover:
  - mount/unmount (polling stops, the DOM is cleared, a second mount replaces the first)
  - tab keyboard navigation
  - the quiz job flow, with the range text and bar, the tie and clear-winner verdicts, and model output rendered as text, not HTML
  - 409 demo and failed-job messages, and Stop/cancel
  - the full blind compare: validation of model count, empty prompts, the 4001-character counter and error, and the 5-prompt cap. The request body; **no label leakage in `outerHTML` before reveal**, even after a vote whose response contains labels; the reveal mapping and tallies; and restarting
  - compression defaults, same-model filtering, on-disk marks, results and numbers, and validation
  - the local models list, scan job and empty state, plus the built-in fetch fallback sending `X-Session-Token`
  - a static check for CSP-unsafe APIs
- `npm test` (existing ui-flow, 18 tests) and `python3 -m unittest discover -s tests` (72 tests) pass. The keyring panic needed `pip install --ignore-installed cryptography`.
- Playwright screenshots of a standalone harness (host `style.css` + `quality.css`) at 360 px and 1280 px, light and simulated dark, for every tab state. I fixed the issues they showed: tab labels clipped at 360 px, low-contrast reveal text in dark mode, and Δp units.

## llama.cpp facts assumed

- The `llama-perplexity --kl-divergence` output reports "Same top p" and "Mean Δp" as **percentages**. The UI shows Δp as "points" and doesn't rescale it.
