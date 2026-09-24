# Round 3 notes: `r3-ui`

Branch: `claude/v04-r3-ui`. Files: `src/llm_configurator/static/*`, `tests/ui-run.test.cjs`, `tests/ui-quality.test.cjs`, this file.

`npm test`: 79 tests, all pass (was 71). `python3 -m unittest discover -s tests`: 809 tests, OK (27 skipped: real-runtime tests without a build).

## Fixed

### Speed labels (`app.js`)

- **Short tune** (`evidence: "tuned"`, `tps: null`, `tuned.tps` set). Before this change it fell through to "Not tested yet" or the rough estimate. The card now shows `~14.2 tok/s · Tuned with a short test · not checked at this length yet`, and the badge reads `Not tested yet · tuned with a short test`. It never says "Measured" or "Speed verified locally".
- **Tuned with a speed** already read "Measured after tuning". That stays.
- **Details → Speed evidence** now lists every speed number the card has, each with its own label. There are four kinds:
  - "Measured on this computer"
  - "Measured after tuning, on this computer"
  - "Short tune near the start of a conversation (not checked at this length)"
  - "Estimated from your tests at other lengths (not a test at this length)", and "Other people's results: middle value from N similar computers (not this one)"

  Estimates carry a `~`. The list only appears when there are two or more numbers.
- The interpolation badge now says "estimated from your tests at other lengths", the same words as the speed label. The old wording was "nearby tests", which was vague.

### Tune panel (`run.js`)

- The page's own stop sentence is gone, so the stop reason appears once. The tuner's `notes` always end with it.
- "The best settings are saved…" appears only when `result.saved !== false`. Otherwise the panel says "These settings were not saved, so the model keeps using its earlier settings."

### Community share (`run.js`)

- When `skipped > 0`, the panel shows: "N results were measured on different hardware and were left out (this includes results from before a hardware or driver change)". `community.share` skips results whose fingerprint differs, so a driver change also counts.
- `records: 0` gives a plain "nothing to share" message with no JSON and no link. Today the server answers that case with an error, and the UI shows the error text. Both paths are handled.
- `fits_in_url: false`: the copy-first and paste instructions were already there.

### Local models (`quality.js`, `run.js`)

- `f.filename` and `l.label` were already used. A file with only a `local_variant_id` is now tagged "Your own file" and reads "Not in our list of models, but you can still use it here". Before, it showed "Not in the catalogue".
- The compression check marks `local_variant_id` files as "on your disk" too.
- `/api/downloads/plan` with `local: true` and no `local_copy` shows no Download button. It says the file isn't where it was found and how to scan again, and offers "Check again". The step label reads "Your file is missing".
- Compression check heading: it falls back to `reference_quant` when kl_check's `reference` is a dict (the new app shape).

### Hardware (`app.js`)

- Every `other_gpus` entry is listed with its `reason`, for example "Also found Intel Arc A380 (not counted in memory estimates): Intel drivers do not…".
- With no readable GPU, the summary line counts the chips found ("A graphics chip was found, but free memory can't be read, so estimates use RAM") instead of naming only the first one.
- Names and reasons are escaped. A test covers hostile names.

### From the independent review (a subagent reviewed accessibility, wording and XSS)

- No XSS or CSP problems were found.
- The details dialog gets `aria-label="Deployment details"`.
- "View details ↗" loses the arrow, because it opens a dialog, not an external site.
- Focus is restored with `getElementById` plus a `contains` check, instead of a `[id="…"]` selector built from a string.

## Already done before this round (checked, no change)

- Escaping of `gpu.name`, `cpu_name || cpu`, the "Shared with system RAM" label for unified memory, and `hardware.warnings`.
- `variant_ids` lookup in the compression check (`quality.js`).
- Reveal by candidate id plus `labels`, with the old shape still accepted. Voting "No preference" sends `slot: "tie"`.
- "step x of y" is hidden when `done`/`total` are null. `runtime_install` stages with `total: null` show no "0 B". Byte progress with `unit: "bytes"` is recognised.
- Scan kind `local_scan`: the UI never branches on the kind, so both `scan` and `local_scan` work.
- `GET /api/export` formats, falling back to the built-in list.

## Rejected

- None of the requests were wrong.
- I kept one small change of wording. The request text says "measured on different hardware". The server skips on any fingerprint change, including drivers, so the sentence says so.

## Not done (left as notes)

- Review suggestions not taken, because they are larger changes:
  - Card headings are `<h3>` directly under the page `<h1>`.
  - `#cards` re-renders drop focus to `<body>`.
  - Run-step state changes are not announced.
  - Speed units mix "tok/s" and "tokens/s".
- Older saved tune results that have no "Stopped…" note now show no stop reason. That is acceptable: the tuner has always written one since the fix-up round.

## Requests

- **hardware (no round-3 owner):** `other_gpus[*].reason` for NVIDIA mentions "nvidia-smi" as-is, and the page shows it verbatim. A plainer version: "The NVIDIA driver tool (nvidia-smi) is missing, so free graphics memory can't be read."
- **r3-server:** `/favicon.ico` still returned 403 during my runs. It's on your list.

## Evidence (real app, Chromium via Playwright)

- **`serve --demo --no-browser`:**
  - I clicked through welcome, skip rankings, the five questions, review, results, details, and "Get it running" with every step opened.
  - I took screenshots at 360 px and 1280 px, in light and dark.
  - No page errors. The only non-2xx responses were `409` on `/api/downloads/plan`, which is the designed demo refusal (the UI shows "Not available in demo mode").
  - At 360 px the cards stack cleanly with no sideways scroll. The demo notice sits once at the top of the run panel.
- **Normal run with an empty home folder:** the same flow and widths/themes. I rewrote `/api/state` and `/api/recommend` in the browser (route interception) so the real page showed:
  - two `other_gpus` entries;
  - a short tune;
  - community evidence;
  - a tuned card with an interpolation and a community number.

  All read as described above. Details list each number with its own label.
- **Keyring panic:** the first demo run hit it on `/api/credentials` (empty response). Fixed with `pip install --ignore-installed cryptography`, as documented.
