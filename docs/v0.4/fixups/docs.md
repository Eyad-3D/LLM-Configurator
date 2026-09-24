# Fix-up: `docs`

Branch `claude/v04-fix-docs`, from `claude/keen-volta-7mnrm5` (076bab8).
Owns `README.md`, `docs/*.md`, `CHANGELOG.md`, `pyproject.toml`, `.github/workflows/*`.

## What changed

Each page was checked claim by claim against the code by a sub-agent (every command's `--help`, the implementing
module, and UI labels in `static/*`). A "fresh user" sub-agent read only README + getting-started; its 28 findings
were all applied. Highlights:

- **README.md** (101 → 79 lines): uvx/pipx moved below "From source" and marked "after the PyPI release (not yet)";
  separate ZIP steps (`LLM-Configurator-main`); how to open a terminal per OS; one line at a time; keep the window
  open, Ctrl+C stops; 127.0.0.1/port explained; says plainly the app has **no chat window** (Try my prompts, or start
  the server and connect a chat app); screenshots placeholder removed. Verified `llm-config serve --demo --no-browser`
  serves on 127.0.0.1:8765.
- **getting-started.md**: real step names from `static/run.js` (**Get the engine / Install the engine**, **Download the
  model**, **Test it**, **Tune it**); "jobs tray" → **Background tasks**; removed "Quality panel lists local files"
  (`LocalModels.mount` is never called) → `llm-config local --scan`; reuse of existing files limited to what the code
  does (models folder or a previous scan); new **gated models / `HF_TOKEN`** section quoting `downloads.py`; tips use
  `.venv/bin/llm-config` so they work for source installs; numbered uninstall.
- **cli.md**: every command/sub-command/option from `--help` (29 checked): added `recommend --kv`, `test --tuned`,
  `export --output`, `runtime status --json`, `local --json`, `tune --budget 60–1800 (300)`/`--goal`, `quiz --workload`
  choices, real `bench` defaults; exit code 2; `models remove` also removes built-ins; gated models: `HF_TOKEN`, or add
  `"config_repo"` by hand to the data folder's `catalogue.json` (`models add` has no option for it); `HF_ENDPOINT`
  affects downloads only; `download --directory` files need `local --scan --dir`.
- **how-it-works.md**: memory allowances are current (not "v0.3"); exact Apple wired-limit rule from `hardware.py`;
  verdict badges only from measured/tuned speed; community needs ≥2 reports; calibration NVIDIA-only (Apple/AMD have
  none); learning rules (≥2 tests, 90 days, ±10% floor); speed-ranking order; AA key order; catalogue = the 39 entries
  really in `catalogue.json`; offered quant levels; `config_repo` + `HF_TOKEN` note.
- **testing-and-tuning.md**: real button labels; smoke question 17+25=42 and its five checks; depth exactly
  `context − 640`; verdict wording "It works / It works, but slowly / It didn't work"; what tune really varies, 3% rule,
  up to 3 rounds, re-check of the winner; `--tuned` only on `test`/`export`/`run` (CLI), the app's Use it step ignores
  tunes; must stop a running server first.
- **quality-checks.md**: real tab/button labels; quiz size 45–55 (was 30–60); quiz topics as built; needle test;
  compare limits (4,000 chars, 20 kept); compression check corpus ~44 KB, temp file often > 1 GB, always deleted.
- **using-your-model.md**: **Start server / Stop server / Copy**; app picks a free port, exports assume **8080**
  (`export.DEFAULT_PORT`) — added "change the port to match"; suggested file names per export; Docker is local-only too.
- **troubleshooting.md**: quoted only real strings; added entries for "Could not reach GitHub…", cudart for hand
  downloads, "Stop the running model server first", 401/403 gated downloads with `HF_TOKEN`, size/content conflicts;
  real **Extra RAM/VRAM headroom** labels; backend choice rule as coded.
- **privacy-and-safety.md**: model info from huggingface.co only (`HF_ENDPOINT` is downloads only); community
  share/import are **CLI only**; shared fields = `anonymize()` (month precision); Try-my-prompts prompts/answers are
  stored locally (last 20); `AA_API_KEY`; data files listed; pointer to `community/README.md` kept (file exists).
- **faq.md**: Docker export does not share on a network; calibration NVIDIA-only; `HF_TOKEN`; licence still "not
  chosen yet".
- **contributing.md**: Node ≥ 22 (glob in `node --test`); keyring/cryptography fix; exact real-runtime recipe from the
  harness handoff with `JOBS` / `LLAMA_WORK_DIR` / `LLAMA_CPP_PYTHON_VERSION`; the new CI job.
- **CHANGELOG.md**: no "pause" (cancel + resume only); tune budgets 1/5/15 min UI, 60–1800 s CLI; community CLI-only;
  CI line updated. Every other 0.4.0 claim checked in code.
- **pyproject.toml**: no change needed (build, metadata, scripts and package-data verified below).

## Claims that depend on in-flight fixes

The docs describe the code **as it is on 076bab8**. When these fixes land, the lead should update the named lines.

| Claim (page) | Today | Owner / review item |
|---|---|---|
| "It works, but slowly" is described, with a note that the app currently always says "It works" (testing-and-tuning) | `test_job` never passes `min_tps` | api-cli #9, api-http #10 |
| `quiz --needle` documented plainly | CLI prints "see --json" | api-cli #8 |
| `download --directory` → run `local --scan --dir DIR` (cli) | file not registered | api-cli #7 |
| `models add` → then run `refresh` (cli) | entry not fetched | api-cli #17 |
| `bench` finds `llama-bench` on PATH only (cli) | ignores installed runtime | api-cli #12 |
| `local --dir` "use full paths"; `local --add` "give the first shard", `.gguf` only (cli) | relative paths saved as-is | api-cli #14, #3, #2 |
| `test` has no `--min-tps` (cli: not listed) | — | api-cli #9 |
| `run --json` accepted but ignored (cli) | own finding | api-cli |
| App server "picks a free port each time"; exports assume 8080 (using-your-model, troubleshooting) | random port | api-http #5 |
| App's Use it / server buttons don't use saved tunes (using-your-model, testing-and-tuning) | recommended settings | api-http #4 |
| Existing model files are reused only after `llm-config local --scan`; LocalModels panel not shown (getting-started) | `mount` never called | ui; api-http #9 |
| "Ctrl+C stops the server too" (no claim about closing the window) | SIGHUP/SIGTERM not handled | api-http / server-fake |
| Real-runtime CI job and the normal matrix: 2 fake-vs-real failures | fake accepts `--draft-max`, bench `--version` exits 0 | server-fake |

## CI changes

- `.github/workflows/tests.yml`: new `real-runtime` job. It runs **only** on `workflow_dispatch` (Actions → Tests →
  Run workflow) and a weekly `schedule` (Mondays 04:00 UTC); the job has `if:` on the event name, so pushes and pull
  requests skip it. Steps: `pip install -e . gguf` → restore cache of `$LLAMA_WORK_DIR` (key: OS, arch,
  `LLAMA_CPP_PYTHON_VERSION`, hash of `scripts/build_llama_cpp.sh`) → `scripts/build_llama_cpp.sh` (last line = bin dir,
  exported as `LLM_CONFIG_REAL_RUNTIME`) → `scripts/make_tiny_models.py $RUNNER_TEMP/tiny-models --bin …` (exported as
  `LLM_CONFIG_TINY_MODELS`) → `python -m unittest -v tests.integration.test_real_runtime tests.integration.test_fake_matches_real`.
  60-minute timeout.
  - The build uses `GGML_NATIVE=ON` (tuned to the CPU that built it). GitHub runners differ in CPU, so a cached build
    could crash with "illegal instruction". The job runs `llama-server --version` on the cached copy first and deletes
    the cached `build/` folder if it fails, so the script rebuilds. (A cleaner fix is `-DGGML_NATIVE=OFF` in the
    script when `CI` is set; `scripts/*` is the lead's.)
- The other jobs (`test` matrix, `ui`, `package`) are unchanged and also run on the manual/weekly triggers.
- `npm test` already runs `node --test "tests/ui-*.test.cjs"` (the ui session changed `package.json`); locally that is
  56 tests, 56 pass on Node 22. Node's `--test` glob needs Node ≥ 21; CI uses 22.
- Both workflow files parse with PyYAML (`tests.yml` jobs: test, ui, package, real-runtime; `release.yml`: build, publish).
- Not run on GitHub from here (no Actions access from this container). The steps were run locally where possible:
  the integration modules import and skip cleanly without the env vars (32 tests, 24 skipped, 2 failing — see below).

**Expected red until `server-fake` lands:** `test_fake_matches_real` has 2 failures on 076bab8
(`test_bench_has_no_version_flag_like_real`, `test_server_rejects_removed_draft_max_like_real`). They run in the normal
matrix too (no real runtime needed), so the `test` jobs are red until that fix merges. Nothing in my area can fix them.

## Packaging evidence

`python -m build` from a clean `git archive` copy in a scratch dir (fresh venv, build 1.x, twine):

- `llm_configurator-0.4.0.tar.gz` and `llm_configurator-0.4.0-py3-none-any.whl` built.
- `twine check --strict`: both PASSED.
- Wheel data files (15, none missing against the `static/*`, `evals/*.json`, `data/*`, `catalogue.json` globs):
  `catalogue.json`, `data/quantcheck_corpus.txt`, `evals/{agentic,coding,documents,general}.json`,
  `static/{app.js,index.html,jobs.js,quality.css,quality.js,run.js,selects.js,style.css,wizard.js}`.
- Installed the wheel in a fresh venv: `llm-config --help` and `llm-configurator --help` work; `importlib.resources`
  finds the evals, the corpus and `static/run.js`.
- Note: `static/*` does not recurse. If the UI ever adds a sub-folder under `static/`, add `static/**/*`.
- **No LICENSE** — still not chosen. Nothing licence-related is in `pyproject.toml`. Must be added before a PyPI release.

## Tests

`python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: **604 tests, 2 failures, 25 skipped** (after the documented `pip install --ignore-installed cryptography`). The 2 failures are the pre-existing fake-vs-real ones owned by `server-fake` (same on the base branch); nothing in this branch touches code. `npm test`: 56/56 pass.

## Requests

- **lead (`scripts/build_llama_cpp.sh`)**: consider `-DGGML_NATIVE=OFF` when `CI` is set, so a cached CI build runs on
  any runner CPU (the workflow works around it by re-testing and rebuilding).
- **lead**: add a LICENSE before any PyPI release, then `license = "…"` + `license-files = ["LICENSE"]` in
  `pyproject.toml` and update `docs/faq.md` ("not chosen yet").
- **lead**: after the PyPI release, drop the "after the PyPI release (not yet)" notes in README, getting-started and
  troubleshooting.
- **api-cli**: `models add` has no `--config-repo`; docs tell people to edit `catalogue.json` by hand. An option would
  be friendlier (catalogue.add_entry already takes `config_repo`).
- **catalogue**: `catalogue.py` model-info requests send `HF_TOKEN` via plain `urlopen`, which would keep the header on a
  cross-host redirect (downloads strip it). Probably harmless; worth matching `downloads.py`.
