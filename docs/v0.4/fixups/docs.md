# Fix-up: `docs`

Branch `claude/v04-fix-docs`, from `claude/keen-volta-7mnrm5` (076bab8).
Owns `README.md`, `docs/*.md`, `CHANGELOG.md`, `pyproject.toml`, `.github/workflows/*`.

## What changed

PAGES

## Claims that depend on in-flight fixes

The docs describe the code **as it is on 076bab8**. When these fixes land, the lead should update the named lines.

INFLIGHT

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

`python3 -m pip install -e .` then `python3 -m unittest discover -s tests`: TESTS

## Requests

REQUESTS
