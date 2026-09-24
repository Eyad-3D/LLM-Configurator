# Handoff: `docs` workstream

Branch: `claude/v04-docs` (from `claude/keen-volta-7mnrm5`).

## ⚠️ Before any PyPI release: there is no LICENSE

The repository has **no licence file**. Without one, nobody may legally reuse the code, and PyPI users can't tell the terms. I did **not** invent one, and set **nothing licence-related** in `pyproject.toml` (no `license`, no licence classifier). The owner must choose a licence (e.g. MIT / Apache-2.0), add `LICENSE`, then add `license = "…"` (SPDX string) + `license-files = ["LICENSE"]` to `pyproject.toml`. `release.yml` and `docs/contributing.md` both say this. `docs/faq.md` says "a licence for this project's own code has not been chosen yet" — update it when one is added.

## What I built

- **README.md** rewritten: ~110 lines. The loop (pick → download → test → tune → use), quick start (uvx one-liner, pipx two-liner, from source for Windows and macOS/Linux in collapsible blocks), a "What happens on your computer" box, screenshot placeholders describing what to capture (no images), and a table of doc links. Links are **absolute GitHub URLs** (`…/blob/main/docs/…`) so they work on the PyPI page too.
- **docs/**: `getting-started.md`, `how-it-works.md`, `testing-and-tuning.md`, `quality-checks.md`, `using-your-model.md`, `cli.md`, `privacy-and-safety.md`, `troubleshooting.md`, `faq.md`, `contributing.md`. Every caveat from the old README is kept (memory allowances, "not guaranteed upper bounds", USS 75% reclaim, swap never counted, calibration ranges 35–80% / 10–55% / 1–4×, 30-day expiry, AA key precedence, match-entries-start-empty, index versions not mixed, documents uses general index, etc.), moved into the relevant page.
- **CHANGELOG.md**: 0.4.0 (unreleased) entry per the contract, and history 0.1.0 → 0.3.0 reconstructed from git log and pyproject versions in each commit.
- **pyproject.toml**: `readme`, better `description`, `authors`, `keywords`, `classifiers` (Alpha, Win/macOS/Linux, Py 3.10–3.13), `[project.urls]`. `requires-python`, dependencies, version and package-data lines unchanged. **Added a second console script `llm-configurator`** (same entry point) so `uvx llm-configurator serve` and `pipx run llm-configurator` work — uvx runs the command named after the package, and only `llm-config` existed.
- **.github/workflows/tests.yml**: matrix ubuntu/windows/macos × Python 3.10/3.13 (6 jobs, `fail-fast: false`), UI job unchanged, new `package` job (build, `twine check --strict`, assert every file under `static/*`, `evals/*.json`, `data/*` and `catalogue.json` is in the wheel, install the wheel in a fresh venv, run `llm-config --help` and `llm-configurator --help`). Kept the existing demo smoke commands. Added `permissions: contents: read`.
- **.github/workflows/release.yml**: triggered **only** by `push` of tags `v*` (plus an `if: startsWith(github.ref, 'refs/tags/v')` guard). Build job checks the tag equals `v<pyproject version>`, runs unit tests, builds, `twine check`; publish job uses `pypa/gh-action-pypi-publish@release/v1` with `id-token: write` in environment `pypi`. Header comment explains the one-time PyPI trusted-publisher + GitHub environment setup.

## Public API as built

None (docs only). User-facing command names: `llm-config` and the new alias `llm-configurator`.

## Deviations from the contract and why

- Added the `llm-configurator` console script (above). It is in `[project.scripts]`, not a version/package-data line, so within my ownership. Needed for the requested `uvx llm-configurator` one-liner.
- `docs/contributing.md` mentions the internal contract/handoff process briefly, as asked.

## Known gaps

- **Not on PyPI yet.** README and getting-started say `uvx`/`pipx` work once v0.4.0 is published; the name `llm-configurator` was free on PyPI on 2026-09-24 (`pip index versions` found nothing). Remove those notes after the first release.
- **No screenshots.** Placeholders list what to capture.
- The README links point at `main`; they 404 until this lands on `main`.
- Docs describe v0.4 features from the contract, written in parallel with the code. See the double-check list below.

## Requests

- **api (cli.py) — CI will be red until fixed:** `python -m llm_configurator recommend --demo --context 2048` fails **on the base branch already** with `Error: 'Namespace' object has no attribute 'kv_cache_type'`. `cli.main` builds `Requirements` from every dataclass field via `getattr(args, name)`, and the lead added `Requirements.kv_cache_type` without a matching `recommend --kv` option. Fix: add `rec.add_argument("--kv", dest="kv_cache_type", choices=["f16","q8_0","q4_0"], default="f16")` (or use `getattr(args, name, default)`). `docs/cli.md` does **not** yet list a `recommend --kv` option; add a row if you add it.
- **ui-run (package.json):** `npm test` currently runs only `tests/ui-flow.test.cjs`. Make it run all `tests/ui-*.test.cjs` (e.g. `node --test tests/`) so ui-run/ui-quality tests run in CI.
- **harness:** `docs/contributing.md` says real-runtime tests live in `tests/integration/` and run when `LLM_CONFIG_REAL_RUNTIME` is set, "see `scripts/`". Adjust if the script name/usage differs; consider adding a manual (`workflow_dispatch`) CI job for it later — I did not add one.
- **community:** `docs/privacy-and-safety.md` points to `community/README.md` for the format.
- **lead:** after merge, update the version-specific notes in README/getting-started once PyPI is live; add LICENSE when the owner decides.

## How I tested

- `python3 -m unittest discover -s tests`: 72 tests OK (needed the documented `cryptography` reinstall).
- `python -m build` (sdist + wheel from sdist) and `twine check --strict dist/*`: both PASSED.
- Inspected the wheel: `static/*`, `catalogue.json` present. Simulated the other workstreams' files (`evals/general.json`, `data/quantcheck_corpus.txt`, `static/run.js`) in a scratch copy and rebuilt: all shipped in the wheel via the existing package-data patterns, and `importlib.resources` finds them after install.
- Installed the wheel in a fresh venv: `llm-config --help` and `llm-configurator --help` work.
- Parsed both workflow files with PyYAML; ran the package job's wheel-contents check locally.
- A "fresh user" subagent read only README + getting-started and reported confusing sentences; a second subagent checked the docs against the contract and code. Fixes applied (see commits).

## llama.cpp facts assumed (for the harness to confirm)

These appear in user-facing docs:

- `llama-server` serves an OpenAI-compatible API at `http://127.0.0.1:PORT/v1`, ignores the API key, and accepts any model name when one model is loaded.
- A compressed **V** cache requires flash attention (from `launch.normalize`).
- `--n-cpu-moe N` keeps the expert tensors of N layers in RAM.
- Official releases publish a `digest: sha256:…` per asset in the GitHub API; Windows CUDA builds need the separate `cudart` asset.
- Docker image `ghcr.io/ggml-org/llama.cpp:server` (and a CUDA variant) exists.
- `llama-perplexity --kl-divergence-base FILE` writes large logits files (docs say "can be several GB").
- Ollama expresses KV cache type only server-wide via `OLLAMA_KV_CACHE_TYPE`.

## Statements the lead must double-check after merge

Each is written from the contract, not from merged code. Adjust the docs if the implementation differs.

1. **UI labels**: "Get it running", its six step names, "jobs tray", "Install runtime", "Start"/"Stop", "Copy", verdict badges "Runs well / Runs slowly / Too slow / Not tested yet", Quality tabs "Quick quiz / Try my prompts / Compression check", "Reveal", "Import community results" (getting-started, quality-checks, using-your-model, privacy).
2. **Tune budgets** 1/5/15 min in the UI and 60–1800 s on the CLI `--budget` (API range; CLI validation may differ).
3. **`llm-config run`** without `--port` picks a free port and prints the address (testing-and-tuning, using-your-model, troubleshooting).
4. **`llm-config local`** with no flags lists found files; `--scan` searches; `--add` registers (cli.md).
5. **`download --directory`** is optional and defaults to the models folder (contract shows `[--directory DIR]`; the current CLI still has it `required=True`).
6. **Test verdict** wording "Works / Works, but slowly / Failed" (maps `works|works_slowly|failed`).
7. **Speed test** depth "about context − 640 tokens" and `-p 512 -n 128`.
8. **Memory allowances** in how-it-works (10%, 0.75 + 0.20/user GiB, 0.5 GiB staging) are the v0.3 values; the engine ws may change them. The page already says they may change.
9. **Apple Silicon wired limit** rule ("`iogpu.wired_limit_mb` when set, otherwise macOS's default share") — link or copy the exact rule from the hardware handoff.
10. **Catalogue contents** listed in how-it-works/CHANGELOG (Qwen3 dense+MoE, Qwen2.5-Coder, Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss, DeepSeek-R1 distills, small models) — confirm against the final `catalogue.json` (and whether Qwen3-235B made it).
11. **Quiz** item counts (30–60) and "Wilson 95%" range example.
12. **Community** anonymised fields and "month of the test" precision.
13. **Speed estimate quants**: `speed.py` currently only knows Q4_K_M/Q5_K_M/Q6_K/Q8_0; docs list all `QUANT_BYTES_PER_PARAMETER` levels as having known **sizes** (memory), not speed estimates. Fine unless engine drops a level.
14. **Troubleshooting error texts** are paraphrased, not quoted, except "Secure credential storage is unavailable" and "Reload the application to refresh the session" (both from current code) and the contract's "Compare again first".
15. **Learning label** "adjusted from N local measurements" and evidence labels list.
16. **`uvx --python 3.12 llm-configurator serve`** in troubleshooting — standard uv flag.
