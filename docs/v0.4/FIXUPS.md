# v0.4 fix-up round

The 18 build branches are merged into `claude/keen-volta-7mnrm5`. Unit tests pass (572 Python, 56 UI), and 29 of 32 real-runtime tests pass. But each piece was built without seeing the others. This round makes every seam correct and matches the real llama.cpp.

Each fix-up session owns the files below and **nothing else**. Other sessions edit other files at the same time. If you need a change elsewhere, write it under **Requests** in `docs/v0.4/fixups/<area>.md`.

| Area (branch `claude/v04-fix-<area>`) | Owns | Requests addressed to (in `docs/v0.4/handoff/*.md`) | Review file |
|---|---|---|---|
| `api-http` | `server.py`, `tests/test_server.py`, `tests/test_api_v04.py` | api (server/HTTP parts) | `docs/v0.4/review/api-http.md` |
| `api-cli` | `cli.py`, `app.py`, `tests/test_cli.py` | api (CLI/app parts), "api / cli", "api/app" | `docs/v0.4/review/api-cli.md` |
| `server-fake` | `llama_server.py`, `tests/fixtures/*`, `tests/test_llama_server.py`, `tests/integration/test_fake_matches_real.py` | server | `docs/v0.4/review/server-fake.md` |
| `testing-tuner` | `testing.py`, `tuner.py`, `tests/test_testing.py`, `tests/test_tuner.py` | testing, tuner | self-review |
| `evals` | `evals.py`, `evals/*.json`, `tests/test_evals.py` | evals | self-review |
| `quantcheck` | `quantcheck.py`, `data/quantcheck_corpus.txt`, `tests/test_quantcheck.py` | quantcheck | self-review |
| `runtime-downloads` | `runtime_install.py`, `downloads.py`, `runtime.py`, their tests | runtime, downloads, "engine / downloads (runtime.py)" | self-review |
| `discover` | `discover.py`, `gguf.py`, their tests | discover | self-review |
| `engine` | `engine.py`, `speed.py`, `learning.py`, their tests (`test_engine`, `test_speed`, `test_learning`, `test_quality_comparison`, `test_guided_requirements`, `test_calibration`) | engine | self-review |
| `hardware-catalogue` | `hardware.py`, `catalogue.py`, `catalogue.json`, `test_hardware.py`, `test_catalogue.py`, `test_adapters.py` | hardware, catalogue | self-review |
| `community-export` | `community.py`, `community/*`, `export.py`, `test_community.py`, `test_export.py`, `.github/ISSUE_TEMPLATE/*` | community, export | self-review |
| `ui` | `static/*`, `tests/ui-*.test.cjs`, `package.json`, `package-lock.json` | ui-run, ui-quality | self-review |
| `docs` | `README.md`, `docs/*.md` (not `docs/v0.4/`), `CHANGELOG.md`, `pyproject.toml`, `.github/workflows/*` | docs, "lead / docs" | self-review |

The lead keeps `domain.py`, `storage.py`, `jobs.py`, `launch.py`, `tests/test_foundations.py`, `scripts/*` and `tests/integration/test_real_runtime.py`.

## Pinned data shapes (so parallel fixes don't create new seams)

- **Tuned record** (`store.append("tuned", …)`, written by `server.py` / `cli.py`, read by `engine.py`):
  `{**tune_result, "variant_id", "sha256", "fingerprint", "context", "gpu_layers", "n_cpu_moe", "kv_cache_type", "timestamp"}`
- **Tune measurement** (optional, written by the api next to the tuned record): a §2.3 record with `kind="tune"`, `id`, `settings` from `best`, and `depth`.
- **Quiz result** (`quality_results`): `{"kind": "quiz", "variant_id", "candidate_id", "workload", "score", "ci_low", "ci_high", "needle": result|None, "result": full run_quiz result, "timestamp"}`. Quant check: `{"kind": "quant_check", "variant_id": <reference>, "result": …, "timestamp"}`.
- **Reveal response** (`POST /api/quality/reveal`): `{"mapping": [{"A": candidate_id, "B": …} per item], "tallies": {candidate_id: n}, "labels": {candidate_id: "friendly name"}}`. The UI should also accept the pre-fix-up shape.
- **Serve status** (`GET /api/serve`): the existing keys plus `candidate_id` and `variant_id`, with no absolute paths anywhere.
- **Every other HTTP response shape stays as it is now.** `api-http` may only add keys, never rename or remove them.
- **Job `kind`s:** `runtime_install`, `download`, `local_scan`, `test`, `tune`, `quiz`, `compare`, `quant_check`, `serve`, `community_import`. Download/test/tune/serve/quiz jobs carry `subject: {candidate_id?, variant_id}` and a human-readable `title`.

## Real llama.cpp

Sessions that touch llama.cpp behaviour (`server-fake`, `testing-tuner`, `quantcheck`, `runtime-downloads`, `api-http`, `api-cli`, `export`) should build it early in the background (about 10–15 minutes):

```bash
python3 -m pip install gguf
JOBS=4 bash scripts/build_llama_cpp.sh /opt/llama-work        # last line = bin dir
python3 scripts/make_tiny_models.py /opt/llama-work/models --bin <bin dir>
LLM_CONFIG_REAL_RUNTIME=<bin dir> LLM_CONFIG_TINY_MODELS=/opt/llama-work/models \
  python3 -m unittest tests.integration.test_real_runtime tests.integration.test_fake_matches_real
```

Status on the merged branch (4b8e02d): 32 tests, 3 failing. They are `runtime_install` build parsing, and the fake accepting `--draft-max` and answering `llama-bench --version`.
