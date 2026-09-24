# Handoff: harness (real llama.cpp ground truth)

## What I built

- **A real llama.cpp** (commit `4df29be`, 16 Aug 2026) built from the PyPI sdist `llama-cpp-python==0.3.35`
  (`vendor/llama.cpp`), because this container can't reach GitHub. CPU only.
- **Tiny GGUF models** made with `gguf` 0.19.0 from PyPI, then quantized and split by the real tools: dense `llama`
  (F16, Q8_0, Q4_K_M), MoE `qwen3moe` (F16, Q4_K_M), a 4-shard split, plus `unknown-arch.gguf` and `corrupt.gguf`.
  Greedy output is wired to be plain ASCII (chat reply `42`, the smoke-test answer; raw text ` the quick brown fox.`).
  With purely random weights llama-server returns HTTP 500 because its reply parser rejects the garbage.
- **`docs/v0.4/llama-cpp-facts.md`**: verified flags and output formats, with a **Mismatches** section at the top.
- **`tests/integration/samples/`**: raw transcripts (help, `--version`, HTTP bodies, bench JSON, perplexity/KLD text, error text).
- **`tests/integration/test_real_runtime.py`**: 24 tests against the real binaries (skipped unless env vars are set).
- **`tests/integration/test_fake_matches_real.py`**: 8 tests comparing the fake llama.cpp with the real samples. No real
  runtime needed, so they run in the normal suite (they skip until `tests/fixtures` is merged).

Files: `scripts/build_llama_cpp.sh`, `scripts/make_tiny_models.py`, `scripts/capture_llama_facts.py`,
`scripts/tiny_corpus.txt` (original text for perplexity), `tests/integration/{__init__.py,.gitignore,test_*.py,samples/}`,
`docs/v0.4/llama-cpp-facts.md`, this file.

## Public API as built

Scripts (no Python API):

- `scripts/build_llama_cpp.sh [WORK_DIR]` → builds `llama-server llama-bench llama-perplexity llama-cli llama-gguf-split
  llama-quantize`; idempotent; the **last stdout line is the bin directory**. `WORK_DIR` defaults to `$LLAMA_WORK_DIR` or
  `/opt/llama-work`; `LLAMA_CPP_PYTHON_VERSION` pins the sdist (default 0.3.35); `JOBS` sets parallelism. Takes about
  12 minutes on 4 cores.
- `python3 scripts/make_tiny_models.py OUT_DIR [--bin BIN_DIR]` → deterministic (seed 1234) models plus `manifest.json`.
  Needs `pip install gguf`. Without `--bin` only the F16 models and the broken files are written.
- `python3 scripts/capture_llama_facts.py BIN_DIR MODELS_DIR OUT_DIR [step ...]` → rewrites the samples. Steps:
  `versions server server_errors bench perplexity`.

Test switches: `LLM_CONFIG_REAL_RUNTIME=<bin dir>`, `LLM_CONFIG_TINY_MODELS=<models dir>`.

## Where things live in this container

- Source and build: `/opt/llama-work/llama_cpp_python-0.3.35/vendor/llama.cpp/` (build in `build/`)
- Binaries: `/opt/llama-work/llama_cpp_python-0.3.35/vendor/llama.cpp/build/bin` (symlink **`/opt/llama-work/bin`**)
- Tiny models: **`/opt/llama-work/models`** (split copy in `models/split/`)
- Scratch merge of every branch's module files, used to run the module tests: `/opt/llama-work/integ`
  (a detached git worktree; not pushed; delete with `git worktree remove --force /opt/llama-work/integ`)

## Reproduce

```bash
python3 -m pip install -e . gguf                       # gguf brings numpy (test/script-only, not a runtime dependency)
BIN=$(scripts/build_llama_cpp.sh /opt/llama-work | tail -1)
python3 scripts/make_tiny_models.py /opt/llama-work/models --bin "$BIN"
export LLM_CONFIG_REAL_RUNTIME="$BIN" LLM_CONFIG_TINY_MODELS=/opt/llama-work/models
python3 -m unittest -v tests.integration.test_real_runtime
python3 -m unittest -v tests.integration.test_fake_matches_real
# refresh samples after a llama.cpp upgrade:
python3 scripts/capture_llama_facts.py "$BIN" /opt/llama-work/models tests/integration/samples
```

## How I tested (results at the end of this session)

- Normal suite on this branch: `python3 -m unittest discover -s tests` → **OK (skipped=25)** (needed the documented
  `pip install --ignore-installed cryptography` for keyring first).
- Real runtime, this branch (only lead + existing modules present): 24 tests → **13 pass, 1 fail**, 10 module tests skip (not merged here).
  - FAIL `LaunchTests.test_draft_model_is_accepted_and_used`: real mismatch, `launch.server_args` emits removed `--draft-max`.
  - All other `launch.server_args` cases start a real server and answer: defaults, `-fa on` + q8_0 K/V cache,
    `-fa off` + q4_0 K cache, batch/ubatch/parallel, `--no-mmap` + alias with a space, `--mlock`, `-dev none`,
    full offload count on a CPU build, `--n-cpu-moe 2` on the MoE model, split GGUF first shard.
  - `runtime.bench` (base-branch version) with the real llama-bench: passes.
- Real runtime + fake-vs-real against a scratch overlay of every branch's module files (fetched 12:50 UTC; see
  "Where things live"): 32 tests → **28 pass, 4 fail**, and all 4 failures are the mismatches below. Passing:
  llama_server (start/chat/timings/complete/tokenize/stop, failure reasons), testing.smoke_test (answers `42`, all checks
  pass), testing.speed_test, tuner.tune (20–30 s budget, converged, 12 trials), tuner.bench_args, quantcheck.kl_check and
  parse_kld_output on the real sample, export llama-server script (runs and serves), runtime.bench (downloads branch),
  6 of 8 fake-vs-real checks. Failing: `runtime_install.detect` (`build` is None), launch draft flags, fake accepts
  `--draft-max`, fake `llama-bench --version` exits 0 (Requests 1–3).

## Deviations from the contract

- `tests/integration/__init__.py` makes the folder a package, so `unittest discover -s tests` also picks these files up
  (they skip without the env vars / the fake). Harmless, and it makes the fake-vs-real check part of the normal suite.
- `scripts/capture_llama_facts.py` and `scripts/tiny_corpus.txt` are extra (both in my `scripts/*`).
- `gguf` (PyPI) is used only by `scripts/make_tiny_models.py`, never by the app or the tests.

## Known gaps

- **No GPU here**: CUDA/Vulkan/Metal offload, VRAM measurement, `--list-devices` GPU format, `-dev <GPU>` and GPU OOM
  messages are unverified. The fake's OOM text can't be checked against anything real yet.
- Build number reads `1` (shallow clone in the sdist). Parsers are right to read it, the value is just not meaningful.
- The tiny tokenizer is crude (≈1.3 characters/token), so the speed test runs with `context=8192`.
- `llama-cli` was built and its `--version` checked, nothing else (no workstream uses it).
- The `llama-perplexity` truncation race is timing-dependent; `test_perplexity_kl_divergence_output` retries up to 3 times.

## Requests

1. **lead, `launch.py`**: for a draft model emit `-md PATH --spec-type draft-simple` and `--spec-draft-n-max N` instead of
   `--draft-max N` (removed; llama-server exits 1). Optionally `--load-mode mlock` / `--load-mode none` instead of the
   deprecated `--mlock` / `--no-mmap` (these still work). Older llama.cpp builds may not know the new flags, so it may be
   worth checking the build number.
2. **runtime, `runtime_install.parse_version`**: accept `version: 0.1.0-dev (build 1234, commit abc1234)` (build and commit
   inside the parentheses) as well as `version: 1234 (abc1234)`. The semver part is always `0.1.0-dev` for nightlies,
   so report `b<build>` as the version. Detect the backend with `llama-server --list-devices`
   (CPU-only prints `Available devices:` then `  (none)`), since a static CPU build prints no backend hint.
3. **server, fake llama.cpp**: reject `--draft-max`/`--draft`/`--draft-min` like the real one (exit 1 with
   `the argument has been removed. use --spec-draft-n-max ...`), accept `--spec-draft-n-max` and `--spec-type`, add
   `draft_n`/`draft_n_accepted` to timings when speculative decoding is on, and make `llama-bench --version` print usage
   and exit 1. Real `/completion` also has `stop_type` (`eos`/`limit`), `tokens_cached`, `has_new_line`, `truncated`.
4. **quantcheck**: when stdout lacks `Same top p:` or `Mean    KLD:`, re-run the candidate step (reuse the base file)
   before falling back to per-chunk rows; it happens about 1 in 10 runs under load and happened once in my session.
5. **downloads, `runtime.bench`**: don't hardcode `-dev CUDA0`; device names depend on the backend (`Vulkan0`, …), and an
   unknown one is fatal.
6. **testing / tuner / evals**: send `"cache_prompt": false` when timing prompt reading or first-word delay; repeated
   prompts are otherwise served from cache (`cache_n` > 0). testing: consider checking the filler prompt length with
   `/tokenize` before sending it.

## llama.cpp facts I assumed

None unverified: everything I relied on was run against the real build, and it's all in `docs/v0.4/llama-cpp-facts.md`.
