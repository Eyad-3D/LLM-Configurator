# Contributing

Thanks for helping. Bug reports with real measurements from your hardware are especially useful: they are how the memory and speed maths gets better.

## Reporting a problem

Open an issue on [GitHub](https://github.com/Eyad-3D/LLM-Configurator/issues). Include your operating system, Python version, `llm-config runtime status`, what you did, and what happened. Remove private details (paths, user names) first.

If an estimate was far from a test result, include both numbers (the Test step shows them side by side).

## Developer setup

```bash
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
python3 -m venv .venv
.venv/bin/python -m pip install -e .      # Windows: .\.venv\Scripts\python.exe -m pip install -e .
```

`-e` ("editable") means code changes take effect without reinstalling. For the UI tests you also need Node.js 22 or newer.

## Running the tests

```bash
python -m unittest discover -s tests
```

Add `-v` to see each test's name. Tests that need a real llama.cpp (below) are reported as skipped.

If importing `keyring` crashes with `pyo3_runtime.PanicException` (a clash with a system-installed `cryptography` package, seen on some Linux setups), reinstall `cryptography` first:

```bash
python3 -m pip install --ignore-installed cryptography
```

UI tests (need Node.js 22 or newer; CI uses 22):

```bash
npm ci
npm test          # runs every tests/ui-*.test.cjs file
```

Tests need **no internet, no graphics card, no API key and no model download**. External services are replaced by fixtures, and llama.cpp by a small fake (`tests/fixtures/fake_llama.py`) that imitates its programs and output, like a flight simulator standing in for the plane. Please keep it that way: never make a real network call or run a real model in the normal test suite.

### Tests against a real llama.cpp

Some tests run the app against real llama.cpp programs and tiny test models, to check that the fake still behaves like the real thing. They are skipped unless `LLM_CONFIG_REAL_RUNTIME` (the folder with the llama.cpp programs) and `LLM_CONFIG_TINY_MODELS` (the folder with the tiny models) are set:

- `tests/integration/test_real_runtime.py`: llama.cpp's own programs and output formats, then the app's launch settings, model server, smoke and speed tests, tuner, compression check, export script, runtime detection and `bench`.
- `tests/integration/test_fake_matches_real.py`: most of it compares the fake with real output recorded in `tests/integration/samples/` and runs in the normal suite. Two groups run the real programs and need the variables: the fake and the real llama.cpp side by side (load errors, out-of-memory text, `/slots`), and the speed test and tuner at several context sizes.
- `tests/test_api_v04.py` (`RealRuntimeHttpTests`): the whole app over HTTP, with a data folder whose name has spaces.

On Linux or macOS with a C/C++ compiler and internet access to PyPI:

```bash
python3 -m pip install -e . gguf
BIN=$(bash scripts/build_llama_cpp.sh /opt/llama-work | tail -1)
python3 scripts/make_tiny_models.py /opt/llama-work/models --bin "$BIN"
export LLM_CONFIG_REAL_RUNTIME="$BIN" LLM_CONFIG_TINY_MODELS=/opt/llama-work/models
python3 -m unittest -v tests.integration.test_real_runtime tests.integration.test_fake_matches_real
python3 -m unittest -v tests.test_api_v04.RealRuntimeHttpTests      # optional; CI doesn't run this one
```

- `scripts/build_llama_cpp.sh` builds a CPU-only llama.cpp (`llama-server`, `llama-bench`, `llama-perplexity`, `llama-cli`, `llama-gguf-split`, `llama-quantize`) from the source bundled in the `llama-cpp-python` package on PyPI, so it works without GitHub access. It installs `cmake` with pip if it's missing and prints the folder with the programs as its last line. The first build takes about 12 minutes on 4 cores; running it again only rebuilds what changed. The build is tuned to the processor that made it, so don't copy it to a different machine.
- Settings: `JOBS` (parallel build jobs; default: all cores), `LLAMA_WORK_DIR` (work folder when none is given; default `/opt/llama-work`), `LLAMA_CPP_PYTHON_VERSION` (which `llama-cpp-python` source to use; default `0.3.35`).
- `scripts/make_tiny_models.py OUT_DIR --bin BIN` writes small, always-identical models: a dense and a mixture-of-experts model, compressed copies, a split copy, a file of an unknown model type and a cut-off (broken) file. They are wired to answer `42` to the smoke test. Without `--bin` it writes only the uncompressed models and the broken files.
- `gguf` is only needed by `scripts/make_tiny_models.py`, never by the app.
- After a llama.cpp upgrade, refresh the recorded output with `python3 scripts/capture_llama_facts.py "$BIN" /opt/llama-work/models tests/integration/samples`.

More detail: [`docs/v0.4/handoff/harness.md`](v0.4/handoff/harness.md) (how the harness works) and [`docs/v0.4/llama-cpp-facts.md`](v0.4/llama-cpp-facts.md) (llama.cpp flags and output checked on a real build).

### Continuous integration

`.github/workflows/tests.yml` has four jobs:

- **test**: installs the package and runs the unit tests on Windows, macOS and Linux with Python 3.10 and 3.13, then checks `--help` and a demo `recommend`.
- **ui**: runs the UI tests with Node.js 22.
- **package**: builds the package, runs `twine check --strict`, checks that every data file is inside, and installs it in a fresh environment.
- **real-runtime**: builds llama.cpp with `scripts/build_llama_cpp.sh`, makes the tiny models and runs `tests.integration.test_real_runtime` and `tests.integration.test_fake_matches_real`. It is slow, so it runs only when started by hand (Actions → Tests → Run workflow) and once a week (Mondays), never on every push. The build is cached between runs; a cached build that won't start on the runner's processor is rebuilt. Time limit: 60 minutes.

## Checking the package

```bash
python -m pip install build twine
python -m build
python -m twine check --strict dist/*
```

## Code guidelines

- **Never invent numbers.** Unknown stays `None` / "unknown"; estimates are labelled as estimates.
- **Plain language** in anything a user reads: short sentences; explain jargon with a simple comparison.
- **No new runtime dependencies** without discussion (currently `psutil`, `keyring`, `numpy`, with exact versions pinned in `pyproject.toml`).
- **Safety:** the browser never supplies paths, URLs, programs or command-line flags, and replies to the page never contain folder names. Subprocesses use argument lists (never a shell) and a timeout. Local servers listen on `127.0.0.1` only. Start llama.cpp programs with the environment from `launch.server_env`, which drops inherited `LLAMA_ARG_*` settings and limits llama-server's CORS to `localhost`.
- **llama.cpp behaviour:** if a change depends on llama.cpp's options or output, check it on a real build (above) and keep the fake in step.
- Match the surrounding style: short module docstring saying what the module promises, compact functions, comments that explain *why*.

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and `src/llm_configurator/__init__.py`, and add a `CHANGELOG.md` entry.
2. Tag and push: `git tag v0.4.0 && git push origin v0.4.0`.
3. `.github/workflows/release.yml` builds and publishes to PyPI. It runs **only** for `v*` tags. See the comment at the top of that file for the one-time PyPI setup.

A licence must be chosen and added before the first PyPI release.

## Internal process

Larger versions are built by several parallel workstreams. `docs/v0.4/CONTRACT.md` records who owns which files and the agreed interfaces. Each workstream leaves notes in `docs/v0.4/handoff/`, and each review round in `docs/v0.4/fixups/`. You don't need these to contribute.
