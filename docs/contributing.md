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

`-e` ("editable") means code changes take effect without reinstalling.

## Running the tests

```bash
python -m unittest discover -s tests -v
```

If importing `keyring` crashes with `pyo3_runtime.PanicException` (a clash with a system-installed `cryptography` package, seen on some Linux setups), reinstall `cryptography` first:

```bash
python3 -m pip install --ignore-installed cryptography
```

UI tests (need Node.js 22 or newer; CI uses 22):

```bash
npm ci
npm test          # runs every tests/ui-*.test.cjs file
```

Tests need **no internet, no graphics card, no API key and no model download**. External services are replaced by fixtures, and llama.cpp by a small fake (`tests/fixtures/fake_llama.py`) that imitates its programs and output. Please keep it that way: never make a real network call or run a real model in the normal test suite.

### Tests against a real llama.cpp

`tests/integration/test_real_runtime.py` runs the app against real llama.cpp programs and tiny test models. It is skipped unless `LLM_CONFIG_REAL_RUNTIME` and `LLM_CONFIG_TINY_MODELS` are set. (`test_fake_matches_real.py` compares the fake with recorded real output and also runs in the normal suite.)

On Linux or macOS with a C/C++ compiler and internet access to PyPI:

```bash
python3 -m pip install -e . gguf
BIN=$(bash scripts/build_llama_cpp.sh /opt/llama-work | tail -1)
python3 scripts/make_tiny_models.py /opt/llama-work/models --bin "$BIN"
export LLM_CONFIG_REAL_RUNTIME="$BIN" LLM_CONFIG_TINY_MODELS=/opt/llama-work/models
python3 -m unittest -v tests.integration.test_real_runtime tests.integration.test_fake_matches_real
```

- The build takes about 12 minutes on 4 cores. It builds a CPU-only llama.cpp from the source bundled in the `llama-cpp-python` package on PyPI (so it works without GitHub access), installs `cmake` with pip if it's missing, and prints the folder with the programs as its last line. Running it again only rebuilds what changed.
- Settings: `JOBS` (parallel build jobs; default: all cores), `LLAMA_WORK_DIR` (work folder when none is given; default `/opt/llama-work`), `LLAMA_CPP_PYTHON_VERSION` (which `llama-cpp-python` source to use; default `0.3.35`).
- `gguf` is only needed by `scripts/make_tiny_models.py`, never by the app.

### Continuous integration

`.github/workflows/tests.yml` runs the unit tests on Windows, macOS and Linux with Python 3.10 and 3.13, runs the UI tests, and checks that the package builds and installs. A separate `real-runtime` job runs the real llama.cpp tests above; it is slow, so it runs only when started by hand (Actions → Tests → Run workflow) and once a week, never on every push.

## Checking the package

```bash
python -m pip install build twine
python -m build
python -m twine check dist/*
```

## Code guidelines

- **Never invent numbers.** Unknown stays `None` / "unknown"; estimates are labelled as estimates.
- **Plain language** in anything a user reads: short sentences; explain jargon with a simple comparison.
- **No new runtime dependencies** without discussion (currently `psutil`, `keyring`, `numpy`, with exact versions pinned in `pyproject.toml`).
- **Safety:** the browser never supplies paths, URLs, programs or command-line flags. Subprocesses use argument lists (never a shell) and a timeout. Local servers listen on `127.0.0.1` only.
- Match the surrounding style: short module docstring saying what the module promises, compact functions, comments that explain *why*.

## Releasing (maintainers)

1. Update the version in `pyproject.toml` and `src/llm_configurator/__init__.py`, and add a `CHANGELOG.md` entry.
2. Tag and push: `git tag v0.4.0 && git push origin v0.4.0`.
3. `.github/workflows/release.yml` builds and publishes to PyPI. It runs **only** for `v*` tags. See the comment at the top of that file for the one-time PyPI setup.

A licence must be chosen and added before the first PyPI release.

## Internal process

Larger versions are built by several parallel workstreams. `docs/v0.4/CONTRACT.md` records who owns which files and the agreed interfaces, and each workstream leaves notes in `docs/v0.4/handoff/`. You don't need these to contribute.
