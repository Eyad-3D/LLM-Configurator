# v0.4 build contract

v0.4 turns LLM Configurator from "this should fit" into a full loop on the user's own computer:

**pick → download → test → tune → run/export**, plus wider hardware and model coverage and local quality checks.

About 18 parallel workstreams build this at the same time. Each one owns specific files. This document is the single source of truth for who owns what and how the pieces connect. If something here is ambiguous, pick the simplest reading that keeps the interfaces below intact, and record the decision in your handoff file.

---

## 1. Ground rules (every workstream)

1. **Only edit files you own** (section 3). Need a change in someone else's file? Don't edit it. Write it under **Requests** in your handoff file `docs/v0.4/handoff/<ws>.md`. The lead merges everything afterwards.
2. **Import other workstreams' modules only through the names in this contract.** They may not exist on your branch yet. In tests, use `unittest.mock` / fakes, or inject callables. **Never create placeholder files for modules you don't own**, because that causes merge conflicts. Where you must call a not-yet-existing module at runtime, import it inside the function, not at module top level.
3. **Dependencies:** Python ≥ 3.10, standard library plus the existing `psutil`, `keyring` and `numpy`. **No new runtime dependencies.** Tests use `unittest`, not pytest-only features. UI tests use `node --test` with jsdom (already present).
4. **No network in tests.** Mock `urllib.request.urlopen` / openers and `subprocess`. This container **cannot reach huggingface.co or GitHub release downloads** (PyPI works). Never download real models.
5. **Honesty stance** (the project's core value): never invent numbers. Unknown stays `None`/"unknown", and estimates are labelled as estimates. Show plain-language errors (`ValueError("...")`) that tell the user what to do next.
6. **Plain language in anything a user reads** (UI strings, CLI output, error messages): short sentences, and explain jargon with a metaphor (for example "KV cache: the model's short-term notepad for this conversation").
7. **Style:** match the existing code. Use a short module docstring stating what the module promises (safety/honesty), compact functions, few comments that explain *why*, and no heavy class hierarchies.
8. **Subprocesses:** use list arguments, never `shell=True`, and always a timeout. On Windows pass `creationflags=subprocess.CREATE_NO_WINDOW` (see `calibration.py`). An "executable" parameter is always **`command: str | list[str]`**, an argv prefix. That lets tests pass `[sys.executable, "tests/fixtures/fake_llama.py", "--as", "server"]`, and lets Windows/macOS/Linux behave the same.
9. **Long operations** take keyword args `progress=None, cancel=None`.
   - `progress(dict)` receives `{"stage": str, "done": number, "total": number | None, "message": str}`. Extra keys are allowed.
   - `cancel` is a `threading.Event`. Call `domain.check_cancel(cancel)` in loops. It raises `domain.Cancelled`. Clean up (kill processes, keep resumable partial downloads) on `Cancelled`.
10. **Security:** the browser never supplies file paths, URLs, executables or command-line flags. HTTP endpoints accept only IDs, enums and bounded numbers or strings. Paths come from the app data dir, discovered model folders, or CLI arguments. Servers started by the app listen on `127.0.0.1` only.
11. **Tests:** put them in `tests/test_<module>.py` (Python) or `tests/ui-*.test.cjs` (UI). They must pass with `python -m unittest discover -s tests` and `npm test`. Before your final push, run **the whole suite**. You must not break tests that exist on the base branch. If the keyring import panics with `pyo3_runtime.PanicException`, run `python3 -m pip install --ignore-installed cryptography` (container quirk).
12. **Git:** commit early and often. Push to **your** branch only (given in your task). Don't rebase or force-push anything you didn't create. Don't open pull requests.
13. **Handoff (required):** `docs/v0.4/handoff/<ws>.md` with these sections:
    - what you built
    - the public API as built (signatures and return shapes)
    - deviations from this contract and why
    - known gaps
    - **Requests** to other owners
    - how you tested
    - which llama.cpp facts you assumed (flags, output formats) that the integration harness should confirm

## 2. Shared foundations (already on the base branch; owned by the lead)

| File | What it gives you |
|---|---|
| `domain.py` | `GIB`, `now()`, `Cancelled`, `check_cancel(cancel)`, `ARCHITECTURES`, `DENSE_ARCHITECTURES`, `MOE_ARCHITECTURES`, `QUANT_BYTES_PER_PARAMETER`, `KV_BYTES_PER_ELEMENT`, `Requirements` (new: `kv_cache_type` f16/q8_0/q4_0), `Variant` (new fields below, `.moe`, `.all_files()`) |
| `storage.py` | `Store.get/put`, **`Store.update(key, fn, default)`** (atomic), **`Store.append(key, item, limit=1000)`** (atomic, use it for list records), `models_dir(store)`, `runtime_dir(store)` |
| `jobs.py` | `JobManager.submit(kind, title, fn, subject=None, exclusive=None)`, where `fn(progress, cancel) -> result`. Also `.get(id)`, `.list()`, `.active(...)`, `.cancel(id)`, `.wait(id, timeout)`. Job dict: `{id, kind, title, state: queued/running/done/failed/cancelled, progress, result, error, created_at, finished_at, subject, exclusive}` |
| `launch.py` | **Launch config** helpers: `normalize(config)`, `server_args(config)` (argv after `llama-server`), `server_env(config)`, `runtime_gpu_layers(gpu_layers, total_layers)`, `from_candidate(candidate, model_path, hardware, **overrides)` |
| `tests/test_foundations.py` | Tests for the above |

### 2.1 `Variant` fields (v0.4 additions, all defaulted so cached v0.3 data still loads)

- `files`: `[{"filename", "size_bytes", "sha256"}]` for sharded models, first shard first. Empty means a single file (`filename`, `size_bytes`, `sha256`). `size_bytes` is always the **total**. Use `variant.all_files()`.
- `parameters`, `active_parameters`: ints or `None`.
- `experts`, `active_experts` (0 means dense), `expert_fraction`: the share of weight **bytes** in routed-expert tensors. These are the parts `--n-cpu-moe` can keep in RAM.
- `sliding_window`, `sliding_layers`: layers whose KV is capped at the window.
- `family`, `license`, `source` (`catalogue` | `custom` | `local`).
- `architecture` must be in `domain.ARCHITECTURES`.

### 2.2 Launch config (dict; `launch.DEFAULTS` lists every key)

```
model_path, context (per user/slot), parallel, gpu_layers (transformer layers), total_layers,
gpu_uuid, gpu_backend (cuda|metal|rocm|vulkan|cpu|None), device, threads, batch, ubatch,
flash_attn (on|off|auto), cache_type_k, cache_type_v (f16|q8_0|q4_0), n_cpu_moe,
mlock, mmap, draft_model_path, draft_max, host (127.0.0.1 only), port, alias
```

This is the exchange format between the **engine → testing → tuner → export → serve** steps. If you need a new key, request it (see Requests) instead of adding ad-hoc keys.

### 2.3 Measurement record (store key `"measurements"`, via `store.append`)

Existing keys (keep them; `engine.matching_speed` relies on them): `variant_id, sha256, fingerprint, timestamp, context, users, gpu_layers, gpu_uuid, threads, tps, runtime_build, raw, note`.

New optional keys:

- `kind`: `bench` | `speed_test` | `tune`
- `pp_tps`: prompt reading speed, in tokens/s
- `ttft_s`: time to first token
- `depth`: tokens already in context when generation was measured
- `peak_ram_bytes`, `peak_vram_bytes`
- `estimated_ram_bytes`, `estimated_vram_bytes`
- `settings`: `{flash_attn, cache_type_k, cache_type_v, batch, ubatch, n_cpu_moe}`
- `runtime`: `{"version", "backend"}`
- `id`: a short random hex string used to reference it

### 2.4 Other store keys

| Key | Owner | Shape |
|---|---|---|
| `runtime` | runtime | detect() result cache |
| `local_files` | discover | list of local file records (§4.3) |
| `hash_cache` | discover | `{path: {size, mtime, sha256}}` |
| `tuned` | tuner | list of tune results (§4.6) |
| `quality_results` | evals / quantcheck | list of `{kind, variant_id, ..., timestamp}` |
| `comparisons` | evals | `{comparison_id: {...}}` blinded compare sessions |
| `community` | community | `{source, fetched_at, records: [...]}` |
| `settings` | api-cli | `{models_dir, runtime_dir?, share_opt_in?}` (written only by the CLI) |

---

## 3. Workstreams and file ownership

Branch for workstream `<ws>`: **`claude/v04-<ws>`**, created from `claude/keen-volta-7mnrm5`.

| ws | Owns (create/edit only these) | Summary |
|---|---|---|
| `runtime` | `src/llm_configurator/runtime_install.py`, `tests/test_runtime_install.py` | Find or install llama.cpp automatically |
| `downloads` | `src/llm_configurator/downloads.py`, `src/llm_configurator/runtime.py`, `tests/test_downloads.py` | Resumable, verified, sharded downloads with progress/cancel |
| `discover` | `src/llm_configurator/discover.py`, `src/llm_configurator/gguf.py`, `tests/test_discover.py`, `tests/test_gguf.py` | Reuse models already on disk (HF cache / LM Studio / Ollama) and read GGUF headers |
| `server` | `src/llm_configurator/llama_server.py`, `tests/fixtures/fake_llama.py`, `tests/fixtures/__init__.py`, `tests/test_llama_server.py` | Start/stop llama-server, chat, timings, peak memory, the user's long-running server, the **fake llama.cpp** used by all tests |
| `testing` | `src/llm_configurator/testing.py`, `tests/test_testing.py` | Smoke test, speed test (reading speed, first-word delay, writing speed), memory check |
| `tuner` | `src/llm_configurator/tuner.py`, `tests/test_tuner.py` | Auto-tune settings under a time budget |
| `evals` | `src/llm_configurator/evals.py`, `src/llm_configurator/evals/*.json`, `tests/test_evals.py` | Quick task quizzes, long-document recall test, blind "try my prompts" compare |
| `quantcheck` | `src/llm_configurator/quantcheck.py`, `src/llm_configurator/data/quantcheck_corpus.txt`, `tests/test_quantcheck.py` | Compression-loss check (KL divergence via llama-perplexity) |
| `export` | `src/llm_configurator/export.py`, `tests/test_export.py` | llama-server / Ollama / docker-compose / client snippets |
| `hardware` | `src/llm_configurator/hardware.py`, `tests/test_hardware.py` | Apple Silicon, AMD, Intel, friendly CPU names and features |
| `catalogue` | `src/llm_configurator/catalogue.py`, `src/llm_configurator/catalogue.json`, `tests/test_adapters.py`, `tests/test_catalogue.py` | 30–50 popular models, MoE, sharded files, more architectures and quants, add-your-own repo |
| `engine` | `src/llm_configurator/engine.py`, `src/llm_configurator/speed.py`, `src/llm_configurator/learning.py`, `tests/test_engine.py`, `tests/test_quality_comparison.py`, `tests/test_guided_requirements.py`, `tests/test_speed.py`, `tests/test_learning.py` | Memory/speed model for KV compression, MoE offload, unified memory, sliding window; plain verdicts; learning from measurements; community evidence |
| `community` | `src/llm_configurator/community.py`, `community/README.md`, `community/results.json`, `tests/test_community.py` | Opt-in anonymous result sharing and import |
| `ui-run` | `src/llm_configurator/static/index.html`, `static/app.js`, `static/style.css`, `static/wizard.js`, `static/selects.js`, `static/run.js` (new), `static/jobs.js` (new), `tests/ui-flow.test.cjs`, `tests/ui-run.test.cjs`, `package.json`, `package-lock.json` | "Get it running" panel: runtime, download, test, tune, export, start server; jobs tray; plain verdict badges |
| `ui-quality` | `src/llm_configurator/static/quality.js`, `static/quality.css`, `tests/ui-quality.test.cjs` | Quality panel: quiz, blind prompt compare, compression-loss check, local models list |
| `api` | `src/llm_configurator/server.py`, `src/llm_configurator/cli.py`, `src/llm_configurator/app.py`, `tests/test_server.py`, `tests/test_cli.py`, `tests/test_api_v04.py` | HTTP endpoints (§5) and CLI commands (§6) wiring every module together |
| `docs` | `README.md`, `docs/*.md` (not `docs/v0.4/`), `CHANGELOG.md`, `pyproject.toml` (except version/package-data lines, which are already set), `.github/workflows/*`, `LICENSE` only if the owner asks | Short plain README, detailed docs pages, CI matrix (macOS added), packaging for `pipx`/`uvx`, a release workflow that is **never** auto-triggered except by a version tag |
| `harness` | `scripts/*`, `tests/integration/*`, `docs/v0.4/llama-cpp-facts.md` | Build a **real** llama.cpp from PyPI sources plus a tiny generated GGUF. Run real-runtime integration tests (skipped unless `LLM_CONFIG_REAL_RUNTIME` is set). Record verified CLI flags and output formats |

Everyone also owns `docs/v0.4/handoff/<ws>.md`.

The **lead** owns `domain.py`, `storage.py`, `jobs.py`, `launch.py`, `__init__.py`, `tests/test_foundations.py`, `docs/v0.4/CONTRACT.md`, and does all merging and integration.

---

## 4. Module contracts

The signatures are binding. Internal details are yours. Extra optional keyword args and extra dict keys are fine.

### 4.1 `runtime_install.py` (ws `runtime`)

Makes llama.cpp "just there". It downloads the official prebuilt release from `https://api.github.com/repos/ggml-org/llama.cpp/releases/latest`.

- `detect(store) -> dict`
  - Returns: `{"installed": bool, "source": "managed"|"configured"|"path"|None, "directory": str|None, "version": str|None, "build": int|None, "backend": "cuda"|"vulkan"|"metal"|"rocm"|"cpu"|"unknown"|None, "binaries": {"llama-server": argv|None, "llama-bench": argv|None, "llama-perplexity": argv|None, "llama-cli": argv|None}, "warnings": [str]}`
  - Search order: CLI-configured dir (`settings.runtime_dir`), managed install under `runtime_dir(store)`, then `PATH` (includes Homebrew).
  - The version comes from `llama-server --version` (short timeout).
- `choose_asset(release: dict, hardware: dict, system: str = platform.system(), machine: str = platform.machine()) -> dict`
  - Returns `{"name", "url", "size", "sha256", "backend", "reason"}`. Raises `ValueError` when nothing fits.
  - Choice rules:
    - Windows: CUDA build if an NVIDIA GPU is present (plus the matching `cudart` asset), otherwise Vulkan, otherwise CPU.
    - macOS: arm64 (Metal) or x64.
    - Linux: Vulkan if a GPU is present, otherwise CPU; arm64 where published.
  - Use the asset's `digest` (`"sha256:…"`) field. **Refuse assets without a digest** unless the caller passes `allow_unverified=True` (CLI flag only).
- `install(store, hardware, progress=None, cancel=None, release=None, allow_unverified=False) -> dict`
  - Downloads (resumable if practical), verifies SHA256, and does a **safe extract** (reject absolute paths, `..` and symlinks escaping the target) into `runtime_dir(store)/<tag>-<backend>/`.
  - Then marks binaries executable, runs a `--version` smoke check, writes `manifest.json`, and returns `detect(store)`.
- `install_archive(store, archive_path, progress=None, cancel=None) -> dict`: offline install from a local zip or tar.gz.
- `use_directory(store, directory) -> dict`: point at an existing build (CLI only). Validates that it contains `llama-server`.
- `binary(store, name) -> list[str]`: argv prefix. Raises `ValueError("llama.cpp is not installed yet. Click Install runtime or run: llm-config runtime install")`.

### 4.2 `downloads.py` + `runtime.py` (ws `downloads`)

- `plan(variant, directory) -> dict`: `{"files": [{"filename", "size_bytes", "present": bool, "partial_bytes": int}], "total_bytes", "remaining_bytes", "disk_free", "enough_space": bool}`. Requires 1 GiB of headroom.
- `download_variant(variant, directory, progress=None, cancel=None, token=None) -> Path`
  - Downloads every file in `variant.all_files()`. **Resumes** from `<name>.part` via HTTP Range; if the server ignores Range (`200` instead of `206`), it restarts safely.
  - Verifies size and SHA256 per file. Allows HTTPS redirects only and strips `Authorization` on cross-host redirects (keep the existing behaviour). Honours `HF_ENDPOINT` (default `https://huggingface.co`) and `HF_TOKEN`.
  - Progress reports total bytes across shards and speed (`bytes_per_second`, `eta_seconds`).
  - Returns the **first file's** path, which is what llama.cpp loads for split GGUFs.
  - On cancel, keeps `.part` files for a later resume.
- `remove_variant(variant, directory) -> int`: bytes freed. Deletes only this variant's files and `.part` files inside `directory`.
- `runtime.download(variant, directory)` keeps working (it now delegates). `runtime.bench` stays compatible.

### 4.3 `gguf.py` + `discover.py` (ws `discover`)

- `gguf.read_metadata(path) -> dict`
  - Pure-Python GGUF v2/v3 header parser. It skips big arrays (the token list) without loading them and has a hard cap on bytes read.
  - Returns `{"version", "tensor_count", "metadata": {scalar keys only}, "architecture", "summary": {"name", "layers", "kv_heads", "head_dim", "context_length", "experts", "active_experts", "sliding_window", "file_type", "quant", "split_count"}}`.
- `gguf.variant_from_file(path, sha256=None) -> Variant`: builds a `source="local"` variant from the header so users can bring any GGUF. It rejects unsupported architectures with a plain message.
- `discover.locations() -> list[{"source": "hf_cache"|"lmstudio"|"ollama"|"models_dir"|"custom", "path": str, "exists": bool}]`. Covers Windows, macOS and Linux defaults, and honours `HF_HOME`/`HF_HUB_CACHE`, `OLLAMA_MODELS` and LM Studio's folders.
- `discover.scan(store, extra_dirs=(), progress=None, cancel=None) -> list[dict]`
  - Returns local file records `{"path", "size_bytes", "sha256"|None, "source", "variant_id"|None, "gguf": summary|None, "mtime", "verified": bool}` and saves them under `local_files`.
  - Matches catalogue variants cheaply first:
    - Hugging Face cache blobs and Ollama `sha256-…` blobs already carry the hash.
    - Otherwise matches on filename and size.
  - It never hashes gigabytes during a scan unless asked.
- `discover.find_for_variant(store, variant, verify=True) -> Path | None`: a verified local copy, including all shards, or `None`. Uses `hash_cache`.
- `discover.hash_cached(store, path, progress=None, cancel=None) -> str`

### 4.4 `llama_server.py` + fake llama.cpp (ws `server`)

- `free_port() -> int`
- `class LlamaServer(command, config, log_path=None)`
  - `command` is the argv prefix for llama-server. `config` is a launch config.
  - `start(timeout=300, progress=None, cancel=None)`: spawns with `launch.server_args` and `launch.server_env`, picks a free port if `config["port"]` is None, and polls `/health` until ready. Raises `ValueError(plain reason)` on exit or timeout.
  - `.base_url`, `.pid`, `.ready()`
  - `chat(messages, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300) -> {"text", "finish_reason", "timings": {"prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"}, "usage": {...}}`. Uses `/v1/chat/completions` and reads llama.cpp's `timings` field.
  - `complete(prompt, max_tokens=..., ...)` for the raw `/completion` endpoint (same shape), `tokenize(text) -> list[int]`, `stop(timeout=10)`, context manager, `log_tail(chars=4000)`.
  - `failure_reason() -> str | None` maps stderr to plain messages: out of memory, unsupported model architecture, file missing or corrupt, port in use, and GPU backend unavailable.
- `class PeakMemory(pid, gpu_backend=None, interval=0.25)`: `start()`, `stop() -> {"peak_ram_bytes", "peak_vram_bytes"|None, "samples"}`. RAM is the process-tree RSS. VRAM comes from `nvidia-smi --query-compute-apps=pid,used_memory` for CUDA, and is `None` otherwise (unified memory is counted as RAM).
- `class ServerRegistry()`: the user's one long-running server. `start(command, config, progress=None, cancel=None) -> status`, `stop() -> status`, `status() -> {"running", "base_url", "openai_base_url", "pid", "config", "started_at", "model", "log_tail"}`.
- **`tests/fixtures/fake_llama.py`**: an executable Python script that imitates llama.cpp. Every other workstream will use it after merge, so make it faithful to real formats.
  - It is selected with `--as server|bench|perplexity|cli`, or by argv[0] name.
  - `server` mode: parses real llama-server flags; HTTP `/health` (503 while "loading", then 200), `/v1/chat/completions` (with `timings` and `usage`), `/completion`, `/tokenize`; deterministic answers (for example it answers arithmetic, and echoes a "needle" if present in the prompt).
  - `bench` mode: accepts comma lists for `-p -n -d -ngl -t -b -ub -fa -ctk -ctv -ncmoe -r` and prints `-o json` rows in llama-bench's real field names (`n_prompt`, `n_gen`, `n_depth`, `avg_ts`, `stddev_ts`, `n_gpu_layers`, `n_threads`, `type_k`, `type_v`, `flash_attn`, `n_batch`, `n_ubatch`, `build_commit`, `build_number`, `model_filename`, …).
  - `perplexity` mode: prints realistic `--kl-divergence` output.
  - `--version` output is realistic too.
  - Env knobs: `FAKE_LLAMA_FAIL=oom|arch|port`, `FAKE_LLAMA_TPS=<float>`, `FAKE_LLAMA_LOAD_SECONDS=<float>`, and speed that depends on flags (so the tuner has something to find, for example `-t` near 8 and `-fa 1` faster).
  - Also `tests/fixtures/__init__.py` with `fake_command(mode) -> list[str]`.

### 4.5 `testing.py` (ws `testing`)

- `smoke_test(server_command, config, progress=None, cancel=None) -> dict`
  - Returns `{"ok", "stage_failed": None|"start"|"reply", "load_seconds", "reply", "checks": [{"name", "ok", "detail"}], "message", "log_tail"}`.
  - One fixed question with a checkable answer, plus sanity checks: the reply isn't empty, isn't the same token repeated, isn't a template leak (for example raw `<|im_start|>` text), and is valid UTF-8.
- `speed_test(bench_command, server_command, variant, config, hardware, progress=None, cancel=None) -> dict`
  - Returns `{"measurement": <record §2.3, kind="speed_test">, "summary": {"pp_tps", "tps", "ttft_s", "depth"}, "memory": {"estimated_ram_bytes", "estimated_vram_bytes", "peak_ram_bytes", "peak_vram_bytes", "within_estimate": bool|None, "note"}}`.
  - Reading and writing speed come from llama-bench at the requested depth (`-p 512 -n 128 -d <context-640>`, clamped). The first-word delay and peak memory come from one llama-server run with a realistic prompt.
  - The memory estimate comes from `engine.allocations` (import inside the function).
- `run_tests(store, variant, config, commands, kind="full"|"smoke"|"speed", hardware=None, progress=None, cancel=None) -> dict`: runs the chosen tests and `store.append("measurements", ...)` for speed. Returns `{"smoke": ..., "speed": ..., "verdict": "works"|"works_slowly"|"failed", "verdict_text": plain sentence}`.

### 4.6 `tuner.py` (ws `tuner`)

- `bench_args(config, n_prompt=512, n_gen=128, depth=0, repetitions=2, sweep=None) -> list[str]`: maps a launch config and a sweep (`{"threads": [4, 8], "flash_attn": ["on", "off"], ...}`) to llama-bench flags using comma lists.
- `tune(bench_command, variant, base_config, hardware, budget_seconds=300, goal="generation"|"balanced"|"prompt", memory_check=None, progress=None, cancel=None) -> dict`
  - Returns `{"best": launch config, "baseline": {"tps", "pp_tps"}, "best_result": {"tps", "pp_tps"}, "improvement": float (ratio), "trials": [{"changes": {...}, "tps", "pp_tps", "seconds", "status": "ok"|"failed"|"skipped_memory"}], "stopped": "budget"|"converged"|"cancelled", "notes": [plain str]}`.
  - Method: start from `base_config`, then coordinate search (one knob at a time, keep improvements).
  - Knobs: `gpu_layers` (only for split placements), `threads`, `batch`/`ubatch`, `flash_attn`, `cache_type_k/v` (only when the user allowed compressed notes, or when memory is tight), and `n_cpu_moe` (MoE only).
  - Uses llama-bench comma sweeps to test several values per process, and runs a confirmation pass on the winner.
  - `memory_check(config) -> bool` (injected; defaults to `engine.allocations` against current free memory) skips unsafe trials.
  - Results are stored with `store.append("tuned", ...)` by the caller (api), not the tuner. Tuning never exceeds the budget by more than one trial.

### 4.7 `evals.py` + `evals/*.json` (ws `evals`)

- `evals/general.json`, `coding.json`, `agentic.json`, `documents.json`
  - **Original** items only: don't copy benchmark datasets. 30–60 items each, format `{"version", "items": [{"id", "prompt", "answer" | "accept": [regex], "kind"}]}`.
  - Items must be checkable **without executing model-written code**. Coding uses "what does this print" and "fix this line" with exact-match answers. Agentic checks tool-call JSON formatting. Documents uses synthetic long-text recall.
- `run_quiz(chat, workload, limit=None, progress=None, cancel=None) -> dict`
  - `chat(messages, max_tokens) -> {"text", ...}` is an injected callable (`LlamaServer.chat` in production).
  - Returns `{"workload", "correct", "total", "score", "ci_low", "ci_high" (Wilson 95%), "items": [{"id", "ok", "expected", "got"}], "seconds", "note"}`.
- `needle_test(chat, tokenize, context_tokens, positions=(0.1, 0.5, 0.9), progress=None, cancel=None) -> dict`: builds synthetic filler text and checks recall of an inserted fact at each position.
- `start_comparison(store, prompts, labels) -> comparison_id`, `record_output(store, comparison_id, label, prompt_index, text, timings)`, `blinded(store, comparison_id) -> {"items": [{"prompt", "outputs": [{"slot": "A", "text"}]}]}` with a shuffled per-item order and no labels, `vote(store, comparison_id, item, slot) -> dict`, `reveal(store, comparison_id) -> dict` (mapping plus tallies).
- `run_comparison(store, prompts, runners: dict[label -> chat callable factory], max_tokens=512, progress=None, cancel=None) -> comparison_id`: runs models **one after another**, never at the same time, because of memory.

### 4.8 `quantcheck.py` (ws `quantcheck`)

- `parse_kld_output(text) -> dict`: `{"mean_kld", "median_kld", "kld_99", "same_top_p", "ppl_base", "ppl", "mean_delta_p"}`. Values missing from the output stay `None`. It must be robust to llama.cpp version differences.
- `kl_check(perplexity_command, reference_path, candidates: dict[label -> path], corpus_path=None, context=512, chunks=None, config=None, progress=None, cancel=None) -> dict`
  - Returns `{"reference", "results": {label: parsed + {"plain": sentence}}, "corpus", "notes"}`.
  - Step 1 writes the reference logits with `--kl-divergence-base <tmpfile>`. Step 2 runs `--kl-divergence` per candidate.
  - It deletes temp files, warns about their size and disk usage, and interprets results in plain words (for example "picks a different top word about 4% of the time").
- `data/quantcheck_corpus.txt`: about 30–60 KB of **original or public-domain** English prose plus a little code.

### 4.9 `export.py` (ws `export`)

- `formats() -> list[{"id", "label", "description"}]`
- `export(config, variant, fmt, platform="posix"|"windows", server_command=None) -> {"format", "filename", "content", "instructions": [plain str], "notes": [str]}`. The formats:
  - `llama-server`: a script for bash or PowerShell with correct quoting. Uses `launch.server_args`.
  - `ollama`: a Modelfile with `FROM <path>`, `PARAMETER num_ctx`, `num_gpu` (runtime layers), `num_thread` and `num_batch`, plus the `ollama create` command. Note what Ollama can't express (for example KV compression per model, which is an Ollama server env setting, or `n_cpu_moe`).
  - `docker-compose`: `ghcr.io/ggml-org/llama.cpp:server` or the CUDA variant, with a volume and port.
  - `openai-python`: a client snippet using the local base URL.
  - `continue`: a Continue config snippet.
  - `open-webui`: connection steps.
  - `lmstudio`: plain settings to enter in LM Studio's load panel.
- Content must be deterministic (no timestamps) so tests can snapshot it.

### 4.10 `hardware.py` (ws `hardware`)

`scan()` keeps every existing key (other code depends on them).

- GPU dicts gain `backend` (`cuda`|`metal`|`rocm`|`vulkan`|`unknown`), `vendor`, and `unified` (bool).
- New top-level keys:
  - `cpu_name`: friendly, from the registry on Windows, `sysctl machdep.cpu.brand_string` on macOS, `/proc/cpuinfo` on Linux
  - `cpu_features`: `{"avx2", "avx512", "neon", ...}` booleans or `None`
  - `unified_memory`: bool
  - `platform`: `{"system", "machine", "release"}`
- **Apple Silicon:** one GPU entry `{"index": 0, "uuid": "apple-<chip>", "name", "backend": "metal", "unified": True, "total": <GPU wired limit>, "available": min(wired limit, ram_available)}`. The wired limit comes from `sysctl iogpu.wired_limit_mb` when set, otherwise macOS's default share of RAM. Document the rule you use and why.
- **AMD on Linux:** read `/sys/class/drm/card*/device/mem_info_vram_total|used` (no tools needed) and the name via `rocm-smi`/`amd-smi` if present.
- **Intel / other and Windows non-NVIDIA:** list them with unknown free memory, and never invent numbers.
- Keep NVIDIA behaviour identical. The fingerprint may change only for newly detected GPUs.

### 4.11 `catalogue.py` + `catalogue.json` (ws `catalogue`)

- Grows `catalogue.json` to **30–50** popular local models, each with `{"base_repo", "gguf_repo", "aa_slug": null, "family", "tags": [...], "license"?}`.
  - Covers Qwen3 dense + MoE (30B-A3B, and 235B only if sharded support works), Qwen2.5-Coder, Llama 3.x, Gemma 3, Mistral / Mistral Small, Phi-4, gpt-oss-20b/120b, DeepSeek-R1 distills, and small models (SmolLM, Llama 3.2 1B/3B).
  - Uses reputable GGUF publishers (official orgs, `ggml-org`, `unsloth`, `bartowski`, `lmstudio-community`).
  - **Verify each repo exists** with WebFetch/WebSearch if those tools work for you. Record unverifiable ones in your handoff.
- `fetch_variants(entry)` gains:
  - **sharded** files: group `-0000k-of-0000n`, sum sizes, and fill `files` with per-shard sha256; the variant `filename` is the first shard
  - **MoE** fields (`num_experts`/`num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size`, and so on → `experts`, `active_experts`, `expert_fraction`, `active_parameters`)
  - **sliding window** (`sliding_window`, `layer_types`/`sliding_window_pattern`)
  - nested `text_config` (Gemma 3)
  - more quants (`domain.QUANT_BYTES_PER_PARAMETER` keys)
  - `parameters` from the safetensors metadata when present
- Also handles Hugging Face quant folders (for example `Q4_K_M/xxx-00001-of-00002.gguf`).
- `add_entry(store, base_repo, gguf_repo) -> dict` and `remove_entry(store, base_repo) -> dict`: user-added repos, validated and saved to the user catalogue copy.
- Keeps `demo_variants()`, `definitions()`, `refresh()`, `apply_scores()` and `test_connection()` signatures.

### 4.12 `engine.py` + `speed.py` + `learning.py` (ws `engine`)

- `allocations(variant, context, users, gpu_layers, kv_cache_type="f16", n_cpu_moe=0, unified=False) -> {"ram", "vram", "kv_total", ...}`
  - KV uses `KV_BYTES_PER_ELEMENT` and sliding-window layers.
  - For MoE, `n_cpu_moe` layers keep their expert tensors in RAM.
  - `unified=True` (Apple) puts everything in one pool and avoids double counting.
  - Existing positional calls must still work.
- `recommend(variants, hardware, requirements, measurements=(), calibration=None, community=(), tuned=()) -> report`.
- New candidate keys:
  - `kv_cache_type`, `n_cpu_moe`, `unified_memory`
  - `launch` (a launch-config fragment without `model_path`, for `launch.from_candidate`)
  - `verdict`: `runs_well` | `runs_slowly` | `too_slow` | `unknown`
  - `verdict_text`: one plain sentence
  - `evidence`: `measured` | `interpolated` | `tuned` | `community` | `estimated` | `none`
  - `community`: `{median_tps, n}` or `None`
  - `tuned`: the best tune result summary or `None`
- The candidate `id` stays stable and unique. Extend the format if needed (for example add a `|kv|moe` suffix).
- **Measurement matching** is looser but honest: exact matches first. Otherwise interpolate between measurements of the same variant, hardware and placement at other contexts (labelled "interpolated", never "measured").
- MoE speed uses active bytes, not total bytes. Candidates include MoE CPU-offload placements when they fit.
- `learning.py`: `fit_efficiency(measurements, hardware, calibration) -> dict | None` narrows `speed.estimate` ranges using this machine's real results. It is labelled as "adjusted from N local measurements".
- `recommend` must stay fast (under 0.5 s for 50 models × 4 quants on typical hardware). Add a test.

### 4.13 `community.py` (ws `community`)

- `anonymize(measurement, hardware, variant) -> dict`
  - Allowlisted fields only: `variant` (repo, filename, sha256, quant), hardware class (CPU name, GPU name, backend, VRAM/RAM rounded to 4 GiB buckets, OS family), settings, `tps`, `pp_tps`, `ttft_s`, `context`, `depth`, `runtime` build, and the `schema` version.
  - No hostnames, usernames, paths, PIDs, UUIDs, fingerprints, or exact timestamps (month precision only).
- `share_payload(store, measurement_ids) -> {"json": str, "issue_url": str}`: a prefilled GitHub issue URL on `Eyad-3D/LLM-Configurator` with the "community-results" label. The user reviews and submits it themselves, so the app never posts.
- `import_records(store, source=None, text=None, progress=None, cancel=None) -> dict`
  - Default source: `https://raw.githubusercontent.com/Eyad-3D/LLM-Configurator/main/community/results.json`.
  - HTTPS only, a size cap, strict schema validation, and it drops bad rows.
  - Returns `{"source", "fetched_at", "records", "rejected"}` and saves it to the store.
- `evidence(records, variant, hardware, config) -> {"median_tps", "n", "similar": "same_gpu"|"same_cpu"|"same_class", "range": [lo, hi]} | None`
- `community/results.json`: `{"schema": 1, "records": []}`. `community/README.md` explains the format and privacy.

### 4.14 UI (ws `ui-run`, `ui-quality`)

- The UI stays vanilla JS with no build step and the strict CSP (`script-src 'self'`, **no inline scripts or handlers**).
- Static files served (the api ws adds them to the allowlist): `index.html, app.js, wizard.js, selects.js, style.css, run.js, jobs.js, quality.js, quality.css`.
- `ui-run` adds to each recommendation card a **"Get it running"** button that opens a step panel:
  1. Runtime ready? (Install button)
  2. Download (size, disk space, progress bar, pause/cancel, "found on your disk" reuse)
  3. Test (smoke plus speed; shows reading speed, first-word delay, writing speed, and memory vs estimate)
  4. Tune (time budget: 1/5/15 min; progress; before/after)
  5. Check quality (mounts `QualityPanel`)
  6. Use it (export tabs with a copy button; Start/Stop server showing the OpenAI address)

  Plus a small **jobs tray**, and plain **verdict badges** on cards (`Runs well` / `Runs slowly` / `Too slow` / `Not tested yet`).
- `ui-quality` exposes `window.QualityPanel = { mount(container, ctx) }` where `ctx = { api(path, options) -> Promise<json>, candidate, candidates, workload, onJob(job) }`. Tabs: **Quick quiz**, **Try my prompts** (blind A/B/C voting, then reveal), **Compression check** (pick reference and candidates). It also exposes `window.LocalModels = { mount(container, ctx) }` to list discovered local files and trigger a scan.
- `jobs.js` (owned by ui-run) exposes `window.Jobs = { watch(jobId, onUpdate) -> stop(), list() }`, polling `/api/jobs/<id>` every ~700 ms. `ui-quality` may call it through `ctx.onJob`, or poll `/api/jobs/<id>` directly.
- Use the existing token header (`X-Session-Token`) and fetch conventions from `app.js`.
- Respect `prefers-reduced-motion` and the light/dark theme. Keep the layout usable at 360 px wide. Keyboard accessible.

---

## 5. HTTP API (ws `api` implements; UI consumes)

All endpoints are loopback-only and require the session token (existing rules). Bodies are JSON (≤ 64 KiB). Errors return `{"error": "plain message"}` with status 400, 404 or 409. Long operations return **202 plus a job** (§2 job dict).

| Method & path | Body | Returns |
|---|---|---|
| GET `/api/runtime` | – | `runtime_install.detect()` |
| POST `/api/runtime/install` | `{}` | 202 job → result detect() |
| GET `/api/jobs` | – | `{"jobs": [...]}` |
| GET `/api/jobs/<id>` | – | job |
| POST `/api/jobs/<id>/cancel` | `{}` | job |
| GET `/api/local-models` | – | `{"files": [...], "locations": [...]}` |
| POST `/api/local-models/scan` | `{}` | 202 job → files |
| POST `/api/downloads/plan` | `{"variant_id"}` | plan + `{"local_copy": path-free bool}` |
| POST `/api/downloads` | `{"variant_id"}` | 202 job → `{"reused": bool, "bytes"}` (never returns absolute paths to the page) |
| POST `/api/downloads/remove` | `{"variant_id"}` | `{"freed_bytes"}` |
| POST `/api/test` | `{"candidate_id", "kind": "smoke"|"speed"|"full"}` | 202 job → testing result |
| POST `/api/tune` | `{"candidate_id", "budget_seconds": 60–1800, "goal"}` | 202 job → tune result |
| POST `/api/quality/quiz` | `{"candidate_id", "workload"?, "include_needle"?: bool}` | 202 job |
| POST `/api/quality/compare` | `{"candidate_ids": [2–3], "prompts": [1–5 strings ≤ 4000 chars]}` | 202 job → `{"comparison_id"}` |
| GET `/api/quality/compare/<id>` | – | blinded items |
| POST `/api/quality/vote` | `{"comparison_id", "item", "slot"}` | tallies |
| POST `/api/quality/reveal` | `{"comparison_id"}` | mapping + tallies |
| POST `/api/quality/quant-check` | `{"reference_variant_id", "variant_ids": [1–3]}` | 202 job |
| GET `/api/quality/results` | – | `{"results": [...]}` |
| POST `/api/export` | `{"candidate_id", "format", "platform": "posix"|"windows"}` | export result (uses a placeholder path and a note if the model isn't downloaded yet) |
| GET `/api/serve` | – | ServerRegistry.status() |
| POST `/api/serve/start` | `{"candidate_id"}` | 202 job → status |
| POST `/api/serve/stop` | `{}` | status |
| GET `/api/community` | – | `{"records", "source", "fetched_at"}` |
| POST `/api/community/import` | `{}` | 202 job |
| POST `/api/community/share` | `{"measurement_ids": [...]}` | `{"json", "issue_url"}` |

- `candidate_id` refers to a candidate in the **most recent** `/api/recommend` report. The server keeps the latest report's candidates in memory; if the id is unknown, it returns 409 "Compare again first".
- Compute-heavy jobs (test, tune, quiz, compare, quant-check, serve start) use `exclusive="compute"`. Downloads use `exclusive="download:<variant_id>"`.
- `/api/recommend` passes `community` and `tuned` records to `engine.recommend`.
- Demo mode never downloads, tests or serves. Those endpoints return 409 with a plain explanation.

## 6. CLI (ws `api`)

Existing commands keep working. New and extended commands:

```
llm-config runtime status | install [--allow-unverified] | install-archive PATH | use DIR
llm-config download VARIANT_ID [--directory DIR] [--yes]      # resumable, progress bar, reuses local copies
llm-config local [--scan] [--dir DIR ...] [--add PATH]        # discover / register local GGUFs
llm-config test VARIANT_ID [--context N] [--gpu-layers N] [--kv f16|q8_0|q4_0] [--kind smoke|speed|full]
llm-config tune VARIANT_ID [--context N] [--gpu-layers N] [--budget SECONDS] [--goal generation|balanced|prompt]
llm-config quiz VARIANT_ID [--workload W] [--needle]
llm-config quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...]
llm-config export VARIANT_ID --format FMT [--platform posix|windows] [--context N] [--gpu-layers N] [--tuned]
llm-config run VARIANT_ID [--context N] [--gpu-layers N] [--port P] [--tuned]   # foreground OpenAI-compatible server
llm-config community import [--source URL] | share MEASUREMENT_ID ... | status
llm-config settings --models-dir DIR
llm-config models add BASE_REPO GGUF_REPO | remove BASE_REPO
```

Every command that uses llama.cpp resolves binaries via `runtime_install.binary(store, name)`. `--tuned` applies the best stored tune for that variant, context and placement.

## 7. Waves

1. **Wave 1 (now, parallel):** every workstream builds and tests its own files against this contract.
2. **Wave 2 (lead):** merge all branches into `claude/keen-volta-7mnrm5`, resolve seams, and run the harness's real-runtime tests. Fix contract mismatches found in handoffs.
3. **Wave 3:** multi-agent review (correctness, security, cross-platform, UX copy), then fixes, the full test suite, and push.
