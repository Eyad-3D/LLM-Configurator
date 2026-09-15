# LLM Configurator

A local Python application that compares **model + quantisation + context + CPU/GPU placement** against current resources and workload requirements. Includes a browser interface, CLI, Hugging Face metadata retrieval, Artificial Analysis integration, and an optional local llama.cpp benchmark runner.

This is a working **v0.2 engineering prototype**. Memory estimates require calibration against real inference workloads. Unknown speed and quantisation quality are explicitly labelled; the application does not invent benchmark scores.

## Run it

Requires Python 3.10 or newer. Run on the computer whose hardware you want to configure. A remotely hosted instance measures the server, not your laptop.

### Windows / PowerShell

```powershell
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\python.exe -m llm_configurator serve
```

### Linux / macOS

```bash
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python -m llm_configurator serve
```

The interface opens at **http://127.0.0.1:8765**. It is local-only and is not a hosted website. Stop with Ctrl+C. Use `--port 8766` if the port is already occupied.

To explore without internet access or an API key:

```bash
python -m llm_configurator serve --demo
```

Use the Python executable from your environment. Demo mode uses **fictional models with real hardware measurements**, clearly marked throughout. No demo scores are provided; demo artifacts cannot be downloaded or benchmarked.

## First real comparison

1. Click **Get started**. Hardware detection runs in the background.
2. Answer one question per screen: main use, quality/speed priority, context needs, active users, and whether other applications will stay open. Exact tokens, tok/s and memory reserves live under advanced controls.
3. Review your answers. Each **Edit** button opens just that question and returns to the review.
4. After the review, optionally connect Artificial Analysis. Choose **Continue without rankings** to skip benchmark requests and exclude even previously cached scores from this comparison. You can enable rankings later.
5. See up to **three recommendations** with distinct models where possible. Cards show workload rank (when available), context, local speed evidence, deployment mode and a brief explanation. **View details & setup** reveals memory breakdown, sources and launch instructions. **Compare all configurations** expands the list.
6. Use **Adjust my answers** to edit a specific answer and recalculate, keeping your other answers and ranking choice. **Ranking settings** lets you change that choice separately.

The app automatically fetches missing model metadata when generating the first recommendations. No weights are downloaded. A saved key can be reused at the optional ranking step; the key prompt never appears before the questions. Benchmark mappings still require choosing the correct model/evaluation entry under **Match benchmark entries**; missing scores are not fabricated.

The initial curated catalogue covers the dense Qwen3 0.6B, 1.7B, 4B, 8B, 14B and 32B repositories with Q4_K_M, Q5_K_M, Q6_K and Q8_0 variants where single-file artifacts are available. Availability is fetched from Hugging Face rather than hardcoded. Sharded variants are excluded in this release.

## Artificial Analysis integration

Open **Benchmark settings** in the interface and paste your Artificial Analysis key into the masked field. Use **Test connection** to validate it, then **Save key** and refresh metadata. Testing alone does not save or apply a newly entered key.

**Remember on this computer** stores it in the native OS credential store (Windows Credential Manager, macOS Keychain or supported Linux vault). Uncheck it for explicit process-only storage. If the vault is unavailable or locked, saving fails with a clear message; the app never silently writes a plaintext credential file. The field clears after saving. **Remove key** clears session and accessible saved credentials. Saved keys are shared by this app's instances under the same OS account, independently of the catalogue cache directory.

The saved key works on subsequent app launches and CLI refreshes. An explicitly entered session key takes precedence over a saved key; a saved key takes precedence over the optional `AA_API_KEY` environment variable. Removing credentials from the GUI does not remove an environment variable. Session-only use does not replace a previously saved key: that saved key becomes active again after restart.

The adapter uses the paginated `/api/v2/language/models/free` endpoint. Keys are passed from the local form to the loopback Python server, then to Artificial Analysis over HTTPS. They are never returned by status endpoints, placed in browser storage, or stored in SQLite/logs. Hugging Face public metadata normally needs no token; `HF_TOKEN` remains optional for repositories requiring access.

After refreshing, expand **Match benchmark entries** and choose the exact evaluation entry, including reasoning mode, for each base model. Matches deliberately start empty: similar names are insufficient evidence of identical models/settings. Save the match and compare again. CLI equivalents are `benchmarks` and `map`.

Each recommendation card prominently shows the workload index, **#rank of rated models**, with the gap in index points from the best eligible score available under details. The comparison counts distinct base models across all eligible results, not quantisation/context duplicates or only the displayed shortlist. Ties share competition ranks. Missing scores are labelled **Not ranked**, with a link to benchmark settings; incompatible benchmark versions withhold ranks. Rank is a base-model quality comparison, not local throughput or quantisation quality. The summary reports missing-score coverage. Speed-verified configurations still take priority in card ordering.

Scores are attributed to **[Artificial Analysis](https://artificialanalysis.ai)**. General, coding and agentic indices are used for their corresponding workloads. Documents currently use general intelligence as a proxy. Different index versions are not numerically ranked together. Missing values stay unknown, including a missing quantisation-specific evaluation. API data usage remains subject to Artificial Analysis's terms; no third-party score dataset is bundled in this repository.

## How memory estimation works

All internal measurements use bytes; the UI displays **GiB** (2^30 bytes).

The estimator supports dense `qwen2`, `qwen3`, and `llama` configurations with full-attention FP16 KV-cache accounting:

```text
KV bytes = 2 × layers × KV heads × head dimension × 2 bytes × context × active users
```

The first factor of two accounts for K and V. It does not use the total number of query heads for grouped-query attention. Context includes input, history, reasoning and generated output. No shared-prefix savings are assumed.

The current estimator adds:

- 10% over file weight size for placement/metadata uncertainty.
- 0.75 GiB plus 0.20 GiB per active user for buffers in each active memory pool.
- 0.5 GiB of host staging allowance for GPU execution.
- User-configurable headroom, defaulting to 2 GiB RAM and 0.5 GiB VRAM.

These are **explicit engineering assumptions, not fitted constants or guaranteed upper bounds**. Buffers, non-layer tensors, loading peaks, runtime versions and architecture-specific behaviour can exceed them. GPU layer placement is approximated proportionally; file-backed mappings are not assumed to require a permanent full duplicate of weights in RAM. Test a chosen configuration before relying on it.

Available OS memory already accounts for current usage; OS usage is not subtracted twice. Reclaim scenarios add only 75% of accessible process USS (private memory). RSS is shown for reference but is not summed as reclaimable memory. Missing USS produces no reclaim credit. GPU memory does not receive speculative reclaim credit. Swap is never added to usable RAM.

The quality priority orders by available base-model score; speed priority orders by matching local tok/s measurements; balanced prioritises verified speed fits then quality. Unknown evidence stays unknown.

The search evaluates CPU-only, full GPU, and the largest fitting partial GPU allocation for each context. All intermediate layer allocations are checked for feasibility, but not all are returned. It returns contexts at or above the user's requested minimum, plus a **memory-only context ceiling** per configuration. That ceiling is not a speed guarantee or a measured long-context quality limit.

## Speed measurements

Hardware specifications and hosted API speed are **not** converted into fabricated local tok/s estimates. Without matching evidence, a candidate is marked **speed unverified**. The strict-speed checkbox excludes these candidates.

Install a compatible [llama.cpp](https://github.com/ggml-org/llama.cpp) build containing `llama-bench` for your hardware. It must support JSON output, `-d` / `--n-depth`, and device selection. No runtime binary is bundled or installed automatically.

1. Get an exact ID from `python -m llm_configurator models` or the interface.
2. Explicitly download the pinned GGUF with `download`. A SHA256 and size check verifies it.
3. Run `bench` on that file. It rechecks available resources, generates 128 tokens near the requested context limit, and records the average of three repetitions.
4. Compare again. A matching measurement can qualify a candidate against the requested tok/s.

```bash
python -m llm_configurator download 'EXACT_VARIANT_ID' --directory ./models
python -m llm_configurator bench 'EXACT_VARIANT_ID' \
  --model ./models/model.gguf --context 8192 --gpu-layers 0 \
  --executable /path/to/llama-bench
```

The interface supplies commands with actual IDs. On PowerShell use one line rather than Bash's backslash continuation. `--gpu-layers` counts transformer layers; a fully offloaded model automatically adds the output layer for the llama.cpp invocation. CPU-only uses explicit device `none`; NVIDIA execution selects one GPU by UUID. `bench` currently reserves 2 GiB RAM / 0.5 GiB VRAM regardless of exploratory UI settings.

Measurements are matched by exact file hash and variant revision, hardware/driver fingerprint, context, GPU placement, CPU threads and single-user operation. They expire after 30 days. The runtime build is retained in the result. They are historical synthetic measurements; changing workload, runtime, background load, thermals or power mode can alter speed. They do not measure TTFT, agent completion time, intelligence, peak memory, or concurrent-user throughput. Multi-user memory can be estimated, but multi-user speed stays unverified.

## CLI

```bash
python -m llm_configurator --help
python -m llm_configurator scan
python -m llm_configurator refresh
python -m llm_configurator models
python -m llm_configurator benchmarks
python -m llm_configurator map Qwen/Qwen3-8B EXACT_AA_SLUG
python -m llm_configurator recommend --workload coding --context 8192 --users 1 --min-tps 20
python -m llm_configurator recommend --reclaim-pids 1234 5678 --json
python -m llm_configurator recommend --strict-speed --output comparison.json
```

`llm-config` is an equivalent shortcut when your environment's scripts directory is on PATH. `--data-dir PATH` must precede the subcommand. The local SQLite cache is stored under `%LOCALAPPDATA%/LLMConfigurator` on Windows or `$XDG_DATA_HOME/llm-configurator` (default `~/.local/share/llm-configurator`) elsewhere. Override with `LLM_CONFIG_HOME`.

To extend the curated set, copy the bundled `src/llm_configurator/catalogue.json` into your data directory and edit repository pairs. Only supported architectures and single-file quantisations are accepted. Base and GGUF relationships are curated; the program does not assert that arbitrary repositories with matching names are equivalent. Failed refreshes preserve cached variants and display the problem. Metadata dates remain visible.

## Architecture

| Module | Responsibility |
|---|---|
| `domain.py` | Validated requirements and model variants |
| `hardware.py` | psutil process/system scan and NVIDIA telemetry |
| `catalogue.py` | Hugging Face + Artificial Analysis adapters |
| `engine.py` | Memory allocation, context search, constraints and ranking |
| `runtime.py` | Explicit verified downloads and llama-bench execution |
| `storage.py` | Local SQLite cache |
| `app.py` | Shared orchestration and benchmark mappings |
| `server.py`, `static/` | Loopback-only browser interface |
| `cli.py` | Command-line workflows |

The HTTP interface is loopback-only, checks Host/Origin/session tokens, and does not expose arbitrary shell commands, process termination, or model downloads. Resource/process data stays local. Exported browser reports omit process identities; raw CLI scans/reports include them. No telemetry is sent to a service; remote requests fetch model metadata and optional benchmark data.

## Tests and current limits

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
# UI navigation tests (development only; Node.js 20+)
npm ci
npm test
```

GitHub Actions runs tests on Windows and Linux with Python 3.10 and 3.12. Tests use fixtures for external APIs and do not need a GPU, API key or model download. They cover memory budgets, concurrency, context limits, workload ranking, benchmark identity, sharded-file exclusion, cache behaviour and local HTTP requests.

Not yet implemented: AMD/Intel/Apple GPU telemetry, unified-memory GPU placement, multi-GPU sharding, sharded model downloads, image understanding/generation workflows, licence filtering, quantisation quality evaluations, TTFT benchmarking, calibrated speed prediction for unseen hardware, automatic runtime installation, or an executable installer. The browser interface is responsive but the program itself must run locally. The hardware and performance paths need real Windows/NVIDIA validation before this should be treated as production-ready.

## Reference documentation

- [psutil](https://psutil.readthedocs.io/)
- [Hugging Face API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api)
- [Artificial Analysis API and attribution](https://artificialanalysis.ai/data-api/docs)
- [llama-bench usage](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md)

### Dropdowns and concurrency

All single-choice menus use the app's green theme, including dynamically loaded GPU and benchmark choices. They support arrow keys, Home/End, typing to locate options, Enter to select, and Escape to dismiss without changing the selection.

The setup asks for **concurrent sessions or agents**, including requests from human users. One person running three agents concurrently counts as three; sequential use of one session counts as one. This is the same concurrency budget used for KV-cache memory. The existing CLI/API field `users` is retained for compatibility and means concurrent active model requests. Context and speed targets are per active session, not per person.

The application list is ordered by measured resident memory usage (RSS), largest first. Each row shows used memory and a separately labelled conservative reclaim estimate; missing reclaim estimates do not hide measured usage.

### Background preparation

Starting guided setup launches a background Hugging Face metadata refresh while the user answers questions. This retrieves model sizes and architecture information, not model weights. A quiet status line reports progress. Editing answers reuses the same preparation job. The final comparison scans current hardware again and applies the latest answers.

Model and benchmark retrieval are separate stages: the optional ranking step refreshes only Artificial Analysis data and applies it to the prepared catalogue. Skipping rankings makes no benchmark request. With cached model data, results never wait for discovery. A first run without cached models must wait for discovery; slow networks cannot guarantee zero wait. Rankings update the displayed results asynchronously, using cached scores in the meantime when available. Failed sources retain cached model metadata and expose warnings with the results. Manual metadata refresh remains available.

The refresh API accepts independent boolean `include_models` and `include_scores` flags (both default to true for compatibility). The server serializes refresh jobs; a busy worker returns HTTP 409 so the client can retry its requested scope rather than mistake a different refresh for its own.

### Results latency

Version 0.2.3 removes network refreshes from the cached-results path. The final hardware scan reads current RAM and GPU availability and inspects memory details only for selected processes. NVIDIA telemetry has a two-second timeout. Recommendations remain usable during a slow or failed benchmark refresh; late responses cannot overwrite an edited or newer comparison. The five-second target depends on machine load and having a populated catalogue; it is not a guaranteed cold-start SLA.
