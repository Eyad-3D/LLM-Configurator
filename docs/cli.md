# Command line (`llm-config`)

Everything in the app can also be done from a terminal.

```bash
llm-config --help
llm-config COMMAND --help
```

`llm-config` and `python -m llm_configurator` are the same. (`llm-configurator` is an alias so `uvx llm-configurator` works.)

**Global option:** `--data-dir PATH` uses a different data folder. It must come **before** the command: `llm-config --data-dir ./test-data models`.

**IDs.** Many commands take a `VARIANT_ID`: the exact ID of one model file (model + compression level). Get IDs from `llm-config models`. Put IDs in quotes; they can contain characters your shell treats specially.

**Exit codes.** `0` success, `1` error (with a plain message on stderr), `130` cancelled with Ctrl+C.

## Start here

| Command | What it does |
|---|---|
| `serve [--port 8765] [--demo] [--no-browser]` | Start the browser app on `http://127.0.0.1:8765`. |
| `scan` | Print your hardware and running apps' memory use as JSON. |
| `calibrate` | Run the ~12-second hardware speed check and save it. Reads no model files. |
| `refresh` | Fetch model information (and rankings if you have a key). Does **not** download models. |
| `models` | List known models with their exact IDs. |
| `models add BASE_REPO GGUF_REPO` | Add your own Hugging Face model: the original repo and the repo with GGUF files. |
| `models remove BASE_REPO` | Remove a model you added. |

## Recommendations

```bash
llm-config recommend [options]
```

| Option | Meaning | Default |
|---|---|---|
| `--workload general\|coding\|agentic\|documents` | Main use | `general` |
| `--context N` | Tokens per chat | `8192` |
| `--users N` | Chats or agents running at the same time | `1` |
| `--min-tps N` | Speed you want, tokens per second | `15` |
| `--reserve-gib N` / `--gpu-reserve-gib N` | RAM / VRAM to keep free | `2` / `0.5` |
| `--gpu-index N` | Which graphics card | `0` |
| `--reclaim-pids PID ...` | Apps you'd close to free memory | none |
| `--priority balanced\|quality\|speed` | What matters most | `balanced` |
| `--strict-speed` | Only show configurations with a measured speed | off |
| `--no-rankings` | Ignore quality rankings | rankings on |
| `--demo` | Made-up models, real hardware | off |
| `--json` / `--output FILE` | Print / save the full report as JSON | off |

## Rankings

| Command | What it does |
|---|---|
| `benchmarks` | List Artificial Analysis entries you can match to models. |
| `map BASE_REPO SLUG` | Match a model to an exact ranking entry. Use `-` as the slug to clear. |

## The engine (llama.cpp)

| Command | What it does |
|---|---|
| `runtime status` | Is llama.cpp installed? Where, which version, which backend (CUDA, Metal, Vulkan, CPU…). |
| `runtime install [--allow-unverified]` | Download and install the official llama.cpp release for your system. Checks its fingerprint (SHA256). Refuses files without a published fingerprint unless you add `--allow-unverified`; only use that if you trust the source. |
| `runtime install-archive PATH` | Install from a llama.cpp zip / tar.gz you already downloaded (for offline machines). |
| `runtime use DIR` | Use a llama.cpp build you already have (it must contain `llama-server`). |

Search order: a folder set with `runtime use`, then the app's own install, then your `PATH` (including Homebrew).

## Models on disk

| Command | What it does |
|---|---|
| `download VARIANT_ID [--directory DIR] [--yes]` | Download a model (all pieces if it is split). Shows the size first; `--yes` skips the question. Resumes if interrupted. Reuses a copy already on your disk. Checks every file's fingerprint. Default folder: your models folder. |
| `local` | List GGUF model files found on your computer. |
| `local --scan [--dir DIR ...]` | Search the usual places (Hugging Face cache, LM Studio, Ollama, your models folder) plus any `--dir`. |
| `local --add PATH` | Register one GGUF file you already have. |
| `settings --models-dir DIR` | Change where models are downloaded. |

## Test, tune and check

| Command | What it does |
|---|---|
| `test VARIANT_ID [--context N] [--gpu-layers N] [--kv f16\|q8_0\|q4_0] [--kind smoke\|speed\|full]` | Start the model, check it answers, measure speed and memory. |
| `tune VARIANT_ID [--context N] [--gpu-layers N] [--budget SECONDS] [--goal generation\|balanced\|prompt]` | Try settings and save the fastest safe ones. |
| `quiz VARIANT_ID [--workload W] [--needle]` | Run the quick quiz; `--needle` adds the long-document recall test. |
| `quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...]` | Compare compressed versions with a reference. |

`--gpu-layers` counts the model's layers on the graphics card (0 = processor only). `--kv` sets notepad compression (see [How it works](how-it-works.md#the-notepad-kv-cache)).

See [Testing and tuning](testing-and-tuning.md) and [Quality checks](quality-checks.md).

## Use it

| Command | What it does |
|---|---|
| `export VARIANT_ID --format FMT [--platform posix\|windows] [--context N] [--gpu-layers N] [--tuned]` | Print settings for another tool. Formats: `llama-server`, `ollama`, `docker-compose`, `openai-python`, `continue`, `open-webui`, `lmstudio`. |
| `run VARIANT_ID [--context N] [--gpu-layers N] [--port P] [--tuned]` | Run an OpenAI-compatible server in the foreground on `127.0.0.1`. Ctrl+C stops it. |

`--tuned` uses your best saved tune for that model, context and placement. See [Using your model](using-your-model.md).

## Community results

| Command | What it does |
|---|---|
| `community status` | Show imported community results. |
| `community import [--source URL]` | Download shared speed results (HTTPS only; bad rows are dropped). |
| `community share MEASUREMENT_ID ...` | Print an anonymised copy of your results and a link to a pre-filled GitHub issue. **Nothing is sent**: you open the link, check it, and submit it yourself. |

See [Privacy and safety](privacy-and-safety.md#community-results).

## Older commands (still work)

| Command | What it does |
|---|---|
| `bench VARIANT_ID --model PATH [--executable llama-bench] [--context N] [--gpu-layers N] [--gpu-index N] [--timeout 600]` | The v0.3 benchmark on a GGUF file you point to. `test --kind speed` is the newer, fuller version. |

## Environment variables

| Variable | Effect |
|---|---|
| `LLM_CONFIG_HOME` | Data folder (settings, cache, results, runtime). |
| `LLM_CONFIG_MODELS` | Models folder (overrides `settings --models-dir`). |
| `HF_TOKEN` | Hugging Face token, only for models that require you to log in or accept terms. |
| `HF_ENDPOINT` | Use a Hugging Face mirror (default `https://huggingface.co`). |
| `HF_HOME`, `HF_HUB_CACHE`, `OLLAMA_MODELS` | Where the local model search looks for existing files. |
| `AA_API_KEY` | Artificial Analysis key (a key saved in the app wins over this). |

## Examples

```bash
# What fits for coding with a 16k context?
llm-config recommend --workload coding --context 16384

# The whole loop from the terminal
llm-config runtime install
llm-config download "EXACT_VARIANT_ID" --yes
llm-config test "EXACT_VARIANT_ID" --context 16384
llm-config tune "EXACT_VARIANT_ID" --context 16384 --budget 300
llm-config run "EXACT_VARIANT_ID" --context 16384 --tuned
```

On PowerShell write each command on one line (PowerShell does not use `\` to continue lines).
