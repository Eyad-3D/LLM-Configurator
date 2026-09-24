# Command line (`llm-config`)

Everything in the app can also be done from a terminal.

```bash
llm-config --help
llm-config COMMAND --help
```

`llm-config` and `python -m llm_configurator` are the same. (`llm-configurator` is an alias so `uvx llm-configurator` works.)

**Global option:** `--data-dir PATH` uses a different data folder. It must come **before** the command: `llm-config --data-dir ./test-data models`.

**IDs.** Many commands take a `VARIANT_ID`: the exact ID of one model file (model + compression level). Get IDs from `llm-config models`. Put IDs in quotes; they can contain characters your shell treats specially.

**Exit codes.** `0` success, `1` error (with a plain message on stderr; `test` also returns `1` when the model fails the test), `2` a mistyped command or option, `130` cancelled with Ctrl+C.

**`--json`.** `scan`, `calibrate`, `refresh`, `models`, `benchmarks`, `map` and `settings` always print JSON. `runtime status`, `local`, `recommend`, `test`, `tune`, `quiz`, `quant-check` and `export` print plain text; add `--json` for the full result. (`run` accepts `--json` but ignores it.)

## Start here

| Command | What it does |
|---|---|
| `serve [--port 8765] [--demo] [--no-browser]` | Start the browser app on `http://127.0.0.1:8765` and open it in your browser (`--no-browser` skips that). `--demo` uses made-up models. |
| `scan` | Print your hardware and running apps' memory use as JSON. |
| `calibrate` | Run the hardware speed check (at most about 12 seconds) and save it. Reads no model files. |
| `refresh` | Fetch model information from Hugging Face (and rankings if you have an Artificial Analysis key). Does **not** download models. |
| `models` | List known models with their exact IDs (as JSON). |
| `models add BASE_REPO GGUF_REPO` | Add your own Hugging Face model: the original repo (for example `Qwen/Qwen3-8B`) and the repo with GGUF files (for example `Qwen/Qwen3-8B-GGUF`). Then run `refresh` so its files show up in `models`. |
| `models remove BASE_REPO` | Take a model out of your list, built-in ones too (`models add` brings one back). Files on disk are kept. |

**Gated models.** Some original repos make you log in and accept a licence first. `refresh` then needs `HF_TOKEN` (see [Environment variables](#environment-variables)). Without a token, you can point the model at an ungated copy of its `config.json`: open `catalogue.json` in your data folder and add `"config_repo": "owner/name"` to that model's entry. (`models add` has no option for this.)

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
| `--reserve-gib N` / `--gpu-reserve-gib N` | RAM / VRAM (GiB) to keep free | `2` / `0.5` |
| `--gpu-index N` | Which graphics card | `0` |
| `--reclaim-pids PID ...` | Apps you'd close to free memory | none |
| `--priority balanced\|quality\|speed` | What matters most | `balanced` |
| `--kv f16\|q8_0\|q4_0` | Notepad compression, to fit longer chats ([what is this?](how-it-works.md#the-notepad-kv-cache)) | `f16` |
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
| `runtime status [--json]` | Is llama.cpp installed? Where, which version, which backend (CUDA, Metal, Vulkan, CPU…). |
| `runtime install [--allow-unverified]` | Download and install the official llama.cpp release for your system. Checks its fingerprint (SHA256). Refuses files without a published fingerprint unless you add `--allow-unverified`; only use that if you trust the source. |
| `runtime install-archive PATH` | Install from a llama.cpp zip / tar.gz you already downloaded (for offline machines). |
| `runtime use DIR` | Use a llama.cpp build you already have. `DIR` must contain `llama-server` (directly, or in `bin/` or `build/bin/`). |

Search order: a folder set with `runtime use`, then the app's own install, then your `PATH` (including Homebrew).

## Models on disk

| Command | What it does |
|---|---|
| `download VARIANT_ID [--directory DIR] [--yes]` | Download a model (all pieces if it is split). Shows the size first and asks; `--yes` skips the question (needed in scripts). Resumes if interrupted. Checks every file's fingerprint (SHA256). Default folder: your models folder; if the model is already on your computer it says so and downloads nothing. With `--directory`, it downloads there without that check, and other commands won't find the file until you run `local --scan --dir DIR`. Gated models need `HF_TOKEN`. |
| `local [--json]` | List GGUF model files found by the last search, and the files you registered. |
| `local --scan [--dir DIR]` | Search the usual places (Hugging Face cache, LM Studio, Ollama, your models folder). Repeat `--dir DIR` to add folders (it also starts a search on its own). Use full paths for `--dir`. |
| `local --add PATH` | Register one `.gguf` file you already have and print its model ID. For a split model, give the first piece (`…-00001-of-…`). |
| `settings [--models-dir DIR]` | Show settings, or change where models are downloaded. |

## Test, tune and check

| Command | What it does |
|---|---|
| `test VARIANT_ID [--kind smoke\|speed\|full] [--tuned]` | Start the model, check it answers (`smoke`), measure speed and memory (`speed`), or both (`full`, the default). |
| `tune VARIANT_ID [--budget SECONDS] [--goal generation\|balanced\|prompt]` | Try settings and save the fastest safe ones. Budget 60–1800 seconds (default `300`). Goal: faster writing (`generation`, the default), faster reading of long inputs (`prompt`), or both (`balanced`). |
| `quiz VARIANT_ID [--workload general\|coding\|agentic\|documents] [--needle]` | Run the quick quiz (default workload `general`); `--needle` adds the long-document recall test. |
| `quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...]` | Compare compressed versions with a reference (use the least-compressed one you have, such as Q8_0). All must be downloaded. |

`test`, `tune`, `quiz`, `export` and `run` also take:

| Option | Meaning | Default |
|---|---|---|
| `--context N` | Tokens per chat | `8192`, or the model's limit if lower |
| `--gpu-layers N` | How many of the model's layers go on the graphics card (0 = processor only) | the most that fit |
| `--kv f16\|q8_0\|q4_0` | Notepad compression; `q8_0`/`q4_0` use less memory (see [How it works](how-it-works.md#the-notepad-kv-cache)) | `f16` |
| `--json` | Print the full result as JSON | off |

See [Testing and tuning](testing-and-tuning.md) and [Quality checks](quality-checks.md).

## Use it

| Command | What it does |
|---|---|
| `export VARIANT_ID --format FMT [--platform posix\|windows] [--tuned] [--output FILE]` | Print settings for another tool, or save them to `FILE`. Formats: `llama-server`, `ollama`, `docker-compose`, `openai-python`, `continue`, `open-webui`, `lmstudio`. `--platform` defaults to the system you're on. |
| `run VARIANT_ID [--port P] [--tuned]` | Run an OpenAI-compatible server in the foreground on `127.0.0.1` (default port: any free one; the address is printed). Ctrl+C stops it. |

`--tuned` (on `test`, `export` and `run`) uses your best saved tune for that model, context and placement; it fails if there is none yet. See [Using your model](using-your-model.md).

## Community results

| Command | What it does |
|---|---|
| `community status` | Show imported community results. |
| `community import [--source URL]` | Download shared speed results from the project's list, or from `URL` (HTTPS only; bad rows are dropped). |
| `community share MEASUREMENT_ID ...` | Print an anonymised copy of your results and a link to a pre-filled GitHub issue. **Nothing is sent**: you open the link, check it, and submit it yourself. |

See [Privacy and safety](privacy-and-safety.md#community-results).

## Older commands (still work)

| Command | What it does |
|---|---|
| `bench VARIANT_ID --model PATH [--executable llama-bench] [--context 8192] [--gpu-layers 0] [--gpu-index 0] [--timeout 600]` | The v0.3 benchmark on a GGUF file you point to. It looks for `llama-bench` on your `PATH` only (not the copy from `runtime install`), so pass `--executable` if needed. `test --kind speed` is the newer, fuller version. |

## Environment variables

| Variable | Effect |
|---|---|
| `LLM_CONFIG_HOME` | Data folder (settings, cache, results, runtime). `--data-dir` wins over it. |
| `LLM_CONFIG_MODELS` | Models folder (overrides `settings --models-dir`). |
| `HF_TOKEN` | Hugging Face access token, only for gated models (ones that make you log in and accept a licence first). Used by `refresh` and `download`. |
| `HF_ENDPOINT` | Download model files from a Hugging Face mirror (`https://` only; default `https://huggingface.co`). `refresh` still uses huggingface.co. |
| `HF_HOME`, `HF_HUB_CACHE`, `OLLAMA_MODELS` | Where the local model search looks for existing files. |
| `AA_API_KEY` | Artificial Analysis key (a key entered or saved in the app wins over this). |

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
