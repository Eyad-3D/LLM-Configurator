# Command line (`llm-config`)

You can do everything the app does from a terminal.

```bash
llm-config --help
llm-config COMMAND --help
llm-config COMMAND ACTION --help     # for runtime, models and community
```

`llm-config` and `python -m llm_configurator` do the same thing. (`llm-configurator` is an alias so that `uvx llm-configurator` works.)

## Basics

**Global option.** `--data-dir PATH` uses a different data folder (see [Where things are saved](#where-things-are-saved)). It must come **before** the command: `llm-config --data-dir ./test-data models`. Put after the command, it is an error.

**Model IDs.** Many commands take a `VARIANT_ID`: the exact ID of one model file (one model at one compression level). `llm-config models` lists them. They look like `Qwen/Qwen3-8B-GGUF@<commit>:Qwen3-8B-Q4_K_M.gguf`, or `local:<code>:<file name>` for your own files. Put IDs in quotes, because they can contain characters your shell treats specially.

**What you see while it works.** Long commands (such as refresh, download, local search, runtime install, test, tune, quiz, quant-check and run) print a one-line progress bar with plain words, a percentage and, for downloads, speed and time left:

```
Downloading Qwen3-8B-Q4_K_M.gguf ·  43% · 2.10 GiB of 4.90 GiB · 42.0 MB/s · about 1m 10s left
```

Progress and messages go to the error stream (stderr). Results go to normal output (stdout), so `llm-config models > models.json` saves only the result. When the output goes to a file or another program, you get one plain line at each 10% step instead of a moving bar.

**Ctrl+C.** Long commands stop cleanly: you see `Stopping… (press Ctrl+C again to quit immediately)`, the job is cancelled, and any llama.cpp program the command started is closed. Then `Cancelled.`, or for downloads `Paused. Run the same command again to resume.`: what was already downloaded is kept and the next run picks up where it stopped. Closing the terminal or stopping the command with `kill` also closes the llama.cpp programs it started.

**Exit codes** (a number scripts can check):

| Code | Meaning |
|---|---|
| `0` | Success. Also when you answer "no" to the download question. |
| `1` | Error, with a plain `Error: …` message on stderr. `test` also returns `1` when the model fails the test ("works, but slowly" still returns `0`), and `run` returns `1` if the server stops by itself. |
| `2` | A mistyped command or option (or `--data-dir` after the command). |
| `130` | Stopped with Ctrl+C. |

**JSON.** `scan`, `calibrate`, `refresh`, `models`, `models remove`, `benchmarks`, `map`, `settings` and `bench` always print JSON. `runtime status`, `local`, `recommend`, `test`, `tune`, `quiz`, `quant-check` and `export` print plain text; add `--json` for the full result. `run` accepts `--json` but ignores it. The rest print plain text only.

## Start here

| Command | What it does |
|---|---|
| `serve [--port 8765] [--demo] [--no-browser]` | Start the browser app on `http://127.0.0.1:8765` and open it in your browser (`--no-browser` skips that). `--demo` uses made-up models on your real hardware. It only answers on this computer. If the port (a numbered door on your computer) is taken, pick another with `--port`; `--port 0` takes any free one. Ctrl+C stops the app and any model it started. |
| `scan` | Print your hardware (processor, memory, graphics cards, free disk) and how much memory running apps use, as JSON. |
| `calibrate` | Run the short hardware speed check (at most about 12 seconds), save it and print it. Reads no model files. A saved check is used for 7 days, or until your hardware changes. |
| `refresh` | Fetch model information from Hugging Face, and rankings if you have an Artificial Analysis key. Does **not** download models. Shows `Checked N of M model sources`, then prints a summary with any warnings. Ctrl+C stops it and keeps your old list. |

## Your model list

| Command | What it does |
|---|---|
| `models` | List every model file the app knows, with its exact `id` (JSON): the built-in list (after `refresh`), models you added, and your own files registered with `local`. |
| `models add BASE_REPO GGUF_REPO [--config-repo REPO]` | Add your own Hugging Face model: the original repo (for example `Qwen/Qwen3-8B`) and the repo with the GGUF files, the model file format llama.cpp uses (for example `Qwen/Qwen3-8B-GGUF`). It fetches the file list straight away and says `Added …: N downloadable versions`. If that fetch fails, the model is still added; run `refresh` later. Up to 200 of your own models. |
| `models remove BASE_REPO` | Take a model off your list, built-in ones too (`models add` brings one back). Files on disk are kept. Prints JSON. |

**Gated models.** Some original repos make you log in and accept a licence first. The app reads the model's `config.json` (its spec sheet) from the original repo, so for those `refresh` and `models add` need `HF_TOKEN` (see [Environment variables](#environment-variables)). Without a token, add `--config-repo OWNER/NAME` with an open copy of the original repo; only `config.json` is read from there. Built-in models already do this. For a model you already added, `models remove` it and add it again with `--config-repo`. If the GGUF repo itself is gated, `download` needs `HF_TOKEN` too.

## Recommendations

```bash
llm-config recommend [options]
```

| Option | Meaning | Default |
|---|---|---|
| `--workload general\|coding\|agentic\|documents` | Main use | `general` |
| `--context N` | Tokens per chat (256 to 1,048,576). A token is about three quarters of a word. | `8192` |
| `--users N` | Chats or agents running at the same time (1 to 64) | `1` |
| `--min-tps N` | Speed you want, in tokens per second | `15` |
| `--reserve-gib N` | RAM (GiB) to keep free for everything else | `2` |
| `--gpu-reserve-gib N` | Graphics memory (VRAM, GiB) to keep free | `0.5` |
| `--gpu-index N` | Which graphics card (`0` is the first) | `0` |
| `--reclaim-pids PID ...` | Process IDs (the PID numbers in Task Manager or Activity Monitor) of apps you would close to free memory. Adds an "after closing" option next to "now". | none |
| `--priority balanced\|quality\|speed` | What matters most | `balanced` |
| `--kv f16\|q8_0\|q4_0` | Notepad compression, to fit longer chats ([what is this?](how-it-works.md#the-notepad-kv-cache)) | `f16` |
| `--strict-speed` | Only show options whose speed was measured on this computer and reaches `--min-tps` | off |
| `--no-rankings` | Ignore quality rankings | rankings on |
| `--demo` | Made-up models, real hardware | off |
| `--json` | Print the full report as JSON instead of the summary | off |
| `--output FILE` | Also save the full report as JSON to `FILE` | none |

The summary shows the shortlist. For each option: model, compression level, where it runs (`cpu`, `gpu` or `split`), context, speed (measured, or an estimated range marked low confidence, or "speed unverified"), RAM and VRAM use, and its Model ID. The notes at the end are part of the answer: memory fit is an estimate, not a guarantee, and only tests on your computer verify speed. `--json` also lists the options that did not make the shortlist.

## Rankings

| Command | What it does |
|---|---|
| `benchmarks` | List the Artificial Analysis ranking entries (name and slug, the entry's short name) from the last `refresh`, and each model's current match. Empty until you `refresh` with an Artificial Analysis key. |
| `map BASE_REPO SLUG` | Match a model to one exact ranking entry. The slug must be one that `benchmarks` lists. Use `-` as the slug to clear the match. |

See [Quality checks](quality-checks.md) for how rankings are used.

## The engine (llama.cpp)

| Command | What it does |
|---|---|
| `runtime status [--json]` | Is llama.cpp installed? Shows its version, backend (which chips it was built to use: CUDA, Metal, Vulkan, ROCm or CPU), folder, how it was found (`configured` = a folder you chose with `runtime use`, `managed` = the app's own install, `path` = on your `PATH`, the list of folders your terminal searches for programs) and whether each tool (`llama-server`, `llama-bench`, `llama-perplexity`, `llama-cli`) is ready. |
| `runtime install [--allow-unverified]` | Download and install the official llama.cpp release that suits your system and graphics card, and say why it picked that build. Checks the download's fingerprint (SHA256, a code that changes if even one byte is different) and refuses files without a published fingerprint unless you add `--allow-unverified` (only if you trust the source). If a graphics-card build will not start, it installs the CPU build instead and says so. Ctrl+C keeps what was downloaded; running it again continues. |
| `runtime install-archive PATH` | Install from a llama.cpp `.zip`, `.tar.gz` or `.tgz` you already downloaded (for offline computers). There is no published fingerprint to compare with, so only use a file you trust. |
| `runtime use DIR` | Use a llama.cpp build you already have. `DIR` must contain `llama-server`, directly or in `bin/`, `build/bin/` or a folder below. It is checked that it starts. |

Search order: the folder set with `runtime use`, then the app's own install, then your `PATH` (and the usual Homebrew folders). `runtime install` and `install-archive` switch the app to the new install and forget a `runtime use` folder; run `runtime use` again to go back to it. Nothing is added to your `PATH` or system folders.

## Models on disk

### `download VARIANT_ID [--directory DIR] [--yes]`

Download a model (all its parts if it is split into several files).

1. It first looks for a copy already on your computer: in your models folder and among files found by `local`. If there is one, it prints `Already on this computer: PATH` and downloads nothing.
2. It shows the size and how much is still to fetch, checks there is enough disk space (plus 1 GiB spare), and asks `Download this model? [y/N]`. `--yes` skips the question; scripts need it, because without a keyboard it stops with an error.
3. It downloads with a progress bar. Ctrl+C pauses; run the same command to resume. Every file is checked against its published fingerprint (SHA256) before it gets its final name.
4. It prints the path of the model file (the first part, for a split model).

Default folder: your models folder (see `settings`). `--directory DIR` saves there instead and registers the file, so other commands find it; if that fails, it tells you to run `local --scan --dir DIR`. It never overwrites a different file with the same name. Models with no published fingerprint cannot be downloaded. Gated models need `HF_TOKEN`. Your own files (`local:` IDs) have nothing to download.

### `local [--scan] [--dir DIR] [--add PATH] [--json]`

Find and register model files that are already on your computer.

| Command | What it does |
|---|---|
| `local` | Show the folders it searches (`[x]` exists, `[ ]` missing), the model files found by the last search with their Model IDs, and files you registered. |
| `local --scan` | Search the usual places: the Hugging Face cache, LM Studio, Ollama and your models folder. It does not read whole files, so it is quick; a file is checked against its fingerprint the first time you use it (until then it may say `not verified yet`). |
| `local --dir DIR` | Also search `DIR`. Repeat `--dir` for more folders. It starts a search on its own. A later search without the same `--dir` forgets files it found only there. |
| `local --add PATH` | Register one GGUF file and print its Model ID. It reads the whole file once to fingerprint it. For a split model, give any of its parts. Ollama's model files without a `.gguf` name work too. If the file is a known model it says `Matches a catalogue model`, otherwise `Registered as your own model`. Registered files stay listed while they exist. |

Your own files can then be used with `recommend`, `test`, `tune`, `quiz`, `quant-check`, `export`, `run` and `bench`.

### `settings [--models-dir DIR]`

Show your settings as JSON, including the models folder in use. `--models-dir DIR` changes where models are downloaded (the folder is created if needed). The `LLM_CONFIG_MODELS` variable wins over this setting.

## Test, tune and check

These need llama.cpp (`runtime install`) and a model that is on your computer.

| Command | What it does |
|---|---|
| `test VARIANT_ID [--kind smoke\|speed\|full] [--tuned] [--min-tps 15]` | `smoke`: start the model and check it answers one question correctly. `speed`: measure reading speed, first-word delay, writing speed and peak memory. `full` (the default): both. Below `--min-tps` tokens per second the verdict is "works, but slowly" (`0` skips that check). Speed results are saved and `recommend` uses them. |
| `tune VARIANT_ID [--budget 300] [--goal generation\|balanced\|prompt]` | Try engine settings (threads, flash attention, batch sizes and, for split setups, the layers on the graphics card) and save the fastest ones. Only a clear win counts, and settings that would not fit in memory are never tried. Budget: 60 to 1800 seconds. Goal: faster writing (`generation`, the default), faster reading of long inputs (`prompt`), or both (`balanced`). Ctrl+C stops it and saves nothing. |
| `quiz VARIANT_ID [--workload general\|coding\|agentic\|documents] [--needle]` | Ask short questions with known answers (default workload `general`) and show the score with the range the true score is likely in. `--needle` adds the long-document test: three facts are hidden in a document as long as `--context` (at least 512 tokens) and the model must find them. |
| `quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...] [--json]` | Measure how much quality compression loses, like comparing a low-quality JPEG with the original photo. Use the least-compressed file you have (such as Q8_0) as the reference. Every file reads the same sample text (about 6,000 tokens), and you get how often each one picks a different next word than the reference, with a plain verdict. All files must be on your computer. Uses `llama-perplexity`. Verdicts are rough guidance. |

`test`, `tune`, `quiz`, `export` and `run` also take:

| Option | Meaning | Default |
|---|---|---|
| `--context N` | Tokens per chat, from 256 up to the model's limit | `8192`, or the model's limit if lower |
| `--gpu-layers N` | How many of the model's layers go on the graphics card (`0` = processor only) | the most that fit in free memory now |
| `--kv f16\|q8_0\|q4_0` | Notepad compression; `q8_0`/`q4_0` use less memory (see [How it works](how-it-works.md#the-notepad-kv-cache)) | `f16` |
| `--json` | Print the full result as JSON | off |

Results of `test`, `tune`, `quiz` and `quant-check` are saved on your computer. See [Testing and tuning](testing-and-tuning.md) and [Quality checks](quality-checks.md).

## Use it

| Command | What it does |
|---|---|
| `export VARIANT_ID --format FMT [--platform posix\|windows] [--tuned] [--port 8080] [--output FILE]` | Print ready-to-use settings for another tool, or save them to `FILE`. Formats: `llama-server`, `ollama`, `docker-compose`, `openai-python`, `continue`, `open-webui`, `lmstudio`. `--platform` writes for macOS/Linux (`posix`, bash) or Windows (`windows`, PowerShell); the default is the system you're on. `--port` is the port the server will use (default `8080`). The steps to follow are printed after it (on stderr). With `--output`, a `.sh` script is made runnable and a `.ps1` script is saved so Windows PowerShell reads it correctly. The model does not have to be downloaded yet; the path shown is where it will be. |
| `run VARIANT_ID [--port P] [--tuned]` | Run the model as an OpenAI-compatible server (apps made for OpenAI's API can talk to it) in this terminal. It prints `Running. OpenAI-compatible address: http://127.0.0.1:PORT/v1`. It only answers on this computer, and any API key works. Default port: any free one. Ctrl+C stops it. |

`--tuned` (on `test`, `export` and `run`) uses your saved tune for the same model, context, placement and notepad compression. It stops with an error if there is none yet, or if the tune moved the model to a split that no longer fits in free memory. See [Using your model](using-your-model.md).

## Community results

| Command | What it does |
|---|---|
| `community status` | Show how many community results are loaded, and where and when they were fetched. |
| `community import [--source URL]` | Download shared speed results from the project's list, or from `URL`. HTTPS only, at most 10 MB. Rows that fail the checks are dropped; the new list replaces the old one (if nothing can be read, the old one is kept). |
| `community share MEASUREMENT_ID [MEASUREMENT_ID ...]` | Print an anonymised copy of up to 50 of your speed results and a link to a pre-filled GitHub issue. **Nothing is sent**: you open the link, check it, and submit it yourself. If the data is too long for the link, the form opens empty and you paste the text. Only results measured on this computer as it is now can be shared; others are left out and counted. |

Result IDs are the `id` that `bench` prints, or `speed` → `measurement` → `id` in `test --json`. The browser app lets you pick results too. See [Privacy and safety](privacy-and-safety.md#community-results).

## Older commands (still work)

| Command | What it does |
|---|---|
| `bench VARIANT_ID --model PATH [--executable CMD] [--context 8192] [--gpu-layers 0] [--gpu-index 0] [--timeout 600]` | The v0.3 benchmark: writing speed with the chat almost full, on the GGUF file at `PATH`. It uses the `llama-bench` that `runtime status` shows, unless you pass `--executable`. First it reads the whole file to check it matches the model ID's fingerprint (the model needs a published one; your own files get one from `local --add`), and it checks memory (keeping 2 GiB of RAM and 0.5 GiB of VRAM free). `--timeout` is in seconds. Prints and saves the result as JSON. `test --kind speed` is the newer, fuller version. |

## Where things are saved

| What | Where |
|---|---|
| Data folder | Windows: `%LOCALAPPDATA%\LLMConfigurator`. macOS and Linux: `~/.local/share/llm-configurator` (or `$XDG_DATA_HOME/llm-configurator`). Change it with `--data-dir` or `LLM_CONFIG_HOME`. |
| Settings, model list, test, tune and quiz results | `cache.sqlite3` in the data folder |
| Models you added, ranking matches | `catalogue.json` in the data folder |
| Downloaded models | `models/` in the data folder, or the folder from `settings --models-dir` / `LLM_CONFIG_MODELS`. All parts of a split model sit side by side. An unfinished download is kept as `NAME.part` until it completes. |
| llama.cpp | `runtime/` in the data folder |
| Compression check's temporary file | `tmp/` in the data folder, deleted when the check ends |
| `recommend --output`, `export --output` | The file you name |

## Environment variables

| Variable | Effect |
|---|---|
| `LLM_CONFIG_HOME` | Data folder (settings, cache, results, runtime). `--data-dir` wins over it. |
| `LLM_CONFIG_MODELS` | Models folder (wins over `settings --models-dir`). |
| `HF_TOKEN` | Hugging Face access token, only for gated models (ones that make you log in and accept a licence first). Used by `refresh`, `models add` and `download`, and sent only to Hugging Face (or to your `HF_ENDPOINT` mirror). |
| `HF_ENDPOINT` | Download model files from a Hugging Face mirror (`https://` only; default `https://huggingface.co`). `refresh` and `models add` still use huggingface.co. |
| `HF_HOME`, `HF_HUB_CACHE`, `OLLAMA_MODELS` | Where `local --scan` looks for existing Hugging Face and Ollama files. |
| `AA_API_KEY` | Artificial Analysis key for rankings (a key entered or saved in the app wins over this). |

## Examples

```bash
# What fits for coding with a 16k context?
llm-config recommend --workload coding --context 16384

# Try it with made-up models and a separate data folder
llm-config --data-dir ./test-data recommend --demo --context 2048

# Reuse model files you already have from LM Studio, Ollama or Hugging Face
llm-config local --scan

# The whole loop from the terminal
llm-config runtime install
llm-config refresh
llm-config download "EXACT_VARIANT_ID" --yes
llm-config test "EXACT_VARIANT_ID" --context 16384
llm-config tune "EXACT_VARIANT_ID" --context 16384 --budget 300
llm-config run "EXACT_VARIANT_ID" --context 16384 --tuned
```

On PowerShell, write each command on one line (PowerShell does not use `\` to continue lines).
