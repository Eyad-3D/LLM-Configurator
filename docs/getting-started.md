# Getting started

This page takes you from nothing to your first recommendation, and then to a running model.

## What you need

- **Python 3.10 or newer.** Check with `python3 --version` (Windows: `py --version`).
- **The computer you want to run models on.** The app measures the machine it runs on. If you run it on a server, it measures the server, not your laptop.
- **Disk space** for the models you choose. A small model is 1–5 GB; big ones are 20–100+ GB. The app shows the size before downloading.
- **Internet** for model information and downloads. Not needed for demo mode.
- No graphics card is required. Models run on the processor (CPU) too, just slower.

## Install

Pick one.

### Option 1: uv (one command)

[uv](https://docs.astral.sh/uv/) is a fast Python tool runner. `uvx` downloads the app into a private cache and runs it, like a pop-up shop that leaves nothing behind in your main Python.

```bash
uvx llm-configurator serve
```

### Option 2: pipx

[pipx](https://pipx.pypa.io/) installs Python apps in their own box so they don't clash with anything else.

```bash
pipx install llm-configurator
llm-config serve
```

If your terminal says `llm-config: command not found`, run `pipx ensurepath` and open a new terminal.

> **Note:** Options 1 and 2 need v0.4.0 to be published on PyPI (the Python package store). Until then, use Option 3.

### Option 3: from source

Windows (PowerShell):

```powershell
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\llm-config.exe serve
```

macOS / Linux:

```bash
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/llm-config serve
```

`.venv` is a "virtual environment": a private folder of Python packages just for this app.

You can always use `python -m llm_configurator` instead of `llm-config`; they do the same thing.

## First run

`llm-config serve` starts the app and opens **http://127.0.0.1:8765** in your browser. That address only works on your own computer. Stop the app with **Ctrl+C** in the terminal.

- Port already in use? `llm-config serve --port 8766`
- Don't want a browser tab to open? `llm-config serve --no-browser`
- Just exploring? `llm-config serve --demo` uses **made-up models on your real hardware**. It is clearly marked, needs no internet, and never downloads, tests or starts anything.

## The guided questions

1. Click **Get started**. The app checks your hardware in the background.
2. Choose whether to **include quality rankings**. Rankings come from Artificial Analysis and need a free API key (see [Quality checks](quality-checks.md#rankings-from-artificial-analysis)). You can skip this.
3. Answer one question per screen:
   - what you'll mainly use it for (general chat, coding, agents/tools, long documents)
   - whether you care more about quality or speed
   - how much text it needs to keep in view at once (the **context**, like the size of its desk)
   - how many chats or agents run **at the same time** (one person running three agents counts as three)
   - whether other apps will stay open
   Exact numbers (tokens, words per second, memory to keep free) are under **advanced** controls.
4. Check your answers. **Edit** jumps back to one question.
5. See up to **three recommendations**. Each card shows:
   - a **verdict badge**: *Runs well*, *Runs slowly*, *Too slow* or *Not tested yet*
   - how much memory it needs
   - speed: measured on your machine, or a clearly labelled estimate
   - the quality rank, if you turned rankings on

While you answer, the app fetches model information (sizes and shapes, **not** the model files) and runs a short hardware speed check (about 12 seconds, reads no model files).

## Get it running

Each card has a **Get it running** button. It opens six steps:

1. **Runtime** – installs [llama.cpp](https://github.com/ggml-org/llama.cpp), the engine that actually runs the model. One click. See [Troubleshooting](troubleshooting.md#installing-llamacpp) if it fails.
2. **Download** – shows the size and your free disk space, then downloads with a progress bar. You can pause and resume. If the file is **already on your disk** (for example from LM Studio, Ollama or a Hugging Face download), it is reused instead.
3. **Test** – loads the model, asks it a simple question, and measures speed and memory. See [Testing and tuning](testing-and-tuning.md).
4. **Tune** – spends 1, 5 or 15 minutes trying settings and keeps the fastest safe ones.
5. **Check quality** – optional quizzes and blind comparisons. See [Quality checks](quality-checks.md).
6. **Use it** – start a local server, or copy settings for Ollama, LM Studio, Docker and others. See [Using your model](using-your-model.md).

A small **jobs tray** shows everything running in the background. You can cancel any job.

## Where your data lives

The app keeps a small local database (settings, cached model info, your test results):

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\LLMConfigurator` |
| macOS / Linux | `$XDG_DATA_HOME/llm-configurator` (usually `~/.local/share/llm-configurator`) |

Inside it: `models/` (downloaded models) and `runtime/` (llama.cpp). Change the main folder with the `LLM_CONFIG_HOME` environment variable or `llm-config --data-dir PATH ...`. Change only the models folder with `llm-config settings --models-dir DIR` or `LLM_CONFIG_MODELS`.

To remove everything, delete that folder, then uninstall (`pipx uninstall llm-configurator`, or delete the source folder). A saved API key lives in your system password store; remove it first with **Remove key** in the app.

## Next

- [How it works](how-it-works.md) – why a model does or doesn't fit
- [Command line](cli.md) – do everything without the browser
