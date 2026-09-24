# LLM Configurator

Run an AI chat model on **your own computer**, with less guesswork.

LLM Configurator looks at your hardware and walks you through one loop:

1. **Pick** – it suggests models that should fit your memory and speed needs.
2. **Download** – it fetches the model file and checks it is not damaged.
3. **Test** – it starts the model and measures how fast it really is.
4. **Tune** – it tries different settings and keeps the fastest safe ones.
5. **Use** – it gives the model an address that chat apps and code on your computer can connect to (the same way they'd connect to ChatGPT), or ready-made settings for Ollama, LM Studio and more.

It never makes up numbers. If something is unknown, it says "unknown". If something is an estimate, it says "estimate".

> **Status:** v0.4 is new. The memory and speed maths still need checking on more real machines. Test a setup before you rely on it.

## Quick start

You need **[Python](https://www.python.org/downloads/) 3.10 or newer**. Type the commands below in a terminal (**PowerShell** on Windows, **Terminal** on macOS). Most computers with 8 GB of memory or more can run small models; more memory means bigger, smarter models.

> **Right now, use "From source".** The shorter `uvx` / `pipx` options work once v0.4.0 is published on PyPI (the Python package store).

**From source** (needs [Git](https://git-scm.com/downloads), or download the ZIP from GitHub and unzip it):

<details open>
<summary>Windows (PowerShell)</summary>

```powershell
git clone https://github.com/Eyad-3D/LLM-Configurator.git; cd LLM-Configurator
py -m venv .venv; .\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\llm-config.exe serve
```
</details>

<details open>
<summary>macOS / Linux</summary>

```bash
git clone https://github.com/Eyad-3D/LLM-Configurator.git && cd LLM-Configurator
python3 -m venv .venv && .venv/bin/python -m pip install .
.venv/bin/llm-config serve
```
</details>

To start it again later, open a terminal in the `LLM-Configurator` folder and run just the last line.

**One command with [uv](https://docs.astral.sh/uv/getting-started/installation/)** (install uv first):

```bash
uvx llm-configurator serve
```

**Or with [pipx](https://pipx.pypa.io/stable/installation/)** (install pipx first; it keeps the app in its own box, away from your other Python tools):

```bash
pipx install llm-configurator
llm-config serve
```

Your browser opens **http://127.0.0.1:8765** (if it doesn't, paste that address into your browser). Stop the app with **Ctrl+C** in the terminal. Want to look around first, with no internet and no downloads? Add `--demo` to the last command.

More detail: [Getting started](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/getting-started.md).

## What happens on your computer

> - **Runs locally.** The app and any model server it starts listen on `127.0.0.1` only (your own computer). Other devices on your network cannot reach them.
> - **Nothing is downloaded without you asking.** Model files and the llama.cpp engine are only fetched when you click **Download** / **Install the engine** or run the matching command. Downloads are checked against their published fingerprint (a unique code that changes if even one byte is different) before use.
> - **No tracking.** The app sends no usage data. It only contacts Hugging Face (model info and files), GitHub (the llama.cpp engine, community results you choose to import) and, if you add a key, Artificial Analysis (quality rankings).
> - **Your key stays in your system's password store** (Windows Credential Manager, macOS Keychain, or a Linux keyring). Never in a plain file.
> - **Sharing is manual.** Community sharing only opens a pre-filled GitHub page. You review it and press submit yourself.
>
> Full details: [Privacy and safety](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/privacy-and-safety.md).

## Screenshots

_Screenshots to be added. What to capture:_

1. _The guided questions screen (one question, e.g. "What will you mainly use it for?")._
2. _The results screen with three recommendation cards showing verdict badges ("Runs well", "Not tested yet")._
3. _The "Get it running" panel, mid-download, with the progress bar and the six steps._
4. _The test results: reading speed, first-word delay, writing speed and memory used vs estimate._
5. _The "Use it" step with the OpenAI address shown and the export tabs._
6. _The Quality panel's blind "Try my prompts" comparison._

## Learn more

| Page | What it covers |
|---|---|
| [Getting started](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/getting-started.md) | Install, first run, the guided questions |
| [How it works](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/how-it-works.md) | The memory and speed maths, and its limits |
| [Testing and tuning](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/testing-and-tuning.md) | What the tests measure and how tuning picks settings |
| [Quality checks](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/quality-checks.md) | Quizzes, blind comparisons, compression-loss check, rankings |
| [Using your model](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/using-your-model.md) | Local server, OpenAI address, Ollama / LM Studio / Docker exports |
| [Command line](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/cli.md) | Every `llm-config` command |
| [Privacy and safety](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/privacy-and-safety.md) | Network use, keys, downloads, sharing |
| [Troubleshooting](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/troubleshooting.md) | Install problems, out of memory, antivirus, ports |
| [FAQ](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/faq.md) | Short answers to common questions |
| [Contributing](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/contributing.md) | Developer setup and tests |
| [Changelog](https://github.com/Eyad-3D/LLM-Configurator/blob/main/CHANGELOG.md) | What changed in each version |

Quality rankings, when enabled, come from [Artificial Analysis](https://artificialanalysis.ai). Models run on [llama.cpp](https://github.com/ggml-org/llama.cpp).
