# LLM Configurator

Run an AI chat model on **your own computer**, with less guesswork.

LLM Configurator looks at your hardware and walks you through one loop:

1. **Pick** – it suggests models that should fit your memory and speed needs.
2. **Download** – it fetches the model file and checks it is not damaged.
3. **Test** – it starts the model and measures how fast it really is.
4. **Tune** – it tries different settings and keeps the fastest safe ones.
5. **Use** – it gives the model an address that chat apps and code on your computer can connect to (the same way they'd connect to ChatGPT), or ready-made settings for Ollama, LM Studio and more.

The app has no chat window of its own: you chat through an app you connect to it, such as Open WebUI, Ollama or LM Studio.

It never makes up numbers. If something is unknown, it says "unknown". If something is an estimate, it says "estimate".

> **Status:** v0.4 is new. The memory and speed maths still need checking on more real machines. Test a setup before you rely on it.

## Quick start

You need **[Python](https://www.python.org/downloads/) 3.10 or newer** (on Windows, tick **"Add python.exe to PATH"** when installing). Open a terminal (**PowerShell** on Windows, not Command Prompt; **Terminal** on macOS) and type these lines **one at a time**. The install line takes about a minute and ends with `Successfully installed ...`. Most computers with 8 GB of memory or more can run small models; more memory means bigger, smarter models.

<details open>
<summary>Windows (PowerShell)</summary>

```powershell
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
.\.venv\Scripts\llm-config.exe serve
```
</details>

<details open>
<summary>macOS / Linux</summary>

```bash
git clone https://github.com/Eyad-3D/LLM-Configurator.git
cd LLM-Configurator
python3 -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/llm-config serve
```
</details>

No [Git](https://git-scm.com/downloads)? Download the ZIP from GitHub, unzip it, open a terminal in the `LLM-Configurator-main` folder and start from the third line. [Getting started](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/getting-started.md) walks through this step by step.

Your browser opens **http://127.0.0.1:8765** (`127.0.0.1` means "this computer"; paste the address into your browser if nothing opens). **Keep the terminal open** while you use the app; press **Ctrl+C** in it to stop. To start again later, open a terminal in the project folder and run just the last line. To look around first with made-up models (no internet, no downloads), add `--demo` to the end of that line. Stuck? See [Troubleshooting](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/troubleshooting.md).

**After v0.4.0 is published on PyPI** (the Python package store; not yet), one command will do: `uvx llm-configurator serve` (with [uv](https://docs.astral.sh/uv/getting-started/installation/)) or `pipx install llm-configurator` then `llm-config serve` (with [pipx](https://pipx.pypa.io/stable/installation/)).

## What happens on your computer

> - **Runs locally.** The app and any model server it starts listen on `127.0.0.1` only (your own computer). Other devices on your network cannot reach them.
> - **Nothing is downloaded without you asking.** Model files and the llama.cpp engine are only fetched when you click **Download** / **Install the engine** or run the matching command. Downloads are checked against their published fingerprint (a unique code that changes if even one byte is different) before use.
> - **No tracking.** The app sends no usage data. It only contacts Hugging Face (the main website where AI models are shared: model info and files), GitHub (the llama.cpp engine, community results you choose to import) and, if you add a key, Artificial Analysis (quality rankings).
> - **Your key stays in your system's password store** (Windows Credential Manager, macOS Keychain, or a Linux keyring). Never in a plain file.
> - **Sharing is manual.** Community sharing only opens a pre-filled GitHub page. You review it and press submit yourself.
>
> Full details: [Privacy and safety](https://github.com/Eyad-3D/LLM-Configurator/blob/main/docs/privacy-and-safety.md).

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
