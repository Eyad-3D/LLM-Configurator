# Getting started

This page takes you from nothing to your first recommendation, and then to a running model.

## What you need

- **Python 3.10 or newer** (see [Step 1](#step-1-open-a-terminal-and-check-python)).
- **Memory:** most computers with 8 GB of RAM or more can run small models. More memory (or a graphics card) lets you run bigger, smarter ones.
- **The computer you want to run models on.** The app measures the machine it runs on. If you run it on a server, it measures the server, not your laptop.
- **Disk space** for the models you choose. A small model is 1–5 GB; big ones are 20–100+ GB. The app shows the size before downloading.
- **Internet** for model information and downloads. Not needed for demo mode.
- No graphics card is required. Models run on the processor (CPU) too, just slower.

## Step 1: open a terminal and check Python

A terminal is a window where you type commands.

- **Windows:** open the Start menu, type `PowerShell` and open **Windows PowerShell**. Use PowerShell, not "Command Prompt": the commands below are written for PowerShell.
- **macOS:** press **Cmd+Space**, type `Terminal` and press Enter.
- **Linux:** open your Terminal app (on many systems, **Ctrl+Alt+T**).

Now type `py --version` (Windows) or `python3 --version` (macOS / Linux) and press Enter. You should see something like `Python 3.12.4`.

If you get an error or a number below 3.10, install Python from [python.org](https://www.python.org/downloads/). On Windows, tick **"Add python.exe to PATH"** on the first installer screen. Then close the terminal, open a new one and check again.

## Step 2: install

> **Right now, use "From source" below.** The shorter uv and pipx options only work after v0.4.0 is published on PyPI (the Python package store).

### From source

Type the lines **one at a time**, pressing Enter after each and waiting for it to finish. The `pip install` line takes about a minute and ends with `Successfully installed ...`. A notice that "a new release of pip is available" is harmless; you can ignore it.

**With Git** ([Git](https://git-scm.com/downloads) is a tool that copies the project from GitHub):

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

**Without Git (ZIP):**

1. On the [GitHub page](https://github.com/Eyad-3D/LLM-Configurator), click **Code → Download ZIP**.
2. Unzip it. You get a folder called **`LLM-Configurator-main`**. (Windows sometimes puts it inside a second folder of the same name; use the one that contains `pyproject.toml`.)
3. Open a terminal **in that folder**. Windows: open the folder in File Explorer, click the address bar, type `powershell` and press Enter. macOS: right-click the folder and choose **New Terminal at Folder** (under Services). Linux: right-click inside the folder and choose **Open in Terminal**.
4. Run the lines above **from the third line on** (skip `git clone` and `cd`).

`.venv` is a "virtual environment": a private folder of Python packages just for this app, so it doesn't mix with anything else.

**Linux (Debian / Ubuntu):** if `python3 -m venv .venv` says `ensurepip is not available`, run `sudo apt install python3-venv`, then try that line again.

If a step fails, see [Troubleshooting: Installing the app](troubleshooting.md#installing-the-app).

### After the PyPI release: uv or pipx

These will work once v0.4.0 is on PyPI.

- **uv** ([install uv](https://docs.astral.sh/uv/getting-started/installation/) first) downloads the app into a private cache and runs it, leaving your main Python untouched: `uvx llm-configurator serve`
- **pipx** ([install pipx](https://pipx.pypa.io/stable/installation/) first) installs the app in its own box: `pipx install llm-configurator`, then `llm-config serve`. If your terminal says `llm-config: command not found`, run `pipx ensurepath` and open a new terminal.

## First run

The last line (`... serve`) starts the app and opens **http://127.0.0.1:8765** in your browser. If no browser opens, copy that address into one. `127.0.0.1` means "this computer", and `8765` is the port: think of it as a door number on your computer that the app answers at. Only your own computer can reach it.

**Keep the terminal window open** while you use the app; closing it stops the app. To stop it on purpose, click the terminal and press **Ctrl+C**.

**Starting it again later:** open a terminal in the project folder (`LLM-Configurator`, or `LLM-Configurator-main` from the ZIP) and run only the last line again. From source, always use that full `.venv...` path; the short `llm-config` only works with pipx (or after activating the virtual environment).

Handy extras. Add them to the **end of your start line** (shown in the macOS / Linux form; on Windows write `.\.venv\Scripts\llm-config.exe` instead of `.venv/bin/llm-config`):

- **Just exploring?** Stop the app with Ctrl+C, then start it again with `--demo`: `.venv/bin/llm-config serve --demo`. Demo mode uses **made-up models on your real hardware**. It is clearly marked, needs no internet, and never downloads, tests or starts anything.
- **"Address already in use"?** Another program is using that door. Pick another: `.venv/bin/llm-config serve --port 8766` (see [Troubleshooting: Ports](troubleshooting.md#ports)).
- **Don't want a browser tab to open?** `.venv/bin/llm-config serve --no-browser`

## The guided questions

1. Click **Get started**. The app checks your hardware in the background.
2. Choose **Include rankings** or **Continue without rankings**. Rankings are optional quality scores from [Artificial Analysis](https://artificialanalysis.ai), a website that tests AI models. They need an API key (a personal access code) from that site; the app links to where you get one. Skip this if unsure. See [Quality checks](quality-checks.md#rankings-from-artificial-analysis).
3. Answer one question per screen:
   - what you'll mainly use it for (general chat, coding, agents, long documents). **Agents** are AI helpers that take actions for you, such as calling tools.
   - whether you care more about quality or speed
   - how much text it needs to keep in view at once (the **context**, like the size of its desk, measured in **tokens**: pieces of words, about ¾ of a word each)
   - how many chats or agents run **at the same time** (one person running three agents counts as three)
   - whether other apps will stay open
   Exact numbers (tokens, speed target, memory to keep free) can be set on those screens and under **Advanced hardware settings**.
4. Check your answers. **Edit** jumps back to one question.
5. See up to **three recommendations**. Each card shows:
   - a **verdict badge**: *Runs well*, *Runs slowly*, *Too slow* or *Not tested yet*
   - how much memory it needs
   - speed: measured on your machine, or a clearly labelled estimate
   - the quality rank, if you turned rankings on

While you answer, the app fetches model information from [Hugging Face](https://huggingface.co) (the main website where AI models are shared): sizes and shapes only, **not** the model files. It also runs a short hardware speed check (about 12 seconds, reads no model files).

**"Not tested yet"** means the app has only estimated that model; nothing has run on your computer yet. That is normal on a first run. Pick the card that looks best and click **Get it running**: its Test step turns the estimate into a real measurement. **Always run Test before you trust a verdict.**

## Get it running

Each card has a **Get it running** button. It opens six steps:

1. **Get the engine** – installs [llama.cpp](https://github.com/ggml-org/llama.cpp), the engine that actually runs the model. One click (**Install the engine**). If it fails, see [Troubleshooting](troubleshooting.md#installing-llamacpp). Antivirus programs (such as Windows Defender) sometimes block the engine; see [Antivirus and quarantine](troubleshooting.md#antivirus-and-quarantine).
2. **Download the model** – shows the size and your free disk space, then downloads with a progress bar. You can pause and resume. The file is checked against its published fingerprint (a unique code that changes if even one byte is different). A copy already in the app's models folder is reused. To reuse files you already have from LM Studio, Ollama or a Hugging Face download, run `.venv/bin/llm-config local --scan` once first: it looks in those apps' usual folders (see [Command line](cli.md)).
3. **Test it** – loads the model, asks it a simple question, and measures speed and memory. See [Testing and tuning](testing-and-tuning.md).
4. **Tune it** – spends 1, 5 or 15 minutes trying settings and keeps the fastest safe ones. **Pick 5 minutes if unsure** (the default).
5. **Check quality** – optional quizzes and blind comparisons. See [Quality checks](quality-checks.md).
6. **Use it** – start a local server, or copy settings for Ollama, LM Studio, Docker and others. See [Using your model](using-your-model.md).

A small **Background tasks** tray shows everything running in the background. You can cancel any of them.

### Can I chat with it inside the app?

Not really. The app picks, tests and sets up the model; it has no chat window of its own. The **Try my prompts** tab (under Check quality) lets you type your own questions and compare answers, which is a quick way to try a model. For everyday chatting, click **Start server** in the Use it step and connect a chat app (for example Open WebUI) to the address it shows, or copy the settings into Ollama or LM Studio. [Using your model](using-your-model.md) shows how.

## Gated models (Hugging Face licence)

Some model makers (for example Meta, for Llama) "gate" their models on Hugging Face: you must log in and accept their licence before you can download. The built-in list uses open copies where it can, so most people never need this. If a model you add yourself is gated, the app stops with a message like *"Hugging Face refused access … If the model is gated, accept its licence on huggingface.co and set HF_TOKEN to an access token, then try again."* To fix it:

1. Open the model's page on [huggingface.co](https://huggingface.co), log in and accept the licence.
2. Create a **read** access token (a password just for apps) under **Settings → Access Tokens**.
3. Stop the app (Ctrl+C), set the token in the same terminal, then start the app again:
   - Windows (PowerShell): `$env:HF_TOKEN = "hf_..."`
   - macOS / Linux: `export HF_TOKEN=hf_...`

The token is only sent to Hugging Face.

## Where your data lives

The app keeps a small local database (settings, cached model info, your test results) in one folder:

| System | Folder |
|---|---|
| Windows | `%LOCALAPPDATA%\LLMConfigurator` (paste this into the File Explorer address bar to open it) |
| macOS / Linux | `~/.local/share/llm-configurator`: a hidden folder inside your home folder (`$XDG_DATA_HOME/llm-configurator` if you set that variable) |

Inside it: `models/` (downloaded models) and `runtime/` (llama.cpp). Change the main folder with the `LLM_CONFIG_HOME` environment variable or `llm-config --data-dir PATH ...`. Change only the models folder with `llm-config settings --models-dir DIR` or `LLM_CONFIG_MODELS`.

### Uninstall

1. If you saved an Artificial Analysis key, click **Remove key** in the app first. The key lives in your system's password store (the "keyring": Windows Credential Manager, macOS Keychain or a Linux keyring), not in the data folder.
2. Stop the app (Ctrl+C).
3. Delete the data folder above. This also deletes downloaded models, unless you moved them elsewhere.
4. Remove the app: delete the project folder (from source), or run `pipx uninstall llm-configurator` (pipx).

## Next

- [How it works](how-it-works.md) – why a model does or doesn't fit
- [Command line](cli.md) – do everything without the browser
- [Troubleshooting](troubleshooting.md) – when something goes wrong
