# Troubleshooting

Start with the error message: the app tries to say what went wrong and what to do next. If that isn't enough, look for your problem below.

## Installing the app

**`llm-config: command not found`**
- pipx: run `pipx ensurepath`, then open a new terminal.
- From source: use the full path, `.venv/bin/llm-config` (Windows `.\.venv\Scripts\llm-config.exe`), or `python -m llm_configurator`.

**"requires a different Python" / install fails on old Python**
You need Python 3.10 or newer. Check with `python3 --version` (Windows `py --version`). uv can fetch a suitable Python for you: `uvx --python 3.12 llm-configurator serve`.

**Saving the API key fails: "Secure credential storage is unavailable"**
Your system password store is locked or missing. On Linux, install and unlock a keyring (GNOME Keyring or KWallet); on headless servers there often is none. Either untick **Remember on this computer** (key lasts for this session) or set the `AA_API_KEY` environment variable.

## Installing llama.cpp

**The app says there is no llama.cpp build for your computer**
llama.cpp doesn't publish a ready-made build for every system. Options:
- Install it yourself (for example `brew install llama.cpp` on macOS, or build it from source), then run `llm-config runtime use DIR`, or just make sure `llama-server` is on your `PATH`.
- Download a release archive by hand and run `llm-config runtime install-archive PATH`.

**Install refused because the download has no published fingerprint (SHA256)**
The app couldn't find a fingerprint to check the download against. Wait for the next release, install llama.cpp yourself (see above), or, only if you trust the source, run `llm-config runtime install --allow-unverified`.

**Which build do I get?**
- Windows: CUDA if you have an NVIDIA card, otherwise Vulkan, otherwise CPU.
- macOS: Apple Silicon (Metal) or Intel.
- Linux: Vulkan if a graphics card is found, otherwise CPU.

`llm-config runtime status` shows what is installed. If the graphics card isn't used ("GPU backend unavailable"), update your graphics driver. On Linux the Vulkan build needs Vulkan drivers (for example the `mesa-vulkan-drivers` package, or your vendor's driver).

**The download of llama.cpp or a model keeps failing**
Some company networks block GitHub release downloads or Hugging Face. Try another network, or use `runtime install-archive` with a file you fetched elsewhere.

## Antivirus and quarantine

Freshly downloaded programs sometimes get blocked.

**Windows (Defender / SmartScreen).** Defender may quarantine `llama-server.exe` or other llama.cpp files, and the Test step then says the runtime is missing or fails to start. The app already checked the file's fingerprint against the official release. If you trust it: open **Windows Security → Virus & threat protection → Protection history**, restore the file, and optionally add an exclusion for the app's `runtime` folder (`%LOCALAPPDATA%\LLMConfigurator\runtime`). Then run `llm-config runtime status`.

**macOS (Gatekeeper).** Downloads made by the app itself are normally not quarantined. If you downloaded an archive in your browser and used `runtime install-archive`, macOS may say the program "cannot be opened" or is from an "unidentified developer". Go to **System Settings → Privacy & Security** and click **Open Anyway**, or, for files you trust, remove the quarantine flag from that folder: `xattr -dr com.apple.quarantine "DIR"`.

**Linux.** "Permission denied" on a runtime you installed yourself: make it executable with `chmod +x llama-server`.

## Out of memory

Symptoms: the Test step fails with "not enough memory", the computer becomes very slow, or the model crashes while loading.

Try, from easiest:
1. Close other big apps (browsers with many tabs, games, other AI apps) and test again.
2. Use a **smaller context** (less text in view).
3. Use a **compressed notepad** (`--kv q8_0`).
4. Put **fewer layers on the graphics card** (`--gpu-layers`), or for MoE models, keep more experts in RAM.
5. Pick a **smaller compression level** (for example Q4_K_M instead of Q6_K) or a smaller model.
6. Raise the memory to keep free in the advanced options, so recommendations leave more room.

The memory numbers are estimates. The Test step shows real peak memory next to the estimate; if the real number is much higher on your machine, please tell us (see [Contributing](contributing.md)).

On Apple Silicon, macOS limits how much memory the graphics side can use. See [How it works](how-it-works.md#apple-silicon-and-unified-memory-new-in-v04).

## Ports

**The app won't start: "address already in use"**
Something else uses port 8765 (maybe another copy of the app). Run `llm-config serve --port 8766`.

**The model server says "port in use"**
Choose another port (`llm-config run ... --port 8081`) or leave `--port` out so a free one is picked.

**Another device can't reach the server**
That is intentional: everything listens on `127.0.0.1`, your own computer only. See [Using your model](using-your-model.md#only-on-this-computer).

**The page says "Reload the application to refresh the session"**
The app was restarted and the page has an old session token. Reload the page.

## Models

**"Unsupported model architecture"**
Either the app's memory maths doesn't know this model type yet (see [supported models](how-it-works.md#supported-models)), or your llama.cpp is too old for it. Update llama.cpp (`llm-config runtime install`) and the app.

**Model download needs login (401/403)**
Some models require you to accept terms on Hugging Face. Accept them on the model's page, create a read token, and set `HF_TOKEN`.

**A model I already have isn't found**
Run `llm-config local --scan --dir "FOLDER"`, or register the file with `llm-config local --add PATH`.

**Fingerprint mismatch after download**
The file is damaged or changed. Delete it (the app's **Remove**, or delete the file) and download again.

## Speed looks wrong

- "Speed unverified" / "Not tested yet": no test has been run for this setup. Run **Test**.
- Estimates are wide on purpose. Real tests always replace them.
- Laptops on battery, hot machines and busy computers are slower. Test in the conditions you'll really use.
- Measurements expire after 30 days, and only match the exact same file, hardware and settings.

## Still stuck?

Open an issue on [GitHub](https://github.com/Eyad-3D/LLM-Configurator/issues) with: your operating system, `llm-config runtime status` output, the error message, and the last lines of the log shown in the app. Remove anything private first.
