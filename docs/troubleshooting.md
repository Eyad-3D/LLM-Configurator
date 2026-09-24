# Troubleshooting

Start with the error message: the app tries to say what went wrong and what to do next. If that isn't enough, look for your problem below.

## Installing the app

**`llm-config: command not found`**
- pipx: run `pipx ensurepath`, then open a new terminal.
- From source: use the full path, `.venv/bin/llm-config` (Windows `.\.venv\Scripts\llm-config.exe`), or `python -m llm_configurator`.

**"requires a different Python" / install fails on old Python**
You need Python 3.10 or newer. Check with `python3 --version` (Windows `py --version`). Once the app is on PyPI, uv can fetch a suitable Python for you: `uvx --python 3.12 llm-configurator serve`.

**Saving the API key fails: "Could not save to the OS credential store"**
Your system password store is locked or missing. On Linux, install and unlock a keyring (GNOME Keyring or KWallet); on headless servers there often is none. Either untick **Remember on this computer** (key lasts for this session) or set the `AA_API_KEY` environment variable.

## Installing llama.cpp

**The app says there is no llama.cpp build for your computer**
llama.cpp doesn't publish a ready-made build for every system. Options:
- Install it yourself (for example `brew install llama.cpp` on macOS, or build it from source), then run `llm-config runtime use DIR`, or just make sure `llama-server` is on your `PATH`.
- Download a release archive by hand and run `llm-config runtime install-archive PATH`.

**Install refused because GitHub did not publish a checksum (a fingerprint, SHA256)**
The app couldn't find a fingerprint to check the download against. Try again later, install llama.cpp yourself (see above), or, only if you trust the source, run `llm-config runtime install --allow-unverified`.

**"Could not reach GitHub to look up llama.cpp releases"**
Check your internet connection, or install from a file you downloaded elsewhere with `llm-config runtime install-archive PATH`.

**Which build do I get?**
Only official builds from llama.cpp's GitHub releases page, checked against their published fingerprint (unless you used `--allow-unverified`).
- Windows and Linux: with an NVIDIA card, the CUDA build when llama.cpp publishes one for your system and your driver is new enough (the separate CUDA runtime files, "cudart", are downloaded with it). Otherwise Vulkan if any graphics card is found, otherwise the CPU build. ARM (arm64) builds where llama.cpp publishes them.
- macOS: Apple Silicon (Metal) or Intel.

If a graphics-card build won't start on your computer, the app installs the CPU build instead and tells you; updating your graphics driver may let the faster build work.

`llm-config runtime status` shows what is installed. If the graphics card isn't used (the app says llama.cpp could not use the graphics card, or found no usable one), update your graphics driver. On Linux the Vulkan build needs Vulkan drivers (for example the `mesa-vulkan-drivers` package, or your vendor's driver).

**Installing a CUDA build by hand**
`runtime install-archive` takes one file. A CUDA build also needs the matching `cudart` archive: unpack both into the same folder, then run `llm-config runtime use DIR`.

**The download of llama.cpp or a model keeps failing**
Downloads continue where they stopped, so first just try again. Some company networks block GitHub release downloads or Hugging Face. Try another network, use `runtime install-archive` with a llama.cpp file you fetched elsewhere, or, for models, set `HF_ENDPOINT` to a Hugging Face mirror (it must start with `https://`).

## Antivirus and quarantine

Freshly downloaded programs sometimes get blocked.

**Windows (Defender / SmartScreen).** Defender may quarantine `llama-server.exe` or other llama.cpp files, and the Test step then says the runtime is missing or fails to start. If you installed with `runtime install`, the app already checked the file's fingerprint against the official release. If you trust it: open **Windows Security → Virus & threat protection → Protection history**, restore the file, and optionally add an exclusion for the app's `runtime` folder (`%LOCALAPPDATA%\LLMConfigurator\runtime`). Then run `llm-config runtime status`.

**macOS (Gatekeeper).** If you downloaded an archive in your browser and used `runtime install-archive`, macOS may say the program "cannot be opened" or is from an "unidentified developer". Go to **System Settings → Privacy & Security** and click **Open Anyway**, or, for files you trust, remove the quarantine flag from that folder: `xattr -dr com.apple.quarantine "DIR"`. `llm-config runtime status` warns when a folder is marked this way and shows the command with the right folder.

**Linux.** "Permission denied" on a runtime you installed yourself: make it executable with `chmod +x llama-server`.

## Out of memory

Symptoms: the Test step says it ran out of memory, the computer becomes very slow, or the model crashes while loading.

Try, from easiest:
1. Close other big apps (browsers with many tabs, games, other AI apps) and test again.
2. Use a **smaller context** (less text in view).
3. Use a **compressed notepad** (`--kv q8_0`).
4. Put **fewer layers on the graphics card** (`--gpu-layers`), or for MoE models, keep more experts in RAM (a tune tries this for you).
5. Pick a **smaller compression level** (for example Q4_K_M instead of Q6_K) or a smaller model.
6. Raise **Extra RAM headroom** (or **Extra VRAM headroom**) under **Advanced hardware settings**, so recommendations leave more room.

The memory numbers are estimates. The Test step shows real peak memory next to the estimate; if the real number is much higher on your machine, please tell us (see [Contributing](contributing.md)).

On Apple Silicon, macOS limits how much memory the graphics side can use. See [How it works](how-it-works.md#apple-silicon-and-unified-memory-new-in-v04).

## Ports

**The app won't start: "address already in use"** (Windows: "only one usage of each socket address", error 10048)
Something else uses port 8765 (maybe another copy of the app). Run `llm-config serve --port 8766`.

**The model server says its port is in use**
The message says the network port is already used by another program. Choose another port (`llm-config run ... --port 8081`) or leave `--port` out so a free one is picked.

**A copied setup can't reach the server**
Exports assume port 8080, but the server started in the app uses a free port that can change after a restart. Change the port in your setup to match the address shown under **Use it**.

**Another device can't reach the server**
That is intentional: everything listens on `127.0.0.1`, your own computer only. See [Using your model](using-your-model.md#only-on-this-computer).

**The page says "Reload the application to refresh the session"**
The app was restarted and the page has an old session token. Reload the page.

**"Compare again first"**
The app only remembers the most recent set of recommendations (and forgets them when it restarts). Run the comparison again, then use **Get it running**.

## Models

**"Unsupported memory architecture" or the model type isn't supported**
Either the app's memory maths doesn't know this model type yet (see [supported models](how-it-works.md#supported-models)), or your llama.cpp is too old for it. Update llama.cpp (`llm-config runtime install`) and the app.

**Model download refused (HTTP 401 or 403)**
Some models are "gated": you must accept their licence on Hugging Face first. Accept it on the model's page, create an access token (read access is enough), set it as the `HF_TOKEN` environment variable, and try again.

**"Stop the running model server first"**
Tests, tunes, quizzes and comparisons load a model of their own, which may not fit in memory next to the running server. Click **Stop server** under **Use it**, then try again.

**A model I already have isn't found**
Run `llm-config local --scan --dir "FOLDER"`, or register the file with `llm-config local --add PATH`.

**Fingerprint (SHA256) mismatch after download**
The file was damaged or changed on the way. The app deletes the bad file for you; just download again.

**"already exists ... with a different size" (or "different contents")**
A different file with the same name is in the models folder. Move that file, or choose another models folder.

## Speed looks wrong

- "Speed unverified" / "Not tested yet": no test has been run for this setup. Run **Test**.
- Estimates are wide on purpose. Real tests always replace them.
- Laptops on battery, hot machines and busy computers are slower. Test in the conditions you'll really use.
- Measurements expire after 30 days, and only match the exact same file, hardware and settings.

## Still stuck?

Open an issue on [GitHub](https://github.com/Eyad-3D/LLM-Configurator/issues) with: your operating system, `llm-config runtime status` output, the error message, and the last lines of the log shown in the app. Remove anything private first.
