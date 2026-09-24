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

**"This llama.cpp release has no ready-made build (or no suitable build) for …"**
llama.cpp doesn't publish a ready-made build for every system. Options:
- Install it yourself (for example `brew install llama.cpp` on macOS, or build it from source), then run `llm-config runtime use DIR`, or just make sure `llama-server` is on your `PATH`.
- Download a release archive by hand and run `llm-config runtime install-archive PATH`.

**"GitHub did not publish a checksum (a digital fingerprint) for …"**
The app checks every llama.cpp download against a fingerprint (SHA256) that GitHub publishes, like matching a parcel against its seal. Without one it can't tell whether the file was damaged or swapped, so it refuses. Try again later, install llama.cpp yourself (see above), or, only if you trust the source, run `llm-config runtime install --allow-unverified`. The browser can't skip this check.

**"Could not reach GitHub to look up llama.cpp releases"**
Check your internet connection, or install from a file you downloaded elsewhere with `llm-config runtime install-archive PATH`.

**Which build do I get?**
Only official builds from llama.cpp's GitHub releases page, checked against their published fingerprint (unless you used `--allow-unverified`). `llm-config runtime status` shows the reason under "Why this build".
- **NVIDIA card** (Windows or Linux, with a working `nvidia-smi`): the CUDA build, when llama.cpp publishes one for your system, your driver is new enough, and the matching CUDA runtime files ("cudart") are published too; both are downloaded. Otherwise the Vulkan build, otherwise the CPU build.
- **AMD card on Linux** whose memory the app can read: the Vulkan build. ROCm builds are never picked automatically; to use one, install it yourself and run `llm-config runtime use DIR`.
- **Graphics chips whose free memory the app can't read** (Intel graphics, AMD or Intel cards on Windows, built-in AMD graphics, NVIDIA without `nvidia-smi`): the CPU build. The app would never put a model on such a chip, so a graphics build would only be a bigger download. The reason says so: "… was found, but its free memory cannot be read …".
- **No graphics card**: the CPU build.
- **macOS**: Metal on Apple Silicon, the CPU build on Intel Macs.
- ARM (arm64) builds where llama.cpp publishes them.

If a graphics-card build won't start on your computer, the app installs the CPU build instead and tells you; updating your graphics driver may let the faster build work.

If the graphics card isn't used (the app says llama.cpp could not use the graphics card, or found no usable one), update your graphics driver. On Linux the Vulkan build needs Vulkan drivers (for example the `mesa-vulkan-drivers` package, or your vendor's driver).

**The version shows as a number like `b8123`**
That is llama.cpp's build number, the same name its GitHub releases use. New llama.cpp builds all report the same version name (`0.1.0-dev`), so the build number is the only way to tell them apart. A build you made yourself from a source download without git history reports `b1`; that is normal.

**"Found llama-server in …, but it would not start or report its version"**
The program is damaged or needs system files it can't find. Reinstall with `llm-config runtime install`. On Windows and macOS, also check the next section (antivirus and quarantine).

**"Windows could not find a .dll file" or "files were removed right after unpacking"**
llama.cpp is one program plus helper files (`.dll` files such as `llama-server-impl.dll`) that must sit in the same folder. The official download includes all of them, so a missing one was almost always removed by antivirus software right after unpacking. Restore it and allow the folder as described in the next section, then install again. (Older versions of this app showed a Windows "System Error" box saying `llama-server-impl.dll was not found`; that is the same problem.)

**"The app now uses this new llama.cpp instead of the folder you chose earlier"**
You installed llama.cpp while the app was pointed at your own build (`runtime use`). The fresh install wins. To go back, run `llm-config runtime use DIR` with your folder again.

**Installing a CUDA build by hand**
`runtime install-archive` takes one file. A CUDA build also needs the matching `cudart` archive: unpack both into the same folder, then run `llm-config runtime use DIR`.

## Downloads

**A download stopped, failed or was cancelled**
Just start it again: it continues where it stopped. The unfinished part is kept as a `.part` file next to the model, checked, and then extended. Model downloads retry network hiccups by themselves and only give up after several tries in a row without progress ("The network kept failing while downloading …"). If the server's file no longer lines up with the saved part, the part is deleted and the download starts from zero.

**It keeps failing**
Some company networks block GitHub release downloads or Hugging Face. Try another network, use `runtime install-archive` with a llama.cpp file you fetched elsewhere, or, for models, set `HF_ENDPOINT` to a Hugging Face mirror (it must start with `https://`).

**"Not enough free disk space"**
The app wants the download size plus 1 GiB spare. Free some space, or move the models folder (`llm-config settings --models-dir DIR`).

**Model download refused (HTTP 401 or 403)**
Some models are "gated": you must accept their licence on Hugging Face first. Accept it on the model's page, create an access token (read access is enough), set it as the `HF_TOKEN` environment variable, and try again.

**"… did not match its published SHA256"**
The file was damaged or changed on the way. The app deletes the bad part for you; just download again.

**"already exists ... with a different size" (or "different contents")**
A different file with the same name is in the models folder. Move that file, or choose another models folder.

## Antivirus and quarantine

Freshly downloaded programs sometimes get blocked.

**Windows (Defender / SmartScreen).** Defender may quarantine `llama-server.exe` or other llama.cpp files, and the Test step then says the runtime is missing or fails to start. If you installed with `runtime install`, the app already checked the file's fingerprint against the official release. If you trust it: open **Windows Security → Virus & threat protection → Protection history**, restore the file, and optionally add an exclusion for the app's `runtime` folder (`%LOCALAPPDATA%\LLMConfigurator\runtime`). Then run `llm-config runtime status`.

**macOS (Gatekeeper).** Your browser marks files it downloads as "from the internet" (a quarantine flag). If you downloaded llama.cpp in your browser and used `runtime install-archive` or `runtime use`, macOS may say the program "cannot be opened" or is from an "unidentified developer". Go to **System Settings → Privacy & Security** and click **Open Anyway**, or, for files you trust, remove the flag from that folder: `xattr -dr com.apple.quarantine "DIR"`. `llm-config runtime status` warns when llama.cpp is marked this way ("macOS has marked this llama.cpp as downloaded from the internet …") and shows the command with the right folder.

**Linux.** "Permission denied" on a runtime you installed yourself: make it executable with `chmod +x llama-server`.

## Out of memory

Symptoms: the Test step says it ran out of memory, the computer becomes very slow, or the model crashes while loading.

What the messages mean:
- **"Not enough memory to load this model with these settings"**, **"The model ran out of memory while loading"**, **"The speed test ran out of memory"** or, while tuning, **"Ran out of memory with these settings"**: llama.cpp itself said it ran out of memory.
- **"The system stopped the model server, most likely because memory ran out"** (or "… stopped llama-bench, possibly because memory ran out"): your operating system ended the program without a word from llama.cpp. Memory is the usual reason, but it is a guess.
- **"llama.cpp could not load the model with these settings and gave no reason"**: it may need more memory than is free, or the file may be damaged.
- **"Probably not enough memory: a lighter setting of the same model worked just before"**: only during a tune, when a lighter setting of the same model had just worked.
- A tune note saying your **full conversation length** did not run: the context you chose may not fit in memory with those settings.

About llama-bench: the speed test and tune use llama.cpp's benchmark program, llama-bench. On its own it only says "failed to load model" or "failed to create context" and hides the cause. The app always runs it with `-v` ("verbose": tell me more), so the messages above can name the cause. `llm-config bench` does not add `-v`, so its errors show only that short line; use the Test step or `llm-config test` to see why.

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

A port is a numbered door on your computer that a program listens behind; only one program can use each door.

**The app won't start: "address already in use"** (Windows: "only one usage of each socket address", error 10048)
Something else uses port 8765 (maybe another copy of the app). Run `llm-config serve --port 8766`.

**"The network port is already used by another program"**
The model server's port is taken. With `llm-config run`, choose another port (`--port 8081`) or leave `--port` out so a free one is picked. In the app, **Start server** uses port 8080 when it is free and otherwise picks a free one by itself.

**A copied setup can't reach the server**
Exports assume port 8080. **Start server** uses 8080 when it can, but moves to another free port if 8080 was taken, and `llm-config run` picks a free port unless you give `--port`. Change the port in your setup to match the address shown under **Use it** (or start with `llm-config run ... --port 8080`, or export with `llm-config export ... --port N`).

**Another device can't reach the server**
That is intentional: everything listens on `127.0.0.1`, your own computer only. See [Using your model](using-your-model.md#only-on-this-computer).

**The page says "Reload the application to refresh the session"**
The app was restarted and the page has an old session token. Reload the page.

**"Compare again first"**
The app only remembers the most recent set of recommendations (and forgets them when it restarts). Run the comparison again, then use **Get it running**.

## Models

**"Unsupported memory architecture" or the model type isn't supported**
Either the app's memory maths doesn't know this model type yet (see [supported models](how-it-works.md#supported-models)), or your llama.cpp is too old for it. Update llama.cpp (`llm-config runtime install`) and the app.

**"The text is longer than the model's context window"**
The context window is the model's short-term notepad. Your text (for example in **Try my prompts**) doesn't fit in the context you picked. Pick a larger context (it needs more memory), or use a shorter text. The speed test sizes its own text to fit, so you shouldn't see this there.

**"Stop the running model server first"**
Tests, tunes, quizzes and comparisons load a model of their own, which may not fit in memory next to the running server. Click **Stop server** under **Use it**, then try again.

**A model I already have isn't found**
In the app, open **Model files already on this computer** and click **Scan my disk**. On the command line, run `llm-config local --scan --dir "FOLDER"`, or register the file with `llm-config local --add PATH`.

**"This model came from a file on your computer, and the file is no longer there"**
(On the command line: "The file for … is no longer where it was found".) The app never downloads a file you brought yourself, so it can only reuse it. Put the file back, or, if you moved it, scan again (**Scan my disk**, or `llm-config local --scan`) and pick the model from the new list. In the app, the download step shows **Your file is missing** with a **Check again** button.

## Speed looks wrong

- "Speed unverified" / "Not tested yet": no test has been run for this setup. Run **Test**.
- "Tuned with a short test · not checked at this length yet": the tune measured near the start of a conversation, not at your full length. Run **Test** to measure it properly.
- Estimates are wide on purpose. Real tests always replace them.
- Laptops on battery, hot machines and busy computers are slower. Test in the conditions you'll really use.
- Measurements expire after 30 days, and only match the exact same file, hardware and settings.

## Still stuck?

Open an issue on [GitHub](https://github.com/Eyad-3D/LLM-Configurator/issues) with: your operating system, `llm-config runtime status` output, the error message, and the last lines of the log shown in the app. Remove anything private first.
