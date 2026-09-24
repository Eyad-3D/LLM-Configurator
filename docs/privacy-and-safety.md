# Privacy and safety

Short version: **everything runs on your computer, nothing is tracked, and nothing is downloaded or shared unless you ask.**

## What goes over the network

The app sends **no usage data or telemetry**. It only makes these requests, all over HTTPS (encrypted web connections):

| Destination | When | What is sent / fetched |
|---|---|---|
| **Hugging Face** (`huggingface.co`) | Guided setup, **Refresh model metadata**, `refresh`, `models add` | Public model information: sizes, layers, file list. Not the model files. Your `HF_TOKEN`, if you set one. |
| **Hugging Face** (`huggingface.co`, or your `HF_ENDPOINT` mirror) | Only when you click **Download** or run `download` | The model file(s) you chose. Your `HF_TOKEN`, if you set one. |
| **GitHub** (`api.github.com` and GitHub release downloads) | Only when you click **Install the engine** or run `runtime install` | Asks for the newest official llama.cpp release and downloads the build for your system (for an NVIDIA CUDA build, also the matching CUDA runtime file). No account or computer details are sent. |
| **GitHub** (`raw.githubusercontent.com`) | Only when you click **Get the latest shared results** or run `community import` | Downloads the public community results file (or the `https://` address you give with `--source`). Nothing about your computer is sent. |
| **Artificial Analysis** (`artificialanalysis.ai`) | Only if you turned on rankings and gave a key, or clicked **Test connection** | Your key (to authorise) and requests for public benchmark scores. |

Your hardware details, prompts, test results, file paths and keys are **not** sent anywhere, except each key to the one service it belongs to. Sharing speed results is a separate step that you do yourself (see [Community results](#community-results)); the app never posts anything.

**Redirects.** Sometimes a site answers "that file is over there" and points to another address (a *redirect*). The app follows it only if the new address is also HTTPS. If it points to a **different site**, the app first removes your Hugging Face token or Artificial Analysis key, like handing your house key to the front desk but not to the courier they call. This applies to model information, rankings and downloads.

A Hugging Face token (`HF_TOKEN`) is optional and only needed for "gated" models, which require you to log in and accept a licence. It goes only to Hugging Face, or for downloads to the mirror you set with `HF_ENDPOINT`.

## Keys and passwords

- The Artificial Analysis key is kept in your **system password store**: Windows Credential Manager, macOS Keychain, or a Linux Secret Service / KWallet keyring.
- If that store is locked or missing, saving fails with a clear message. The app **never** falls back to writing the key into a plain file.
- Untick **Remember on this computer** to keep the key in memory only until the app closes.
- Instead of saving a key, you can set the `AA_API_KEY` environment variable. A key saved in the app, or entered for this session, wins over it.
- The key is never shown back to the page, never put in browser storage, never written to the app's database or logs.
- **Remove key** in the app deletes it from the password store.
- Saved keys are shared by every copy of the app under the same system user account.

## Downloads

**Models:**

- Downloaded only when you ask, after showing the size and your free disk space.
- The app only offers model files that have a published **SHA256 fingerprint** for every piece. A fingerprint is a unique code for a file's contents: if a single byte differs, the code differs.
- Every file (every piece of a split model) is checked against its fingerprint. A download that doesn't match is deleted, never used.
- The app never writes more than the published size, and never overwrites a different file that has the same name.
- Interrupted downloads keep a `.part` file and resume later.
- Removing a model deletes only that model's files (and its `.part` files) inside the models folder, and only when their size matches.
- Model files found elsewhere on your computer (by a scan or with `local --add`) are reused, never deleted. A file that isn't in the built-in list is never downloaded either: if it has gone, the app says so.

**The llama.cpp engine:**

- Comes only from the official `ggml-org/llama.cpp` GitHub releases. A file whose address is not on that release page is refused.
- After saving, the app checks the file on disk against the SHA256 fingerprint GitHub publishes for it. If no fingerprint is published, it **refuses** to install, unless you run `llm-config runtime install --allow-unverified` on the command line. (The browser can't do this.)
- A download bigger than GitHub announced is stopped and deleted. Unpacking stops at 8 GiB or 20,000 files, which protects against "zip bombs" (tiny archives that unpack to a huge size).
- Archives are unpacked safely: files that try to escape the target folder (absolute paths, drive letters, `..`, links pointing outside) and special device files are rejected.
- The new build is unpacked into a hidden staging folder and must start (answer `--version`) before it replaces anything.
- It is installed inside the app's own data folder (`runtime/`), not system-wide, and not added to your `PATH`.
- A fresh install becomes the one the app uses, even if you had pointed it at your own build with `runtime use`. The app tells you when that happens.
- You can use your own build instead: `llm-config runtime use DIR`, or `llm-config runtime install-archive PATH` for an archive you downloaded. These have no official fingerprint to compare with, so use them only with files you trust.

**Models you already have:** the local search only reads file names, sizes and the small header at the start of GGUF files. It does not hash (fingerprint) big files unless needed to confirm a match. It never moves, renames or deletes them.

## The local app and servers

The app runs a small web server that only your own browser should talk to. It protects it like this:

- It listens on **`127.0.0.1` only**, a private address that only your own computer can reach.
- Every request must be addressed to `127.0.0.1` or `localhost` on the app's port. This blocks a trick where an outside website pretends to be your computer ("DNS rebinding").
- Every data request needs a random **session token** that changes each start. Only the page files themselves load without it. Actions must also come from the app's own page (or from no web page at all) and carry a small request (at most 64 KB). Together, this stops other websites open in your browser from controlling the app.
- The page loads only its own scripts and styles, and can't be shown inside another website.
- The page **cannot** send file paths, web addresses, program names or command-line options. It only sends IDs from lists the app made itself, choices from fixed menus, and bounded numbers or text. Anything that touches files or programs comes from the app's own folders or from command-line arguments you type.
- The page never sees your folders: replies show only file names, and known folders such as your home folder appear as `~`. The one exception is export text you ask for, since a start script has to name the model file.

Model servers and other llama.cpp programs the app starts:

- Listen on `127.0.0.1` only. The app refuses any other address.
- llama-server normally lets **any** website open in your browser read its answers. (Its "CORS" setting is the guest list of websites allowed to read its replies, and by default everyone is on it.) The app sets `LLAMA_ARG_CORS_ORIGINS=localhost`, so only pages on your own computer can. Exported start scripts set it too.
- llama.cpp also reads settings from `LLAMA_ARG_…` environment variables. The app removes any you have set before it starts a llama.cpp program (and `LLAMA_API_KEY` for model servers), so a forgotten variable can't quietly change what was tested.
- llama-server's `/slots` status page (settings and token counts of each request) is switched off on servers the app starts.
- They stop when the app stops, also when you close its terminal window.

Also:

- Heavy jobs (tests, tuning, quizzes) run one at a time and can be cancelled.
- Demo mode never downloads, tests or starts anything.
- Reports exported from the browser leave out process IDs and names. Raw `scan` and `recommend --json` output from the command line include them (it is your own machine's data; check before you share it).

## Your local data

Stored in the app's data folder (see [Getting started](getting-started.md#where-your-data-lives)). Most of it lives in one small database file, `cache.sqlite3`:

- cached model information and rankings,
- your hardware speed check, test, tune and quality results,
- your last 20 "try my prompts" comparisons, **including the prompts you typed and the answers**,
- imported community results,
- settings, a list of model files found on disk, and their fingerprints (so big files aren't re-checked every time).

Next to it:

- `catalogue.json`: only if you added or removed models,
- `models/`: downloaded models,
- `runtime/`: llama.cpp,
- `logs/llama-server.log`: llama.cpp's messages from the server you last started with **Start server**, replaced at each start,
- `tmp/`: work space for the compression check; its large temporary file is deleted when the check ends.

The list of running apps is read for the memory view but never saved. Other model servers (for tests, quizzes and `llm-config run`) write llama.cpp's messages to a temporary file that is deleted when they stop.

Delete the data folder to remove everything. If you moved the models folder (`settings --models-dir` or `LLM_CONFIG_MODELS`), delete that folder too.

## Community results

Sharing helps others know what speed to expect. It is **opt-in and manual**:

1. Choose a result to share. In the app: after a speed test, click **Show what would be shared** under **Share this result (optional)**. On the command line: `llm-config community share MEASUREMENT_ID ...`.
2. The app builds an **anonymised** copy and a link to a pre-filled GitHub issue on this project. Nothing is sent at this point.
3. **You** read exactly what will be posted, open the link (**Open GitHub to post it**) and submit it yourself with your GitHub account. If the text is too long for a link, copy it and paste it into the form.

Results measured before a hardware or driver change are left out, because the description of your computer would no longer match them. The app tells you how many were left out.

The anonymised copy contains only:

- the model: fingerprint, compression level and layer count, plus repository and file name **only for models from the built-in list** (for models you added or found on disk, only the fingerprint, since a private name could identify you),
- a hardware **class**: processor and graphics card names with serial numbers, IDs and your computer and user names removed; the kind of llama.cpp build that ran it (a Vulkan build on an NVIDIA card is shared as Vulkan); RAM/VRAM rounded to 4 GiB steps; whether memory is shared (Apple); operating system family with only the major version for Windows/macOS (for example `Windows 11`, `macOS 14`; none for Linux); and chip family (`x86_64` / `arm64`),
- the kind of test, the settings used (placement, graphics-card layers, number of users at once, threads, flash attention, KV cache type, batch sizes, MoE experts kept in RAM), the speeds measured (writing speed, reading speed, first-word delay), context and depth,
- the llama.cpp build number and backend, the **month** of the test (for example `2026-09`), and a new random ID.

It never contains your computer's name, user name, file paths, process IDs, device serial numbers (UUIDs), hardware fingerprint or exact time.

Note: the GitHub issue is public and linked to your GitHub account.

**Importing** (**Get the latest shared results** under **Speed results shared by other people**, or `llm-config community import`) downloads a public file over HTTPS only, with a 10 MB and 50,000-row limit and strict checks. Rows that don't match the expected format are dropped. If no row can be read, your saved results are kept. Community speeds are labelled **community**, never "measured". See `community/README.md` for the file format.

## Reporting a security problem

Please open a GitHub issue without exploit details, or contact the maintainer privately through GitHub, and we'll arrange a private channel.
