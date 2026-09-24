# Privacy and safety

Short version: **everything runs on your computer, nothing is tracked, and nothing is downloaded or shared unless you ask.**

## What goes over the network

The app sends **no usage data or telemetry**. It only makes these requests, all over HTTPS:

| Destination | When | What is sent / fetched |
|---|---|---|
| **Hugging Face** (`huggingface.co`) | Guided setup, **Refresh model metadata**, `refresh`, `models add` | Public model information: sizes, layers, file list. Not the model files. |
| **Hugging Face** (`huggingface.co`, or your `HF_ENDPOINT` mirror) | Only when you click **Download** or run `download` | The model file(s) you chose. |
| **GitHub** (`api.github.com` and GitHub release downloads) | Only when you click the install button or run `runtime install` | Asks for the latest official llama.cpp release and downloads the build for your system (for an NVIDIA CUDA build, also the matching CUDA runtime file). |
| **GitHub** (`raw.githubusercontent.com`) | Only when you run `community import` | Downloads the public community results file (or the `https://` address you give with `--source`). |
| **Artificial Analysis** (`artificialanalysis.ai`) | Only if you turned on rankings and gave a key, or clicked **Test connection** | Your key (to authorise) and requests for public benchmark scores. |

Your hardware details, prompts, test results, file paths and keys are **not** sent anywhere, except the key to Artificial Analysis itself.

A Hugging Face token (`HF_TOKEN`) is optional and only needed for models that require login. It is sent to Hugging Face only, and removed if a download is redirected to another host.

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
- Every file (every piece of a split model) is checked against its published **SHA256 fingerprint** (a unique code; if a single byte differs, the code differs). A file that doesn't match is not used.
- Interrupted downloads keep a `.part` file and resume later.
- Only HTTPS redirects are followed.
- Removing a model deletes only that model's files (and its `.part` files) inside the models folder.

**The llama.cpp engine:**

- Comes from the official `ggml-org/llama.cpp` GitHub releases.
- The app checks the file against the SHA256 fingerprint GitHub publishes for it. If no fingerprint is published, it **refuses** to install, unless you run `llm-config runtime install --allow-unverified` on the command line. (The browser can't do this.)
- Archives are unpacked safely: files that try to escape the target folder (absolute paths, `..`, links pointing outside) are rejected.
- It is installed inside the app's own data folder (`runtime/`), not system-wide, and not added to your `PATH`.
- You can use your own build instead: `llm-config runtime use DIR`.

**Models you already have:** the local search only reads file names, sizes and the small header at the start of GGUF files. It does not hash (fingerprint) big files unless needed to confirm a match.

## The local app and servers

- The app listens on **`127.0.0.1` only**, so only your own computer can reach it. It rejects requests with a different host or origin, and every data request needs a random session token that changes each start (only the page files themselves load without it). This stops other websites open in your browser from controlling it.
- The page **cannot** send file paths, web addresses, program names or command-line options. It only sends IDs from lists the app made itself, choices from fixed menus, and bounded numbers or text. Anything that touches files or programs comes from the app's own folders or from command-line arguments you type.
- Model servers the app starts also listen on `127.0.0.1` only.
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

Next to it: `catalogue.json` (only if you added or removed models), `models/` (downloads) and `runtime/` (llama.cpp).

The list of running apps is read for the memory view but never saved. While a model server runs, its log goes to a temporary file that is deleted when the server stops.

Delete the data folder to remove everything. If you moved the models folder (`settings --models-dir` or `LLM_CONFIG_MODELS`), delete that folder too.

## Community results

Sharing helps others know what speed to expect. It is **opt-in and manual**:

1. You choose which test results to share: `llm-config community share MEASUREMENT_ID ...` (on the command line; the app page has no share button yet).
2. The app builds an **anonymised** copy and a link to a pre-filled GitHub issue on this project.
3. **You** open the link, read exactly what will be posted, and submit it yourself with your GitHub account. The app never posts anything.

The anonymised copy contains only:

- the model: fingerprint, compression level and layer count, plus repository and file name **only for models from the built-in list** (for models you added or found on disk, only the fingerprint, since a private name could identify you),
- a hardware **class**: cleaned processor and graphics card names, backend, RAM/VRAM rounded to 4 GiB steps, whether memory is shared (Apple), operating system family with only the major version for Windows/macOS (for example `Windows 11`, `macOS 14`; none for Linux), and chip family (`x86_64` / `arm64`),
- the settings used (graphics-card layers, threads, flash attention, KV cache type, batch sizes, MoE placement), the speeds measured (writing speed, reading speed, first-word delay), context and depth,
- the llama.cpp build and backend, the **month** of the test (for example `2026-09`), and a new random ID.

It never contains your computer's name, user name, file paths, process IDs, device serial numbers (UUIDs), hardware fingerprint or exact time.

Note: the GitHub issue is public and linked to your GitHub account.

**Importing** (`llm-config community import`) downloads a public file over HTTPS only, with a 10 MB limit and strict checks. Rows that don't match the expected format are dropped. Community speeds are labelled **community**, never "measured". See `community/README.md` for the file format.

## Reporting a security problem

Please open a GitHub issue without exploit details, or contact the maintainer privately through GitHub, and we'll arrange a private channel.
