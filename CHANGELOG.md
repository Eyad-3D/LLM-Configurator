# Changelog

All notable changes to LLM Configurator. Versions follow [semantic versioning](https://semver.org/) loosely while the project is below 1.0.

## [0.4.0] – unreleased

v0.4 turns "this should fit" into a full loop on your own computer: **pick → download → test → tune → use**.

### New
- **Get it running** panel on every recommendation, one step at a time: get the engine, download, test, tune, check quality and use it. Long jobs show their progress in a **Background tasks** tray and can be cancelled.
- **Automatic llama.cpp install** (llama.cpp is the engine that runs the models) from its official GitHub releases, matched to your computer: CUDA, Vulkan, Metal or CPU, with the reason shown. If a graphics build won't start, the CPU build is installed instead. You can also use your own build (`llm-config runtime use`) or install from a downloaded archive (`runtime install-archive`).
- **Resumable, verified model downloads**, including models split into several files. Cancelling keeps the partial file, and the next download continues from there. Gated models work with an `HF_TOKEN`.
- **Reuse models you already have** from the Hugging Face cache, LM Studio, Ollama and the app's own folder, or any GGUF file you add: **Model files already on this computer → Scan my disk** in the app, or `llm-config local`. Your own files are only ever reused, never downloaded or deleted.
- **Real tests**: a smoke test (does it answer sensibly?), reading speed, writing speed near your chosen context, first-word delay, and peak memory (loading included) next to the estimate. Plain verdicts: "It works", "It works, but slowly" (below your minimum speed) or "It didn't work".
- **Auto-tuning** of graphics-card layers, MoE expert placement, threads, flash attention and batch sizes within a time limit you choose: 1, 5 or 15 minutes in the app, 1–30 minutes with `llm-config tune --budget`. Settings that would not fit in memory are skipped. The winner is measured again at your full conversation length and kept only if it is really faster there. In the app, Test, **Start server** and export use a matching saved tune automatically; on the command line, add `--tuned`.
- **Quality checks**: short original quizzes (general, coding, agentic, documents) with honest error ranges, a long-document recall ("needle") test, a blind "try my prompts" comparison, and a compression-loss check using llama.cpp's KL divergence (a measure of how much compression changes the model's word choices).
- **Use it**: start a local OpenAI-compatible server (**Start server**, or `llm-config run`), or export ready-made settings for llama-server, Ollama, docker-compose, the OpenAI Python library, Continue, Open WebUI and LM Studio. **Start server** uses port 8080, the one the exports expect, when it is free.
- **Memory model**: compressed notepad (KV cache: q8_0 / q4_0), sliding-window layers, mixture-of-experts (MoE) models with experts kept in RAM, and Apple Silicon shared memory.
- **Wider hardware**: Apple Silicon (Metal), AMD graphics memory on Linux, friendly processor names and features. Graphics chips whose free memory can't be read are listed with the reason and never guessed.
- **Bigger catalogue**: 39 popular open models (Qwen3 dense and MoE, Qwen2.5-Coder, Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss, DeepSeek-R1 distills, small models), more compression levels, and your own repositories (`llm-config models add`, with `--config-repo` for gated originals, and `models remove`).
- **Plain verdict badges** on cards (Runs well / Runs slowly / Too slow / Not tested yet) and speed labels that say where each number comes from: measured, measured after tuning, tuned with a short test, estimated from your tests at other lengths, other people's results (community), or estimated.
- **Learning from your measurements** to narrow speed estimates for untested models.
- **Community results (opt-in)**: after a speed test, see exactly what would be shared and post it yourself as a GitHub issue (or `llm-config community share`). Download other people's results (**Get the latest shared results**, or `llm-config community import`) as a rough guide, always labelled "community".
- **New commands**: `runtime`, `local`, `test`, `tune`, `quiz`, `quant-check`, `export`, `run`, `community`, `settings`, and `models add` / `models remove`. See [docs/cli.md](docs/cli.md).
- **Packaging**: install with `pipx install llm-configurator` or run with `uvx llm-configurator serve` (once published). New `llm-configurator` command alias.
- **CI** on Windows, macOS and Linux, UI tests, a packaging check, a manual/weekly job that tests against a real llama.cpp build, and a tag-only release workflow for PyPI.
- **Docs**: a short README and detailed pages under `docs/`.

### Changed
- README shortened; the detailed explanations moved to `docs/`.
- Launch settings, background jobs and all-or-nothing saving are shared by testing, tuning, export and serving, so a tested configuration is exactly the one exported or served.
- The installed llama.cpp build decides the graphics settings. A Vulkan build on an NVIDIA card gets Vulkan settings, not CUDA's, and exports and shared results say Vulkan too. A processor-only run on a graphics build keeps the graphics card out completely.
- Memory estimates follow what llama.cpp really does, checked against its logs: the notepad is rounded up to 256 tokens, and on sliding-window models each user gets their own notepad.
- Downloading (in the app or with `download`) reuses a verified copy already on your disk instead of fetching it again.
- `llm-config bench` uses the installed llama-bench unless you pass `--executable`.
- The llama.cpp version is shown as its build number (for example `b8123`): new builds all report the same version name, so the number is what tells them apart.
- AMD cards are reported with the Vulkan backend, the build the app installs.
- The catalogue only offers model files that have a published fingerprint (SHA256) for every piece.

### Fixed
- One NVIDIA card with unreadable memory (`[N/A]` in `nvidia-smi`, for example on DGX Spark) made every NVIDIA card vanish from the scan. Now only that card is set aside, with the reason.
- Memory estimates for a model split between graphics card and RAM now count the output layer, which llama.cpp puts on the graphics card and which is often much bigger than a normal layer.
- `llm-config bench` put one layer too few on the graphics card in a split (llama.cpp counts the output layer as a layer), always asked for the device `CUDA0` (which fails on Vulkan and Metal builds), and let `LLAMA_ARG_*` environment variables change its settings.

### Security
- Model information and ranking requests no longer pass your `HF_TOKEN` or Artificial Analysis key on to another website when they are redirected, and they refuse redirects from HTTPS to plain HTTP. (Downloads already did this.)
- llama.cpp installs take only files from the official release page, check them against GitHub's published SHA256 (skippable only on the command line, with `--allow-unverified`), stop oversized downloads and "zip bombs" (tiny archives that unpack to a huge size), and unpack safely: paths that escape the folder, links pointing outside and device files are rejected.
- Model servers the app starts listen on `127.0.0.1` only. They let only pages on your own computer read their answers (`LLAMA_ARG_CORS_ORIGINS=localhost`; by default llama-server lets any website do it), switch off the `/slots` status page, and ignore any `LLAMA_ARG_*` or `LLAMA_API_KEY` variables you have set. Exported start scripts use the same CORS rule.
- Model servers stop with the app, also when its terminal is closed or it is stopped with `kill`, so no server is left holding memory and a port.
- The page never receives folder names from the app, only file names (except in export text you ask for), and the session-token check no longer gives hints through response timing.
- Community sharing sends nothing by itself and shares only a fixed list of fields, with serial numbers and your computer and user names removed. Community import accepts HTTPS only, at most 10 MB and 50,000 rows, and drops every row that fails strict checks.
- Model downloads never write more than the published size, never overwrite a different file with the same name, and refuse unsafe file names.

## [0.3.0] – 2026-09-15
### Added
- Bounded ~12-second hardware calibration (processor memory and 4-bit matrix proxy; NVIDIA memory and transfer speed) cached for 7 days.
- Low-confidence generation speed ranges on cards, with a likely-meets / borderline / likely-below assessment. Estimates never count as verified speed.
- `llm-config calibrate` and **Recalibrate speed**.

## [0.2.4] – 2026-09-15
### Changed
- Optional rankings chosen before the questions; the choice is remembered (never the key) and live loading progress is shown.

## [0.2.3] – 2026-09-15
### Changed
- Cached results show immediately; rankings update in the background. No network refresh on the cached-results path.

## [0.2.2] – 2026-09-15
### Changed
- Model information is prepared during guided setup; rankings are fetched separately.

## [0.2.1] – 2026-09-15
### Changed
- Themed, keyboard-accessible dropdowns; "concurrent sessions or agents" wording; applications sorted by memory use.

## [0.2.0] – 2026-09-15
### Added
- Guided setup (one question per screen), optional rankings, and concise comparisons with up to three recommendations.

## [0.1.1] – 2026-09-14
### Added
- Artificial Analysis key settings in the app, stored in the system password store, with connection testing.

## [0.1.0] – 2026-09-14
### Added
- First working version: hardware scan, Hugging Face metadata, memory estimates for dense Qwen/Llama models, local browser interface, CLI, verified single-file downloads and a llama-bench runner.
