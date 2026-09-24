# Changelog

All notable changes to LLM Configurator. Versions follow [semantic versioning](https://semver.org/) loosely while the project is below 1.0.

## [0.4.0] – unreleased

v0.4 turns "this should fit" into a full loop on your own computer: **pick → download → test → tune → use**.

### Added
- **Get it running** panel on every recommendation: runtime, download, test, tune, quality check and use, with a jobs tray for background work and cancel buttons.
- **Automatic llama.cpp install** from official GitHub releases, matched to your system (CUDA, Vulkan, Metal or CPU), checked against its published SHA256 and unpacked safely. Also: use your own build, or install from a local archive (`llm-config runtime ...`).
- **Resumable, verified downloads**, including models split into several files. Cancelling keeps the partial file, and the next download picks up where it stopped.
- **Reuse models already on disk** from the Hugging Face cache, LM Studio and Ollama, or any GGUF file you add (`llm-config local`).
- **Real tests**: smoke test (does it answer sensibly?), reading speed, writing speed near your chosen context, first-word delay, and peak memory next to the estimate. Plain verdicts: works / works slowly / failed.
- **Auto-tuning** of threads, batch sizes, flash attention, graphics-card layers, notepad (KV cache) compression and MoE expert placement within a time limit you choose (1, 5 or 15 minutes in the app; 1–30 minutes with `llm-config tune --budget`), skipping settings that would not fit in memory.
- **Quality checks**: short original quizzes (general, coding, agentic, documents) with honest error ranges, long-document recall ("needle") test, blind "try my prompts" comparison, and a compression-loss check using llama.cpp's KL divergence.
- **Use it**: start a local OpenAI-compatible server (`llm-config run`) or export settings for llama-server, Ollama, docker-compose, the OpenAI Python library, Continue, Open WebUI and LM Studio.
- **Memory model**: KV cache compression (q8_0 / q4_0), sliding-window layers, mixture-of-experts models with experts kept in RAM, Apple Silicon unified memory.
- **Wider hardware**: Apple Silicon (Metal), AMD graphics memory on Linux, friendly processor names and features; unknown graphics memory stays unknown.
- **Bigger catalogue**: popular open models (Qwen3 dense and MoE, Qwen2.5-Coder, Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss, DeepSeek-R1 distills, small models), more compression levels, and add-your-own repositories (`llm-config models add`).
- **Plain verdict badges** on cards (Runs well / Runs slowly / Too slow / Not tested yet) and evidence labels (measured, tuned, interpolated, community, estimated).
- **Learning from your measurements** to narrow speed estimates for untested models.
- **Opt-in community results** (command line: `llm-config community share` / `import`): anonymised sharing via a pre-filled GitHub issue you submit yourself, and import of shared results, shown as "community" speeds.
- **Packaging**: install with `pipx install llm-configurator` or run with `uvx llm-configurator serve` (once published). New `llm-configurator` command alias.
- **CI** on Windows, macOS and Linux, UI tests, a packaging check, a manual/weekly job that tests against a real llama.cpp build, and a tag-only release workflow for PyPI.
- **Docs**: a short README and detailed pages under `docs/`.

### Changed
- README shortened; the detailed explanations moved to `docs/`.
- Launch settings, background jobs and atomic storage are shared foundations used by testing, tuning, export and serving, so a tested configuration is exactly the one exported or served.

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
