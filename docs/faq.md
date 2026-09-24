# FAQ

**Does it cost anything?**
No, the app costs nothing, and the models it recommends are open models. Check each model's own licence before commercial use; the app doesn't filter by licence yet. (A licence for this project's own code has not been chosen yet.)

**Do I need a graphics card?**
No. Models run on the processor too, just slower. Small models (1–4B, meaning 1 to 4 billion parameters, a rough measure of a model's size) are often fine on a modern laptop processor.

**Does it send my prompts or data anywhere?**
No. Models run on your computer. The app only fetches public model information and the files you ask for. See [Privacy and safety](privacy-and-safety.md).

**Do I need an API key?**
No. The key is only for optional quality rankings from Artificial Analysis. Everything else works without it. (A Hugging Face token, `HF_TOKEN`, is only needed for models that require you to log in and accept a licence.)

**Why does it say "unknown" or "not tested yet" so often?**
Because it refuses to make up numbers. An unknown is honest; a guess dressed up as a fact is not. Run **Test** to replace unknowns with real measurements.

**How accurate are the memory estimates?**
They are careful engineering guesses, not guarantees. Loading peaks and different llama.cpp versions can use more. The Test step shows real peak memory next to the estimate so you can check. See [How it works](how-it-works.md).

**How accurate are the speed estimates?**
Low confidence, with wide ranges. They get narrower as you test models on your machine. Real tests always win.

**What is the difference between this and Ollama or LM Studio?**
Those are great for **running** models. LLM Configurator helps you **choose and set up** one for your hardware and needs, measures it, and can then hand the settings to Ollama or LM Studio (see [Using your model](using-your-model.md)). It uses llama.cpp, the same engine many of those tools build on.

**Can I use models I already downloaded?**
Yes. Under your recommendations, open **Model files already on this computer** and click **Scan my disk** (or run `llm-config local --scan`). It finds GGUF model files in the Hugging Face cache, LM Studio and Ollama. After that, **Download** uses your copy instead of fetching it again. Files that aren't in the app's list can still be tested and run. To add one file from anywhere: `llm-config local --add PATH`.

**Can I add a model that isn't in the list?**
Yes, from the command line: `llm-config models add BASE_REPO GGUF_REPO` with the original Hugging Face repository and the one containing GGUF files. It then fetches the file list; `llm-config models` shows the new IDs. Its architecture must be one the app understands ([list](how-it-works.md#supported-models)). If the original repository asks you to log in first, add `--config-repo` with a public copy of it (see [Gated models](getting-started.md#gated-models-hugging-face-licence)).

**Which compression level (quantisation) should I pick?**
Quantisation stores the model with fewer digits per number, like saving a photo as a smaller JPEG: smaller and faster, slightly less sharp. Q4_K_M is a common sweet spot. Q5_K_M/Q6_K are a little better and bigger; Q8_0 is close to the original. Use the [compression check](quality-checks.md#compression-check) to measure the difference for a model you care about.

**What does "sessions or agents at once" (or "chats at once") mean?**
How many conversations or agents run **at the same time**. One person running three agents in parallel counts as three. Using one chat after another counts as one. Each needs its own notepad (memory).

**Can other computers on my network use the model server?**
Not directly: servers listen on your own computer only (`127.0.0.1`), for safety. Exports keep that too, including the Docker one. If you really need network access, export the `llama-server` script and change its `--host` yourself; anyone who can reach that address can then use your model.

**Can a website I visit read my model's answers?**
Not from servers the app starts, or from the exported `llama-server` start script or Docker setup. By default llama.cpp lets any web page read its answers; these servers only let pages from your own computer (such as Open WebUI running locally) do that.

**Why are my own `LLAMA_ARG_...` settings ignored?**
llama.cpp can take settings from environment variables whose names start with `LLAMA_ARG_`. The app ignores any you have set when it tests, tunes or starts a model, so what runs is exactly what was tested. To change a setting, export the `llama-server` start script and edit it.

**Does it work offline?**
Demo mode works fully offline. With models and llama.cpp already downloaded, testing, tuning and running work offline too. Fetching model information and downloads need internet.

**Will it slow my computer down?**
Only while it tests, tunes or runs a model. Those jobs use your processor and graphics card heavily; they run one at a time and can be cancelled.

**Where is my data, and how do I remove it?**
See [Getting started → Where your data lives](getting-started.md#where-your-data-lives).

**Does it support Mac / AMD / Intel graphics?**
Partly. The app only puts a model on a graphics card whose free memory it can read: NVIDIA cards (through NVIDIA's `nvidia-smi` tool), AMD cards on Linux, and Apple Silicon Macs (where the processor and graphics share one pool of memory). Other cards, such as AMD or Intel graphics on Windows, are listed with their free memory shown as unknown instead of guessed, and models run on the processor there.

The engine it installs is matched to that: the Metal build on Apple Silicon, the CUDA build on NVIDIA (Vulkan if the NVIDIA driver is too old), the Vulkan build on AMD, otherwise a processor-only build. (Metal, CUDA and Vulkan are different "languages" for talking to a graphics card.) The installed engine then decides how the graphics card is used; the **Get the engine** step shows which. The quick hardware speed check (calibration) measures NVIDIA graphics cards only; on other cards, run **Test** for real speeds.

**Can one model use two graphics cards?**
Not yet.
