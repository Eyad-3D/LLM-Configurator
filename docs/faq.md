# FAQ

**Does it cost anything?**
No, the app costs nothing, and the models it recommends are open models. Check each model's own licence before commercial use; the app doesn't filter by licence yet. (A licence for this project's own code has not been chosen yet.)

**Do I need a graphics card?**
No. Models run on the processor too, just slower. Small models (1–4B) are often fine on a modern laptop processor.

**Does it send my prompts or data anywhere?**
No. Models run on your computer. The app only fetches public model information and the files you ask for. See [Privacy and safety](privacy-and-safety.md).

**Do I need an API key?**
No. The key is only for optional quality rankings from Artificial Analysis. Everything else works without it.

**Why does it say "unknown" or "not tested yet" so often?**
Because it refuses to make up numbers. An unknown is honest; a guess dressed up as a fact is not. Run **Test** to replace unknowns with real measurements.

**How accurate are the memory estimates?**
They are careful engineering guesses, not guarantees. Loading peaks and different llama.cpp versions can use more. The Test step shows real peak memory next to the estimate so you can check. See [How it works](how-it-works.md).

**How accurate are the speed estimates?**
Low confidence, with wide ranges. They get narrower as you test models on your machine. Real tests always win.

**What is the difference between this and Ollama or LM Studio?**
Those are great for **running** models. LLM Configurator helps you **choose and set up** one for your hardware and needs, measures it, and can then hand the settings to Ollama or LM Studio (see [Using your model](using-your-model.md)). It uses llama.cpp, the same engine many of those tools build on.

**Can I use models I already downloaded?**
Yes. It finds GGUF files from the Hugging Face cache, LM Studio and Ollama, and you can add any GGUF file: `llm-config local --add PATH`.

**Can I add a model that isn't in the list?**
Yes: `llm-config models add BASE_REPO GGUF_REPO` with the original Hugging Face repository and the one containing GGUF files. Its architecture must be one the app understands ([list](how-it-works.md#supported-models)).

**Which compression level (quantisation) should I pick?**
Q4_K_M is a common sweet spot. Q5_K_M/Q6_K are a little better and bigger; Q8_0 is close to the original. Use the [compression check](quality-checks.md#compression-check) to measure the difference for a model you care about.

**What does "chats at once" (users) mean?**
How many conversations or agents run **at the same time**. One person running three agents in parallel counts as three. Using one chat after another counts as one. Each needs its own notepad (memory).

**Can other computers on my network use the model server?**
Not directly: servers listen on your own computer only, for safety. Use an export (for example Docker) and set up access yourself if you need that.

**Does it work offline?**
Demo mode works fully offline. With models and llama.cpp already downloaded, testing, tuning and running work offline too. Fetching model information and downloads need internet.

**Will it slow my computer down?**
Only while it tests, tunes or runs a model. Those jobs use your processor and graphics card heavily; they run one at a time and can be cancelled.

**Where is my data, and how do I remove it?**
See [Getting started → Where your data lives](getting-started.md#where-your-data-lives).

**Does it support Mac / AMD / Intel graphics?**
v0.4 adds Apple Silicon (unified memory, Metal), AMD graphics memory on Linux, and lists Intel and other graphics cards. Where free graphics memory can't be read, it shows unknown instead of guessing. Speed calibration of the graphics card itself is still NVIDIA-only.

**Can one model use two graphics cards?**
Not yet.
