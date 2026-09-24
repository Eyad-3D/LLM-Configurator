# How it works

This page explains how LLM Configurator decides whether a model **fits** (memory) and how **fast** it will be (speed), and how honest each number is.

Everything is counted in bytes internally. The app shows **GiB** (1 GiB = 2^30 bytes, about 7% more than a "GB").

## A few words first

| Term | Plain meaning |
|---|---|
| **Model file (GGUF)** | The model's "brain" saved as one file (or a few pieces). GGUF is the format llama.cpp reads. |
| **Quantisation** (Q4_K_M, Q8_0 …) | Compression of the model file, like saving a photo as a smaller JPEG. Smaller = less memory and faster, but a bit less accurate. |
| **Context** | How much text the model can keep in view at once, like the size of its desk. Measured in **tokens** (roughly ¾ of a word each). |
| **KV cache** | The model's short-term notepad for the current conversation. It grows with the context. |
| **VRAM** | The graphics card's own memory. Much faster than normal memory (RAM), but usually smaller. |
| **Layers** | A model is a stack of layers. Some can sit on the graphics card and the rest in RAM (a "split"). |
| **MoE (mixture of experts)** | A model made of many "specialists" where only a few work on each word. Big on disk, but each word only reads a small part. |
| **Unified memory** | Apple Silicon Macs: the processor and graphics share one pool of memory. |
| **tok/s** | Tokens per second: how fast the model writes. Around 10+ tok/s feels like fast reading. |

## Part 1: will it fit? (memory)

For each model, compression level, context size and placement (all on processor, all on graphics card, or a split), the app adds up:

1. **The weights** – the model file itself.
2. **The notepad (KV cache)** – for every conversation running at once.
3. **Working buffers** – scratch space the engine needs.
4. **Headroom** – memory you asked to keep free for other apps.

It then checks the total against what is free **right now** on your computer.

### The notepad (KV cache)

```text
notepad bytes = 2 × layers × KV heads × head size × bytes per value × context × chats at once
```

- The first **2** is because the notepad has two parts (K and V).
- It uses the model's KV heads, not its total attention heads (models with "grouped-query attention" share notepads between heads).
- Context includes everything: your input, history, the model's thinking and its reply.
- Like llama.cpp, the app rounds the context up to the next multiple of 256 tokens.
- Each chat at once gets its own full-size notepad. The app starts llama.cpp with context × chats in total, and llama.cpp gives each chat an equal share.
- No savings from shared prompts are assumed.

**Notepad compression (new in v0.4).** llama.cpp can store the notepad in a smaller format. Bytes per value:

| Setting | Bytes per value | Size vs normal |
|---|---|---|
| `f16` (normal) | 2 | 100% |
| `q8_0` | 34 ÷ 32 ≈ 1.06 | about 53% |
| `q4_0` | 18 ÷ 32 ≈ 0.56 | about 28% |

Compressed notepads let you fit a much longer context, at a small quality cost that is usually minor for `q8_0` and more noticeable for `q4_0`. llama.cpp needs **flash attention** (a faster way to compute attention) to compress the V half, so the app turns it on when you choose this. If a model only fits with a compressed notepad, the results say so.

**Sliding-window layers (new in v0.4).** Some models (Gemma 2, Gemma 3 and gpt-oss) have layers that only look back a fixed number of tokens, the "window", like reading through a letterbox. Those layers' notepad stops growing at the window. For each chat, the app counts those layers at the window plus one batch of incoming text (512 tokens by default), rounded up to a multiple of 256, and never more than the full context. For example, with a 1,024-token window those layers need room for 1,536 tokens per chat, however long the context is. This matches the sizes llama.cpp itself reports.

### Weights, buffers and headroom

The estimator uses these allowances. They are **engineering guesses, not measured constants or guaranteed upper limits**:

- 10% on top of the file size for placement and metadata uncertainty.
- 0.75 GiB plus 0.20 GiB per chat-at-once for buffers, in each memory pool used (processor and graphics card each get their own).
- 0.5 GiB of extra RAM for staging when a separate graphics card is used (not on Apple Silicon).
- Headroom you choose; default 2 GiB of RAM and 0.5 GiB of VRAM.

The allowances may change as they are checked against real measurements. The Test step shows the **real peak memory next to the estimate** so you can see how close it was.

Loading peaks, unusual model parts, and different llama.cpp versions can go above the estimate. **Test a setup before relying on it.**

### Splitting between graphics card and RAM

The app tries three placements: everything on the processor, everything on the graphics card, and the largest split that fits.

- **The output layer goes on the graphics card first.** The output layer is the model's last step, which turns its thinking into word choices. llama.cpp counts it as one extra layer and puts it on the graphics card before any other. So when the app puts N layers on the graphics card, it asks llama.cpp for N + 1 (you will see `-ngl N+1` in exported commands). It keeps room on the card for the output layer: 15% of the file, capped at 2 GiB, but never less than one average layer. Models with large vocabularies have big output layers.
- The rest of the weights and the notepad are shared out in proportion to the number of layers on each side.
- Model files are "memory-mapped" (read from disk as needed), so the app does not assume a full second copy in RAM.

### MoE models (new in v0.4)

In an MoE model most of the file is "experts". llama.cpp can keep the experts of some layers in RAM while the rest runs on the graphics card (the `--n-cpu-moe` setting). Because each word only uses a few experts, this costs less speed than you might expect. The app:

- works out what share of the model is experts, from the expert counts and sizes in the model's `config.json`,
- counts the expert bytes of those layers in RAM and everything else in VRAM (each layer's notepad stays on the graphics card with the layer),
- when the whole model does not fit on the graphics card, offers the "experts on CPU" placement: all layers on the graphics card, with the experts of as few layers as possible moved to RAM.

### Apple Silicon and unified memory (new in v0.4)

On Apple Silicon, the processor and graphics chip share memory. The app treats it as **one pool** so nothing is counted twice. macOS only lets the graphics side use part of the memory (the "wired limit"):

- If you have set `sysctl iogpu.wired_limit_mb` above 0, that is the limit (never more than your installed RAM).
- Otherwise the app uses macOS's usual default: **2/3 of RAM on Macs with up to 32 GiB, 3/4 above that** (so a 36 GiB Mac gets 27 GiB). These are the limits llama.cpp reports on such Macs, but they can differ with the chip and macOS version.

The graphics side can never use more than the RAM that is free right now.

### What "free memory" means

- Free memory already accounts for what other apps use. It is not subtracted twice.
- **Swap** (disk used as overflow memory) is never counted as usable. It is far too slow.
- If you mark apps you'd close, the app credits only 75% of their private memory (USS). If that number is unavailable, it credits nothing. A separate graphics card never gets a guessed "after closing" credit. On Apple Silicon, closing apps can raise the graphics share up to the wired limit, because it is the same memory.
- If a graphics card does not report its free memory, the app shows only processor options rather than guess.

### Context ceiling

Each recommendation also shows a **memory-only context ceiling**: the largest context that would still fit. It is not a promise about speed or about how well the model handles long text.

## Part 2: how fast? (speed)

Speed comes from the best evidence available. The app uses the first kind in this list that it has, and each card says which kind it is:

| Evidence | Meaning |
|---|---|
| **measured** | A speed test of exactly this model file, computer, placement and settings, at this context. |
| **tuned** | The tuner's best settings, measured with (nearly) this context already in the chat. |
| **interpolated** | Worked out from your own tests of the same file and placement at other context sizes (see below). Never labelled "measured". |
| **tuned with a short test** | A tune that only measured near the start of a conversation. Its speed is scaled to this context as an estimate. |
| **community** | Median speed from at least two people with similar hardware, the same model file and the same settings (only if you imported community results). |
| **estimated** | Worked out from a short hardware check. Low confidence. |
| **none** | Not enough information. Speed shows as "Unavailable". |

The **verdict badge** compares speed with the speed you asked for. Only **measured** or **tuned** speed can earn *Runs well* (meets your target), *Runs slowly* (at least half your target) or *Too slow* (under half). Everything else shows *Not tested yet*, with a sentence such as "likely fast enough" and where the number came from.

**How interpolation works.** Between two tests, the app draws a straight line. If it only has a shorter test, it scales the speed by how much more each word must read as the notepad grows, takes off 10% to stay cautious, and never stretches a test to more than 4× its length. If it only has a longer test, it reuses that speed, because a shorter chat is not slower.

### Why speed is mostly about memory speed

To write each word, the model reads (roughly) all of its active weights plus the notepad once. So writing speed is limited by how fast memory can be read, like how fast you can flip through a book, not by how fast you can think.

- **Dense models:** bytes read per word ≈ file size + notepad.
- **MoE models (new in v0.4):** only the **active** experts are read, so the estimate uses active bytes, not total bytes. A 30B MoE model with 3B active can be many times faster than a 30B dense one.

### The hardware check (calibration)

At the start of the questions, a separate process runs a synthetic test, stopped after at most 12 seconds. It reads no model files.

- **Processor:** memory copy speed (a 128 MiB buffer copied to another, smaller if RAM is short) and a small "unpack 4-bit numbers and multiply" test, using up to 8 threads. These stand in for real inference; they are not the real llama.cpp code.
- **NVIDIA graphics cards:** graphics memory copy speed, transfers between RAM and graphics card, and a small-transfer delay, using your installed driver. Up to four cards. No CUDA toolkit or PyTorch needed.
- Other graphics chips (including Apple Silicon and AMD) are not calibrated yet, so placements that use them get no speed estimate until you test them. Advertised speeds are never used in their place.
- Results are kept for 7 days and tied to your hardware. **Recalibrate speed** in the app, or `llm-config calibrate`, runs it again.

### Turning the check into a range

The estimate assumes real inference reaches:

- 35–80% of the processor's measured memory speed (the slow end can be lower still if the 4-bit test says the processor is the bottleneck),
- 10–55% of the graphics card's measured memory speed,
- plus, in a split, the time for data to cross between processor and graphics card, counted 1× for the fast end and 4× for the slow end.

These are **wide, low-confidence ranges, not statistical confidence intervals or guarantees**. Real speed can fall outside them.

An estimate needs the file's compression level to be one of those listed under [Supported models](#supported-models), or the model's parameter count.

**Learning from your tests (new in v0.4).** Once you have at least two speed tests on this computer (from the last 90 days, one chat at a time, each run with nearly its full context in the chat), the app compares them with its own estimates and adjusts the range for models you haven't tested yet. If a placement (processor, graphics card or split) has two tests of its own, they shift and narrow its range. Otherwise the range is only shifted, keeping its full width. It never claims better than about ±10%. Such estimates are labelled "adjusted from N local measurements". They are still estimates, never "verified".

### What is not estimated

- Speed for several chats at once. It depends on how llama.cpp batches the chats together, and the speed test measures one chat at a time. Memory for several chats **is** estimated.
- Speed estimates never reject a configuration, and never count as "verified" for the strict speed filter. Only a real test at this context does (measured, or tuned with the full context).

Reading speed (how fast it reads your prompt) and first-word delay are **measured** by the Test step, not estimated. See [Testing and tuning](testing-and-tuning.md).

### How measurements are matched

A measurement counts for a recommendation only when all of these match, with one chat at a time:

- the model file (exact checksum; a file found on your disk that has not been checksummed yet matches by its ID),
- the computer (hardware and driver),
- the placement and processor threads,
- the settings: notepad compression, experts on CPU, batch sizes and flash attention.

For **measured**, the context must match exactly too, and the test must have run with nearly that much text already in the chat (within 1,024 tokens). Speed tests stop at 32,768 tokens in the chat, so a test of a context over about 33,800 tokens counts as **interpolated** (scaled from 32,768), not measured.

Measurements and tuning results expire after 30 days. They reflect the conditions at the time: other apps, heat, power mode and llama.cpp version can change speed.

## Part 3: ranking

- **Quality priority** orders by the base model's quality score (if you turned rankings on), then speed.
- **Speed priority** puts configurations with a real test that meets your target first. Then it orders by the best speed number available: measured or tuned, then interpolated, then a short tune scaled to this context, then the community median, then the low end of the estimate.
- **Balanced** puts configurations with verified speed that meets your target first, then ones not known to miss it, then quality.

Unknown stays unknown. See [Quality checks](quality-checks.md) for what the quality scores do and don't mean.

### Where quality scores come from

Scores come from Artificial Analysis and need an API key. Which key wins: a key entered for this session only, then a key saved on this computer, then the `AA_API_KEY` environment variable.

A model gets a score only after you match it to exactly one Artificial Analysis entry under **Match benchmark entries** in the rankings setup (or with `llm-config map`). Matches start empty on purpose, and the app never guesses from similar names, so for example a "thinking" and a normal version are never mixed up. Models without a match have no score. Scores from different versions of the benchmark are never ranked against each other. Scores describe the original model, not the compressed file.

## Supported models

Architectures the memory model understands:

- **Dense:** Qwen2, Qwen3, Llama, Mistral, Gemma 2, Gemma 3, Phi-3/Phi-4, Granite, OLMo 2
- **MoE:** Qwen2-MoE, Qwen3-MoE, Mixtral, gpt-oss

Compression levels with known sizes: Q2_K, Q3_K_M, IQ4_XS, Q4_0, Q4_K_M, MXFP4, Q5_K_M, Q6_K, Q8_0, F16, BF16. For built-in models the app offers Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0 and MXFP4 when the repository has them, plus F16/BF16 for small models (up to about 4.5 billion parameters). A GGUF file already on your computer can use other levels too, because the app reads its real size. Models split into several files ("sharded") are supported from v0.4.

The built-in list (39 models) covers Qwen3 (including the Qwen3-30B-A3B and 235B MoE models and Qwen3-Coder), Qwen2.5-Coder, DeepSeek-R1 distills (Qwen and Llama), Llama 3.1, 3.2 and 3.3, Gemma 3, Mistral (7B, Nemo, Small 24B), Phi-4 and Phi-4-mini, gpt-oss (20B and 120B), SmolLM2, Granite 3.3 and OLMo 2. You can add a model with `llm-config models add BASE_REPO GGUF_REPO` (the original Hugging Face repository and the one with its GGUF files), or use a GGUF file you already have (`llm-config local --add PATH`).

The app reads each model's shape (layers, heads, context length) from the original repository's `config.json`. Some original repositories are **gated** (you must log in and accept a licence first); set `HF_TOKEN` to your Hugging Face token to use them. Some catalogue entries also have a `config_repo`: an ungated copy of the same model, used only to read its `config.json` when you have no token. `llm-config models add` takes `--config-repo REPO` for the same purpose.

## Not covered yet

- Spreading one model across several graphics cards.
- Speed estimates for Apple Silicon and AMD graphics (test to get a real number).
- Image input or image generation.
- Filtering by licence.
- Guaranteed memory limits: estimates are estimates until you test.
