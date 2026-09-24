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
- No savings from shared prompts are assumed.

**Notepad compression (new in v0.4).** llama.cpp can store the notepad in a smaller format. Bytes per value:

| Setting | Bytes per value | Size vs normal |
|---|---|---|
| `f16` (normal) | 2 | 100% |
| `q8_0` | 34 ÷ 32 ≈ 1.06 | about 53% |
| `q4_0` | 18 ÷ 32 ≈ 0.56 | about 28% |

Compressed notepads let you fit a much longer context, at a small quality cost that is usually minor for `q8_0` and more noticeable for `q4_0`. llama.cpp needs **flash attention** (a faster way to compute attention) to compress the V half, so the app turns it on when you choose this.

**Sliding-window layers (new in v0.4).** Some models (for example Gemma 3 and gpt-oss) have layers that only look back a fixed number of tokens. Their notepad stops growing at that window, so the app caps those layers' notepad at the window size.

### Weights, buffers and headroom

The v0.3 estimator used these allowances. They are **engineering guesses, not measured constants or guaranteed upper limits**:

- 10% on top of the file size for placement and metadata uncertainty.
- 0.75 GiB plus 0.20 GiB per chat-at-once for buffers, in each memory pool used.
- 0.5 GiB of extra RAM for staging when the graphics card is used.
- Headroom you choose; default 2 GiB of RAM and 0.5 GiB of VRAM.

> v0.4 revises the memory model (see below). The exact allowances may change as they are checked against real measurements; the Test step shows the **real peak memory next to the estimate** so you can see how close it was.

Loading peaks, unusual model parts, and different llama.cpp versions can go above the estimate. **Test a setup before relying on it.**

### Splitting between graphics card and RAM

The app tries: everything on the processor, everything on the graphics card, and the largest split that fits. Memory for a split is shared out in proportion to the number of layers on each side. Model files are "memory-mapped" (read from disk as needed), so the app does not assume a full second copy in RAM.

### MoE models (new in v0.4)

In an MoE model most of the file is "experts". llama.cpp can keep the experts of some layers in RAM while the rest runs on the graphics card (the `--n-cpu-moe` setting). Because each word only uses a few experts, this costs less speed than you might expect. The app:

- knows what share of the file is experts (from the model's metadata),
- counts expert bytes for those layers in RAM and the rest in VRAM,
- offers these "experts in RAM" placements when they fit.

### Apple Silicon and unified memory (new in v0.4)

On Apple Silicon, the processor and graphics card share memory. The app treats it as **one pool** so nothing is counted twice. macOS only lets the graphics side use part of the memory (the "wired limit"). The app reads it from `sysctl iogpu.wired_limit_mb` when you have set it, and otherwise uses macOS's default share. The exact rule is documented by the hardware module and may be refined.

### What "free memory" means

- Free memory already accounts for what other apps use. It is not subtracted twice.
- **Swap** (disk used as overflow memory) is never counted as usable. It is far too slow.
- If you mark apps you'd close, the app credits only 75% of their private memory (USS). If that number is unavailable, it credits nothing. The graphics card never gets a guessed "after closing" credit.

### Context ceiling

Each recommendation also shows a **memory-only context ceiling**: the largest context that would still fit. It is not a promise about speed or about how well the model handles long text.

## Part 2: how fast? (speed)

Speed comes from the best evidence available. Each card says which kind it is:

| Evidence label | Meaning |
|---|---|
| **measured** | You tested exactly this model, file, hardware and settings on this computer. |
| **tuned** | Measured with settings found by the tuner. |
| **interpolated** | Worked out from your own measurements of the same model and placement at other context sizes. Never labelled "measured". |
| **community** | Median speed other people reported on similar hardware (only if you imported community results). |
| **estimated** | Worked out from a short hardware check. Low confidence. |
| **none** | Not enough information. Shown as "unknown". |

The **verdict badge** (*Runs well*, *Runs slowly*, *Too slow*, *Not tested yet*) compares this evidence with the speed you asked for.

### Why speed is mostly about memory speed

To write each word, the model reads (roughly) all of its active weights plus the notepad once. So writing speed is limited by how fast memory can be read, like how fast you can flip through a book, not by how fast you can think.

- **Dense models:** bytes read per word ≈ file size + notepad.
- **MoE models (new in v0.4):** only the **active** experts are read, so the estimate uses active bytes, not total bytes. A 30B MoE model with 3B active can be many times faster than a 30B dense one.

### The hardware check (calibration)

When you start the questions, a separate process runs a ~12-second synthetic test. It reads no model files.

- **Processor:** memory copy speed (up to two 128 MiB buffers) and a small "unpack 4-bit numbers and multiply" test, using up to 8 threads. These stand in for real inference; they are not the real llama.cpp code.
- **NVIDIA graphics cards:** graphics memory copy speed, transfers between RAM and graphics card, and a small-transfer delay, using your installed driver. Up to four cards. No CUDA toolkit or PyTorch needed.
- Other graphics cards are marked **uncalibrated**. Advertised speeds are never used in their place.
- Results are kept for 7 days and tied to your hardware. **Recalibrate speed** in the app, or `llm-config calibrate`, runs it again.

### Turning the check into a range

The estimate assumes real inference reaches:

- 35–80% of the processor's measured memory speed,
- 10–55% of the graphics card's measured memory speed,
- plus 1–4× extra for data crossing between processor and graphics card in a split.

These are **wide, low-confidence ranges, not statistical confidence intervals or guarantees**. Real speed can fall outside them.

**Learning from your tests (new in v0.4).** After you test models on this computer, the app uses those results to narrow the ranges for models you haven't tested yet. Such estimates are labelled "adjusted from N local measurements".

### What is not estimated

- Speed for several chats at once (it depends on batching, which needs a real test). Memory for several chats **is** estimated.
- Speed estimates never reject a configuration, and never count as "verified" for the strict speed filter. Only real tests do.

Reading speed (how fast it reads your prompt) and first-word delay are **measured** by the Test step, not estimated. See [Testing and tuning](testing-and-tuning.md).

### How measurements are matched

A measurement counts for a recommendation only when the file (exact fingerprint), hardware and driver, placement, processor threads and settings match. Measurements expire after 30 days. They reflect the conditions at the time: other apps, heat, power mode and llama.cpp version can change speed.

## Part 3: ranking

- **Quality priority** orders by the base model's quality score (if you turned rankings on).
- **Speed priority** orders by measured speed first, then the low end of estimates.
- **Balanced** prefers configurations with verified speed, then quality.

Unknown stays unknown. See [Quality checks](quality-checks.md) for what the quality scores do and don't mean.

## Supported models

Architectures the memory model understands:

- **Dense:** Qwen2, Qwen3, Llama, Mistral, Gemma 2, Gemma 3, Phi-3/Phi-4, Granite, OLMo 2
- **MoE:** Qwen2-MoE, Qwen3-MoE, Mixtral, gpt-oss

Compression levels with known sizes: Q2_K, Q3_K_M, IQ4_XS, Q4_0, Q4_K_M, MXFP4, Q5_K_M, Q6_K, Q8_0, F16, BF16. Models split into several files ("sharded") are supported from v0.4.

The built-in list covers popular open models (Qwen3, Qwen2.5-Coder, Llama 3.x, Gemma 3, Mistral, Phi-4, gpt-oss, DeepSeek-R1 distills and small models). You can add any Hugging Face repository with `llm-config models add`, or use a GGUF file you already have (`llm-config local --add PATH`).

## Not covered yet

- Spreading one model across several graphics cards.
- Image input or image generation.
- Filtering by licence.
- Guaranteed memory limits: estimates are estimates until you test.
