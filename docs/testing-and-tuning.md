# Testing and tuning

Estimates are only a starting point. The **Test** and **Tune** steps run the model on your computer and measure what really happens.

Both need the llama.cpp engine (the **Runtime** step installs it) and the downloaded model. Both use your processor and graphics card heavily while they run, so close heavy apps first. Only one heavy job runs at a time; others wait their turn in the jobs tray.

## Test

The test has two parts. In the app choose **Test**; on the command line use `llm-config test VARIANT_ID --kind smoke|speed|full`.

### 1. Smoke test ("does it work at all?")

The app starts the model and asks one question with a checkable answer. It then checks the reply:

- it is not empty,
- it is not the same word repeated over and over,
- it does not contain leaked template text (raw control markers like `<|im_start|>`, a sign the model's chat format is wrong),
- it is valid text.

If the model fails to **start**, you get a plain reason where possible: not enough memory, model type not supported by this llama.cpp version, file missing or damaged, port in use, or graphics backend unavailable. The last lines of the engine's log are shown too.

### 2. Speed test ("how fast, really?")

| Result | Plain meaning | How it is measured |
|---|---|---|
| **Reading speed** | How fast it reads your prompt, in tokens per second | `llama-bench`, 512 prompt tokens |
| **Writing speed** | How fast it writes the reply, in tokens per second | `llama-bench`, 128 new tokens, with the notepad already filled near your chosen context |
| **First-word delay** | How long you wait before the reply starts | One real request to `llama-server` with a realistic prompt |
| **Peak memory** | The most RAM (and NVIDIA VRAM) it actually used | Watched while that request runs |

"Near your chosen context" matters: models slow down as the notepad fills, so the test measures writing speed at roughly `context − 640` tokens already in view (kept within sensible limits), not on an empty notepad.

The result shows **measured memory next to the estimate**, and says whether it stayed within the estimate. This is the best way to see how accurate the memory maths is for your machine.

Peak graphics memory is only measured on NVIDIA cards (via `nvidia-smi`). On Apple Silicon, graphics memory is part of RAM and is counted there. On other cards it shows as unknown.

### The verdict

After the test you get one of:

- **Works** – started, passed the checks, and is fast enough.
- **Works, but slowly** – started and passed the checks, but is slow.
- **Failed** – with a plain reason and what to try next.

Speed results are saved and used by future recommendations (see "measured" in [How it works](how-it-works.md#part-2-how-fast-speed)).

### Limits of the test

- It measures one chat at a time. Speed with several chats at once is not measured.
- It is a short synthetic test. Your real conversations, other apps, heat and power settings can change speed.
- A correct answer to one question says the model **works**, not that it is **good**. For quality, see [Quality checks](quality-checks.md).

## Tune

Tuning tries different engine settings and keeps the ones that make the model fastest **without running out of memory**.

In the app, pick a time budget of **1, 5 or 15 minutes**. On the command line: `llm-config tune VARIANT_ID --budget SECONDS` (the app allows 60 to 1800 seconds).

### Goal

- **generation** – fastest writing speed.
- **prompt** – fastest reading speed (good for long documents).
- **balanced** – a mix of both.

### What it tries

| Setting | Plain meaning |
|---|---|
| Threads | How many processor cores work on it. More is not always faster. |
| Batch sizes | How many prompt tokens are processed in one go. |
| Flash attention | A faster way to compute attention, if your hardware supports it. |
| Layers on graphics card | Only for split setups: how much of the model sits in VRAM. |
| Notepad compression | Only if you allowed a compressed notepad, or memory is tight. |
| Experts in RAM | MoE models only: how many layers keep their experts in RAM. |

### How it searches

1. Measure your current settings (the "baseline").
2. Change **one setting at a time**. Keep a change only if it is faster.
3. Test several values per run to save time.
4. **Skip** any setting the memory check says would not fit, rather than risk a crash.
5. Re-run the winner once to confirm it wasn't a lucky reading.

It stops when time runs out, when nothing improves, or when you cancel. It never goes over the budget by more than one trial.

### The result

You see before and after speeds, the improvement (for example "1.3× faster"), each trial, and short notes. The best settings are saved. Use them later with `--tuned` on `export` and `run`.

### Limits of tuning

- Small differences (a few percent) can be noise.
- It tunes for one chat at a time.
- Settings tuned with one llama.cpp version may not be best on another.
