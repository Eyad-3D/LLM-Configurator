# Testing and tuning

Estimates are only a starting point. The **Test it** and **Tune it** steps run the model on your computer and measure what really happens.

In the app, click **Get it running →** on a model card. The panel walks through the steps: get the engine, download the model, **Test it**, **Tune it**, **Check quality** and **Use it**. Testing and tuning need the llama.cpp engine and the downloaded model, so those steps wait until the first two are done.

Both use your processor and graphics card heavily while they run, so close heavy apps first. Only one heavy job runs at a time; others wait their turn in the jobs tray. If your own model server is running (**Use it**), stop it first: two models at once may not fit in memory.

If the app says **"Compare again first"**, the model you picked is no longer in your latest list of results (for example after the app restarted). Go back to the results page, compare again, and open the card again.

## Test

The test has two parts. In the app, **Run the full test** does both; **Quick check only** does just the first. On the command line use `llm-config test VARIANT_ID --kind smoke|speed|full` (the default is `full`).

### 1. Smoke test ("does it work at all?")

The app starts the model and asks one question with a checkable answer: "What is 17 + 25?". It then checks the reply:

- it is not empty (hidden "thinking" text doesn't count),
- it has no broken characters,
- it is not the same thing repeated over and over,
- it does not contain leaked template text (raw control markers like `<|im_start|>`, a sign the model's chat format is wrong),
- it answers 42.

If the model fails to **start**, you get a plain reason where possible: not enough memory, model type not supported by this llama.cpp version, file missing or damaged, port in use, or graphics card unavailable. On the command line, `--json` also includes the last lines of the engine's log.

If the smoke test fails, the speed test is skipped.

### 2. Speed test ("how fast, really?")

| Result | Plain meaning | How it is measured |
|---|---|---|
| **Reading speed** | How fast it reads your prompt, in tokens per second | `llama-bench`, 512 prompt tokens (each `llama-bench` test runs twice and is averaged) |
| **Writing speed** | How fast it writes the reply, in tokens per second | `llama-bench`, 128 new tokens, with the notepad already filled near your chosen context |
| **First-word delay** | How long you wait before the reply starts | One real request to `llama-server` with about 1,500 tokens of text |
| **Memory used** | The most RAM (and NVIDIA graphics memory) it actually used | Watched while that request and a short reply run |

(A token is a piece of a word, roughly ¾ of a word.)

"Near your chosen context" matters: models slow down as the notepad fills, so the test measures with `context − 640` tokens already in the chat. The test then ends exactly at your chosen size. Very short contexts use a smaller test.

The result shows **measured memory next to the estimate**, and says whether it stayed within the estimate. This is the best way to see how accurate the memory maths is for your machine.

Graphics memory is only measured on NVIDIA cards (via `nvidia-smi`). On Apple Silicon, graphics memory is part of RAM and is counted there. On other cards it is not measured.

### The verdict

After the test you get one of:

- **It works** – started, passed the checks, and (if the speed test ran) how fast it writes.
- **It works, but slowly** – started and passed the checks, but writes slower than your speed target. *This needs a fix that is in progress; for now the app always says "It works" when the checks pass.*
- **It didn't work** – with a plain reason and what to try next.

If the smoke test passes but the speed test fails, you still get **It works**, with a note that the speed test did not finish.

Speed results are saved and used by future recommendations (see "measured" in [How it works](how-it-works.md#part-2-how-fast-speed)). A result is not saved if `llama-bench` reports that it used different settings than the ones asked for.

### Limits of the test

- It measures one chat at a time. Speed with several chats at once is not measured.
- It is a short synthetic test. Your real conversations, other apps, heat and power settings can change speed.
- A correct answer to one question says the model **works**, not that it is **good**. For quality, see [Quality checks](quality-checks.md).

## Tune

Tuning tries different engine settings and keeps the ones that make the model fastest **without running out of memory**.

In the app, pick a time budget of **1, 5 or 15 minutes**, choose what should get faster, and click **Start tuning**. On the command line: `llm-config tune VARIANT_ID --budget SECONDS --goal GOAL` (60 to 1800 seconds, default 300).

### Goal

| App choice | `--goal` | Maximises |
|---|---|---|
| Faster writing (chat, coding) | `generation` (default) | Writing speed |
| A balance of reading and writing | `balanced` | A mix of both, leaning a little towards writing |
| Faster reading (long documents) | `prompt` | Reading speed |

### What it tries

| Setting | Plain meaning | When |
|---|---|---|
| Layers on graphics card | How much of the model sits in graphics memory | Only for split setups (part on the graphics card, part in RAM) |
| Experts in RAM | How many layers keep their "expert" parts in RAM | MoE models that use the graphics card |
| Threads | How many processor cores work on it. More is not always faster. | Always |
| Flash attention | A faster way to do the attention maths; tried on and off | When the notepad is not compressed |
| Batch sizes | How many prompt tokens are read in one go | Only for the **balanced** and **prompt** goals (it doesn't change writing speed) |
| Notepad compression | Stores the notepad (KV cache) at half size | Only after a setting was skipped because memory was tight |

Tuning measures with up to 1,024 tokens already in the chat (less than the test, to save time), so its speeds can be a little higher than the test's.

### How it searches

1. Measure your current settings (the "baseline").
2. Change **one setting at a time**. Keep a change only if it is at least 3% faster (or more, if the measurements were noisy). Go round up to three times.
3. Test several values in one run to save time.
4. **Skip** any setting the memory check says would not fit, rather than risk a crash.
5. Re-run the winner with more repetitions to confirm it wasn't a lucky reading. If it no longer beats the baseline, your starting settings are kept.

It stops when time runs out, when a round finds nothing faster, or when you cancel. It never goes over the budget by more than one trial.

### The result

You see before and after speeds (for example "Found settings about 30% faster"), which settings changed, how many were tried, why it stopped, and short notes. The best settings are saved on this computer. A cancelled tune is not saved.

On the command line, use the saved settings with `--tuned` on `test`, `export` and `run`. The app's **Use it** step uses the recommended settings, not the tuned ones.

### Limits of tuning

- Small differences (a few percent) can be noise.
- It tunes for one chat at a time.
- Settings tuned with one llama.cpp version may not be best on another.
