# Testing and tuning

Estimates are only a starting point. The **Test it** and **Tune it** steps run the model on your computer and measure what really happens.

In the app, click **Get it running →** on a model card. The panel walks through the steps: get the engine, download the model, **Test it**, **Tune it**, **Check quality** and **Use it**. Testing and tuning need the llama.cpp engine and the downloaded model, so those steps wait until the first two are done.

Both use your processor and graphics card heavily while they run, so close heavy apps first. Only one heavy job runs at a time; others wait their turn in the jobs tray. If your own model server is running (**Use it**), stop it first: two models at once may not fit in memory.

If the app says **"Compare again first"**, the model you picked is no longer in your latest list of results (for example after the app restarted). Go back to the results page, compare again, and open the card again.

## Test

The test has two parts. In the app, **Run the full test** does both; **Quick check only** does just the first. On the command line use `llm-config test VARIANT_ID --kind smoke|speed|full` (the default is `full`).

### 1. Smoke test ("does it work at all?")

The app starts the model and asks one question with a checkable answer: "What is 17 + 25?". It asks the model to answer straight away, without "thinking" out loud first. It then checks the reply:

- it is not empty (hidden "thinking" text doesn't count),
- it has no broken characters,
- it is not the same thing repeated over and over,
- it does not contain leaked template text (raw control markers like `<|im_start|>`, a sign the model's chat format is wrong),
- it answers 42.

If a model spends its whole answer thinking and never replies, the test says so: its chat template may not let the app turn thinking off.

If the model fails to **start**, you get a plain reason where possible: not enough memory, model type not supported by this llama.cpp version, file missing or damaged, port in use, or graphics card unavailable. On the command line, `--json` also includes the last lines of the engine's log.

If the smoke test fails, the speed test is skipped.

### 2. Speed test ("how fast, really?")

| Result | Plain meaning | How it is measured |
|---|---|---|
| **Reading speed** | How fast it reads your prompt, in tokens per second | `llama-bench`, 512 prompt tokens, with the notepad already filled near your chosen context (each `llama-bench` test runs twice and is averaged) |
| **Writing speed** | How fast it writes the reply, in tokens per second | `llama-bench`, 128 new tokens, with the same filled notepad |
| **First-word delay** | How long you wait before the reply starts to a new message | One real request to `llama-server` with about 1,500 tokens of fresh text (see below) |
| **Memory used** | The most RAM (and NVIDIA graphics memory) it actually used | Watched from the moment the model starts loading, through that request and a short reply |

(A token is a piece of a word, roughly ¾ of a word.)

"Near your chosen context" matters: models slow down as the notepad fills, like working at a desk that gets more cluttered. So the test measures with `context − 640` tokens already in the chat, then reads 512 and writes 128, ending exactly at your chosen size. Very short contexts (under 768 tokens) use a smaller test. Very long ones stop at 32,768 tokens already in the chat, because going deeper can take hours on a processor. The result says which depth it used. For contexts over about 33,800 tokens, recommendations treat such a test as scaled from 32,768 tokens ("interpolated"), not as measured at your length.

**First-word delay, in detail.** The app sends a tiny warm-up request first, so one-off start-up costs don't count. The timed message is sized with the model's own tokenizer, so it always fits (short contexts get `context − 256` tokens instead of 1,500). The app also tells the server not to reuse text it has read before (`cache_prompt: false`), so the whole message is read, as it would be for a message you have never sent.

The result shows **measured memory next to the estimate**, and says whether it stayed within the estimate (or by how many percent it went over). This is the best way to see how accurate the memory maths is for your machine.

Graphics memory is only measured on NVIDIA cards with llama.cpp's CUDA version (via `nvidia-smi`). On Apple Silicon, graphics memory is part of RAM and is counted there. On other cards it is not measured.

### The verdict

After the test you get one of:

- **It works** – started, passed the checks, and (if the speed test ran) how fast it writes.
- **It works, but slowly** – started and passed the checks, but writes slower than your speed target. In the app, the target is the one from your latest comparison. On the command line use `--min-tps` (default 15; `0` turns the check off).
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
| Notepad compression | Stores the notepad (KV cache) at about half size (`q8_0`), on its own or using the room saved for 2 more layers on the graphics card (or 2 fewer expert layers in RAM) | Only when compression is allowed; the tuner never changes your chosen notepad format on its own |

### How it searches

1. Measure your current settings (the "baseline").
2. Change **one setting at a time**. Keep a change only if it is at least 3% faster (or more, if the measurements were noisy or the computer's speed drifted between runs). Go round up to three times.
3. Test several values in one run to save time.
4. **Skip** any setting the memory check says would not fit, rather than risk a crash.
5. Re-run the winner with more repetitions (5 instead of 2) to confirm it wasn't a lucky reading. If it no longer beats the baseline, your starting settings are kept.
6. **Check at your full length.** To save time, the search runs with only a short conversation in the chat (up to 1,024 tokens). If there is time left in the budget, your starting settings and the winner are measured once more with your full context in the chat (`context − 640` tokens, at most 32,768), like the speed test. The saved speeds then come from this run. If the winner is not clearly faster there, or does not run, your starting settings are kept.

If there is no time for step 6, the notes say the speeds come from the shorter test. Such a tune shows as "tuned with a short test": its speed is only an estimate at your length and can't earn a *Runs well* verdict.

It stops when time runs out, when a round finds nothing faster, or when you cancel. It never goes over the budget by more than one trial.

### The result

You see before and after speeds (for example "Found settings about 30% faster"), which settings changed, how many were tried, why it stopped, and short notes. The best settings are saved on this computer. A cancelled tune is not saved.

**Using the tuned settings.** A saved tune belongs to one model, context, placement and notepad format. The app uses it automatically for **Test it**, **Use it** and **Copy a setup**. If the tuner moved layers between the graphics card and RAM, that change is only used while it still fits in free memory. On the command line, a tune that kept the same placement is used automatically too. Add `--tuned` to `test`, `export` or `run` to also use a tune that moved layers, or to get an error if there is no matching tune.

### Limits of tuning

- Small differences (a few percent) can be noise.
- It tunes for one chat at a time.
- Settings tuned with one llama.cpp version may not be best on another.
- Tunes older than 30 days are ignored, like speed tests.
