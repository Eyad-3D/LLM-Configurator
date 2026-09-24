# Quality checks

Speed tells you if a model is **usable**. Quality tells you if it is **good enough for you**. LLM Configurator offers four kinds of evidence. None of them is a full benchmark; each is labelled for what it is.

All checks except rankings run **locally**, on your computer. They need the llama.cpp engine and the downloaded model(s). Models are run **one after another**, never at the same time, to save memory.

In the app, the local checks are in **Get it running → Check quality**, on three tabs: **Quick quiz**, **Try my prompts** and **Compression check**. You pick models from your latest list of results. If the app says **"Compare again first"**, go back to the results page and compare again.

## Quick quiz

A short set of questions with checkable answers, for your chosen use:

| Quiz | What it checks |
|---|---|
| **general** | General knowledge and simple reasoning (arithmetic, units, dates, word puzzles) |
| **coding** | "What does this Python or JavaScript code print?" and "which line has the bug?", checked by exact answer. The app **never runs code the model writes**. |
| **agentic** | Whether it picks the right tool, fills in its details in correct JSON format, or declines when no tool fits |
| **documents** | Answering questions using only a one-page made-up document (memos, reports and so on) |

All questions were written for this project. They are not copied from public benchmarks (so models are unlikely to have memorised them), but that also means scores **can't be compared** with published benchmark numbers.

Each quiz has 45 to 55 questions, and the app asks all of them. The result is a score plus a range. For example, 30 right out of 50 is 60%, with a likely range of about 46% to 72%. The range (a 95% Wilson interval) is wide because 50 questions is a small sample. Treat differences of a few questions as noise; the app says "Too close to call" when two results' ranges overlap. Models that "think" before answering can run out of room; the note says so when that happens, because the score may then understate the model.

**Long-document recall ("needle test").** Optionally, the app fills the context you chose with a long made-up text, hides one secret word at 10%, 50% and 90% of the way through, and asks the model to find it. This checks the model can actually use the context you chose.

App: **Quick quiz**, then tick **Also test memory for long documents** for the needle test. Command line: `llm-config quiz VARIANT_ID [--workload general|coding|agentic|documents] [--needle]` (default workload: `general`).

## Try my prompts (blind comparison)

Pick 2 or 3 models and type up to 5 of your own prompts (up to 4,000 characters each), then click **Run the models**. The app runs each model on each prompt, then shows the answers **shuffled and labelled A, B, C** (in a new order for each prompt) so you don't know which is which.

For each prompt, vote for the best answer or choose **No preference**. **Skip voting on the rest** skips the remaining prompts. Once every prompt is voted on or skipped, click **Reveal which model wrote each answer**. You then see which model wrote each answer and how many votes each model got. Voting closes after the reveal.

This is the most useful check for your real work, because it uses your prompts and your judgement. Your prompts stay on your computer. With only a handful of prompts, a one-vote lead is a hint, not proof.

App only: **Try my prompts**.

## Compression check

Compressing a model (quantisation, like a smaller JPEG) makes it less accurate. This check measures **how much**.

The app runs a reference version (usually the biggest file you have of the same model, for example Q8_0) and one to three smaller versions over the same text, and compares their word predictions. Results are in plain words, for example "Picks a different top word about 4% of the time compared with Q8_0 — usually hard to notice in chat." **Show the numbers** gives the details.

Behind the scenes: this uses llama.cpp's `llama-perplexity` tool and a measure called **KL divergence** (how different two sets of predictions are; 0 means identical). The test text is a bundled ~44 KB sample of English prose and a little code; each file reads about 6,000 tokens of it, so the answer is quick and rough.

Be aware:

- You need the reference **and** the compressed files downloaded. References are big.
- Each file is loaded in turn, so this can take several minutes per file.
- It writes a temporary file of the reference's predictions, which can be large (often a gigabyte or more). The app checks there is enough free disk space first, tells you the file's size in the results, and always deletes it afterwards.
- It tells you how much a compressed version **differs** from the reference, not how good the model is overall. The verdicts are rough guidance based on llama.cpp's published measurements.

App: **Compression check**. Command line: `llm-config quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...]`.

## Rankings from Artificial Analysis

Optional. If you add an [Artificial Analysis](https://artificialanalysis.ai) API key, cards show the model's rank for your use (general, coding, agentic).

**What a rank means:** a comparison of the **original, uncompressed base model** on public tests. It is not your compressed file, not your speed, and not your hardware.

**Adding a key:**

1. In guided setup choose **Include rankings** (or later, **Ranking settings**) and paste your key.
2. **Test connection** checks it. **Save key** stores it. (Testing alone does not save.)
3. **Remember on this computer** keeps it in your system password store (Windows Credential Manager, macOS Keychain, or a Linux Secret Service/KWallet keyring). Untick it to use the key for this session only. If the password store is locked or missing, saving fails with a clear message; the app never falls back to a plain file.
4. **Remove key** deletes it.

Which key wins: a key entered in this session, then a saved key, then the `AA_API_KEY` environment variable. Removing a key in the app does not remove an environment variable.

**Matching models to scores.** Scores are only used after you pick the exact matching entry (including "reasoning" mode) under **Match benchmark entries**. Matches start empty on purpose: similar names are not proof of the same model. Command line: `llm-config benchmarks` then `llm-config map BASE_REPO SLUG`.

**How ranks are shown:**

- "#3 of 12 rated models": counted over distinct base models, not every compression/context combination. Ties share a rank.
- Missing score: **Not ranked**. Never guessed.
- Scores from different index versions are never ranked together.
- "Documents" uses the general index as a stand-in, because there is no separate documents index.
- Measured-speed configurations still come first in card order.

Scores are attributed to Artificial Analysis and subject to their terms. No score data is bundled in this repository.

## Where results go

Quiz, needle and compression results are saved in the app's local database; the **Quick quiz** tab lists earlier quiz results. Blind comparisons and your votes are also kept locally (the 20 newest comparisons). Nothing is uploaded.
