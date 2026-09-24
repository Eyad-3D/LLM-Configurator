# Quality checks

Speed tells you if a model is **usable**. Quality tells you if it is **good enough for you**. LLM Configurator offers four kinds of evidence. None of them is a full benchmark; each is labelled for what it is.

All checks except rankings run **locally**, on your computer. They need the llama.cpp engine and the downloaded model(s). Models are run **one after another**, never at the same time, to save memory.

## Quick quiz

A short set of questions with checkable answers, for your chosen use:

| Quiz | What it checks |
|---|---|
| **general** | General knowledge and simple reasoning |
| **coding** | "What does this code print?" and "fix this line", checked by exact answer. The app **never runs code the model writes**. |
| **agentic** | Whether it writes tool calls in correct JSON format |
| **documents** | Recalling facts from long synthetic text |

All questions were written for this project. They are not copied from public benchmarks (so models are unlikely to have memorised them), but that also means scores **can't be compared** with published benchmark numbers.

The result is a score plus a range. For example, 18 right out of 30 is 60%, with a likely range of about 42% to 75%. The range (a 95% Wilson interval) is wide because 30–60 questions is a small sample. Treat differences of a few questions as noise.

**Long-document recall ("needle test").** Optionally, the app hides one fact inside a long filler text at 10%, 50% and 90% of the way through, and asks the model to find it. This checks the model can actually use the context you chose.

App: **Quality → Quick quiz**. Command line: `llm-config quiz VARIANT_ID [--workload W] [--needle]`.

## Try my prompts (blind comparison)

Pick 2 or 3 models and type up to 5 of your own prompts. The app runs each model on each prompt, then shows the answers **shuffled and labelled A, B, C** so you don't know which is which. You vote for the best answer to each prompt, then reveal the answers to see which model you preferred.

This is the most useful check for your real work, because it uses your prompts and your judgement. Your prompts stay on your computer.

App only: **Quality → Try my prompts**.

## Compression check

Compressing a model (quantisation, like a smaller JPEG) makes it less accurate. This check measures **how much**.

The app runs a reference version (for example Q8_0 or F16) and one to three compressed versions over the same text, and compares their word predictions. Results are in plain words, for example "picks a different top word about 4% of the time".

Behind the scenes: this uses llama.cpp's `llama-perplexity` tool and a measure called **KL divergence** (how different two sets of predictions are; 0 means identical). The test text is a bundled ~30–60 KB sample of English prose and a little code.

Be aware:

- You need the reference **and** the compressed files downloaded. References are big.
- It writes a temporary file of the reference's predictions, which can be large. The app warns you about its size and deletes it afterwards.
- It tells you how much a compressed version **differs** from the reference, not how good the model is overall.

App: **Quality → Compression check**. Command line: `llm-config quant-check REFERENCE_VARIANT_ID VARIANT_ID [VARIANT_ID ...]`.

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

Quiz, needle and compression results are saved in the app's local database and shown in the Quality panel. Blind comparisons and your votes are also kept locally. Nothing is uploaded.
