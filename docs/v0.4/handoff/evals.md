# Handoff: `evals` workstream

## What I built

- `src/llm_configurator/evals.py`: local quality checks that need no API key.
  - **Quick quiz**: short questions matched to the user's workload, checked by the app itself (never by running model-written code), scored with a Wilson 95% range and a plain note.
  - **Needle test**: a synthetic "long record of town events" sized to a target token count with the injected tokenizer. One secret code word is hidden at each position, and the test reports found/not found per position.
  - **Blind comparison**: "try my own prompts". Models answer one after another, the answers are shown under slot letters A/B/C in a shuffled order per prompt, the user votes, then reveals.
- `src/llm_configurator/evals/{general,coding,agentic,documents}.json`: original quiz items (see the counts below). Each item has `id`, `kind`, `difficulty`, `prompt` and `answer`, plus optional `accept` regexes. Files carry `version: 1`.
  - `general`: arithmetic, word problems, dates/weekdays, units, letter counting, string tasks, logic, and stable common knowledge.
  - `coding`: "what does this Python/JS print", "value of x", "which line has the bug" (line number) and a few one-word concept items. Every snippet item also stores `code` and `language`. The snippets were run with python3/node to confirm the answers; they are our fixed snippets, never model output.
  - `agentic`: `check: "tool_call"`, `tools` (JSON-schema functions) and `answer: {"name", "arguments"}`. About 5 items expect `{"name": "none"}`.
  - `documents`: top-level `passages` (synthetic memos, leases, itineraries, reports), with items referencing a passage by key.
- `tests/test_evals.py`: 28+ tests (see "How I tested").

## Public API as built

```python
WORKLOADS = ("general", "coding", "agentic", "documents")

run_quiz(chat, workload, limit=None, progress=None, cancel=None, max_tokens=256) -> {
    "kind": "quiz", "workload", "version", "correct", "total",
    "score": float 0..1, "ci_low": float 0..1, "ci_high": float 0..1,   # Wilson 95%
    "items": [{"id", "kind", "ok", "expected", "got", "seconds", "truncated"}],
    "by_kind": {kind: {"correct", "total"}}, "truncated": int,
    "seconds", "note": plain str, "timestamp"}

needle_test(chat, tokenize, context_tokens, positions=(0.1, 0.5, 0.9), progress=None, cancel=None,
            max_tokens=64, seed=7) -> {
    "kind": "needle", "context_tokens",
    "results": [{"position", "found", "expected", "got", "prompt_tokens", "seconds"}],
    "found", "total", "score", "seconds", "note", "timestamp"}

start_comparison(store, prompts, labels) -> comparison_id            # 1–5 prompts ≤ 4000 chars, 2–3 unique labels
record_output(store, comparison_id, label, prompt_index, text, timings, error=None)
blinded(store, comparison_id) -> {
    "comparison_id", "state": running|ready|cancelled|revealed, "revealed": bool,
    "items": [{"item", "prompt", "outputs": [{"slot": "A", "text", "ready": bool}], "vote": slot|"tie"|None}],
    "voted", "total", "complete": bool}                              # no labels, timings or error texts
vote(store, comparison_id, item, slot) -> {"comparison_id", "item", "slot", "votes": [slot|None...],
                                           "voted", "total", "complete"}   # slot in A/B/C or "tie"
reveal(store, comparison_id) -> {
    "comparison_id", "mapping": [{"item", "slots": {"A": label, ...}, "vote", "winner": label|None}],
    "tallies": {label: wins}, "ties", "unvoted", "overall": label|None,
    "speed": {label: {"answers", "failed", "tokens_per_second"}}, "note", "prompts"}
run_comparison(store, prompts, runners, max_tokens=512, progress=None, cancel=None) -> comparison_id
```

Helpers that others may use: `visible_text(text)` (removes `<think>` blocks and template tokens), `final_answer(text)`, `check_item(item, reply)`, `parse_tool_call(text)`, `validate_quiz(quiz) -> [problems]`, `load_quiz(workload)` and `wilson(k, n)`.

Behaviour notes:
- **Chat call**: `chat(messages, max_tokens=N)`. When the callable's signature accepts `temperature`/`seed` (or `**kwargs`), it also gets `temperature=0.0, seed=1`. `LlamaServer.chat` qualifies. A plain `str` return is accepted as `{"text": str}`.
- **Answer checking**: removes reasoning (`<think>`, `<thinking>`, `<reasoning>`, gpt-oss `<|channel|>final<|message|>`, and a stray closing tag when the template opened it). If the reply is still inside an unclosed `<think>`, it counts as no answer. The answer is the text after the last `Answer:` (or `Final answer:`, or `**Answer:**`), otherwise the last line. Normalising covers case, whitespace, outer quotes/markdown/punctuation, `\boxed{}`, thousands separators, `$`, `%` and number words 0–20. Kinds ending in `_output` compare strictly, so `1` does not pass for `1.0`.
- **Tool calls**: the first JSON object in the reply that looks like a call. OpenAI `{"function": ...}`, `{"tool_calls": [...]}`, string-encoded `arguments` and Qwen `<tool_call>` wrappers all work. The tool name must match and every expected argument must be equal (strings case/whitespace-insensitive, numbers by value, `{"$regex", "example"}` for real alternates). Extra arguments are allowed only if the tool schema has them.
- **`limit`** picks evenly spaced items, so a short quiz still covers the whole difficulty range.
- **Comparisons**: stored under the `comparisons` store key, and only the 20 newest are kept. The slot order per prompt is `random.Random(sha256(f"{id}:{item}"))`, stored in the record too. Re-voting replaces a vote. Votes close after reveal. A model that fails to start gets an `error` on its outputs (shown blind as "this model failed to run"), and the other models still run. `<think>` text is removed from the stored output.

## Deviations from the contract and why

- `vote()` returns vote progress, not per-model tallies. Tallies by model would reveal who is behind each slot before the reveal, and tallies by slot mean nothing because the slot order changes per prompt. Tallies come from `reveal()`.
- `score`, `ci_low` and `ci_high` are fractions from 0 to 1, not percentages. The note speaks in percentage points.
- Agentic items use `answer` as an object `{"name", "arguments"}` with `check: "tool_call"` instead of `accept` regexes, because tool calls are compared structurally.
- Extra optional kwargs: `max_tokens` on `run_quiz`/`needle_test`, `seed` on `needle_test` and `error` on `record_output`.
- Quiz and needle results are **not** written to `quality_results` by evals (the functions take no store). The caller (api) should `store.append("quality_results", result)`: `result["kind"]` is already `"quiz"` or `"needle"`, and the caller should add `variant_id`.

## Known gaps

- Scores from 30–60 items are noisy. The note says so, with the actual interval width.
- The quiz asks for short answers with `max_tokens=256`. Thinking models may be cut off mid-thought. Those items count as wrong and are reported in `truncated` and the note ("may understate the model"). The api could pass a larger `max_tokens` for known reasoning models.
- The needle check passes when the code word appears anywhere in the visible reply. The filler vocabulary never contains code words (a test enforces this), so a false pass would need the model to guess the exact word.
- `run_comparison` checks for cancel between prompts. A single long generation is not interrupted mid-way (that would need a cancel hook in `LlamaServer.chat`).

## Requests

- **api**: for `POST /api/quality/quiz`, run `run_quiz(server.chat, workload)` and, if `include_needle`, `needle_test(server.chat, server.tokenize, config["context"])` inside one `LlamaServer` context. Then append both results to `quality_results` with `variant_id`. For compare, build `runners = {candidate_label: lambda: LlamaServer(command, config)}`: a `LlamaServer` works directly because evals uses `.chat` when the runner itself isn't callable, but it must be started on `__enter__`. If `LlamaServer.__enter__` doesn't call `start()`, wrap it in a small `contextlib.contextmanager` that does. `GET /api/quality/compare/<id>` → `blinded`, vote → `vote`, reveal → `reveal`. Labels are only revealed through `reveal()`. `ValueError` messages are written for users; map "not found" to 404.
- **ui-quality**: `score`/`ci_*` are 0–1 fractions. Show `note` verbatim. Blinded outputs include `ready` (a spinner until true) and `vote`. The slot choices are `A`/`B`/`C` plus `tie`.
- **server** (fake llama): please answer prompts that contain `The secret code is <word>.` with that word, so integration runs of `needle_test` pass against the fake.

## How I tested

- `python3 -m unittest tests.test_evals` and the full `python3 -m unittest discover -s tests` both pass. No network is used.
- **Quiz files**: every file passes `validate_quiz`. That covers unique ids, answers present, accept regexes that compile, passage references, tool names and required/known arguments, `$regex` examples that match, and every stated answer passing its own checker. Ids are unique across all files, and junk or wrong answers fail every item.
- **Fake models**: always-correct (100% on all four workloads), always-wrong (0%, `ci_low` 0), reasoning-wrapped (still 100%), malformed JSON for agentic (0%), and replies cut off at the length limit.
- **Checker units**: think-stripping edge cases, answer extraction, number formats, strict program output, tool-call shapes, and Wilson values against known numbers.
- **Needle**: a word-count tokenizer checks that the prompt fits the budget and fills at least 85% of it, that the needle lands at the requested depth, that runs are deterministic, that misses are reported, and that validation and cancel work.
- **Comparisons**: the full blind → vote → reveal flow, with no label, timing or error leakage in `blinded()`/`vote()`, input validation, pruning, and `run_comparison` loading one model at a time. A failing model doesn't sink the others, `.chat` server objects and plain callables both work, and cancel stops before the next model starts.
- **Answer review**: quiz answers were written by helper agents that ran every code snippet and recomputed arithmetic, then checked by an independent reviewer agent for correctness and ambiguity.

## llama.cpp facts assumed (for the integration harness to confirm)

- `/v1/chat/completions` returns the model's reasoning inline as `<think>…</think>` in `content` when no reasoning parser is on. If llama-server's `--reasoning-format` moves it into `reasoning_content`, `content` is already clean, and both cases work.
- `finish_reason == "length"` marks replies cut off by `max_tokens`.
- `timings.predicted_per_second` is present in chat responses and is used for the speed line in `reveal()`.
- `/tokenize` token counts are close to what the chat template adds. The needle test leaves 64 tokens of slack plus `max_tokens` for the template and the reply.
