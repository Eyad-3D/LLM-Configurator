# Fix-up: `evals`

Owner files: `src/llm_configurator/evals.py`, `src/llm_configurator/evals/*.json`, `tests/test_evals.py`.

## Summary

- **No answer key was wrong.** All 202 quiz items were solved independently by seven reviewer agents (one per half file), each answering before looking at the key. Every code snippet was executed (Python 3.11, Node 22), arithmetic, dates and letter counts were computed, and the logic puzzles were brute-forced. A second, independent adversarial pass then re-checked every changed item and tried to make the checker accept wrong answers.
- What was wrong was mostly **checker strictness**: correct, normally written replies were marked wrong ("12:05 p.m.", "25,000 cm²", "16½", "Inês", lists in a different order). Two gaps worked the other way and gave credit for wrong answers: `hop-3` passed for `HOP-3`, and `24` passed for the program output `2 4`.
- All quiz files are now `version: 2`.

## Items changed (none removed)

| Item | Change | Why |
|---|---|---|
| gen-003, gen-013 | accept "Tokyo, Japan", "Ottawa, Canada/Ontario" | correct replies failed |
| gen-014 | accept "Iron (Fe)" | correct reply failed |
| gen-037 | accept "No, not necessarily" | correct reply failed |
| gen-041 | accept "2 hours 24 minutes" forms (12/5 and 2 2/5 now pass through the checker) | correct replies failed |
| gen-048 | accept "25,000 cm²", "sq cm" forms | the ² became a second number |
| gen-053 | accept "2026-3-6", "Friday, March 6, 2026", "6th of March 2026" | correct replies failed (numeric d/m forms stay rejected: ambiguous) |
| doc-001 | accept "Mr/Dr Owen Castellan" | correct reply failed |
| doc-004 | accept "Mon 17 March", "17 Mar", "March 17th at 8 am" | correct replies failed |
| doc-008 | accept "the caged bird" | correct reply failed |
| doc-013, doc-016 | accept "12:05 p.m." / "1:55 p.m." and "13:55 (1:55 pm)" | correct replies failed |
| doc-015 | accept "(0161) 550 3020" and a leading "Saltmarsh Travel …" | correct replies failed; the tour leader's number still fails |
| doc-027 | wider accept for E4 ("grind cup not in place", "no grind cup or portafilter in the cradle", …) | correct replies failed |
| doc-046 | accept "Inês" (circumflex); the pattern only had "é" | correct spelling failed |
| agt-001, agt-016 | accept "Lisboa", "Lisbon, PT", "Lyon, FR" | correct arguments failed |
| agt-005 | accept `36*4817`, `(4817*36)`, `·` | equivalent expressions failed |
| agt-008 | prompt reworded to "currently in 'shipped' status (sent out but not yet delivered)" | "which orders have shipped" could fairly include delivered ones |
| agt-018, agt-029 | wider title/text patterns ("renew auto insurance", "remind me to call the plumber") | correct arguments failed |
| agt-019 | query only needs to contain "glass orchard" | many natural query strings failed |
| agt-036, agt-037 | attendee and colour lists are now order-free (`$unordered`) | the order of a set is not part of the answer |
| agt-037 | `max_price` also accepts 59.99 ("under $60") | strict reading failed |
| agt-038 | line-item descriptions accept "flour", "sacks of flour", "delivery", "delivery charge" | correct arguments failed |
| cod-031 | accept Firefox's `Array(3) [ 1, 10, 9 ]` | correct reply failed |
| cod-053 | accept "the stack", "stack data structure" | correct reply failed |
| cod-054 | accept LaTeX `O(\log n)`, `$O(\log n)$`, `O(log_2 n)`, "O(log n) time", Θ | correct replies failed |
| cod-024 | no item change: now wrong-case replies fail (see checker) | `hop-3` passed for `HOP-3` |

## Checker fixes (`evals.py`)

- `.7` keeps its decimal point. Before, the leading dot was stripped as punctuation, so `.7` read as 7 (wrong for 0.7, and accepted for a key of 7).
- Program output (`*_output` kinds) is now **case-sensitive**, and spaces may differ only next to punctuation: `[1,16]` = `[1, 16]`, but `24` ≠ `2 4` and `b a` ≠ `ba`. Accept patterns for these kinds are matched case-sensitively.
- Working shown on the answer line counts: the result after a last `=` ("40 x 12 = 480") and the text before a trailing note ("20.2 (60.6 / 3)"). A note containing "or/maybe/possibly" doesn't count ("4 (or maybe 5)" fails).
- The single-number fallback ("42 apples" answers 42) is refused when the answer is negated ("not 42").
- Numbers: fractions and mixed numbers (`12/5`, `2 2/5`, `16½`), number words up to ninety-nine ("forty-two", "one year"), `€`/`£`, and unit powers (`cm²`, `m^2` no longer count as an extra number).
- `a.m.`/`p.m.` normalise to `am`/`pm`. Multiple-choice replies `(b)` and `b)` read as `b`.
- Tool calls: `{"$unordered": [...]}` for set-like list arguments (the validator checks these too), and a `functions.`/`tools.` name prefix is accepted. A plain-text "no tool fits" still **fails** a `none` item: the prompt asks for `{"name": "none", …}`, and following the format is part of what the agentic quiz tests.
- `truncated` now counts only wrong answers that hit the length limit.

## Seams

- **`LlamaServer.chat`** (read in `llama_server.py`): its signature is `chat(messages, max_tokens, temperature, seed, stop, timeout, cache_prompt, enable_thinking, extra)`. `_ask` forwards only the options the callable accepts (so plain test lambdas still work, as does `app._Chat`'s `**options`).
  - Quiz and needle send `temperature=0, seed=1, enable_thinking=False` (→ `chat_template_kwargs`). This makes Qwen3-style models answer directly instead of burning the token budget on thinking, and the fake honours it. Templates without the switch ignore it. Blind comparisons do **not** switch thinking off, because the user should see the model as it will be used.
  - The quiz `max_tokens` default is now 512 (was 256). "Keep it short" stops normal models early, so only models that can't switch thinking off (for example gpt-oss) use the extra room.
- **Needle sizing**: the budget is `context - max_tokens - max(128, 2% of context)`. The old fixed 64 was too tight for templates that add a system message (Llama 3.1, gpt-oss). The real run showed the template adds ~16 tokens on the tiny Qwen3 model.
  - The deepest needle runs first with `cache_prompt=True`, so later prompts only re-read the text after their needle. Results keep the requested order.
  - A reply cut off before answering is reported as `truncated` ("tells us nothing"), not as a recall miss.
- **Quiz with a small context**: a question whose document doesn't fit (the llama-server "context" error) is now skipped and listed in `skipped`, and the note says so. Before, the whole quiz stopped: the real run with a 2,048-token context stopped on the documents quiz.
- **Blind comparison leak fixed**: models answer one after another, so showing each answer as soon as it arrived told the user which slot on every prompt belonged to the model running now (and the job subject lists candidates in run order). Now a prompt's answers appear only when all models have answered it.
- **Reveal shape**: each `mapping` row now also carries the slot letters at top level (`{"A": label, "B": label, "item", "slots", "vote", "winner"}`), and the response has `"labels": {label: friendly name}`, as pinned in FIXUPS.md. `start_comparison`/`run_comparison` take an optional `names={label: friendly}`. Labels may now be up to 600 characters, to fit candidate ids.
- **Store record shapes**: evals takes no store for quizzes. The pinned quiz record is written by `app.py` (see Requests).

## Requests

- **api-cli (`app.py`)**
  1. `quiz_job`: write the pinned record `{"kind": "quiz", "variant_id", "candidate_id", "workload", "score", "ci_low", "ci_high", "needle": needle_result|None, "result": quiz_result, "timestamp"}`, as one record rather than a separate `needle` record. It currently lacks `candidate_id`, `result` and the nested `needle`.
  2. `compare_job`: key `runners` by **candidate id** and pass `names={candidate_id: "Name Quant"}` to `evals.run_comparison(..., names=...)`. Then `reveal()` returns candidate ids in `mapping`/`tallies` and friendly names in `labels`, as pinned. Today the labels are friendly names, so the UI gets names where it expects ids.
- **api-http (`server.py`)**: `POST /api/quality/vote` only allows `slot` in `A/B/C`. `evals.vote` also accepts `"tie"`, and the reveal counts ties, so please allow `"tie"`.
- **ui-quality**:
  - In `blinded()`, all outputs of a prompt now turn `ready` together.
  - The quiz result has `skipped` (ids too long for the context), and needle rows have `truncated`.
  - Show `note` verbatim. It already explains both.

## Test evidence

- `python3 -m unittest tests.test_evals`: 39 tests, OK (28 before). The new tests cover:
  - tricky replies ("14"/"-4"/".4"/"4 or 5"/"not 4"/"12 = 3 x 4" for key 4, `.7`, `16½`, `25,000 cm²`, the unicode minus, "forty-two", `(b)`)
  - case and spacing in program output
  - reasoning tags and template tokens
  - `$unordered` and prefixed tool names
  - thinking switched off, and too-long questions skipped
  - needle order, caching and truncation
  - no early reveal in the blind comparison, and the pinned reveal shape
  - a **real `LlamaServer` against the fake llama.cpp** (with `FAKE_LLAMA_REASONING=1`, so the test fails if thinking isn't switched off): quiz and needle, 3/3 found.
- Full suite: `python3 -m unittest discover -s tests` gives 615 tests with 2 failures. Both are `integration.test_fake_matches_real` (`--draft-max`, `llama-bench --version`): known fake-vs-real gaps owned by `server-fake`, and identical on the base commit. (`test_adapters` needed the documented `pip install --ignore-installed cryptography`.)
- Every `*_output` snippet was re-run after the changes: stdout equals the key exactly, and passes the case-sensitive checker.
- **Real llama.cpp end to end** (built with `scripts/build_llama_cpp.sh`, `tiny-qwen3moe-Q4_K_M.gguf`, context 2048):
  - `LlamaServer.chat(..., enable_thinking=False)` works, and its reply has `text`, `reasoning`, `finish_reason`, `timings`, `usage`.
  - `run_quiz` ran for all four workloads (limit 4; the documents run skipped 3 questions that were too long, as designed).
  - `needle_test` built prompts of 1,831–1,834 tokens with no context error.
  - `run_comparison` ran two `LlamaServer` runners one after another. `blinded` showed no ids, and `reveal` returned `mapping[0] = {"A": "cand-1", "B": "cand-2", …}`, `labels` and `tokens_per_second`.
  - The tiny random model scored 0, as expected: this checked the mechanics, not quality.
