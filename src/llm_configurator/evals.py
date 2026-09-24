"""Local quality checks that need no API key: task quizzes, long-document recall and blind comparisons.

Honesty rules: every quiz item has one machine-checkable answer, scores always come with a
95% range so small quizzes are not over-read, and model-written code is never executed.
Blind comparisons never show which model wrote which answer until the user asks to reveal.
"""
import contextlib
import hashlib
import inspect
import json
import math
import random
import re
import secrets
import time
import unicodedata
from pathlib import Path

from .domain import Cancelled, check_cancel, now

WORKLOADS = ("general", "coding", "agentic", "documents")
QUIZ_DIR = Path(__file__).with_name("evals")
SLOTS = "ABC"
TIE = "tie"
MAX_PROMPTS, MAX_PROMPT_CHARS, KEEP_COMPARISONS = 5, 4000, 20

ANSWER_INSTRUCTION = "\n\nKeep it short. Reply with the final answer on its own last line, in the form:\nAnswer: <answer>"
TOOL_INSTRUCTION = ('\n\nReply with ONLY a JSON object of the form {"name": "<tool name>", "arguments": {...}} '
                    'that calls the one best tool for this request. If no tool fits, reply '
                    '{"name": "none", "arguments": {}}.')

# ---------------------------------------------------------------- reading replies

_THINK_BLOCK = re.compile(r"<(think|thinking|reasoning)>.*?</\1>", re.S | re.I)
_THINK_CLOSE = re.compile(r"</(?:think|thinking|reasoning)>", re.I)
_THINK_OPEN = re.compile(r"<(?:think|thinking|reasoning)>", re.I)
_SPECIAL_TOKEN = re.compile(r"<\|[^|<>]{1,40}\|>|</s>|<s>")
_ANSWER_LINE = re.compile(r"\b(?:final\s+)?answer\s*(?:\*\*|__)?\s*[:：]\s*(.*)", re.I)
_BOXED = re.compile(r"\\boxed\{([^{}]*)\}")
_NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NUMBER = re.compile(r"[-+]?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|[-+]?\$?\.\d+%?")
_OUTER = " \t\n.,;:!?\"'`*“”‘’"


def visible_text(text):
    """What the model 'said' once hidden reasoning and template tokens are removed."""
    text = text or ""
    if "<|channel|>final<|message|>" in text:  # gpt-oss style: analysis first, answer after
        text = text.rsplit("<|channel|>final<|message|>", 1)[1]
    text = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(text))
    if closes:  # the opening tag was part of the prompt template
        text = text[closes[-1].end():]
    opened = _THINK_OPEN.search(text)
    if opened:  # still thinking when the reply was cut off: no answer was given
        text = text[:opened.start()]
    return _SPECIAL_TOKEN.sub("", text).strip()


def final_answer(text):
    """The text after the last 'Answer:' marker, else the last non-empty line."""
    body = visible_text(text)
    body = _BOXED.sub(lambda m: m.group(1), body)
    lines = [line.strip() for line in body.splitlines()]
    for index in range(len(lines) - 1, -1, -1):
        matches = list(_ANSWER_LINE.finditer(lines[index]))
        if matches:
            answer = matches[-1].group(1).strip(_OUTER)
            if answer:
                return answer
            following = [line for line in lines[index + 1:] if line.strip(_OUTER)]
            return following[0] if following else ""
    rest = [line for line in lines if line.strip(_OUTER)]
    return rest[-1] if rest else ""


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).replace("\u2212", "-").lower()
    return re.sub(r"\s+", " ", text).strip(_OUTER).strip()


def _number(text):
    text = normalize(text)
    if text in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[text])
    if not _NUMBER.fullmatch(text):
        return None
    try:
        return float(text.replace(",", "").replace("$", "").rstrip("%"))
    except ValueError:
        return None


def _numbers_in(text):
    return [value for value in (_number(match) for match in _NUMBER.findall(normalize(text))) if value is not None]


def _close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def check_text(item, reply_text):
    """(ok, extracted answer). Exact after normalising; numbers compared by value unless the item is strict."""
    got = final_answer(reply_text)
    wanted, said = normalize(item["answer"]), normalize(got)
    if said and (said == wanted or said.replace(" ", "") == wanted.replace(" ", "")):
        return True, got
    if any(re.fullmatch(pattern, said, re.I) for pattern in item.get("accept", ())):
        return True, got
    # Program output must match exactly ("1.0" is not "1"); elsewhere "42 apples" answers "42".
    strict = item.get("strict", str(item.get("kind", "")).endswith("_output"))
    expected = _number(wanted)
    if not strict and expected is not None and said:
        value = _number(said)
        if value is None:
            found = _numbers_in(said)
            value = found[0] if len(found) == 1 else None
        if value is not None and _close(value, expected):
            return True, got
    return False, got


def parse_tool_call(text):
    """First JSON object in the reply that looks like a tool call, as {"name", "arguments"}; else None."""
    body = visible_text(text)
    decoder = json.JSONDecoder()
    for start in [m.start() for m in re.finditer(r"\{", body)][:200]:
        try:
            value, _ = decoder.raw_decode(body, start)
        except ValueError:
            continue
        call = _as_call(value)
        if call:
            return call
    return None


def _as_call(value):
    if isinstance(value, dict) and isinstance(value.get("tool_calls"), list) and value["tool_calls"]:
        value = value["tool_calls"][0]
    if isinstance(value, dict) and isinstance(value.get("function"), dict):
        value = value["function"]
    if not isinstance(value, dict):
        return None
    name = value.get("name", value.get("tool"))
    arguments = value.get("arguments", value.get("parameters", value.get("args", {})))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except ValueError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def _same(expected, got):
    if isinstance(expected, dict) and "$regex" in expected:
        return (isinstance(got, (str, int, float)) and not isinstance(got, bool)
                and re.fullmatch(expected["$regex"], normalize(got), re.I) is not None)
    if isinstance(expected, bool) or isinstance(got, bool):
        return expected is got
    if isinstance(expected, (int, float)):
        value = got if isinstance(got, (int, float)) else _number(got) if isinstance(got, str) else None
        return value is not None and _close(float(value), float(expected))
    if isinstance(expected, str):
        return isinstance(got, (str, int, float)) and normalize(got) == normalize(expected)
    if isinstance(expected, list):
        return isinstance(got, list) and len(got) == len(expected) and all(map(_same, expected, got))
    if isinstance(expected, dict):  # nested objects: every expected key must match; extras are tolerated
        return isinstance(got, dict) and all(k in got and _same(v, got[k]) for k, v in expected.items())
    return expected is None and got is None


def check_tool_call(item, reply_text):
    """(ok, what the model called). Right tool, every expected argument equal, no invented argument names."""
    call = parse_tool_call(reply_text)
    if call is None:
        return False, "(no valid JSON tool call)"
    got = json.dumps(call, ensure_ascii=False)
    expected = item["answer"]
    if call["name"].strip().lower() != expected["name"].lower():
        return False, got
    if expected["name"].lower() == "none":
        return True, got
    tool = next((t for t in item.get("tools", ()) if t["name"] == expected["name"]), {})
    allowed = set((tool.get("parameters") or {}).get("properties", {})) | set(expected["arguments"])
    arguments = call["arguments"]
    ok = (set(arguments) <= allowed
          and all(key in arguments and _same(value, arguments[key]) for key, value in expected["arguments"].items()))
    return ok, got


def check_item(item, reply_text):
    if item.get("check") == "tool_call":
        return check_tool_call(item, reply_text)
    return check_text(item, reply_text)


# ---------------------------------------------------------------- quizzes

def load_quiz(workload):
    if workload not in WORKLOADS:
        raise ValueError(f"Unknown quiz '{workload}'. Choose one of: {', '.join(WORKLOADS)}.")
    with open(QUIZ_DIR / f"{workload}.json", encoding="utf-8") as handle:
        return json.load(handle)


def quiz_messages(quiz, item):
    if item.get("check") == "tool_call":
        tools = json.dumps(item["tools"], indent=1, ensure_ascii=False)
        content = f"You can call these tools:\n{tools}\n\nUser request: {item['prompt']}{TOOL_INSTRUCTION}"
    elif item.get("passage"):
        passage = quiz["passages"][item["passage"]]
        content = ("Read the document and answer the question using only the document.\n\n"
                   f"<document>\n{passage}\n</document>\n\nQuestion: {item['prompt']}{ANSWER_INSTRUCTION}")
    else:
        content = item["prompt"] + ANSWER_INSTRUCTION
    return [{"role": "user", "content": content}]


def expected_text(item):
    answer = item["answer"]
    return json.dumps(answer, ensure_ascii=False) if isinstance(answer, dict) else str(answer)


def validate_quiz(quiz):
    """Problems found in a quiz file (empty list means valid). Each stated answer must pass its own checker."""
    problems = []
    if type(quiz.get("version")) is not int:
        problems.append("version must be an integer")
    items = quiz.get("items")
    if not isinstance(items, list) or not items:
        return problems + ["items must be a non-empty list"]
    seen = set()
    for item in items:
        name = item.get("id")
        if not isinstance(name, str) or not name or name in seen:
            problems.append(f"missing or duplicate id: {name!r}")
        seen.add(name)
        for key in ("prompt", "kind"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                problems.append(f"{name}: {key} must be a non-empty string")
        if "answer" not in item:
            problems.append(f"{name}: no answer")
            continue
        try:
            [re.compile(pattern) for pattern in item.get("accept", ())]
        except re.error as error:
            problems.append(f"{name}: bad accept regex ({error})")
            continue
        if item.get("passage") is not None and item["passage"] not in (quiz.get("passages") or {}):
            problems.append(f"{name}: unknown passage {item['passage']!r}")
        if item.get("check") == "tool_call":
            problems += [f"{name}: {p}" for p in _tool_item_problems(item)]
            reply = json.dumps(_canonical_call(item["answer"]))
        else:
            if not isinstance(item["answer"], str) or not item["answer"].strip():
                problems.append(f"{name}: answer must be a non-empty string")
                continue
            reply = f"Answer: {item['answer']}"
        if not check_item(item, reply)[0]:
            problems.append(f"{name}: the stated answer does not pass its own checker")
    return problems


def _example(value):
    """A concrete value for an expected argument: {"$regex"} placeholders carry an `example`."""
    if isinstance(value, dict):
        return value.get("example") if "$regex" in value else {k: _example(v) for k, v in value.items()}
    return [_example(v) for v in value] if isinstance(value, list) else value


def _canonical_call(answer):
    return {"name": answer["name"], "arguments": _example(answer.get("arguments", {}))}


def _tool_item_problems(item):
    problems, answer = [], item.get("answer")
    tools = item.get("tools")
    if not isinstance(tools, list) or not tools:
        return ["tool items need a non-empty tools list"]
    if not isinstance(answer, dict) or not isinstance(answer.get("name"), str) or not isinstance(answer.get("arguments"), dict):
        return ["answer must be {name, arguments}"]
    if answer["name"] == "none":
        return [] if not answer["arguments"] else ["a 'none' answer takes no arguments"]
    tool = next((t for t in tools if t.get("name") == answer["name"]), None)
    if tool is None:
        return [f"answer calls unknown tool {answer['name']}"]
    schema = tool.get("parameters") or {}
    properties = schema.get("properties", {})
    problems += [f"argument {k} is not in the schema" for k in answer["arguments"] if k not in properties]
    problems += [f"required argument {k} missing" for k in schema.get("required", ()) if k not in answer["arguments"]]
    return problems + [f"a $regex value needs an 'example' matching it: {p}" for p in _regex_problems(answer["arguments"])]


def _regex_problems(value):
    if isinstance(value, dict) and "$regex" in value:
        try:
            ok = re.fullmatch(value["$regex"], normalize(value.get("example", "")), re.I) is not None
        except re.error:
            ok = False
        return [] if ok and "example" in value else [value["$regex"]]
    items = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
    return [p for v in items for p in _regex_problems(v)]


def _select(items, limit):
    if limit is None or limit >= len(items):
        return list(items)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a whole number of questions, at least 1")
    return [items[(i * len(items)) // limit] for i in range(limit)]  # evenly spread over easy..hard


def wilson(correct, total, z=1.96):
    """Wilson 95% interval for a proportion; stays sensible for tiny quizzes and 0% or 100% scores."""
    if total <= 0:
        return None, None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _ask(chat, messages, max_tokens):
    """Call an injected chat function; pass greedy settings when it accepts them (LlamaServer.chat does)."""
    options = {"max_tokens": max_tokens}
    try:
        parameters = inspect.signature(chat).parameters
        takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        options.update({k: v for k, v in (("temperature", 0.0), ("seed", 1)) if takes_any or k in parameters})
    except (TypeError, ValueError):
        pass
    reply = chat(messages, **options)
    return {"text": reply} if isinstance(reply, str) else dict(reply or {})


def _report(progress, stage, done, total, message):
    if progress:
        progress({"stage": stage, "done": done, "total": total, "message": message})


def run_quiz(chat, workload, limit=None, progress=None, cancel=None, max_tokens=256):
    """Ask each quiz question once (greedy), check answers locally and score with a 95% range."""
    quiz = load_quiz(workload)
    items = _select(quiz["items"], limit)
    started, results = time.monotonic(), []
    for index, item in enumerate(items):
        check_cancel(cancel)
        _report(progress, "quiz", index, len(items), f"Question {index + 1} of {len(items)}")
        began = time.monotonic()
        try:
            reply = _ask(chat, quiz_messages(quiz, item), max_tokens)
        except Cancelled:
            raise
        except Exception as error:
            raise ValueError(f"The quiz stopped at question {index + 1}: {error}") from error
        ok, got = check_item(item, reply.get("text") or "")
        results.append({"id": item["id"], "kind": item["kind"], "ok": ok, "expected": expected_text(item),
                        "got": got[:300], "seconds": round(time.monotonic() - began, 3),
                        "truncated": reply.get("finish_reason") == "length"})
    _report(progress, "quiz", len(items), len(items), "Quiz finished")
    correct, total = sum(r["ok"] for r in results), len(results)
    low, high = wilson(correct, total)
    by_kind = {}
    for result in results:
        entry = by_kind.setdefault(result["kind"], {"correct": 0, "total": 0})
        entry["correct"] += result["ok"]
        entry["total"] += 1
    truncated = sum(r["truncated"] for r in results)
    return {"kind": "quiz", "workload": workload, "version": quiz.get("version"), "correct": correct, "total": total,
            "score": round(correct / total, 4) if total else None,
            "ci_low": round(low, 4) if low is not None else None, "ci_high": round(high, 4) if high is not None else None,
            "items": results, "by_kind": by_kind, "truncated": truncated,
            "seconds": round(time.monotonic() - started, 3), "note": _quiz_note(correct, total, low, high, truncated),
            "timestamp": now()}


def _quiz_note(correct, total, low, high, truncated):
    if not total:
        return "No questions were asked."
    spread = round((high - low) * 50)
    note = (f"{correct} of {total} right. With only {total} questions the true score is probably between "
            f"{round(low * 100)}% and {round(high * 100)}%, so treat differences under about {spread} points as noise.")
    if truncated:
        note += (f" {truncated} answers hit the length limit (often a model still 'thinking'), "
                 "so this score may understate the model.")
    return note


# ---------------------------------------------------------------- long-document recall ("needle")

_FILLER = {
    "who": ["The committee", "A local farmer", "The river guide", "Our neighbour", "The night clerk", "A travelling tailor",
            "The school board", "An old sailor", "The bakery owner", "The town surveyor", "A quiet student",
            "The harbour master"],
    "did": ["walked past", "wrote a note about", "painted", "measured", "talked about", "repaired", "cleaned",
            "photographed", "visited", "counted the windows of", "described", "planned a picnic near"],
    "what": ["the stone bridge", "the red barn", "the village library", "the narrow lane", "the grain store",
             "the clock tower", "the fishing boats", "the orchard wall", "the post office", "the wooden pier",
             "the market square", "the hill path"],
    "when": ["early on Monday", "after a long lunch", "before the rain started", "during the spring fair",
             "late in the evening", "on a windy afternoon", "at the end of summer", "just after sunrise",
             "while the bells rang", "in the cold season"],
    "tail": ["Nobody thought it was unusual.", "It was an ordinary day.", "The weather stayed mild.",
             "Most people were busy with chores.", "A few birds circled overhead.", "The streets were quiet.",
             "Tea was served afterwards.", "Everyone went home on time."],
}
CODE_WORDS = ["periwinkle", "saxophone", "tamarind", "obsidian", "zeppelin", "quokka", "marzipan", "chrysanthemum",
              "kaleidoscope", "tangerine", "nebula", "harmonica", "pistachio", "labyrinth", "cardamom", "flamingo",
              "vermilion", "trombone", "hazelnut", "glacier", "origami", "mandolin", "sapphire", "walrus"]
_NEEDLE = "The secret code is {code}."
_HEADER = "Below is a long record of everyday events in a small town. Somewhere in it is one secret code.\n\n"
_QUESTION = "\n\nWhat is the secret code mentioned in the record above? Reply with just the code word.{instruction}"


def _sentence(index, seed):
    rng = random.Random(f"{seed}:{index}")
    return " ".join([rng.choice(_FILLER["who"]), rng.choice(_FILLER["did"]), rng.choice(_FILLER["what"]),
                     rng.choice(_FILLER["when"]) + ".", rng.choice(_FILLER["tail"])])


def needle_prompt(sentences, position, code):
    """Filler sentences with the needle inserted at `position` (0 = start, 1 = end)."""
    body = list(sentences)
    body.insert(round(position * len(body)), _NEEDLE.format(code=code))
    paragraphs = [" ".join(body[i:i + 8]) for i in range(0, len(body), 8)]
    return _HEADER + "\n\n".join(paragraphs) + _QUESTION.format(instruction=ANSWER_INSTRUCTION)


def _fit(tokenize, budget, seed, code):
    """Largest filler size whose prompt fits the token budget, found with a few tokenizer calls."""
    count = lambda n: len(tokenize(needle_prompt([_sentence(i, seed) for i in range(n)], 0.5, code)))
    overhead = count(0)
    per_sentence = max(1.0, (count(40) - overhead) / 40)
    n = max(0, int((budget - overhead) / per_sentence))
    for _ in range(8):
        size = count(n)
        if size > budget:
            n = max(0, n - math.ceil((size - budget) / per_sentence) - 1)
        elif budget - size > 2 * per_sentence:
            n += int((budget - size) / per_sentence) - 1
        else:
            break
    while n > 0 and count(n) > budget:
        n -= max(1, n // 100)
    return n


def needle_test(chat, tokenize, context_tokens, positions=(0.1, 0.5, 0.9), progress=None, cancel=None,
                max_tokens=64, seed=7):
    """Hide 'The secret code is <word>' at each depth of a synthetic document sized to `context_tokens`."""
    if type(context_tokens) is not int or not 512 <= context_tokens <= 1048576:
        raise ValueError("The recall test needs a context between 512 and 1,048,576 tokens.")
    positions = list(positions)
    if not positions or not all(isinstance(p, (int, float)) and not isinstance(p, bool) and 0 <= p <= 1
                                for p in positions):
        raise ValueError("Positions must be numbers between 0 (start) and 1 (end).")
    budget = context_tokens - max_tokens - 64  # room for the reply and the chat template's own tokens
    rng = random.Random(f"needle:{seed}")
    codes = rng.sample(CODE_WORDS, len(positions)) if len(positions) <= len(CODE_WORDS) else \
        [rng.choice(CODE_WORDS) for _ in positions]
    _report(progress, "sizing", 0, len(positions), "Building a long test document")
    n = _fit(tokenize, budget, seed, codes[0])
    if n < 8:
        raise ValueError("The context is too small for a meaningful recall test. Try at least 1,024 tokens.")
    sentences = [_sentence(i, seed) for i in range(n)]
    started, results = time.monotonic(), []
    for index, (position, code) in enumerate(zip(positions, codes)):
        check_cancel(cancel)
        _report(progress, "needle", index, len(positions),
                f"Checking recall {round(position * 100)}% of the way in ({index + 1} of {len(positions)})")
        prompt = needle_prompt(sentences, position, code)
        prompt_tokens = len(tokenize(prompt))
        began = time.monotonic()
        reply = _ask(chat, [{"role": "user", "content": prompt}], max_tokens)
        text = reply.get("text") or ""
        found = re.search(rf"\b{code}\b", visible_text(text), re.I) is not None
        results.append({"position": position, "found": found, "expected": code, "got": final_answer(text)[:120],
                        "prompt_tokens": prompt_tokens, "seconds": round(time.monotonic() - began, 3)})
    _report(progress, "needle", len(positions), len(positions), "Recall test finished")
    hits = sum(r["found"] for r in results)
    missed = [f"{round(r['position'] * 100)}%" for r in results if not r["found"]]
    note = (f"Found the hidden code at {hits} of {len(results)} places in a document of about "
            f"{max(r['prompt_tokens'] for r in results):,} tokens.")
    if missed:
        note += (f" Missed at {', '.join(missed)} of the way in: the model may lose details there "
                 "in long documents at this context size.")
    return {"kind": "needle", "context_tokens": context_tokens, "results": results, "found": hits,
            "total": len(results), "score": round(hits / len(results), 4),
            "seconds": round(time.monotonic() - started, 3), "note": note, "timestamp": now()}


# ---------------------------------------------------------------- blind comparisons

def _slot_order(labels, comparison_id, item):
    """Per-item shuffle derived only from the comparison id, so it is stable and needs no global RNG."""
    order = list(labels)
    random.Random(hashlib.sha256(f"{comparison_id}:{item}".encode()).digest()).shuffle(order)
    return order


def _validate_prompts(prompts):
    if not isinstance(prompts, (list, tuple)) or not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"Give between 1 and {MAX_PROMPTS} prompts.")
    for prompt in prompts:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Each prompt must be non-empty text.")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"Each prompt must be at most {MAX_PROMPT_CHARS:,} characters.")


def _validate_labels(labels):
    if not isinstance(labels, (list, tuple)) or not 2 <= len(labels) <= len(SLOTS):
        raise ValueError(f"Compare between 2 and {len(SLOTS)} models.")
    if any(not isinstance(label, str) or not label.strip() or len(label) > 200 for label in labels):
        raise ValueError("Each model needs a short name.")
    if len(set(labels)) != len(labels):
        raise ValueError("Each model in a comparison must be different.")


def _keep_recent(comparisons):
    ordered = sorted(comparisons.items(), key=lambda kv: kv[1].get("created_at", ""))
    return dict(ordered[-KEEP_COMPARISONS:])


def start_comparison(store, prompts, labels):
    _validate_prompts(prompts)
    _validate_labels(labels)
    comparison_id = secrets.token_hex(6)
    record = {"id": comparison_id, "created_at": now(), "state": "running", "prompts": list(prompts),
              "labels": list(labels), "order": [_slot_order(labels, comparison_id, i) for i in range(len(prompts))],
              "outputs": {label: [None] * len(prompts) for label in labels}, "votes": [None] * len(prompts),
              "revealed": False}
    store.update("comparisons", lambda all_: _keep_recent({**(all_ or {}), comparison_id: record}), {})
    return comparison_id


def _get(store, comparison_id):
    record = (store.get("comparisons") or {}).get(comparison_id) if isinstance(comparison_id, str) else None
    if record is None:
        raise ValueError("That comparison was not found. It may be old; start a new one.")
    return record


def _change(store, comparison_id, change):
    """Atomically apply change(record) -> result; the record is edited in place inside the store update."""
    box = {}

    def apply(all_):
        record = (all_ or {}).get(comparison_id)
        if record is None:
            raise ValueError("That comparison was not found. It may be old; start a new one.")
        box["result"] = change(record)
        return all_

    store.update("comparisons", apply, {})
    return box["result"]


def record_output(store, comparison_id, label, prompt_index, text, timings, error=None):
    def change(record):
        if record["revealed"]:
            raise ValueError("This comparison is already revealed; start a new one.")
        if label not in record["labels"]:
            raise ValueError("That model is not part of this comparison.")
        if type(prompt_index) is not int or not 0 <= prompt_index < len(record["prompts"]):
            raise ValueError("That prompt number is not part of this comparison.")
        record["outputs"][label][prompt_index] = {"text": str(text or ""), "timings": dict(timings or {}),
                                                  "error": error}
    _change(store, comparison_id, change)


def _set_state(store, comparison_id, state):
    _change(store, comparison_id, lambda record: record.update(state=state))


def blinded(store, comparison_id):
    """Answers under slot letters only: no model names, timings or errors that could give them away."""
    record = _get(store, comparison_id)
    items = []
    for index, prompt in enumerate(record["prompts"]):
        outputs = []
        for slot, label in zip(SLOTS, record["order"][index]):
            output = record["outputs"][label][index]
            text = "" if output is None else "(No answer: this model failed to run.)" if output.get("error") \
                else output["text"]
            outputs.append({"slot": slot, "text": text, "ready": output is not None})
        items.append({"item": index, "prompt": prompt, "outputs": outputs, "vote": record["votes"][index]})
    voted = sum(v is not None for v in record["votes"])
    return {"comparison_id": comparison_id, "state": record["state"], "revealed": record["revealed"],
            "items": items, "voted": voted, "total": len(items),
            "complete": all(o["ready"] for i in items for o in i["outputs"])}


def vote(store, comparison_id, item, slot):
    """Record which slot's answer was best for one prompt ('tie' allowed). Re-voting replaces the vote."""
    def change(record):
        if record["revealed"]:
            raise ValueError("Voting is closed because the model names were already revealed.")
        if type(item) is not int or not 0 <= item < len(record["prompts"]):
            raise ValueError("That prompt number is not part of this comparison.")
        slots = SLOTS[:len(record["labels"])]
        if slot != TIE and (not isinstance(slot, str) or slot not in slots):
            raise ValueError(f"Vote for one of {', '.join(slots)} or 'tie'.")
        if any(record["outputs"][label][item] is None for label in record["labels"]):
            raise ValueError("Wait until every answer for this prompt is ready.")
        record["votes"][item] = slot
        return list(record["votes"])
    votes = _change(store, comparison_id, change)
    voted = sum(v is not None for v in votes)
    return {"comparison_id": comparison_id, "item": item, "slot": slot, "votes": votes, "voted": voted,
            "total": len(votes), "complete": voted == len(votes)}


def reveal(store, comparison_id):
    """Show which model sat behind each slot, count wins, and close voting."""
    record = _change(store, comparison_id, lambda r: r.update(revealed=True, state="revealed") or dict(r))
    labels = record["labels"]
    tallies = {label: 0 for label in labels}
    mapping, ties, unvoted = [], 0, 0
    for index, order in enumerate(record["order"]):
        slots = dict(zip(SLOTS, order))
        choice = record["votes"][index]
        winner = slots.get(choice)
        if winner:
            tallies[winner] += 1
        ties += choice == TIE
        unvoted += choice is None
        mapping.append({"item": index, "slots": slots, "vote": choice, "winner": winner})
    best = max(tallies.values())
    leaders = [label for label, wins in tallies.items() if wins == best]
    overall = leaders[0] if best and len(leaders) == 1 else None
    speed = {}
    for label in labels:
        rates = [o["timings"].get("predicted_per_second") for o in record["outputs"][label] if o and not o.get("error")]
        rates = [r for r in rates if isinstance(r, (int, float))]
        speed[label] = {"answers": sum(1 for o in record["outputs"][label] if o and not o.get("error")),
                        "failed": sum(1 for o in record["outputs"][label] if o and o.get("error")),
                        "tokens_per_second": round(sum(rates) / len(rates), 2) if rates else None}
    voted = len(mapping) - unvoted
    note = f"You picked a winner on {voted - ties} of {len(mapping)} prompts."
    if overall is None and voted:
        note += " No single model came out ahead."
    if len(mapping) < 5:
        note += " With this few prompts, a one-vote lead is a hint, not proof."
    return {"comparison_id": comparison_id, "mapping": mapping, "tallies": tallies, "ties": ties,
            "unvoted": unvoted, "overall": overall, "speed": speed, "note": note,
            "prompts": record["prompts"]}


def _open_runner(factory):
    runner = factory()
    return runner if hasattr(runner, "__enter__") else contextlib.nullcontext(runner)


def run_comparison(store, prompts, runners, max_tokens=512, progress=None, cancel=None):
    """Answer every prompt with each model in turn. Only one model is loaded at a time, to spare memory.

    `runners` maps a label to a factory returning a context manager (it starts and stops a server)
    that yields a chat callable, or an object with a `.chat` method.
    """
    labels = list(runners)
    comparison_id = start_comparison(store, prompts, labels)
    total, done = len(labels) * len(prompts), 0
    try:
        for number, label in enumerate(labels, 1):
            check_cancel(cancel)
            _report(progress, "loading", done, total, f"Starting model {number} of {len(labels)}")
            try:
                with _open_runner(runners[label]) as runner:
                    chat = runner if callable(runner) else runner.chat
                    for index, prompt in enumerate(prompts):
                        check_cancel(cancel)
                        _report(progress, "answering", done, total,
                                f"Model {number} of {len(labels)}: prompt {index + 1} of {len(prompts)}")
                        began = time.monotonic()
                        reply = _ask(chat, [{"role": "user", "content": prompt}], max_tokens)
                        raw = reply.get("text") or ""
                        text = visible_text(raw)
                        if not text and raw.strip():
                            text = "(No visible answer: the model ran out of room while thinking.)"
                        timings = {**(reply.get("timings") or {}), "seconds": round(time.monotonic() - began, 3),
                                   "finish_reason": reply.get("finish_reason")}
                        record_output(store, comparison_id, label, index, text, timings)
                        done += 1
            except Cancelled:
                raise
            except Exception as error:  # one model failing must not lose the others' answers
                missing = [i for i, o in enumerate(_get(store, comparison_id)["outputs"][label]) if o is None]
                for index in missing:
                    record_output(store, comparison_id, label, index, "", {}, error=str(error) or type(error).__name__)
                done = number * len(prompts)
    except Cancelled:
        _set_state(store, comparison_id, "cancelled")
        raise
    _set_state(store, comparison_id, "ready")
    _report(progress, "done", total, total, "All answers are ready to vote on")
    return comparison_id
