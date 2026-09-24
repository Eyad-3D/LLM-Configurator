import contextlib
import json
import re
import tempfile
import threading
import unittest

from llm_configurator import evals
from llm_configurator.domain import Cancelled
from llm_configurator.storage import Store


def answer_book(workload):
    """Map each exact quiz prompt to a correct reply, so a fake model can 'know' the answers."""
    quiz = evals.load_quiz(workload)
    book = {}
    for item in quiz["items"]:
        content = evals.quiz_messages(quiz, item)[0]["content"]
        if item.get("check") == "tool_call":
            book[content] = json.dumps(evals._canonical_call(item["answer"]))
        else:
            book[content] = f"Answer: {item['answer']}"
    return book


def correct_chat(workload, wrap=lambda text: text):
    book = answer_book(workload)
    return lambda messages, max_tokens: {"text": wrap(book[messages[-1]["content"]]), "finish_reason": "stop"}


def wrong_chat(messages, max_tokens):
    return {"text": "Answer: qqqzzz-wrong", "finish_reason": "stop"}


def malformed_chat(messages, max_tokens):
    return {"text": '{"name": "get_weather", "arguments": {"city": "Oslo"', "finish_reason": "stop"}


class QuizFileTests(unittest.TestCase):
    def test_every_quiz_file_is_valid_and_self_consistent(self):
        for workload in evals.WORKLOADS:
            with self.subTest(workload=workload):
                quiz = evals.load_quiz(workload)
                self.assertEqual(evals.validate_quiz(quiz), [])
                self.assertTrue(30 <= len(quiz["items"]) <= 60, len(quiz["items"]))
                self.assertEqual(quiz.get("workload"), workload)
                for item in quiz["items"]:
                    self.assertIn(item.get("difficulty"), {"easy", "medium", "hard"}, item["id"])

    def test_ids_are_unique_across_all_files(self):
        ids = [item["id"] for w in evals.WORKLOADS for item in evals.load_quiz(w)["items"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_wrong_answers_fail_every_item(self):
        for workload in evals.WORKLOADS:
            for item in evals.load_quiz(workload)["items"]:
                for reply in ("Answer: qqqzzz-wrong", "", '{"name": "qqq_tool", "arguments": {}}'):
                    self.assertFalse(evals.check_item(item, reply)[0], (item["id"], reply))

    def test_agentic_items_use_tool_checks(self):
        quiz = evals.load_quiz("agentic")
        self.assertTrue(all(item.get("check") == "tool_call" for item in quiz["items"]))
        self.assertTrue(any(item["answer"]["name"] == "none" for item in quiz["items"]))

    def test_validator_catches_bad_items(self):
        quiz = {"version": 1, "items": [
            {"id": "x", "kind": "k", "prompt": "p", "answer": "4", "accept": ["("]},
            {"id": "x", "kind": "k", "prompt": "p", "answer": "5"},
            {"id": "y", "kind": "k", "prompt": "p"},
            {"id": "z", "kind": "k", "prompt": "p", "answer": "5", "passage": "missing"},
            {"id": "t", "kind": "k", "prompt": "p", "check": "tool_call", "tools": [{"name": "a", "parameters": {
                "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}}],
             "answer": {"name": "a", "arguments": {"r": 1}}}]}
        problems = " | ".join(evals.validate_quiz(quiz))
        for fragment in ("bad accept regex", "duplicate id", "y: no answer", "unknown passage",
                         "argument r is not in the schema", "required argument q missing"):
            self.assertIn(fragment, problems)
        self.assertIn("version", " ".join(evals.validate_quiz({"items": []})))


class CheckerTests(unittest.TestCase):
    def test_reasoning_is_stripped_before_reading_the_answer(self):
        item = {"id": "a", "kind": "arithmetic", "answer": "42"}
        self.assertTrue(evals.check_text(item, "<think>maybe 41? Answer: 41</think>\nAnswer: 42")[0])
        self.assertFalse(evals.check_text(item, "<think>Answer: 42</think>\nAnswer: 41")[0])
        self.assertTrue(evals.check_text(item, "reasoning here...</think>Answer: 42")[0])
        self.assertFalse(evals.check_text(item, "<think>Answer: 42 is likely but let me")[0])
        self.assertTrue(evals.check_text(item, "<|channel|>analysis<|message|>41?<|channel|>final<|message|>42")[0])

    def test_answer_extraction_and_normalising(self):
        self.assertEqual(evals.final_answer("Some working.\n**Answer:** Paris."), "Paris")
        self.assertEqual(evals.final_answer("Answer:\n  7\n"), "7")
        self.assertEqual(evals.final_answer("line one\nthe result is 9\n"), "the result is 9")
        self.assertEqual(evals.final_answer("So \\boxed{12}"), "So 12")
        item = {"id": "b", "kind": "capital", "answer": "Ottawa", "accept": ["ottawa,? (ontario|canada)"]}
        for reply in ("Answer: ottawa", "ANSWER: `Ottawa`", "Final answer: Ottawa, Canada", "Ottawa"):
            self.assertTrue(evals.check_text(item, reply)[0], reply)
        self.assertFalse(evals.check_text(item, "Answer: Toronto")[0])

    def test_number_formats(self):
        item = {"id": "n", "kind": "arithmetic", "answer": "1250"}
        for reply in ("Answer: 1,250", "Answer: 1250.0", "Answer: $1250", "Answer: 1250 dollars"):
            self.assertTrue(evals.check_text(item, reply)[0], reply)
        for reply in ("Answer: 125", "Answer: 1250 or 1300", "Answer: 12,50"):
            self.assertFalse(evals.check_text(item, reply)[0], reply)
        self.assertTrue(evals.check_text({"id": "w", "kind": "count", "answer": "3"}, "Answer: three")[0])

    def test_program_output_is_strict(self):
        item = {"id": "p", "kind": "python_output", "answer": "1.0"}
        self.assertFalse(evals.check_text(item, "Answer: 1")[0])
        self.assertTrue(evals.check_text(item, "Answer: 1.0")[0])
        listing = {"id": "l", "kind": "python_output", "answer": "[1, 16]"}
        self.assertTrue(evals.check_text(listing, "Answer: [1,16]")[0])

    def test_tool_call_parsing(self):
        item = {"id": "t", "kind": "tool_choice", "check": "tool_call",
                "tools": [{"name": "get_weather", "parameters": {"type": "object", "properties": {
                    "city": {"type": "string"}, "days": {"type": "integer"}, "unit": {"type": "string"}}}}],
                "answer": {"name": "get_weather", "arguments": {"city": {"$regex": "krak(o|ó)w", "example": "Kraków"},
                                                                "days": 3}}}
        good = ['{"name": "get_weather", "arguments": {"city": "Krakow", "days": 3}}',
                '```json\n{"name":"get_weather","arguments":{"city":"KRAKÓW","days":"3","unit":"c"}}\n```',
                '<think>{"name": "wrong"}</think><tool_call>{"name": "get_weather", "arguments": "{\\"city\\": \\"Krakow\\", \\"days\\": 3}"}</tool_call>',
                '{"function": {"name": "get_weather", "arguments": {"city": "Kraków", "days": 3.0}}}']
        for reply in good:
            self.assertTrue(evals.check_tool_call(item, reply)[0], reply)
        bad = ['{"name": "get_weather", "arguments": {"city": "Krakow", "days": 4}}',
               '{"name": "get_weather", "arguments": {"city": "Krakow", "days": 3, "invented": 1}}',
               '{"name": "get_forecast", "arguments": {"city": "Krakow", "days": 3}}',
               '{"name": "get_weather", "arguments": {"city": "Krakow"',
               'I would call get_weather for Krakow.']
        for reply in bad:
            self.assertFalse(evals.check_tool_call(item, reply)[0], reply)
        none_item = {**item, "answer": {"name": "none", "arguments": {}}}
        self.assertTrue(evals.check_tool_call(none_item, '{"name": "none", "arguments": {}}')[0])
        self.assertFalse(evals.check_tool_call(none_item, '{"name": "get_weather", "arguments": {}}')[0])

    def test_wilson_interval(self):
        low, high = evals.wilson(8, 10)
        self.assertAlmostEqual(low, 0.4902, places=3)
        self.assertAlmostEqual(high, 0.9433, places=3)
        self.assertEqual(evals.wilson(0, 0), (None, None))
        low, high = evals.wilson(0, 5)
        self.assertEqual(low, 0.0)
        self.assertGreater(high, 0.4)


class RunQuizTests(unittest.TestCase):
    def test_always_correct_scores_full_marks_on_every_workload(self):
        for workload in evals.WORKLOADS:
            with self.subTest(workload=workload):
                result = evals.run_quiz(correct_chat(workload), workload)
                self.assertEqual(result["correct"], result["total"])
                self.assertEqual(result["score"], 1.0)
                self.assertLess(result["ci_low"], 1.0)
                self.assertEqual(result["ci_high"], 1.0)
                self.assertTrue(all(i["ok"] for i in result["items"]))

    def test_reasoning_wrapped_answers_still_count(self):
        wrap = lambda text: f"<think>Let me think. Answer: 0\n{{\"name\": \"nope\"}}</think>\n{text}"
        for workload in evals.WORKLOADS:
            result = evals.run_quiz(correct_chat(workload, wrap), workload)
            self.assertEqual(result["correct"], result["total"], workload)

    def test_always_wrong_and_malformed(self):
        result = evals.run_quiz(wrong_chat, "general")
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["ci_low"], 0.0)
        self.assertIn("noise", result["note"])
        self.assertEqual(evals.run_quiz(malformed_chat, "agentic")["correct"], 0)

    def test_shape_limit_progress_and_greedy_settings(self):
        seen, updates = [], []

        def chat(messages, max_tokens, temperature=None, seed=None):
            seen.append((max_tokens, temperature, seed))
            return {"text": "Answer: nope", "finish_reason": "length"}

        result = evals.run_quiz(chat, "coding", limit=5, progress=updates.append, max_tokens=99)
        self.assertEqual(result["total"], 5)
        self.assertEqual(set(seen), {(99, 0.0, 1)})
        self.assertEqual(result["truncated"], 5)
        self.assertIn("length limit", result["note"])
        for key in ("workload", "correct", "total", "score", "ci_low", "ci_high", "items", "seconds", "note"):
            self.assertIn(key, result)
        self.assertEqual(set(result["items"][0]), {"id", "kind", "ok", "expected", "got", "seconds", "truncated"})
        self.assertEqual(updates[-1]["done"], 5)
        self.assertTrue(all({"stage", "done", "total", "message"} <= set(u) for u in updates))
        kinds = {item["kind"] for item in result["items"]}
        self.assertGreater(len(kinds), 1)  # the limited sample is spread across the file

    def test_plain_string_replies_and_errors(self):
        self.assertEqual(evals.run_quiz(lambda m, max_tokens: "nope", "general", limit=2)["total"], 2)

        def broken(messages, max_tokens):
            raise ConnectionError("server went away")

        with self.assertRaisesRegex(ValueError, "question 1"):
            evals.run_quiz(broken, "general")
        with self.assertRaisesRegex(ValueError, "Unknown quiz"):
            evals.run_quiz(wrong_chat, "poetry")
        with self.assertRaises(ValueError):
            evals.run_quiz(wrong_chat, "general", limit=0)

    def test_cancel_stops_the_quiz(self):
        cancel = threading.Event()
        calls = []

        def chat(messages, max_tokens):
            calls.append(1)
            if len(calls) == 3:
                cancel.set()
            return {"text": "Answer: x"}

        with self.assertRaises(Cancelled):
            evals.run_quiz(chat, "general", cancel=cancel)
        self.assertEqual(len(calls), 3)


def word_tokenize(text):
    return text.split()


def needle_chat(messages, max_tokens):
    found = re.search(r"The secret code is (\w+)\.", messages[-1]["content"])
    return {"text": f"<think>hmm</think>Answer: {found.group(1) if found else 'unknown'}"}


class NeedleTests(unittest.TestCase):
    def test_finds_code_at_each_position_and_fits_budget(self):
        result = evals.needle_test(needle_chat, word_tokenize, 4096, positions=(0.0, 0.25, 0.5, 1.0))
        self.assertEqual(result["found"], 4)
        self.assertEqual(result["score"], 1.0)
        self.assertEqual([r["position"] for r in result["results"]], [0.0, 0.25, 0.5, 1.0])
        self.assertEqual(len({r["expected"] for r in result["results"]}), 4)
        for row in result["results"]:
            self.assertLessEqual(row["prompt_tokens"], 4096 - 64)
            self.assertGreater(row["prompt_tokens"], 4096 * 0.85)

    def test_position_places_the_needle(self):
        sentences = [evals._sentence(i, 7) for i in range(100)]
        for position in (0.0, 0.5, 1.0):
            prompt = evals.needle_prompt(sentences, position, "walrus")
            before = prompt.split("The secret code is walrus.")[0]
            share = sum(before.count(s) for s in set(sentences)) / 100
            self.assertAlmostEqual(share, position, delta=0.05)

    def test_filler_is_deterministic_and_never_contains_codes(self):
        self.assertEqual(evals._sentence(5, 7), evals._sentence(5, 7))
        vocabulary = " ".join(" ".join(words) for words in evals._FILLER.values()).lower()
        for code in evals.CODE_WORDS:
            self.assertNotIn(code, vocabulary)
        a = evals.needle_test(needle_chat, word_tokenize, 1024)
        b = evals.needle_test(needle_chat, word_tokenize, 1024)
        self.assertEqual([r["expected"] for r in a["results"]], [r["expected"] for r in b["results"]])

    def test_misses_are_reported(self):
        def forgetful(messages, max_tokens):
            content = messages[-1]["content"]
            middle = content.find("The secret code") / len(content)
            return needle_chat(messages, max_tokens) if middle < 0.3 else {"text": "Answer: I don't know"}

        result = evals.needle_test(forgetful, word_tokenize, 2048)
        self.assertEqual([r["found"] for r in result["results"]], [True, False, False])
        self.assertIn("Missed at 50%, 90%", result["note"])

    def test_validation_and_cancel(self):
        with self.assertRaises(ValueError):
            evals.needle_test(needle_chat, word_tokenize, 100)
        with self.assertRaises(ValueError):
            evals.needle_test(needle_chat, word_tokenize, 2048, positions=(1.5,))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            evals.needle_test(needle_chat, word_tokenize, 2048, cancel=cancel)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_blind_vote_reveal_flow(self):
        prompts = ["Say hi", "Say bye", "Count to three", "Name a colour"]
        labels = ["Qwen 8B", "Llama 3B", "Gemma 4B"]
        cid = evals.start_comparison(self.store, prompts, labels)
        with self.assertRaisesRegex(ValueError, "ready"):
            evals.vote(self.store, cid, 0, "A")
        words = {"Qwen 8B": "alpha", "Llama 3B": "bravo", "Gemma 4B": "charlie"}
        for label in labels:
            for index in range(len(prompts)):
                evals.record_output(self.store, cid, label, index, f"{words[label]} says #{index}",
                                    {"predicted_per_second": 10.0})
        view = evals.blinded(self.store, cid)
        dumped = json.dumps(view)
        for label in labels:
            self.assertNotIn(label, dumped)
        self.assertNotIn("predicted_per_second", dumped)
        self.assertTrue(view["complete"])
        self.assertEqual([o["slot"] for o in view["items"][0]["outputs"]], ["A", "B", "C"])
        by_word = {w: label for label, w in words.items()}
        shown = [[by_word[o["text"].split(" says")[0]] for o in item["outputs"]] for item in view["items"]]
        # derived from the id only: stable across reads, and matches the stored order
        self.assertEqual(shown, [evals._slot_order(labels, cid, i) for i in range(len(prompts))])
        self.assertEqual(shown, self.store.get("comparisons")[cid]["order"])
        self.assertEqual(evals.blinded(self.store, cid), view)

        # vote for whichever slot holds "Gemma 4B" on items 0 and 1, tie on 2, skip 3
        def slot_of(item, label):
            return next(o["slot"] for o in view["items"][item]["outputs"] if o["text"].startswith(words[label]))
        evals.vote(self.store, cid, 0, slot_of(0, "Llama 3B"))
        evals.vote(self.store, cid, 0, slot_of(0, "Gemma 4B"))  # re-vote replaces
        evals.vote(self.store, cid, 1, slot_of(1, "Gemma 4B"))
        result = evals.vote(self.store, cid, 2, "tie")
        self.assertEqual(result["voted"], 3)
        self.assertFalse(result["complete"])
        self.assertNotIn("Gemma", json.dumps(result))
        for bad in [(9, "A"), (0, "D"), (0, "a"), ("0", "A"), (True, "A")]:
            with self.assertRaises(ValueError, msg=bad):
                evals.vote(self.store, cid, *bad)

        revealed = evals.reveal(self.store, cid)
        self.assertEqual(revealed["tallies"], {"Qwen 8B": 0, "Llama 3B": 0, "Gemma 4B": 2})
        self.assertEqual((revealed["ties"], revealed["unvoted"], revealed["overall"]), (1, 1, "Gemma 4B"))
        self.assertEqual(revealed["mapping"][1]["winner"], "Gemma 4B")
        self.assertEqual(set(revealed["mapping"][0]["slots"].values()), set(labels))
        self.assertEqual(revealed["speed"]["Qwen 8B"]["tokens_per_second"], 10.0)
        self.assertTrue(evals.blinded(self.store, cid)["revealed"])
        with self.assertRaisesRegex(ValueError, "closed"):
            evals.vote(self.store, cid, 3, "A")
        self.assertEqual(evals.reveal(self.store, cid)["tallies"], revealed["tallies"])

    def test_shuffle_varies_between_items_and_comparisons(self):
        labels = ["a", "b", "c"]
        orders = {tuple(evals._slot_order(labels, f"id{n}", i)) for n in range(5) for i in range(5)}
        self.assertGreater(len(orders), 3)
        self.assertEqual(evals._slot_order(labels, "x", 1), evals._slot_order(labels, "x", 1))

    def test_input_validation(self):
        for prompts, labels in [([], ["a", "b"]), (["p"] * 6, ["a", "b"]), (["x" * 4001], ["a", "b"]),
                                ([""], ["a", "b"]), (["p"], ["a"]), (["p"], ["a", "a"]), (["p"], ["a", "b", "c", "d"])]:
            with self.assertRaises(ValueError):
                evals.start_comparison(self.store, prompts, labels)
        with self.assertRaisesRegex(ValueError, "not found"):
            evals.blinded(self.store, "missing")
        cid = evals.start_comparison(self.store, ["p"], ["a", "b"])
        with self.assertRaises(ValueError):
            evals.record_output(self.store, cid, "zzz", 0, "t", {})
        with self.assertRaises(ValueError):
            evals.record_output(self.store, cid, "a", 3, "t", {})

    def test_old_comparisons_are_pruned(self):
        ids = [evals.start_comparison(self.store, ["p"], ["a", "b"]) for _ in range(evals.KEEP_COMPARISONS + 3)]
        kept = self.store.get("comparisons")
        self.assertEqual(len(kept), evals.KEEP_COMPARISONS)
        self.assertIn(ids[-1], kept)

    def test_run_comparison_runs_models_one_at_a_time(self):
        events, active = [], []

        def factory(label, fail=False):
            @contextlib.contextmanager
            def runner():
                if active:
                    raise AssertionError("two models loaded at once")
                active.append(label)
                events.append(("start", label))
                try:
                    if fail:
                        raise ValueError("Not enough memory to load qwen-secret-name")

                    def chat(messages, max_tokens):
                        return {"text": f"<think>private</think>{label}: {messages[0]['content']}",
                                "timings": {"predicted_per_second": 5.0}, "finish_reason": "stop"}
                    yield chat
                finally:
                    active.pop()
                    events.append(("stop", label))
            return runner

        updates = []
        cid = evals.run_comparison(self.store, ["one", "two"], {"m1": factory("m1"), "m2": factory("m2", fail=True),
                                                                "m3": factory("m3")},
                                   progress=updates.append)
        self.assertEqual([e for e, _ in events], ["start", "stop"] * 3)
        record = self.store.get("comparisons")[cid]
        self.assertEqual(record["state"], "ready")
        self.assertEqual(record["outputs"]["m1"][1]["text"], "m1: two")
        self.assertEqual(record["outputs"]["m1"][0]["timings"]["predicted_per_second"], 5.0)
        self.assertIn("seconds", record["outputs"]["m1"][0]["timings"])
        self.assertIn("memory", record["outputs"]["m2"][0]["error"])
        view = json.dumps(evals.blinded(self.store, cid))
        self.assertNotIn("private", view)
        self.assertNotIn("qwen-secret-name", view)
        self.assertIn("failed to run", view)
        self.assertEqual(updates[-1]["done"], updates[-1]["total"])
        self.assertTrue(all("m1" not in u["message"] for u in updates))
        self.assertEqual(evals.reveal(self.store, cid)["speed"]["m2"]["failed"], 2)

    def test_run_comparison_accepts_server_objects_and_cancel(self):
        class FakeServer:
            def __init__(self):
                self.stopped = False

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.stopped = True

            def chat(self, messages, max_tokens=256, temperature=0.0, seed=1):
                return {"text": "ok"}

        servers = []

        def make():
            servers.append(FakeServer())
            return servers[-1]

        cid = evals.run_comparison(self.store, ["p"], {"a": make, "b": lambda: (lambda m, max_tokens: "plain")})
        self.assertTrue(servers[0].stopped)
        self.assertEqual([o["text"] for o in sorted(evals.blinded(self.store, cid)["items"][0]["outputs"],
                                                    key=lambda o: o["text"])], ["ok", "plain"])

        cancel = threading.Event()

        def cancelling():
            cancel.set()
            return make()

        with self.assertRaises(Cancelled):
            evals.run_comparison(self.store, ["p", "q"], {"a": cancelling, "b": make}, cancel=cancel)
        self.assertTrue(servers[-1].stopped)
        self.assertEqual(len(servers), 2)  # the second model was never started
        states = [c["state"] for c in self.store.get("comparisons").values()]
        self.assertIn("cancelled", states)


if __name__ == "__main__":
    unittest.main()
