"""Test a model on this computer: does it answer, how fast is it, how much memory does it really use.

Every number here is measured on the user's machine, never invented. When a measurement is
missing it stays None. Servers and benchmark processes started here are always stopped,
including when the user cancels or something fails.
"""
import inspect
import json
import math
import os
import re
import secrets
import subprocess
import tempfile
import threading
import time

from .domain import Cancelled, check_cancel, now
from . import launch

SMOKE_QUESTION = "What is 17 + 25? Reply with just the number."
SMOKE_ANSWER = re.compile(r"(?<!\d)42(?!\d)")
SMOKE_MAX_TOKENS = 200
NO_THINKING = {"enable_thinking": False, "reasoning_effort": "low"}
# Raw chat-template markers that should never reach the user; seeing one means the template is broken.
TEMPLATE_MARKERS = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>|<\|start_header_id\|>|<\|end_header_id\|>|"
                              r"<start_of_turn>|<end_of_turn>|\[/?INST\]|<\|endoftext\|>|<\|assistant\|>|<\|user\|>|"
                              r"<\|system\|>|<\|end\|>|<\|channel\|>|<\|message\|>|<\|begin_of_text\|>|</?s>")
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.S | re.I)
# Only llama.cpp's own words count as "out of memory" (llama_server.FAILURES turns them into "Not enough memory …").
# A bare "memory" or "alloc" also appears in other reasons, e.g. "most likely because memory ran out" for a killed
# process, which is a guess and is passed on as one.
OOM_REASON = re.compile(r"^Not enough memory|\bout of memory\b", re.I)

BENCH_PROMPT = 512
BENCH_GEN = 128
BENCH_REPETITIONS = 2
TTFT_PROMPT_TOKENS = 1500
# llama-bench re-reads the whole depth, untimed, before every repetition of both tests. Past this many tokens that
# costs more time than it tells (a 128K context on a CPU would take hours), so the record says the depth it used.
MAX_BENCH_DEPTH = 32768


def _now():
    return time.monotonic()


def _progress(progress, stage, done, total, message, **extra):
    if progress:
        progress({"stage": stage, "done": done, "total": total, "message": message, **extra})


def _loading(progress, stage, done, total):
    """llama-server reports loading as seconds out of its timeout; keep this test's own stage and step counts."""
    if not progress:
        return None
    return lambda event: progress({"stage": stage, "done": done, "total": total,
                                   "message": (event or {}).get("message") or "Loading the model…"})


def _server_class(server_factory):
    if server_factory is not None:
        return server_factory
    from .llama_server import LlamaServer
    return LlamaServer


def _peak_class(peak_factory):
    if peak_factory is not None:
        return peak_factory
    from .llama_server import PeakMemory
    return PeakMemory


def _chat(server, messages, max_tokens, **kwargs):
    """Ask without hidden reasoning. Chat templates that don't know these switches simply ignore them.
    cache_prompt=False (LlamaServer's default, sent explicitly) makes every call read the whole prompt again."""
    return server.chat(messages, max_tokens=max_tokens, cache_prompt=False,
                       extra={"chat_template_kwargs": dict(NO_THINKING)}, **kwargs)


def _stop(server):
    if server is None:
        return
    try:
        server.stop()
    except Exception:  # cleanup must never hide the real outcome
        pass


def _tail(server):
    try:
        return server.log_tail() if server is not None else ""
    except Exception:
        return ""


def _start_message(reason):
    text = (reason or "").strip()
    if OOM_REASON.search(text):
        return "The model ran out of memory while loading. Try fewer GPU layers, a shorter context or a smaller file."
    if not text:
        return "The model did not start. Check that llama.cpp is installed and the model file is complete."
    return f"The model did not start: {text}"


def strip_thinking(text):
    """Remove hidden reasoning. An unclosed <think> means the model never finished thinking."""
    text = THINK_BLOCK.sub("", text or "")
    opened = re.search(r"<think>", text, re.I)
    return (text[:opened.start()] if opened else text).strip()


def reply_checks(raw_text):
    """Sanity checks on one reply. Returns (answer_text, checks)."""
    raw_text = raw_text if isinstance(raw_text, str) else ""
    text = strip_thinking(raw_text)
    checks = []
    checks.append({"name": "not_empty", "ok": bool(text),
                   "detail": "The model wrote an answer." if text else "The reply was empty (or only hidden thinking)."})
    try:
        text.encode("utf-8")
        utf8 = "�" not in text
    except UnicodeEncodeError:
        utf8 = False
    checks.append({"name": "valid_utf8", "ok": utf8,
                   "detail": "The text is readable." if utf8 else "The reply contains broken characters."})
    words = text.split()
    repeated = (len(words) >= 4 and len(set(words)) == 1) or bool(re.search(r"(.)\1{15,}", text)) or \
        bool(re.search(r"(\S{2,}?)\1{7,}", text))
    checks.append({"name": "not_repeating", "ok": not repeated,
                   "detail": "No stuck repetition." if not repeated else "The model repeats the same thing over and over."})
    leak = TEMPLATE_MARKERS.search(raw_text)
    checks.append({"name": "no_template_leak", "ok": leak is None,
                   "detail": "No raw chat-format markers." if leak is None else
                   f"Raw chat-format text leaked into the reply ({leak.group(0)}). The chat template may be wrong for this file."})
    correct = bool(SMOKE_ANSWER.search(text))
    checks.append({"name": "correct_answer", "ok": correct,
                   "detail": "Answered 17 + 25 = 42 correctly." if correct else "Did not answer 42 to 17 + 25."})
    return text, checks


def smoke_test(server_command, config, progress=None, cancel=None, server_factory=None, timeout=300):
    """Start the model, ask one question with a known answer, and check the reply looks sane."""
    config = launch.normalize(config)
    result = {"ok": False, "stage_failed": None, "load_seconds": None, "reply": None, "checks": [], "message": "", "log_tail": ""}
    server = None
    try:
        check_cancel(cancel)
        _progress(progress, "smoke_start", 0, 2, "Loading the model…")
        server = _server_class(server_factory)(server_command, config)
        started = _now()
        try:
            server.start(timeout=timeout, progress=_loading(progress, "smoke_start", 0, 2), cancel=cancel)
        except Cancelled:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            reason = None
            try:
                reason = server.failure_reason()
            except Exception:
                pass
            result.update(stage_failed="start", message=_start_message(reason or str(error)), log_tail=_tail(server))
            return result
        result["load_seconds"] = round(_now() - started, 3)
        check_cancel(cancel)
        _progress(progress, "smoke_reply", 1, 2, "Asking a simple question…")
        try:
            reply = _chat(server, [{"role": "user", "content": SMOKE_QUESTION}], SMOKE_MAX_TOKENS, temperature=0.0)
        except Cancelled:
            raise
        except (ValueError, OSError, RuntimeError) as error:
            result.update(stage_failed="reply", message=f"The model loaded but did not answer: {error}", log_tail=_tail(server))
            return result
        reply = reply or {}
        text, checks = reply_checks(reply.get("text"))
        result.update(reply=text, checks=checks, log_tail=_tail(server))
        failed = [c for c in checks if not c["ok"]]
        if failed and not text and reply.get("reasoning"):
            # llama-server puts hidden reasoning in its own field; the model thought but never answered.
            result.update(stage_failed="reply", message="The model loaded but spent its whole answer thinking and never "
                          "replied, even when asked not to think. Its chat template may not support turning thinking off.")
        elif failed:
            result.update(stage_failed="reply", message=f"The model loaded but its answer looks wrong. {failed[0]['detail']}")
        else:
            result.update(ok=True, message=f"The model loaded in {result['load_seconds']:.1f} s and answered correctly.")
        _progress(progress, "smoke_done", 2, 2, result["message"])
        return result
    finally:
        _stop(server)


def bench_plan(context):
    """Reading/writing test sizes at the user's conversation length.

    Rule: depth = context - prompt - gen (the contract's -p 512 -n 128 -d <context-640>), so the
    test ends exactly at the configured context. Short contexts shrink prompt/gen to fit, and very long
    contexts are measured at MAX_BENCH_DEPTH (the record's `depth` says so).
    """
    n_prompt, n_gen = BENCH_PROMPT, BENCH_GEN
    if context < n_prompt + n_gen + 128:
        n_prompt, n_gen = max(32, context // 2), max(16, context // 4)
    depth = min(max(0, context - n_prompt - n_gen), MAX_BENCH_DEPTH)
    return {"n_prompt": n_prompt, "n_gen": n_gen, "depth": depth}


def _flash_value(value):
    if value in (True, 1, "1", "on", "true"):
        return "on"
    if value in (False, 0, "0", "off", "false"):
        return "off"
    return "auto"


def bench_args(config, n_prompt, n_gen, depth, repetitions=BENCH_REPETITIONS):
    """llama-bench arguments (after the executable) that mirror llama-server's launch settings."""
    c = launch.normalize(config)
    if not c["model_path"]:
        raise ValueError("A downloaded model file is required")
    # -v: without it llama-bench hides llama.cpp's own log, so an out-of-memory load reads only "failed to load model".
    args = ["-m", c["model_path"], "-p", str(n_prompt), "-n", str(n_gen), "-d", str(depth),
            "-r", str(repetitions), "-o", "json", "-v",
            "-ngl", str(launch.runtime_gpu_layers(c["gpu_layers"], c["total_layers"])),
            "-ctk", c["cache_type_k"], "-ctv", c["cache_type_v"]]
    for flag, name in [("-t", "threads"), ("-b", "batch"), ("-ub", "ubatch")]:
        if c[name]:
            args += [flag, str(c[name])]
    if c["flash_attn"] != "auto":
        args += ["-fa", c["flash_attn"]]
    if c["n_cpu_moe"]:
        args += ["-ncmoe", str(c["n_cpu_moe"])]
    device = c["device"] or ("none" if c["gpu_layers"] == 0 and c["gpu_backend"] not in {None, "cpu"} else None)
    if device:
        args += ["-dev", device]
    if not c["mmap"]:
        args += ["-lm", "none"]  # llama-server's --no-mmap; -mmp 0 is deprecated in favour of --load-mode
    return args


def parse_bench_json(text):
    """Rows from llama-bench `-o json` or jsonl, tolerating log lines and an array left open by a crash."""
    from .tuner import parse_rows
    return parse_rows(text or "")


def _speed(row):
    try:
        value = float(row["avg_ts"])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def check_bench_settings(row, config):
    """Plain list of differences between what we asked llama-bench for and what it reports it used."""
    c = launch.normalize(config)
    expected = {"n_gpu_layers": launch.runtime_gpu_layers(c["gpu_layers"], c["total_layers"]),
                "type_k": c["cache_type_k"], "type_v": c["cache_type_v"]}
    for field, name in [("n_threads", "threads"), ("n_batch", "batch"), ("n_ubatch", "ubatch")]:
        if c[name]:
            expected[field] = c[name]
    if c["n_cpu_moe"] and "n_cpu_moe" in row:
        expected["n_cpu_moe"] = c["n_cpu_moe"]
    problems = [f"{field} is {row.get(field)!r}, expected {value!r}" for field, value in expected.items() if row.get(field) != value]
    if c["flash_attn"] != "auto" and "flash_attn" in row and _flash_value(row["flash_attn"]) != c["flash_attn"]:
        problems.append(f"flash_attn is {row['flash_attn']!r}, expected {c['flash_attn']!r}")
    return problems


def _kill_tree(process):
    try:
        import psutil
        children = psutil.Process(process.pid).children(recursive=True)
    except Exception:
        children = []
    for child in children:
        try:
            child.kill()
        except Exception:
            pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=10)
    except Exception:
        pass


def run_process(argv, timeout, env=None, cancel=None, poll=0.2):
    """Run a child with a deadline and cancel support; stdout/stderr go to temp files to avoid pipe stalls."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            process = subprocess.Popen(argv, stdout=out, stderr=err, stdin=subprocess.DEVNULL, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        except OSError as error:
            raise ValueError(f"llama-bench could not be started: {error}") from None
        deadline = _now() + timeout
        try:
            while process.poll() is None:
                check_cancel(cancel)
                if _now() > deadline:
                    raise ValueError(f"llama-bench took longer than {int(timeout)} s and was stopped.")
                time.sleep(poll)
        finally:
            if process.poll() is None:
                _kill_tree(process)
        out.seek(0)
        err.seek(0)
        return process.returncode, out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")


def run_bench(bench_command, config, plan, repetitions=BENCH_REPETITIONS, timeout=1800, cancel=None):
    """Reading (pp) and writing (tg) speed at depth; refuses results that used different settings."""
    argv = (list(bench_command) if isinstance(bench_command, (list, tuple)) else [bench_command]) + \
        bench_args(config, plan["n_prompt"], plan["n_gen"], plan["depth"], repetitions)
    code, stdout, stderr = run_process(argv, timeout, env=launch.server_env(config), cancel=cancel)
    if code:
        from .tuner import bench_failure
        reason = bench_failure(stderr, stdout, code)
        if reason.startswith("Ran out of memory"):
            raise ValueError("The speed test ran out of memory. Try fewer GPU layers, a shorter context or a smaller file.")
        if reason.startswith("This llama.cpp version"):
            raise ValueError("This llama.cpp version does not support the speed test settings. Update llama.cpp and try again.")
        raise ValueError(f"The speed test failed. {reason}")
    rows = parse_bench_json(stdout)
    pp = next((r for r in rows if r.get("n_prompt") == plan["n_prompt"] and r.get("n_gen") == 0
               and r.get("n_depth", 0) == plan["depth"]), None)
    tg = next((r for r in rows if r.get("n_prompt") == 0 and r.get("n_gen") == plan["n_gen"]
               and r.get("n_depth", 0) == plan["depth"]), None)
    if tg is None or _speed(tg) is None:
        raise ValueError("The speed test output was not recognised. Update llama.cpp (JSON output and --n-depth are needed).")
    for row in (pp, tg):
        if row is not None:
            problems = check_bench_settings(row, config)
            if problems:
                raise ValueError("The speed test ran with different settings than requested (" + "; ".join(problems) +
                                 "). The result was not saved.")
    return {"pp_tps": _speed(pp) if pp else None, "tps": _speed(tg), "pp_row": pp, "tg_row": tg}


def filler_prompt(tokens=TTFT_PROMPT_TOKENS):
    """Deterministic, varied text roughly `tokens` long (about 0.75 words per token)."""
    subjects = ["The river", "A small bakery", "The old library", "Our project team", "The mountain road", "A quiet garden"]
    verbs = ["opened", "changed", "welcomed", "measured", "described", "followed"]
    objects = ["new visitors in spring", "a careful plan for winter", "several useful tools", "the weekly report",
               "an unusual pattern of rain", "stories from the harbour"]
    sentences, words, i = [], 0, 0
    while words < tokens * 0.75:
        sentence = f"{subjects[i % 6]} {verbs[(i // 6) % 6]} {objects[(i // 36 + i) % 6]} on day {i + 1}."
        sentences.append(sentence)
        words += len(sentence.split())
        i += 1
    return " ".join(sentences) + "\n\nIn one short sentence, what is this text about?"


def fitted_prompt(server, target):
    """Filler text of at most `target` tokens, counted by the model's own tokenizer (word-based guesses can overflow
    a short context: a small vocabulary can spend 3+ tokens per word). Returns (prompt, tokens); tokens is None when
    the server can't count."""
    size = target
    for _ in range(8):
        prompt = filler_prompt(size)
        try:
            count = len(server.tokenize(prompt))
        except (ValueError, OSError, AttributeError):
            return prompt, None
        if count <= target or size <= 16:
            return prompt, count
        # Shrink in proportion, and always by at least one sentence's worth, so this ends even when counts are lumpy.
        size = max(16, min(size - 16, int(size * target / count) - 8))
    return prompt, count


def ttft_target(config):
    """First-word prompt size in tokens: TTFT_PROMPT_TOKENS, or less so it fits the context with room for the chat
    template and the reply. `context` is per user: launch passes -c context*parallel, so each slot gets all of it."""
    context = config["context"]
    return max(16, min(TTFT_PROMPT_TOKENS, context - 256 if context >= 512 else context // 2))


class _PeakWatcher:
    """Starts the peak-memory sampler as soon as the server has a process id, while start() is still loading."""

    def __init__(self, server, peak_class, gpu_backend):
        self.server, self.peak_class, self.gpu_backend = server, peak_class, gpu_backend
        self.monitor = None
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, name="peak-watch", daemon=True)
        self._thread.start()

    def _run(self):
        while not self._done.is_set():
            try:
                pid = self.server.pid
            except Exception:
                pid = None
            if pid:
                monitor = self.peak_class(pid, gpu_backend=self.gpu_backend)
                monitor.start()
                self.monitor = monitor
                return
            self._done.wait(0.02)

    def stop(self):
        self._done.set()
        self._thread.join(timeout=5)
        if self.monitor is None:
            return {}
        try:
            return self.monitor.stop() or {}
        except Exception:
            return {}


BACKEND_NAMES = (("cuda", "cuda"), ("rocm", "rocm"), ("hip", "rocm"), ("vulkan", "vulkan"), ("metal", "metal"),
                 ("mtl", "metal"), ("cpu", "cpu"))


def runtime_info(row, config):
    """§2.3 `runtime`: llama.cpp build tag (b<N>, the release name) and the backend that did the work, in the
    lower-case names the rest of the app uses. llama-bench reports e.g. "CPU" or "CUDA,CPU"."""
    number = row.get("build_number")
    version = f"b{number}" if type(number) is int and number > 0 else (row.get("build_commit") or None)
    reported = [part.strip().lower() for part in str(row.get("backends") or "").split(",") if part.strip()]
    names = [next((name for key, name in BACKEND_NAMES if part.startswith(key)), "unknown") for part in reported]
    gpu = next((n for n in names if n not in {"cpu", "unknown"}), None)
    if config["gpu_layers"] and gpu:
        backend = gpu
    elif not config["gpu_layers"] or (names and not gpu and "cpu" in names):
        backend = "cpu"
    else:
        backend = config["gpu_backend"] or ("unknown" if names else None)
    return {"version": version, "backend": backend}


def estimate_memory(variant, config, hardware=None, allocations=None):
    """The app's own memory estimate for this launch config, or None when it cannot be computed."""
    if allocations is None:
        from .engine import allocations
    c = launch.normalize(config)
    kwargs = {"kv_cache_type": c["cache_type_k"] if c["cache_type_k"] == c["cache_type_v"] else "f16",
              "n_cpu_moe": c["n_cpu_moe"], "unified": bool((hardware or {}).get("unified_memory"))}
    try:
        parameters = inspect.signature(allocations).parameters
        accepted = {k: v for k, v in kwargs.items() if k in parameters}
    except (TypeError, ValueError):
        accepted = {}
    try:
        memory = allocations(variant, c["context"], c["parallel"], c["gpu_layers"], **accepted)
    except (ValueError, TypeError):
        return None, None
    return memory.get("ram"), memory.get("vram")


def memory_verdict(estimated_ram, estimated_vram, peak_ram, peak_vram):
    """Compare real peak memory with the estimate. Only pairs where both numbers are known count."""
    pairs = [(label, est, peak) for label, est, peak in [("RAM", estimated_ram, peak_ram), ("GPU memory", estimated_vram, peak_vram)]
             if est and peak is not None]
    if not pairs:
        return None, "Memory use could not be compared with the estimate on this computer."
    over = [(label, (peak - est) / est * 100) for label, est, peak in pairs if peak > est]
    if not over:
        return True, "Real memory use stayed within the app's estimate."
    return False, "Real memory use was over the estimate: " + ", ".join(f"{label} by {pct:.0f}%" for label, pct in over) + \
        ". Leave extra room or use a shorter context."


def speed_test(bench_command, server_command, variant, config, hardware, progress=None, cancel=None,
               server_factory=None, peak_factory=None, allocations=None, repetitions=BENCH_REPETITIONS,
               bench_timeout=1800, server_timeout=300):
    """Reading speed, writing speed and first-word delay at the user's context, plus a memory check."""
    config = launch.normalize(config)
    if not config["model_path"]:
        raise ValueError("A downloaded model file is required")
    if config["total_layers"] is None:
        config["total_layers"] = variant.layers
    plan = bench_plan(config["context"])
    _progress(progress, "bench", 0, 3, "Measuring reading and writing speed…", **plan)
    bench = run_bench(bench_command, config, plan, repetitions=repetitions, timeout=bench_timeout, cancel=cancel)
    check_cancel(cancel)

    _progress(progress, "first_word", 1, 3, "Measuring the delay before the first word…")
    server = watcher = None
    peak = {"peak_ram_bytes": None, "peak_vram_bytes": None}
    ttft = timings = prompt_tokens = None
    try:
        server = _server_class(server_factory)(server_command, config)
        # Memory is watched from the moment the process exists, so the peak includes loading the model.
        watcher = _PeakWatcher(server, _peak_class(peak_factory), config["gpu_backend"])
        try:
            server.start(timeout=server_timeout, progress=_loading(progress, "first_word", 1, 3), cancel=cancel)
        except (ValueError, OSError, RuntimeError) as error:
            reason = None
            try:
                reason = server.failure_reason()
            except Exception:
                pass
            raise ValueError(_start_message(reason or str(error))) from None
        check_cancel(cancel)
        # A tiny warm-up request so one-off start-up costs don't land in the timed request.
        _chat(server, [{"role": "user", "content": "Hi"}], 1, temperature=0.0)
        prompt, prompt_tokens = fitted_prompt(server, ttft_target(config))
        check_cancel(cancel)
        started = _now()
        reply = _chat(server, [{"role": "user", "content": prompt}], 1, temperature=0.0)
        wall = _now() - started
        timings = (reply or {}).get("timings") or {}
        prompt_ms = timings.get("prompt_ms")
        if isinstance(timings.get("prompt_n"), int) and timings["prompt_n"] > 0:
            prompt_tokens = timings["prompt_n"]  # what the server really read, chat template included
        # Wall clock includes the HTTP round trip, which is what the user feels; fall back to the server's own timer.
        ttft = round(wall, 3) if wall > 0 else (round(prompt_ms / 1000, 3) if isinstance(prompt_ms, (int, float)) else None)
        check_cancel(cancel)
        # A short generation so the peak includes a working KV cache, not just the loaded weights.
        _chat(server, [{"role": "user", "content": "Write two sentences about the sea."}], 64, temperature=0.0)
    finally:
        if watcher is not None:
            peak.update(watcher.stop())
        _stop(server)

    _progress(progress, "memory", 2, 3, "Comparing memory use with the estimate…")
    estimated_ram, estimated_vram = estimate_memory(variant, config, hardware, allocations)
    within, note = memory_verdict(estimated_ram, estimated_vram, peak.get("peak_ram_bytes"), peak.get("peak_vram_bytes"))
    tg = bench["tg_row"]
    build = tg.get("build_commit")
    record = {"variant_id": variant.id, "sha256": variant.sha256, "fingerprint": (hardware or {}).get("fingerprint"),
              "timestamp": now(),
              "context": config["context"], "users": config["parallel"], "gpu_layers": config["gpu_layers"],
              "gpu_uuid": config["gpu_uuid"] if config["gpu_layers"] else None,
              "threads": config["threads"] if config["threads"] else tg.get("n_threads"),
              "tps": bench["tps"], "runtime_build": build,
              "raw": {"bench": [r for r in (bench["pp_row"], tg) if r], "server_timings": timings, "plan": plan},
              "note": "Measured on this computer with llama-bench (reading and writing speed with the given depth of "
                      "conversation already in memory) and one llama-server run (first-word delay for a fresh prompt of "
                      "ttft_prompt_tokens tokens, and peak memory including loading). Other programs running can change speed.",
              "kind": "speed_test", "pp_tps": bench["pp_tps"], "ttft_s": ttft, "ttft_prompt_tokens": prompt_tokens,
              "depth": plan["depth"],
              "peak_ram_bytes": peak.get("peak_ram_bytes"), "peak_vram_bytes": peak.get("peak_vram_bytes"),
              "estimated_ram_bytes": estimated_ram, "estimated_vram_bytes": estimated_vram,
              "settings": {k: config[k] for k in ["flash_attn", "cache_type_k", "cache_type_v", "batch", "ubatch", "n_cpu_moe"]},
              "runtime": runtime_info(tg, config),
              "id": secrets.token_hex(6)}
    _progress(progress, "speed_done", 3, 3, "Speed test finished.")
    return {"measurement": record,
            "summary": {"pp_tps": bench["pp_tps"], "tps": bench["tps"], "ttft_s": ttft, "ttft_prompt_tokens": prompt_tokens,
                        "depth": plan["depth"]},
            "memory": {"estimated_ram_bytes": estimated_ram, "estimated_vram_bytes": estimated_vram,
                       "peak_ram_bytes": peak.get("peak_ram_bytes"), "peak_vram_bytes": peak.get("peak_vram_bytes"),
                       "within_estimate": within, "note": note}}


def _command(commands, *names):
    for name in names:
        if commands.get(name):
            return commands[name]
    raise ValueError("llama.cpp is not installed yet. Click Install the engine, or run: llm-config runtime install")


def verdict(smoke, speed, speed_error=None, min_tps=None):
    """works / works_slowly / failed plus one plain sentence."""
    if smoke is not None and not smoke.get("ok"):
        return "failed", smoke.get("message") or "The model did not pass the basic test."
    if speed is None:
        if speed_error:
            if smoke is None:
                return "failed", speed_error
            return "works", f"The model works, but the speed test did not finish: {speed_error}"
        return "works", "The model loads and answers correctly. Run the speed test to see how fast it is."
    summary, memory = speed["summary"], speed["memory"]
    speed_text = (f"writes about {summary['tps']:.1f} tokens per second (a token is about three quarters of a word) "
                  f"with {summary['depth']:,} tokens of conversation already in memory.")
    if summary.get("ttft_s") is not None:
        size = f"a {summary['ttft_prompt_tokens']:,}-token" if summary.get("ttft_prompt_tokens") else "a"
        speed_text += f" It starts answering {size} new message after {summary['ttft_s']:.1f} s."
    extra = " " + memory["note"] if memory.get("within_estimate") is False else ""
    if min_tps and summary["tps"] < min_tps:
        return "works_slowly", f"It works but is slower than your target of {min_tps:g}: it {speed_text}{extra}"
    return "works", f"It works: it {speed_text}{extra}"


def run_tests(store, variant, config, commands, kind="full", hardware=None, progress=None, cancel=None, min_tps=None,
              server_factory=None, peak_factory=None, allocations=None):
    """Run smoke and/or speed tests and save speed measurements for the recommendation engine."""
    if kind not in {"full", "smoke", "speed"}:
        raise ValueError("kind must be full, smoke or speed")
    commands = commands or {}
    server_command = _command(commands, "llama-server", "server")
    smoke = speed = speed_error = None
    if kind in {"full", "smoke"}:
        smoke = smoke_test(server_command, config, progress=progress, cancel=cancel, server_factory=server_factory)
    if kind in {"full", "speed"} and (smoke is None or smoke["ok"]):
        if hardware is None:
            from .hardware import scan
            hardware = scan(False)
        try:
            speed = speed_test(_command(commands, "llama-bench", "bench"), server_command, variant, config, hardware,
                               progress=progress, cancel=cancel, server_factory=server_factory,
                               peak_factory=peak_factory, allocations=allocations)
        except (ValueError, OSError) as error:
            speed_error = str(error)
        else:
            store.append("measurements", speed["measurement"])
    label, text = verdict(smoke, speed, speed_error, min_tps)
    result = {"smoke": smoke, "speed": speed, "verdict": label, "verdict_text": text}
    if speed_error:
        result["speed_error"] = speed_error
    return result
