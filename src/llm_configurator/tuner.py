"""Auto-tune llama.cpp settings on this computer within a time budget.

Every number comes from llama-bench runs on the user's own machine; nothing is guessed.
A setting is kept only when it beats the current best by more than measurement noise,
and the winner is re-measured with more repetitions before it is reported. Settings the
memory check calls unsafe are never run. Tuning never runs past the budget by more than
one trial: each bench process gets a deadline of the remaining budget plus one trial.
"""
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
import time

from . import launch
from .domain import GIB, Cancelled, check_cancel

GOALS = {"generation", "balanced", "prompt"}
# Balanced favours writing speed a little, because chat time is mostly spent writing.
BALANCED_WEIGHTS = (0.6, 0.4)  # (writing, reading) exponents of a weighted geometric mean
MIN_GAIN = 0.03  # a change must win by at least 3% (or by the measured noise, if larger)
MAX_PASSES = 3
CONFIRM_REPETITIONS = 5
DEFAULT_TIMEOUT = 900
BATCH_PAIRS = [(512, 512), (1024, 256), (2048, 512)]
MAX_BENCH_DEPTH = 32768  # same cap as testing.MAX_BENCH_DEPTH: deeper costs hours on a CPU
SETTINGS_KEYS = ["flash_attn", "cache_type_k", "cache_type_v", "batch", "ubatch", "n_cpu_moe"]  # as in speed tests
OOM_TEXT = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|ErrorOutOfDeviceMemory|"
                      r"unable to allocate|not enough memory|insufficient memory|std::bad_alloc|OutOfMemory",
                      re.IGNORECASE)

# launch-config name -> (llama-bench flag, llama-bench JSON field)
FLAGS = {"gpu_layers": ("-ngl", "n_gpu_layers"), "n_cpu_moe": ("-ncmoe", "n_cpu_moe"), "threads": ("-t", "n_threads"),
         "batch": ("-b", "n_batch"), "ubatch": ("-ub", "n_ubatch"), "flash_attn": ("-fa", "flash_attn"),
         "cache_type_k": ("-ctk", "type_k"), "cache_type_v": ("-ctv", "type_v")}


def _bench_value(name, value, config):
    if name == "gpu_layers":
        return str(launch.runtime_gpu_layers(value, config["total_layers"]))
    if name == "flash_attn":
        return value  # llama-bench --help documents on|off|auto (0/1 are also accepted); same words as llama-server
    return str(value)


def bench_args(config, n_prompt=512, n_gen=128, depth=0, repetitions=2, sweep=None):
    """Arguments after the llama-bench executable. `sweep` maps launch-config names to value lists,
    which llama-bench runs as every combination (comma lists) in one process."""
    c = launch.normalize(config)
    if not c["model_path"]:
        raise ValueError("A downloaded model file is required")
    sweep = {k: list(v) for k, v in (sweep or {}).items()}
    unknown = set(sweep) - set(FLAGS)
    if unknown:
        raise ValueError(f"These settings cannot be swept: {', '.join(sorted(unknown))}")
    if any(not values for values in sweep.values()):
        raise ValueError("Each swept setting needs at least one value")
    if "flash_attn" in sweep and "auto" in sweep["flash_attn"]:
        raise ValueError("Sweep flash_attn over on and off; auto is the llama.cpp default")
    for combination in itertools.product(*sweep.values()):  # every combination must pass the launch rules
        launch.normalize({**c, **dict(zip(sweep, combination))})
    for flag, value in [("n_prompt", n_prompt), ("n_gen", n_gen), ("depth", depth)]:
        if type(value) is not int or value < 0:
            raise ValueError(f"{flag} must be a non-negative integer")
    if type(repetitions) is not int or not 1 <= repetitions <= 100:
        raise ValueError("repetitions must be an integer between 1 and 100")
    args = ["-m", c["model_path"], "-p", str(n_prompt), "-n", str(n_gen)]
    if depth:
        args += ["-d", str(depth)]
    for name in FLAGS:
        flag = FLAGS[name][0]
        if name in sweep:
            args += [flag, ",".join(_bench_value(name, v, c) for v in sweep[name])]
        elif name == "gpu_layers":
            args += [flag, _bench_value(name, c[name], c)]
        elif name == "flash_attn":
            if c[name] != "auto":
                args += [flag, _bench_value(name, c[name], c)]
        elif name in {"cache_type_k", "cache_type_v"}:
            if c[name] != "f16":
                args += [flag, c[name]]
        elif c[name]:  # threads, batch, ubatch, n_cpu_moe: llama-bench defaults when unset
            args += [flag, str(c[name])]
    zero_layers = c["gpu_layers"] == 0 and "gpu_layers" not in sweep
    device = c["device"] or ("none" if zero_layers and c["gpu_backend"] not in {None, "cpu"} else None)
    if device:
        args += ["-dev", device]
    if not c["mmap"]:
        args += ["-lm", "none"]  # llama-server's --no-mmap; -mmp 0 is deprecated in favour of --load-mode
    # -v: without it llama-bench hides llama.cpp's own log, so an out-of-memory load reads only "failed to load model".
    return args + ["-r", str(repetitions), "-o", "json", "-v"]


def _kill_tree(process):
    try:
        import psutil
        children = psutil.Process(process.pid).children(recursive=True)
    except Exception:  # noqa: BLE001 - psutil missing or the process already gone
        children = []
    for child in children:
        try:
            child.kill()
        except Exception:  # noqa: BLE001
            pass
    try:
        process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


def run_process(argv, env=None, timeout=DEFAULT_TIMEOUT, cancel=None, poll=0.2):
    """Run llama-bench with a deadline; on cancel raises Cancelled. The process and anything it started are
    always killed on the way out (cancel, timeout, errors). Output goes to temp files so pipes never stall."""
    started = time.monotonic()
    options = {"stdout": None, "stderr": None, "stdin": subprocess.DEVNULL, "env": env}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True  # the terminal's Ctrl+C goes to the app, which then cleans up
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        options.update(stdout=out, stderr=err)
        try:
            process = subprocess.Popen(argv, **options)
        except OSError as error:
            raise ValueError(f"llama-bench could not be started ({error}). Reinstall the runtime and try again.") from None
        timed_out = False
        try:
            while process.poll() is None:
                check_cancel(cancel)
                if time.monotonic() - started > timeout:
                    timed_out = True
                    break
                time.sleep(poll)
        finally:
            if process.poll() is None:
                _kill_tree(process)
        out.seek(0)
        err.seek(0)
        return {"returncode": process.returncode, "stdout": out.read().decode("utf-8", "replace"),
                "stderr": err.read().decode("utf-8", "replace"), "seconds": time.monotonic() - started,
                "timed_out": timed_out}


def parse_rows(text):
    """llama-bench JSON rows; tolerates JSONL and output cut short by a crash."""
    try:
        rows = json.loads(text)
        return [r for r in (rows if isinstance(rows, list) else [rows]) if isinstance(r, dict)]
    except ValueError:
        pass
    rows, decoder, index = [], json.JSONDecoder(), 0
    while (index := text.find("{", index)) >= 0:
        try:
            row, end = decoder.raw_decode(text, index)
            if isinstance(row, dict):
                rows.append(row)
            index = end
        except ValueError:
            index += 1
    return rows


def _row_matches(row, name, value, config):
    field = FLAGS[name][1]
    if field not in row:
        return True  # older builds omit some fields; the remaining fields still identify the row
    got = row[field]
    if name == "flash_attn":
        return got in ({1, True} if value == "on" else {0, False})  # int in new builds, bool in old ones
    if name == "gpu_layers":
        return got == launch.runtime_gpu_layers(value, config["total_layers"])
    return got == value


def _speed(row):
    try:
        tps = float(row["avg_ts"])
        sd = float(row.get("stddev_ts") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    return (tps, sd / tps if math.isfinite(sd) and sd >= 0 else 0.0) if math.isfinite(tps) and tps > 0 else None


def score(tps, pp_tps, goal):
    """The number being maximised: writing speed, reading speed, or a weighted geometric mean."""
    if goal == "generation":
        return tps
    if goal == "prompt":
        return pp_tps
    if not tps or not pp_tps:
        return None
    return tps ** BALANCED_WEIGHTS[0] * pp_tps ** BALANCED_WEIGHTS[1]


_PATH = re.compile(r"'[^']*[/\\\\][^']*'|\"[^\"]*[/\\\\][^\"]*\"|(?:[A-Za-z]:)?[/\\\\][^\s'\"]+")


NO_CAUSE = ("llama.cpp could not load the model with these settings and gave no reason. They may need more memory "
            "than is free, or the file may be damaged.")
PROBABLY_MEMORY = ("Probably not enough memory: a lighter setting of the same model worked just before, and llama.cpp "
                   "gave no other reason for this one.")
# Lines that mention memory without being the failure (e.g. CUDA's "failed to allocate … pinned memory", which
# falls back and carries on), so they never decide the reason.
_NOT_FATAL = re.compile(r"warning|falling back|DEPRECATED", re.I)


def bench_failure(stderr, stdout="", returncode=None, timed_out=False):
    """One plain sentence for a failed llama-bench run, without file paths (they stay on this computer).

    Only stderr is read: with `-o json` stdout holds just the opening "[" of the array. A reason is given only
    when llama.cpp printed one (it does with `-v`); "failed to load model" alone says nothing about the cause."""
    if timed_out:
        return "The test ran out of time."
    # "DEPRECATED: … instead." is written without a newline, so it can be glued to the front of the next line.
    text = re.sub(r"(DEPRECATED:[^\n]*? instead\.)", r"\1\n", stderr or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fatal = [line for line in lines if not _NOT_FATAL.search(line)]
    if re.search(r"unknown argument|invalid parameter|error: invalid|invalid device", text, re.I):
        return "This llama.cpp version does not support one of these settings."
    missing = next((m.group(1) for line in fatal if (m := re.search(r"failed to open GGUF file '([^']*)'", line))), None)
    if missing:
        name = re.split(r"[\\/]", missing)[-1]
        return f"The model file is missing or cannot be read ({name})."
    cause = next((m.group(1).strip() for line in reversed(fatal)
                  if (m := re.search(r"error loading model: (.+)", line))), None)
    if cause and OOM_TEXT.search(cause):
        return "Ran out of memory with these settings."
    if cause:
        return f"llama.cpp could not load the model: {_PATH.sub('<file>', cause)[:200]}"
    if any(OOM_TEXT.search(line) for line in fatal):
        return "Ran out of memory with these settings."
    if re.search(r"quantized V cache requires flash.?attn|V cache quantization requires flash.?attn", text, re.I):
        return "A compressed notepad (KV cache) needs flash attention turned on."
    if "failed to load model" in text or "failed to create context" in text:
        return NO_CAUSE
    if returncode in (-9, 137):
        return "The system stopped llama-bench, possibly because memory ran out (llama.cpp gave no reason)."
    tail = next((line for line in reversed(fatal) if "error" in line.lower()), fatal[-1] if fatal else "no error output")
    return f"llama-bench stopped with an error ({returncode}): {_PATH.sub('<file>', tail)[:200]}"


def _plain_failure(result):
    return bench_failure(result.get("stderr"), result.get("stdout"), result.get("returncode"), result.get("timed_out"))


def default_memory_check(variant, hardware=None):
    """Conservative check with engine.allocations against free memory right now."""
    snapshot = {}

    def check(config):
        from . import engine  # engine ws may extend allocations; import late
        if "hw" not in snapshot:
            try:
                from .hardware import scan
                snapshot["hw"] = scan(False)
            except Exception:  # noqa: BLE001 - fall back to the hardware the caller gave us
                snapshot["hw"] = hardware or {}
        hw = snapshot["hw"]
        kv = "f16" if "f16" in {config["cache_type_k"], config["cache_type_v"]} else \
            ("q8_0" if "q8_0" in {config["cache_type_k"], config["cache_type_v"]} else "q4_0")
        try:
            use = engine.allocations(variant, config["context"], config["parallel"], config["gpu_layers"],
                                     kv_cache_type=kv, n_cpu_moe=config["n_cpu_moe"],
                                     unified=bool(hw.get("unified_memory")))
        except TypeError:  # base-branch engine without the v0.4 keyword arguments
            use = engine.allocations(variant, config["context"], config["parallel"], config["gpu_layers"])
        if use["ram"] > (hw.get("ram_available") or 0) - 1 * GIB:
            return False
        if config["gpu_layers"] and use.get("vram"):
            gpus = hw.get("gpus") or []
            gpu = next((g for g in gpus if config["gpu_uuid"] and g.get("uuid") == config["gpu_uuid"]), None) or \
                (gpus[0] if gpus else None)
            if gpu is None or gpu.get("available") is None or use["vram"] > gpu["available"] - 0.5 * GIB:
                return False
        return True
    return check


def _percent(new, old):
    return abs(new / old - 1) * 100 if old else 0


def _describe(old, new):
    """Plain phrases for what changed between two configs."""
    parts = []
    if old["gpu_layers"] != new["gpu_layers"]:
        parts.append(f"putting {new['gpu_layers']} of {new['total_layers']} layers on the graphics card "
                     f"instead of {old['gpu_layers']}")
    if old["n_cpu_moe"] != new["n_cpu_moe"]:
        parts.append(f"keeping the expert weights of {new['n_cpu_moe']} layers in system memory instead of {old['n_cpu_moe']}")
    if old["threads"] != new["threads"]:
        parts.append(f"using {new['threads']} CPU threads instead of {old['threads'] or 'the default'}")
    if old["flash_attn"] != new["flash_attn"]:
        parts.append(f"turning flash attention (a faster way to do the attention maths) {new['flash_attn']}")
    if (old["batch"], old["ubatch"]) != (new["batch"], new["ubatch"]):
        was = f"{old['batch']}/{old['ubatch']}" if old["batch"] and old["ubatch"] else "the default size"
        parts.append(f"reading the prompt in chunks of {new['batch']}/{new['ubatch']} tokens instead of {was}")
    if (old["cache_type_k"], old["cache_type_v"]) != (new["cache_type_k"], new["cache_type_v"]):
        parts.append(f"storing the model's short-term notepad (KV cache) as {new['cache_type_v']} "
                     f"instead of {old['cache_type_v']}")
    text = " and ".join(parts) or "these settings"
    return text[0].upper() + text[1:]


def _gain_text(new, old, goal):
    parts = []
    for key, word, goals in [("tps", "writing", {"generation", "balanced"}), ("pp_tps", "reading", {"prompt", "balanced"})]:
        if goal in goals and new[key] and old[key]:
            change = _percent(new[key], old[key])
            parts.append(f"{word} about the same" if change < 1 else
                         f"{word} {change:.0f}% {'faster' if new[key] >= old[key] else 'slower'}")
    return " and ".join(parts) or "things faster"


def tune(bench_command, variant, base_config, hardware, budget_seconds=300, goal="generation", memory_check=None,
         progress=None, cancel=None, allow_kv_compression=False, run_bench=None, clock=time.monotonic,
         n_prompt=512, n_gen=128, depth=None, repetitions=2, confirm_repetitions=CONFIRM_REPETITIONS,
         min_gain=MIN_GAIN, timeout=DEFAULT_TIMEOUT, verify_full_depth=True):
    """Coordinate search over llama.cpp settings; returns the fastest safe configuration found."""
    if goal not in GOALS:
        raise ValueError("goal must be generation, balanced or prompt")
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) or not 1 <= budget_seconds <= 86400:
        raise ValueError("budget_seconds must be a number between 1 and 86400")
    prefix = [bench_command] if isinstance(bench_command, str) else list(bench_command)
    base = launch.normalize({**base_config, "total_layers": base_config.get("total_layers") or variant.layers})
    if not base["model_path"]:
        raise ValueError("Download the model before tuning it")
    if depth is None:  # measure with some text already in context, like a real chat, without costing much time
        depth = min(1024, base["context"] // 4) // 256 * 256
    run_bench = run_bench or (lambda argv, env, timeout, cancel: run_process(argv, env, timeout, cancel))
    memory_check = memory_check or default_memory_check(variant, hardware)
    hardware = hardware or {}
    started = clock()
    trials, notes, costs = [], [], []
    state = {"skipped_memory": False}

    def remaining():
        return budget_seconds - (clock() - started)

    def per_combo(reps):
        # Pessimistic: the slowest seen cost per combination, scaled to the repetition count (never zero).
        return max(max(costs[-3:]) * (reps + 1) / (repetitions + 1), 0.01) if costs else None

    def report(message, **extra):
        if progress:
            progress({"stage": "tune", "done": round(min(clock() - started, budget_seconds), 1),
                      "total": budget_seconds, "message": message, **extra})

    def safe(config):
        try:
            return bool(memory_check(config))
        except Exception:  # noqa: BLE001 - an unknown answer is treated as unsafe
            return False

    def changes_of(config):
        return {k: v for k, v in config.items() if base.get(k) != v}

    def measure(configs, step, reps, at_depth=None, deadline=None):
        """Run candidate configs (one process when they form a comma-list product); returns results per config.
        `at_depth`/`deadline` are for the final full-length check, whose cost says nothing about a search trial."""
        at_depth = depth if at_depth is None else at_depth
        results = [None] * len(configs)
        runnable, worked = [], []  # worked: the settings that ran in this step
        for i, config in enumerate(configs):
            if safe(config):
                runnable.append(i)
            else:
                state["skipped_memory"] = True
                results[i] = {"status": "skipped_memory", "tps": None, "pp_tps": None, "noise": 0.0, "row": None}
                trials.append({"changes": changes_of(config), "tps": None, "pp_tps": None, "seconds": 0.0,
                               "status": "skipped_memory", "step": step,
                               "error": "Skipped: the memory check says this might not fit."})
        groups = _groups([configs[i] for i in runnable])
        out_of_time = False
        for number, group in enumerate(groups):
            if out_of_time:  # a budgeted check that already ran out of time does not start another run
                for g in group:
                    results[runnable[g]] = {"status": "failed", "tps": None, "pp_tps": None, "noise": 0.0, "row": None,
                                            "timed_out": True, "error": "Not tested: no time was left."}
                continue
            # llama-bench stops at the first setting that fails to load, so the riskiest values go last.
            indexes = sorted((runnable[g] for g in group), key=lambda i: _risk(configs[i]))
            check_cancel(cancel)
            group_configs = [configs[i] for i in indexes]
            sweep = {k: list(dict.fromkeys(c[k] for c in group_configs)) for k in FLAGS
                     if len({c[k] for c in group_configs}) > 1}
            head = group_configs[0]
            argv = prefix + bench_args(head, n_prompt, n_gen, at_depth, reps, sweep or None)
            # Nothing is known about the model's speed before the baseline, so it gets the full timeout.
            limit = deadline(len(groups) - number) if deadline else (timeout if step == "baseline" else min(timeout, max(remaining() + per_combo(reps), 5)))
            outcome = run_bench(argv, launch.server_env(head), limit, cancel)
            rows = parse_rows(outcome.get("stdout") or "")
            seconds = float(outcome.get("seconds") or 0) / len(indexes)
            if deadline is None:
                costs.append(seconds * (repetitions + 1) / (reps + 1))
            failure = _plain_failure(outcome) if outcome.get("returncode") or outcome.get("timed_out") else None
            out_of_time = bool(deadline and outcome.get("timed_out"))
            blamed = False
            for i in indexes:
                config = configs[i]
                mine = [r for r in rows if all(_row_matches(r, k, config[k], config) for k in sweep)]
                tg_row = next((r for r in mine if r.get("n_gen") and not r.get("n_prompt") and _speed(r)), None)
                pp = next((_speed(r) for r in mine if r.get("n_prompt") and not r.get("n_gen") and _speed(r)), None)
                tg = _speed(tg_row) if tg_row else None
                ok = (tg or not n_gen) and (pp or not n_prompt)
                result = {"status": "ok" if ok else "failed", "tps": tg[0] if tg else None,
                          "pp_tps": pp[0] if pp else None,
                          "noise": _noise(tg, pp, goal), "seconds": round(seconds, 2),
                          "row": tg_row or next((r for r in mine if _speed(r)), None),
                          "timed_out": bool(outcome.get("timed_out")) and not ok}
                if not ok:
                    reason = failure
                    if reason == NO_CAUSE and any(_lighter(other, config) for other in worked):
                        # A lighter setting just loaded the same file (a run goes from least to most memory-hungry).
                        reason = PROBABLY_MEMORY
                    result["error"] = ("Not tested: an earlier setting in the same run stopped llama-bench."
                                       if blamed else reason or "llama-bench gave no usable result for these settings.")
                    blamed = blamed or bool(failure)
                if ok:
                    worked.append(config)
                results[i] = result
                trials.append({"changes": changes_of(config), "tps": result["tps"], "pp_tps": result["pp_tps"],
                               "seconds": round(seconds, 2), "status": result["status"], "step": step,
                               **({"error": result["error"]} if not ok else {})})
        return results

    # Baseline: the starting settings must run, or there is nothing to tune from.
    report("Measuring your starting settings")
    first = measure([base], "baseline", repetitions)[0]
    if first["status"] == "skipped_memory":
        raise ValueError("There is not enough free memory for these settings right now. Close other apps or "
                         "pick a smaller context, then try again.")
    if first.get("timed_out"):
        raise ValueError(f"Measuring your starting settings took longer than {int(timeout)} seconds, so this model is "
                         "too slow to tune here. Try a smaller file or a shorter context.")
    if first["status"] != "ok" or not score(first["tps"], first["pp_tps"], goal):
        raise ValueError(f"The starting settings did not run: {first.get('error') or 'no speed was measured.'} "
                         "Try a smaller context or fewer GPU layers.")
    baseline = {"tps": first["tps"], "pp_tps": first["pp_tps"]}
    best, best_result, best_noise, best_row = base, dict(baseline), first["noise"], first["row"]
    baseline_noise, step_notes = first["noise"], []
    # llama-bench's stddev only covers the repetitions inside one process. The same settings re-measured in a later
    # process show how much the machine drifts between runs; a "gain" must beat that too.
    drift = 0.0
    stopped = None

    try:
        for _ in range(MAX_PASSES):
            improved, done = False, set()
            # Steps are re-read after each one: a memory skip switches on the KV-compression step.
            while (step := next((s for s in _steps(variant, goal, allow_kv_compression, state) if s not in done), None)):
                done.add(step)
                candidates = _candidates(step, best, hardware, variant)
                if not candidates:
                    continue
                check_cancel(cancel)
                # The current settings are re-measured in the same step (in their own run when they use a
                # "llama.cpp default", which a comma list cannot say): comparing with a number from minutes ago
                # would crown machine drift as a gain.
                unit = per_combo(repetitions)
                fit = max(0, int((remaining() - per_combo(confirm_repetitions)) // unit))
                if fit < len(candidates):
                    # Short on time: drop the re-measure of the current settings, then the least likely values.
                    candidates = [c for c in candidates if c != best][:fit]
                if not candidates:
                    stopped = "budget"
                    break
                report(f"Trying {_STEP_LABELS[step]}", best_tps=best_result["tps"], best_pp_tps=best_result["pp_tps"])
                results = measure(candidates, step, repetitions)
                current = next((r for c, r in zip(candidates, results) if c == best and r["status"] == "ok"), None)
                if current and score(current["tps"], current["pp_tps"], goal):
                    # A fresh number from the same run is the fairest thing to compare against.
                    drift = max(drift, abs(score(current["tps"], current["pp_tps"], goal) /
                                           score(best_result["tps"], best_result["pp_tps"], goal) - 1))
                    best_result, best_noise = {"tps": current["tps"], "pp_tps": current["pp_tps"]}, current["noise"]
                    best_row = current["row"] or best_row
                ref, winner = score(best_result["tps"], best_result["pp_tps"], goal), None
                for config, result in zip(candidates, results):
                    s = result["status"] == "ok" and score(result["tps"], result["pp_tps"], goal)
                    if s and config != best and s > ref * (1 + max(min_gain, result["noise"] + best_noise, drift)) \
                            and (winner is None or s > winner[2]):
                        winner = (config, result, s)
                if winner:
                    config, result, _ = winner
                    new_result = {"tps": result["tps"], "pp_tps": result["pp_tps"]}
                    step_notes.append(f"{_describe(best, config)} made {_gain_text(new_result, best_result, goal)}.")
                    best, best_result, best_noise, improved = config, new_result, result["noise"], True
                    best_row = result["row"] or best_row
            if stopped:
                break
            if not improved:
                stopped = "converged"
                break
        stopped = stopped or "converged"
    except Cancelled:
        stopped = "cancelled"

    found = best != base  # the search found something faster than the start
    # Confirm the winner with more repetitions so a lucky run cannot crown it.
    confirmed = None
    if best != base and stopped != "cancelled":
        if (per_combo(confirm_repetitions) or 0) <= max(remaining(), 0) + (per_combo(repetitions) or 0):
            report("Double-checking the winner")
            try:
                confirmed = measure([best], "confirm", confirm_repetitions)[0]
            except Cancelled:
                stopped = "cancelled"
        else:
            notes.append("There was no time left to double-check the winner, so its speed comes from a shorter test.")
    if confirmed is not None:
        s = confirmed["status"] == "ok" and score(confirmed["tps"], confirmed["pp_tps"], goal)
        # The longer check must still beat the start by more than the noise, like every step did.
        if s and s > score(baseline["tps"], baseline["pp_tps"], goal) * \
                (1 + max(min_gain, confirmed["noise"] + baseline_noise, drift)):
            best_result = {"tps": confirmed["tps"], "pp_tps": confirmed["pp_tps"]}
            best_row = confirmed["row"] or best_row
        else:
            notes.append("The best settings did not hold up in a longer check, so your starting settings are kept.")
            best = base
    if best == base:
        best_result, best_row = dict(baseline), first["row"]
    search = {"depth": depth, "baseline": dict(baseline), "best_result": dict(best_result),
              "improvement": round(score(best_result["tps"], best_result["pp_tps"], goal) /
                                   score(baseline["tps"], baseline["pp_tps"], goal), 4)}
    measured_depth, depth_notes = depth, []
    full = min(max(0, base["context"] - n_prompt - n_gen), MAX_BENCH_DEPTH)
    shorter = f"so the speeds come from a shorter test with {depth:,} tokens in memory"
    if verify_full_depth and stopped != "cancelled" and full > depth:
        # The search ran with a short conversation in memory to save time. The start and the winner are measured
        # once more at the user's own length (-d context-640, like the speed test), so the saved speed is real there.
        configs = [base] if best == base else [base, best]
        pp = min((r["pp_tps"] for r in (baseline, best_result) if r.get("pp_tps")), default=None)
        tg = min((r["tps"] for r in (baseline, best_result) if r.get("tps")), default=None)
        # llama-bench re-reads the whole depth, untimed, before every repetition (plus a warm-up) of both tests, and
        # reading and writing both slow down as the conversation grows; 1.5x is a rough allowance for that.
        cost = (len(configs) * (repetitions + 1) * 1.5 * ((2 * full + n_prompt) / pp + (n_gen / tg if n_gen else 0))
                if pp and (tg or not n_gen) else None)
        if cost is None:
            depth_notes.append("Reading speed was not measured, so the cost of a test with your full conversation "
                               f"length could not be estimated and it was not run; {shorter}.")
        elif cost <= max(remaining(), 0) + (per_combo(repetitions) or 0):
            report(f"Measuring again with {full:,} tokens of conversation in memory")
            try:
                # Each llama-bench run gets its share of what is left, so the first cannot starve the second.
                long = measure(configs, "full_depth", repetitions, at_depth=full,
                               deadline=lambda runs_left: min(timeout, max(remaining() / runs_left +
                                                                           (per_combo(repetitions) or 0), 5)))
            except Cancelled:
                stopped, long = "cancelled", None
            if long is not None:
                long_base, long_best = long[0], long[-1]
                base_ok = long_base["status"] == "ok" and score(long_base["tps"], long_base["pp_tps"], goal)
                best_ok = long_best["status"] == "ok" and score(long_best["tps"], long_best["pp_tps"], goal)
                why = lambda r: r.get("error") or "no speed was measured"
                if any(r.get("timed_out") for r in long):
                    # Out of time is not a verdict on the settings: keep what the search found.
                    depth_notes.append(f"The test with your full conversation length ran out of time, {shorter}.")
                elif not base_ok and not best_ok:
                    depth_notes.append(f"With your full conversation length ({full:,} tokens in memory) llama.cpp did "
                                       f"not run ({why(long_base)}). This context may not fit in memory with these "
                                       f"settings; {shorter}.")
                elif not base_ok:
                    depth_notes.append(f"With your full conversation length your starting settings did not run "
                                       f"({why(long_base)}), but the tuned settings did: writing "
                                       f"{long_best['tps'] or 0:.1f} and reading {long_best['pp_tps'] or 0:.1f} tokens "
                                       f"per second. The speeds above come from a shorter test with {depth:,} tokens "
                                       "in memory.")
                else:
                    if best != base and not best_ok:
                        depth_notes.append("The tuned settings did not work with your full conversation length "
                                           f"({why(long_best)}), so your starting settings are kept.")
                        best, long_best = base, long_base
                    elif best != base and best_ok <= base_ok * (1 + max(min_gain, long_best["noise"] +
                                                                       long_base["noise"], drift)):
                        depth_notes.append("With your full conversation length in memory the tuned settings were not "
                                           "clearly faster, so your starting settings are kept.")
                        best, long_best = base, long_base
                    baseline = {"tps": long_base["tps"], "pp_tps": long_base["pp_tps"]}
                    best_result = {"tps": long_best["tps"], "pp_tps": long_best["pp_tps"]}
                    best_row, measured_depth = long_best["row"] or best_row, full
        else:
            depth_notes.append(f"There was no time left to measure with your full conversation length, {shorter}.")
    if best == base and not found:
        notes.insert(0, "Your starting settings were already the fastest we found.")
    elif best == base:
        notes.insert(0, "Your starting settings are kept (see below why the faster settings found were not used).")
    else:
        if measured_depth != depth:  # the per-step gains are from the short search test
            step_notes = [f"In the shorter search test: {n[0].lower()}{n[1:]}" for n in step_notes]
        notes[:0] = [f"Overall, the tuned settings made {_gain_text(best_result, baseline, goal)} than where we started."] + \
            step_notes
    if measured_depth == full and full > depth:
        notes.append(f"These speeds were measured with {full:,} tokens of conversation already in memory, "
                     "as in a long chat at your chosen length.")
    notes += depth_notes
    improvement = score(best_result["tps"], best_result["pp_tps"], goal) / score(baseline["tps"], baseline["pp_tps"], goal)
    # Confirm and full-length runs are checks of settings already counted, and their outcome has its own note.
    failed = sum(t["status"] == "failed" for t in trials if t["step"] not in {"confirm", "full_depth"})
    skipped = sum(t["status"] == "skipped_memory" for t in trials)
    if skipped:
        notes.append(f"Skipped {skipped} setting{'s' * (skipped != 1)} that might not fit in memory.")
    if failed:
        notes.append(f"{failed} setting{'s' * (failed != 1)} failed to run and {'were' if failed != 1 else 'was'} ignored.")
    notes.append({"budget": "Stopped because the time budget ran out.",
                  "converged": "Stopped early: another round found nothing faster.",
                  "cancelled": "Stopped because you cancelled; these are the best settings found so far."}[stopped])
    report("Tuning finished", best_tps=best_result["tps"], best_pp_tps=best_result["pp_tps"])
    return {"best": best, "baseline": baseline, "best_result": best_result, "improvement": round(improvement, 4),
            "trials": trials, "stopped": stopped, "notes": notes, "goal": goal, "seconds": round(clock() - started, 1),
            "confirmed": confirmed is not None and best != base,
            # How the search trials were measured (a short conversation, to save time) and what they found.
            "bench": {"n_prompt": n_prompt, "n_gen": n_gen, "depth": depth, "repetitions": repetitions},
            "search": search,
            # §2.3 fields: `depth` is the conversation length best_result/baseline were measured at, repeated in
            # `settings.depth` next to the llama.cpp settings of `best` (engine.matching_tuned reads it there).
            "depth": measured_depth,
            "settings": {**{k: best[k] for k in SETTINGS_KEYS}, "depth": measured_depth},
            "drift": round(drift, 4), **_row_facts(best_row, best)}


def _row_facts(row, config):
    """What llama-bench reported for the winning run, for a §2.3 kind="tune" measurement: the build (runtime,
    runtime_build) and the thread count it really used when the settings left threads to llama.cpp."""
    from .testing import runtime_info
    row = row or {}
    threads = config["threads"] or (row.get("n_threads") if type(row.get("n_threads")) is int else None)
    return {"runtime": runtime_info(row, config), "runtime_build": row.get("build_commit"), "threads": threads}


_STEP_LABELS = {"gpu_layers": "how many layers go on the graphics card", "n_cpu_moe": "where the expert weights live",
                "threads": "the number of CPU threads", "flash_attn": "flash attention on and off",
                "batch": "prompt chunk sizes", "cache": "a compressed short-term notepad (KV cache)"}


def _noise(tg, pp, goal):
    w, r = BALANCED_WEIGHTS
    tg_sd, pp_sd = (tg[1] if tg else 0.0), (pp[1] if pp else 0.0)
    return {"generation": tg_sd, "prompt": pp_sd, "balanced": w * tg_sd + r * pp_sd}[goal]


def _steps(variant, goal, allow_kv, state):
    steps = ["gpu_layers"]
    if variant.moe:
        steps.append("n_cpu_moe")
    steps += ["threads", "flash_attn"]
    if goal != "generation":  # batch sizes only change prompt reading, never token-by-token writing
        steps.append("batch")
    # The notepad format is the user's choice (compression costs a little quality); only try it when allowed.
    # A tune that silently switched format could never be applied to the candidate it started from.
    if allow_kv:
        steps.append("cache")
    return steps


def _candidates(step, config, hardware, variant):
    """Candidate configs for one knob, the current config first; invalid combinations are dropped."""
    total = config["total_layers"]
    split = 0 < config["gpu_layers"] < total
    if step == "gpu_layers":
        if not split:
            return []  # only split CPU/GPU placements have a layer count to tune
        g = config["gpu_layers"]
        changes = [{"gpu_layers": v} for v in [g + 2, g + 4, g - 2] if 1 <= v <= total]
    elif step == "n_cpu_moe":
        if not config["gpu_layers"]:
            return []
        c = config["n_cpu_moe"]
        changes = [{"n_cpu_moe": v} for v in [c - 2, c - 4, c + 2] if 0 <= v <= total]
    elif step == "threads":
        cores = hardware.get("cores") or hardware.get("threads")
        logical = hardware.get("threads") or cores
        if not cores:
            return []
        changes = [{"threads": v} for v in dict.fromkeys([cores, cores - 1, cores // 2, logical]) if v and v >= 1]
    elif step == "flash_attn":
        if config["cache_type_v"] != "f16":
            return []  # a compressed V cache needs flash attention on
        changes = [{"flash_attn": "on"}, {"flash_attn": "off"}]
    elif step == "batch":
        changes = [{"batch": b, "ubatch": u} for b, u in BATCH_PAIRS]
    elif step == "cache":
        other = "q8_0" if config["cache_type_v"] == "f16" else "f16"
        changes = [{"cache_type_k": other, "cache_type_v": other}]
        if other == "q8_0":
            # The point of a smaller notepad is often room for more of the model on the graphics card.
            if split:
                changes.append({"cache_type_k": other, "cache_type_v": other, "gpu_layers": min(total, config["gpu_layers"] + 2)})
            if variant.moe and config["gpu_layers"] and config["n_cpu_moe"]:
                changes.append({"cache_type_k": other, "cache_type_v": other, "n_cpu_moe": max(0, config["n_cpu_moe"] - 2)})
    else:
        return []
    result = [config]
    for change in changes:
        try:
            candidate = launch.normalize({**config, **change})
        except ValueError:
            continue
        if candidate not in result:
            result.append(candidate)
    return result if len(result) > 1 else []


def _risk(config):
    """Sort key from least to most memory-hungry, for the order inside one llama-bench run."""
    return (config["gpu_layers"], -config["n_cpu_moe"], config["batch"] or 0, config["ubatch"] or 0,
            config["cache_type_v"] != "f16")


_WEIGHT_DEFAULTS = {"batch": 2048, "ubatch": 512}  # llama.cpp's defaults when unset


def _lighter(light, heavy):
    """True when `light` needs no more memory than `heavy` for certain: they differ only in how much sits on the
    graphics card or in the prompt chunk sizes, and `light` is smaller (or equal) in each. Flash attention and the
    cache format change more than memory (a backend may not support them), so a difference there proves nothing."""
    size = lambda c, k: c[k] if c[k] is not None else _WEIGHT_DEFAULTS.get(k)
    if any(light[k] != heavy[k] for k in light if k not in {"gpu_layers", "n_cpu_moe", "batch", "ubatch"}):
        return False
    return light != heavy and light["gpu_layers"] <= heavy["gpu_layers"] and light["n_cpu_moe"] >= heavy["n_cpu_moe"] \
        and all(size(light, k) <= size(heavy, k) for k in ("batch", "ubatch"))


def _groups(configs):
    """Split configs into llama-bench runs: one run when they form a full comma-list product.
    Unset values and flash_attn "auto" mean "llama.cpp default", which a comma list cannot say,
    so a config carrying one of them in a swept setting runs on its own."""
    indexes = list(range(len(configs)))
    swept = [k for k in FLAGS if len({c[k] for c in configs}) > 1]
    alone = [i for i in indexes if any(configs[i][k] in {None, "auto"} for k in swept)]
    if alone:
        rest = [i for i in indexes if i not in alone]
        return [[i] for i in alone] + [[rest[j] for j in g] for g in _groups([configs[i] for i in rest])]
    if len(configs) <= 1:
        return [indexes] if configs else []
    same_rest = all(c[k] == configs[0][k] for c in configs for k in configs[0] if k not in swept)
    values = [list(dict.fromkeys(c[k] for c in configs)) for k in swept]
    have = {tuple(c[k] for k in swept) for c in configs}
    if same_rest and set(itertools.product(*values)) == have and len(have) == len(configs):
        return [indexes]
    return [[i] for i in indexes]
