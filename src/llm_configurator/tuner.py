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
OOM_TEXT = re.compile(r"out of memory|failed to allocate|cudaMalloc failed|ErrorOutOfDeviceMemory|"
                      r"unable to allocate|not enough memory|insufficient memory", re.IGNORECASE)

# launch-config name -> (llama-bench flag, llama-bench JSON field)
FLAGS = {"gpu_layers": ("-ngl", "n_gpu_layers"), "n_cpu_moe": ("-ncmoe", "n_cpu_moe"), "threads": ("-t", "n_threads"),
         "batch": ("-b", "n_batch"), "ubatch": ("-ub", "n_ubatch"), "flash_attn": ("-fa", "flash_attn"),
         "cache_type_k": ("-ctk", "type_k"), "cache_type_v": ("-ctv", "type_v")}


def _bench_value(name, value, config):
    if name == "gpu_layers":
        return str(launch.runtime_gpu_layers(value, config["total_layers"]))
    if name == "flash_attn":
        # "1"/"0" work on every llama-bench: older builds take only 0/1, newer ones treat them as on/off.
        return {"on": "1", "off": "0"}[value]
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
        args += ["-mmp", "0"]
    return args + ["-r", str(repetitions), "-o", "json"]


def run_process(argv, env=None, timeout=DEFAULT_TIMEOUT, cancel=None):
    """Run llama-bench; kills it on cancel (raising Cancelled) or on timeout."""
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env,
                               errors="replace", creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    timed_out = False
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if (cancel is not None and cancel.is_set()) or time.monotonic() - started > timeout:
                process.kill()
                stdout, stderr = process.communicate()
                if cancel is not None and cancel.is_set():
                    raise Cancelled("Cancelled by user") from None
                timed_out = True
                break
    return {"returncode": process.returncode, "stdout": stdout or "", "stderr": stderr or "",
            "seconds": time.monotonic() - started, "timed_out": timed_out}


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


def _plain_failure(result):
    text = (result.get("stderr") or "") + (result.get("stdout") or "")
    if result.get("timed_out"):
        return "The test ran out of time."
    if OOM_TEXT.search(text):
        return "Ran out of memory with these settings."
    if "unknown argument" in text or "invalid parameter" in text or "error: invalid" in text:
        return "This llama.cpp version does not support one of these settings."
    tail = text.strip().splitlines()[-1:] or ["no error output"]
    return f"llama-bench stopped with an error ({result.get('returncode')}): {tail[0][:200]}"


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
    if goal in {"generation", "balanced"} and new["tps"] and old["tps"]:
        parts.append(f"writing {_percent(new['tps'], old['tps']):.0f}% {'faster' if new['tps'] >= old['tps'] else 'slower'}")
    if goal in {"prompt", "balanced"} and new["pp_tps"] and old["pp_tps"]:
        parts.append(f"reading {_percent(new['pp_tps'], old['pp_tps']):.0f}% "
                     f"{'faster' if new['pp_tps'] >= old['pp_tps'] else 'slower'}")
    return " and ".join(parts) or "things faster"


def tune(bench_command, variant, base_config, hardware, budget_seconds=300, goal="generation", memory_check=None,
         progress=None, cancel=None, allow_kv_compression=False, run_bench=None, clock=time.monotonic,
         n_prompt=512, n_gen=128, depth=None, repetitions=2, confirm_repetitions=CONFIRM_REPETITIONS,
         min_gain=MIN_GAIN, timeout=DEFAULT_TIMEOUT):
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
        # Pessimistic: the slowest seen cost per combination, scaled to the repetition count.
        return max(costs[-3:]) * (reps + 1) / (repetitions + 1) if costs else None

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

    def measure(configs, step, reps):
        """Run candidate configs (one process when they form a comma-list product); returns results per config."""
        results = [None] * len(configs)
        runnable = []
        for i, config in enumerate(configs):
            if safe(config):
                runnable.append(i)
            else:
                state["skipped_memory"] = True
                results[i] = {"status": "skipped_memory", "tps": None, "pp_tps": None}
                trials.append({"changes": changes_of(config), "tps": None, "pp_tps": None, "seconds": 0.0,
                               "status": "skipped_memory", "step": step,
                               "error": "Skipped: the memory check says this might not fit."})
        for group in _groups([configs[i] for i in runnable]):
            # llama-bench stops at the first setting that fails to load, so the riskiest values go last.
            indexes = sorted((runnable[g] for g in group), key=lambda i: _risk(configs[i]))
            check_cancel(cancel)
            group_configs = [configs[i] for i in indexes]
            sweep = {k: list(dict.fromkeys(c[k] for c in group_configs)) for k in FLAGS
                     if len({c[k] for c in group_configs}) > 1}
            head = group_configs[0]
            argv = prefix + bench_args(head, n_prompt, n_gen, depth, reps, sweep or None)
            deadline = min(timeout, max(remaining() + (per_combo(reps) or remaining()), 5))
            outcome = run_bench(argv, launch.server_env(head), deadline, cancel)
            rows = parse_rows(outcome.get("stdout") or "")
            seconds = float(outcome.get("seconds") or 0) / len(indexes)
            costs.append(seconds * (repetitions + 1) / (reps + 1))
            failure = _plain_failure(outcome) if outcome.get("returncode") or outcome.get("timed_out") else None
            blamed = False
            for i in indexes:
                config = configs[i]
                mine = [r for r in rows if all(_row_matches(r, k, config[k], config) for k in sweep)]
                tg = next((_speed(r) for r in mine if r.get("n_gen") and not r.get("n_prompt")), None)
                pp = next((_speed(r) for r in mine if r.get("n_prompt") and not r.get("n_gen")), None)
                ok = (tg or not n_gen) and (pp or not n_prompt)
                result = {"status": "ok" if ok else "failed", "tps": tg[0] if tg else None,
                          "pp_tps": pp[0] if pp else None,
                          "noise": _noise(tg, pp, goal), "seconds": round(seconds, 2)}
                if not ok:
                    result["error"] = ("Not tested: an earlier setting in the same run stopped llama-bench."
                                       if blamed else failure or "llama-bench gave no usable result for these settings.")
                    blamed = blamed or bool(failure)
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
    if first["status"] != "ok" or not score(first["tps"], first["pp_tps"], goal):
        raise ValueError(f"The starting settings did not run: {first.get('error') or 'no speed was measured.'} "
                         "Try a smaller context or fewer GPU layers.")
    baseline = {"tps": first["tps"], "pp_tps": first["pp_tps"]}
    best, best_result, best_noise = base, dict(baseline), first["noise"]
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
                unit = per_combo(repetitions)
                fit = int((remaining() - (per_combo(confirm_repetitions) or 0)) // unit)
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
                    best_result, best_noise = {"tps": current["tps"], "pp_tps": current["pp_tps"]}, current["noise"]
                ref, winner = score(best_result["tps"], best_result["pp_tps"], goal), None
                for config, result in zip(candidates, results):
                    s = result["status"] == "ok" and score(result["tps"], result["pp_tps"], goal)
                    if s and config != best and s > ref * (1 + max(min_gain, result["noise"] + best_noise)) \
                            and (winner is None or s > winner[2]):
                        winner = (config, result, s)
                if winner:
                    config, result, _ = winner
                    new_result = {"tps": result["tps"], "pp_tps": result["pp_tps"]}
                    notes.append(f"{_describe(best, config)} made {_gain_text(new_result, best_result, goal)}.")
                    best, best_result, best_noise, improved = config, new_result, result["noise"], True
            if stopped:
                break
            if not improved:
                stopped = "converged"
                break
        stopped = stopped or "converged"
    except Cancelled:
        stopped = "cancelled"

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
        if s and s > score(baseline["tps"], baseline["pp_tps"], goal):
            best_result = {"tps": confirmed["tps"], "pp_tps": confirmed["pp_tps"]}
        else:
            notes.append("The best settings did not hold up in a longer check, so your starting settings are kept.")
            best = base
    if best == base:
        best_result = dict(baseline)
        notes.insert(0, "Your starting settings were already the fastest we found.")
    else:
        notes.insert(0, f"Overall, the tuned settings made {_gain_text(best_result, baseline, goal)} than where we started.")
    improvement = score(best_result["tps"], best_result["pp_tps"], goal) / score(baseline["tps"], baseline["pp_tps"], goal)
    failed = sum(t["status"] == "failed" for t in trials)
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
            "settings": {"n_prompt": n_prompt, "n_gen": n_gen, "depth": depth, "repetitions": repetitions}}


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
    if allow_kv or state["skipped_memory"]:
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


def _groups(configs):
    """Split configs into llama-bench runs: one run when they form a full comma-list product.
    flash_attn "auto" cannot go in a comma list, so such a config runs on its own."""
    indexes = list(range(len(configs)))
    if len({c["flash_attn"] for c in configs}) > 1:
        auto = [i for i in indexes if configs[i]["flash_attn"] == "auto"]
        rest = [i for i in indexes if i not in auto]
        return [[i] for i in auto] + [[rest[j] for j in g] for g in _groups([configs[i] for i in rest])]
    if len(configs) <= 1:
        return [indexes] if configs else []
    swept = [k for k in FLAGS if len({c[k] for c in configs}) > 1]
    same_rest = all(c[k] == configs[0][k] for c in configs for k in configs[0] if k not in swept)
    values = [list(dict.fromkeys(c[k] for c in configs)) for k in swept]
    have = {tuple(c[k] for k in swept) for c in configs}
    if same_rest and set(itertools.product(*values)) == have and len(have) == len(configs):
        return [indexes]
    return [[i] for i in indexes]
