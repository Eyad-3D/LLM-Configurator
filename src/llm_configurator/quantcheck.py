"""Compression-loss check: how far a smaller model file drifts from a reference file.

Think of it like comparing a low-quality JPEG with the original photo, word by word. The
reference model's predictions are saved once, then each compressed file reads the same text
and llama-perplexity reports how often and how much its predictions differ.

Honesty: every number comes from llama-perplexity's own output. Anything it did not print
stays None. The temporary predictions file is estimated before it is written, checked against
free disk space, and always deleted, also on errors and cancel. Verdicts are rough guidance.

Real llama-perplexity sometimes loses the end of its output under load (about 1 run in 10: it
exits 0 but the summary is cut off). A candidate whose summary is missing is run again, reusing
the saved reference predictions; only after MAX_ATTEMPTS does it fall back to the last per-window
row, and the result says so.
"""
import codecs
import math
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from .domain import GIB, Cancelled, check_cancel

CORPUS_PATH = Path(__file__).with_name("data") / "quantcheck_corpus.txt"
TARGET_TOKENS = 6144          # text read per model; llama.cpp scores the second half of each window
ASSUMED_VOCAB = 262144        # largest common vocabulary (Gemma 3); an upper bound when unknown
CHARS_PER_TOKEN = 4.0         # rough for English prose
DISK_MARGIN_BYTES = 256 * 1024**2
MAX_ATTEMPTS = 3              # runs per candidate when the output comes back cut short

# Mean KL divergence bands. Anchors from llama.cpp's LLaMA 3 8B scoreboard against FP16
# (tools/perplexity/README.md): Q8_0 0.0014, Q6_K 0.0055, Q5_K_M 0.011, Q4_K_M 0.031,
# Q3_K_M 0.10, Q2_K 0.33-0.45, IQ1 1.4+. Rough guidance, not a guarantee.
KLD_BANDS = [(0.01, "negligible"), (0.05, "small"), (0.15, "moderate"), (0.5, "large"), (math.inf, "severe")]
# Fallback when only "Same top p" is known: share of positions where the top word differs.
# From the same README's "LLaMA 2 vs. LLaMA 3" table (LLaMA 3 8B vs FP16): Q8_0 2.3%, Q6_K 4.0%,
# Q4_K_M 8.1%, Q2_K 28.9%. Small models and short samples drift more, so these stay rough.
DIFFERENT_TOP_BANDS = [(5.0, "negligible"), (10.0, "small"), (18.0, "moderate"), (35.0, "large"),
                       (math.inf, "severe")]
VERDICTS = {
    "negligible": "practically the same; very unlikely to be noticed",
    "small": "usually hard to notice in chat",
    "moderate": "may show on harder tasks such as maths, code or long reasoning",
    "large": "noticeably worse; expect more mistakes",
    "severe": "much worse; answers may often go wrong",
}

_NUM = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?nan|[-+]?inf)"
_PM = r"(?:\s*(?:±|\+/-|\+-)\s*" + _NUM + r")?"
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# (key, pattern) — first match wins, so the duplicated "99.0%" line of 2024 builds (really 95%) is ignored.
_PATTERNS = [
    ("mean_kld", r"^(?:Mean\s+KLD|Average)\s*:\s*" + _NUM + _PM),
    ("median_kld", r"^Median\s*(?:KLD)?\s*:\s*" + _NUM),
    ("kld_99", r"^(?:99(?:\.0)?\s*%\s*KLD|KLD_99)\s*:\s*" + _NUM),
    ("kld_999", r"^99\.9\s*%\s*KLD\s*:\s*" + _NUM),
    ("max_kld", r"^Maximum\s*(?:KLD)?\s*:\s*" + _NUM),
    ("same_top_p", r"^Same\s+top(?:\s+p)?\s*:\s*" + _NUM + _PM),
    ("ppl", r"^Mean\s+PPL\s*\(\s*Q\s*\)\s*:\s*" + _NUM + _PM),
    ("ppl_base", r"^Mean\s+PPL\s*\(\s*base\s*\)\s*:\s*" + _NUM + _PM),
    ("ppl_ratio", r"^Mean\s+PPL\s*\(\s*Q\s*\)\s*/\s*PPL\s*\(\s*base\s*\)\s*:\s*" + _NUM),
    # Δ can arrive mangled by a console code page, so allow a few odd characters before the "p".
    ("mean_delta_p", r"^Mean\s+\S{1,3}p\s*:\s*" + _NUM + _PM),
    ("rms_delta_p", r"^RMS\s+\S{1,3}p\s*:\s*" + _NUM + _PM),
    ("final_ppl", r"Final estimate:\s*PPL\s*=\s*" + _NUM + _PM),
]
_COMPILED = [(key, re.compile(pattern, re.IGNORECASE)) for key, pattern in _PATTERNS]
_ROW_START = re.compile(r"^\s*(\d+)\s+[-+]?\d")
_ROW_NUMBER = re.compile(_NUM, re.IGNORECASE)
_BASE_PROGRESS = re.compile(r"\[(\d+)\]\s*" + _NUM)
_CHUNK_TOTAL = re.compile(r"(?:computing|calculating perplexity) over (\d+) chunks", re.IGNORECASE)
_QUANT = re.compile(r"(?<![A-Za-z0-9])(I?Q\d_(?:K_[SMLX]L?|K|[01]|[A-Z]{1,3})|MXFP4|BF16|F16|F32)(?![A-Za-z0-9])",
                    re.IGNORECASE)
RESULT_KEYS = ["mean_kld", "median_kld", "kld_99", "same_top_p", "ppl_base", "ppl", "mean_delta_p"]


def _number(text):
    value = float(text)
    return value if math.isfinite(value) else None


def parse_kld_output(text):
    """Pull the KL-divergence summary out of llama-perplexity output (any 2024-2026 layout).

    Percent values (same_top_p, mean_delta_p, rms_delta_p) are 0-100. Missing values stay None.
    Older builds (before mid-2024) print only per-chunk rows for "Same top" (as a 0-1 fraction)
    and PPL, so the last row fills those gaps.

    `complete` is True only when the closing summary was printed in full ("Mean KLD" and, on
    builds that print it, the final "Same top p" line). `from_rows` lists the keys that were
    filled from the last per-window row because the summary did not have them.
    """
    result = {key: None for key in RESULT_KEYS}
    result.update(max_kld=None, kld_999=None, ppl_ratio=None, rms_delta_p=None, mean_kld_uncertainty=None,
                  same_top_p_uncertainty=None, final_ppl=None, chunks_done=None, complete=False, from_rows=[])
    last_row, from_summary, early_layout = None, set(), False
    for raw in _ANSI.sub("", text or "").replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        for key, pattern in _COMPILED:
            match = pattern.search(line)
            if match:
                from_summary.add(key)
                early_layout = early_layout or line.lower().startswith("average")
                if result[key] is None:
                    result[key] = _number(match.group(1))
                    if key in {"mean_kld", "same_top_p"} and match.group(2):
                        result[f"{key}_uncertainty"] = _number(match.group(2))
                break
        else:
            if _ROW_START.match(line):
                numbers = [_number(n) for n in _ROW_NUMBER.findall(line)]
                if len(numbers) in {8, 11}:     # old rows: chunk + 7 numbers; 2024+ rows: chunk + 10
                    last_row = numbers
    early_layout = early_layout or bool(last_row and len(last_row) == 8)
    # Early-2024 builds never print a "Same top" summary line; every later build ends with it.
    result["complete"] = "mean_kld" in from_summary and ("same_top_p" in from_summary or early_layout)
    if last_row:
        result["chunks_done"] = int(last_row[0])
        if len(last_row) == 11:
            fill = {"ppl": last_row[1], "mean_kld": last_row[5], "same_top_p": last_row[9]}
        else:
            fill = {"ppl": last_row[1], "mean_kld": last_row[4],
                    "same_top_p": None if last_row[6] is None else last_row[6] * 100}
        for key, value in fill.items():
            if result[key] is None and value is not None:
                result[key] = value
                if key not in from_summary:
                    result["from_rows"].append(key)
    if result["ppl_ratio"] is None and result["ppl"] and result["ppl_base"]:
        result["ppl_ratio"] = result["ppl"] / result["ppl_base"]
    return result


def quant_label(path):
    """A short name such as "Q4_K_M" from a file name, or the file name itself."""
    match = _QUANT.search(Path(path).name)
    return match.group(1).upper() if match else Path(path).name


def _band(value, bands):
    return next(name for limit, name in bands if value < limit)


def interpret(parsed, reference_label="the reference"):
    """One plain sentence plus a verdict id. Rough guidance only; None when nothing was measured."""
    same_top, kld = parsed.get("same_top_p"), parsed.get("mean_kld")
    if kld is not None:
        verdict = _band(max(kld, 0.0), KLD_BANDS)
    elif same_top is not None:
        verdict = _band(100 - same_top, DIFFERENT_TOP_BANDS)
    else:
        return {"plain": f"No result: the check did not report a comparison with {reference_label}.", "verdict": None}
    if same_top is not None:
        different = max(0.0, 100 - same_top)
        amount = "less than 1%" if different < 1 else f"about {round(different)}%"
        lead = f"Picks a different top word {amount} of the time compared with {reference_label}"
    else:
        lead = f"Its word predictions drift from {reference_label} by a KL divergence of {kld:.3f}"
    return {"plain": f"{lead} — {VERDICTS[verdict]}.", "verdict": verdict}


def estimate_logits_bytes(vocab_size, context, chunks):
    """Size of the --kl-divergence-base file, following llama.cpp's writer (perplexity.cpp): "_logits_"
    and three int32 (n_ctx, n_vocab, n_chunk), the tokens as int32, then per chunk the scored second
    half of the window, (n_ctx - 1 - n_ctx/2) rows of (2*((n_vocab+1)/2) + 4) uint16."""
    scored = context - 1 - context // 2
    per_row = (2 * ((vocab_size + 1) // 2) + 4) * 2
    return 20 + context * chunks * 4 + chunks * scored * per_row


def _vocab_size(path):
    """Vocabulary size from the GGUF header when the reader can tell us: `summary["vocab_size"]`
    (the token list's length) if gguf.read_metadata provides it, else a scalar vocab_size key."""
    try:
        from . import gguf
        info = gguf.read_metadata(path)
        metadata = info.get("metadata") or {}
        found = [(info.get("summary") or {}).get("vocab_size")]
        found += [metadata.get(key) for key in [f"{info.get('architecture')}.vocab_size", "tokenizer.ggml.vocab_size"]]
        for value in found:
            if type(value) is int and value > 0:
                return value
    except Exception:  # missing module or unreadable header: fall back to a safe upper bound
        pass
    return None


def _default_chunks(context, corpus_bytes):
    available = int(corpus_bytes / CHARS_PER_TOKEN) // context
    wanted = max(1, round(TARGET_TOKENS / context))
    return max(1, min(wanted, available, 64))


# Launch settings the check reuses; the rest (cache types, draft model, port, ...) are left out, so
# a server-only setting can neither reach llama-perplexity nor fail validation here.
_CONFIG_KEYS = {"gpu_layers", "total_layers", "gpu_uuid", "gpu_backend", "device", "threads", "n_cpu_moe"}


def _common_args(config, context):
    """Flags llama-perplexity shares with llama-server: GPU layers, device, threads, MoE offload.
    Server-only flags (host, port, alias, jinja, parallel, draft) and KV-cache compression are left
    out so the check measures the weights alone. Batch equals the window: one sequence at a time."""
    args = ["-c", str(context), "-b", str(context)]
    env = None
    if config:
        from . import launch
        c = launch.normalize({k: v for k, v in config.items() if k in _CONFIG_KEYS})
        args += ["-ngl", str(launch.runtime_gpu_layers(c["gpu_layers"], c["total_layers"]))]
        device = c["device"] or ("none" if c["gpu_layers"] == 0 and c["gpu_backend"] not in {None, "cpu"} else None)
        if device:
            args += ["-dev", device]
        if c["threads"]:
            args += ["-t", str(c["threads"])]
        if c["n_cpu_moe"]:
            args += ["--n-cpu-moe", str(c["n_cpu_moe"])]
        env = launch.server_env(c)
    return args, env


class _TimedOut(ValueError):
    pass


def _run_process(argv, env=None, timeout=3600, cancel=None, on_line=None):
    """Run argv, stream merged output to on_line, kill on cancel or timeout. Returns (code, text).

    Pieces are split after newlines and commas: llama-perplexity prints "[1]6.12,[2]6.30," on one
    line while it works, so commas give live progress."""
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env=env, **kwargs)
    pieces, output = queue.Queue(), []

    def pump():
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                data = os.read(process.stdout.fileno(), 65536)
                text = decoder.decode(data, final=not data)
                if text:
                    pieces.put(text)
                if not data:
                    break
        except (OSError, ValueError):  # pipe closed after a cancel or timeout
            pass
        pieces.put(None)
    threading.Thread(target=pump, daemon=True).start()
    deadline, pending = time.monotonic() + timeout, ""
    try:
        while True:
            check_cancel(cancel)
            if time.monotonic() > deadline:
                raise TimeoutError
            try:
                text = pieces.get(timeout=0.2)
            except queue.Empty:
                continue
            if text is None:
                break
            output.append(text)
            parts = re.split(r"(?<=[\n,])", pending + text)
            pending = parts.pop()
            for part in parts:
                if on_line:
                    on_line(part)
        if pending and on_line:
            on_line(pending)
        return process.wait(timeout=30), "".join(output)
    except TimeoutError:
        raise _TimedOut(f"llama-perplexity took longer than {max(1, timeout // 60)} minutes and was stopped. "
                         "Try fewer chunks or a smaller context.") from None
    finally:
        if process.poll() is None:
            process.kill()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:  # keep the original error; the OS reaps it later
                pass
        process.stdout.close()


def _tail(text, lines=6):
    return "\n".join((text or "").strip().splitlines()[-lines:])


def _failure(output):
    """Plain reason from llama-perplexity's log, which often exits 0 even when it did nothing."""
    text = output or ""
    match = re.search(r"you need at least (\d+) tokens", text)
    if match:
        return (f"The sample text is too short for this window size (it needs {match.group(1)} tokens). "
                "Use a smaller context or a longer text.")
    if re.search(r"out of memory|failed to allocate|cudaMalloc failed|unable to allocate", text, re.IGNORECASE):
        return "The model did not fit in memory. Put fewer layers on the GPU or close other programs."
    if re.search(r"unknown model architecture|unsupported model architecture", text, re.IGNORECASE):
        return "This llama.cpp build does not support the model's architecture. Update llama.cpp."
    if re.search(r"inconsistent vocabulary", text, re.IGNORECASE):
        return "The two files use different vocabularies, so they are not versions of the same model."
    if re.search(r"unable to load model|failed to load model|error loading model", text, re.IGNORECASE):
        return "llama.cpp could not load the model file. It may be incomplete or damaged; try downloading it again."
    if re.search(r"has been computed with \d+, while the current context", text):
        return "The saved reference predictions use a different window size. Run the check again."
    if re.search(r"does not look like a file containing log-probabilities|failed reading|failed to open",
                 text, re.IGNORECASE):
        return "The saved reference predictions could not be read. Check free disk space and run the check again."
    if re.search(r"error: (?:invalid argument|unknown argument)", text, re.IGNORECASE):
        return "This llama.cpp build does not accept one of the settings. Update llama.cpp."
    return None


def _redact(text, paths):
    """Replace absolute paths with bare file names, so messages never reveal folder names.
    Whole folders are removed too, which also covers the other parts of a split model."""
    folders = {str(Path(p).parent) for p in paths if p} | {str(p) for p in paths if p and Path(p).is_dir()}
    for folder in sorted(folders, key=len, reverse=True):
        if len(folder) > 1:
            text = text.replace(folder + os.sep, "").replace(folder, Path(folder).name)
    return text


def _whole_lines(text):
    """Drop a last line that has no newline: output lost mid-line can end in "Same top p: 1"."""
    text = text or ""
    return text if not text or text.endswith(("\n", "\r")) else text[:max(text.rfind("\n"), text.rfind("\r")) + 1]


_MISSING_NAMES = [("median_kld", "median"), ("kld_99", "worst 1%"), ("same_top_p", "same top word"),
                  ("mean_delta_p", "change in confidence")]


def _partial_note(parsed, attempts, total_chunks):
    """Plain words for a result whose closing report never arrived in full."""
    times = f"{attempts} time{'s' if attempts != 1 else ''} in a row"
    text = f"llama-perplexity's final report was cut short {times}"
    if parsed.get("from_rows"):
        where = (f"after window {parsed['chunks_done']} of {total_chunks}" if parsed.get("chunks_done")
                 else "part-way through")
        text += f", so some figures are its running average {where}"
    missing = [name for key, name in _MISSING_NAMES if parsed.get(key) is None]
    if missing:
        text += f"; missing: {', '.join(missing)}"
    return text + "."


def kl_check(perplexity_command, reference_path, candidates, corpus_path=None, context=512, chunks=None, config=None,
             progress=None, cancel=None, reference_label=None, vocab_size=None, work_dir=None, timeout=3600, run=None):
    """Compare each candidate file with the reference file on the same text.

    `candidates` maps label -> path. `config` is an optional launch config whose GPU layers,
    device, threads and MoE offload are reused for every model, one model at a time.
    `run(argv, env=, timeout=, cancel=, on_line=) -> (returncode, output)` can be injected for tests.
    """
    run = run or _run_process
    command = [perplexity_command] if isinstance(perplexity_command, str) else list(perplexity_command)
    if not command:
        raise ValueError("llama-perplexity was not found. Install llama.cpp first.")
    if type(context) is not int or not 128 <= context <= 32768:
        raise ValueError("context must be an integer between 128 and 32768")
    if chunks is not None and (type(chunks) is not int or not 1 <= chunks <= 1000):
        raise ValueError("chunks must be an integer between 1 and 1000")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("Choose at least one compressed file to compare")
    reference_path = Path(reference_path)
    paths = {label: Path(path) for label, path in candidates.items()}
    for path in [reference_path, *paths.values()]:
        if not path.is_file():
            raise ValueError(f"Model file not found: {path.name}. Download it first.")
    corpus = Path(corpus_path) if corpus_path else CORPUS_PATH
    if not corpus.is_file():
        raise ValueError(f"Sample text not found: {corpus.name}")
    corpus_bytes = corpus.stat().st_size
    if corpus_bytes / 3 < 2 * context:   # generous 3 characters per token: clearly too short
        raise ValueError("The sample text is too short for this window size. Use a smaller context.")
    chunks = chunks or _default_chunks(context, corpus_bytes)
    reference_label = reference_label or quant_label(reference_path)
    common, env = _common_args(config, context)

    known_vocab = vocab_size or _vocab_size(reference_path)
    estimate = estimate_logits_bytes(known_vocab or ASSUMED_VOCAB, context, chunks)
    temp_root = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    try:
        temp_root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(temp_root).free
    except OSError as exc:
        raise ValueError(f"The folder for temporary files cannot be used ({exc.strerror or type(exc).__name__}).") from None
    where = "the work folder" if work_dir else "the system's temporary folder"
    if free < estimate + DISK_MARGIN_BYTES:
        raise ValueError(f"The check needs about {estimate / GIB:.1f} GB of temporary space in {where}, "
                         f"but only {free / GIB:.1f} GB is free. Free up some space or use fewer chunks.")
    notes = [f"Temporary predictions file: about {estimate / GIB:.2f} GB"
             + ("" if known_vocab else " (vocabulary size unknown, so this assumes a very large one)")
             + f" in {where}; it is deleted afterwards.",
             f"Reads about {chunks * context} tokens of sample text in {chunks} windows of {context} and scores "
             "the second half of each window; a short sample gives a quick, rough answer.",
             "Verdicts are rough guidance based on llama.cpp's published measurements, not a guarantee."]

    order = [("reference", reference_label, reference_path)] + [("candidate", label, path) for label, path in paths.items()]
    total = len(order) * chunks
    state = {"done": 0, "chunks": chunks}

    def report(stage, label, message, extra=0):
        if progress:
            progress({"stage": stage, "done": min(total, state["done"] + extra), "total": total,
                      "message": message, "label": label})

    def on_line_for(stage, label):
        def on_line(line):
            clean = _ANSI.sub("", line)
            found = _CHUNK_TOTAL.search(clean)
            if found:
                state["chunks"] = max(1, int(found.group(1)))
            if stage == "reference":
                numbers = _BASE_PROGRESS.findall(clean)
                chunk = int(numbers[-1][0]) if numbers else None
            else:
                row = _ROW_START.match(clean)
                chunk = int(row.group(1)) if row else None
            if chunk:
                done = min(chunks, round(chunk * chunks / state["chunks"]))
                state["peak"] = max(state.get("peak", 0), done)   # a re-run starts over; the bar does not
                report(stage, label, f"{label}: window {chunk} of {state['chunks']}", state["peak"])
        return on_line

    temp_dir = tempfile.mkdtemp(prefix="llmc-quantcheck-", dir=work_dir)
    logits = Path(temp_dir) / "reference.kld"
    private = [reference_path, *paths.values(), corpus, temp_dir, logits, temp_root]

    def tail(output):
        return _redact(_tail(output), private)

    results, reference = {}, {"label": reference_label, "file": reference_path.name, "ppl": None,
                              "logits_bytes": None}
    try:
        check_cancel(cancel)
        report("reference", reference_label, f"Saving {reference_label}'s predictions on the sample text")
        argv = command + ["-m", str(reference_path), "-f", str(corpus), "--kl-divergence-base", str(logits),
                          "--chunks", str(chunks)] + common
        code, output = run(argv, env=env, timeout=timeout, cancel=cancel, on_line=on_line_for("reference", reference_label))
        check_cancel(cancel)
        if code != 0 or not logits.is_file() or logits.stat().st_size <= 20:
            reason = _failure(output) or f"llama-perplexity stopped without saving predictions:\n{tail(output)}"
            raise ValueError(f"The reference file {reference_label} could not be measured. {reason}")
        parsed = parse_kld_output(output)
        progress_values = _BASE_PROGRESS.findall(_ANSI.sub("", output or ""))
        last_progress = _number(progress_values[-1][1]) if progress_values else None
        reference.update(ppl=parsed["final_ppl"] or parsed["ppl"] or last_progress, logits_bytes=logits.stat().st_size)
        state["done"] += chunks
        reported_chunks = state["chunks"]
        for label, path in paths.items():
            state["peak"] = 0
            check_cancel(cancel)
            report("candidate", label, f"Comparing {label} with {reference_label}")
            argv = command + ["-m", str(path), "-f", str(corpus), "--kl-divergence-base", str(logits),
                              "--kl-divergence"] + common
            best, attempts = None, 0
            while attempts < MAX_ATTEMPTS:
                attempts += 1
                try:
                    code, output = run(argv, env=env, timeout=timeout, cancel=cancel,
                                       on_line=on_line_for("candidate", label))
                except _TimedOut as exc:   # one slow file must not throw away the others
                    code, output = None, str(exc)
                check_cancel(cancel)
                if code is None:
                    best = best or (code, parse_kld_output(""), output)   # keep an earlier partial result
                    break
                output = _whole_lines(output)
                parsed = parse_kld_output(output)
                if best is None or (parsed["complete"], parsed["chunks_done"] or 0) > \
                        (best[1]["complete"], best[1]["chunks_done"] or 0):
                    best = (code, parsed, output)
                # Retry only a clean exit whose summary is cut off; real errors would just repeat.
                if parsed["complete"] or code != 0 or _failure(output):
                    break
                report("candidate", label, f"{label}: llama-perplexity's report came back cut short; "
                                           f"running it again (try {attempts + 1} of {MAX_ATTEMPTS})", state["peak"])
            code, parsed, output = best
            parsed = {key: value for key, value in parsed.items() if key != "final_ppl"}
            if parsed["ppl_base"] is None:
                parsed["ppl_base"] = reference["ppl"]
            error = None
            if code is None:
                error = output
            elif code != 0 or (parsed["mean_kld"] is None and parsed["same_top_p"] is None):
                error = _failure(output) or f"llama-perplexity gave no comparison:\n{tail(output)}"
            summary = interpret(parsed, reference_label)
            partial = None
            if not error and not parsed["complete"]:
                partial = _partial_note(parsed, attempts, state["chunks"])
                summary["plain"] += f" (Partial result: {partial})"
                notes.append(f"{label}: {partial}")
            if error:
                summary = {"plain": f"Could not compare {label} with {reference_label}. {error}", "verdict": None}
            results[label] = {**parsed, **summary, "file": path.name, "error": error, "attempts": attempts,
                              "partial": partial}
            state["done"] += chunks
        report("done", None, "Compression check finished")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {"reference": reference, "results": results,
            "corpus": {"file": corpus.name, "bytes": corpus_bytes, "context": context, "chunks": reported_chunks,
                       "tokens": reported_chunks * context},
            "notes": notes, "estimated_temp_bytes": estimate}
