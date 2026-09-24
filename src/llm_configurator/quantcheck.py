"""Compression-loss check: how far a smaller model file drifts from a reference file.

Think of it like comparing a low-quality JPEG with the original photo, word by word. The
reference model's predictions are saved once, then each compressed file reads the same text
and llama-perplexity reports how often and how much its predictions differ.

Honesty: every number comes from llama-perplexity's own output. Anything it did not print
stays None. The temporary predictions file is estimated before it is written, checked against
free disk space, and always deleted, also on errors and cancel. Verdicts are rough guidance.
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

# Mean KL divergence bands. Anchors from llama.cpp's LLaMA 3 8B scoreboard against FP16
# (tools/perplexity/README.md): Q8_0 0.0014, Q6_K 0.0055, Q5_K_M 0.011, Q4_K_M 0.031,
# Q3_K_M 0.10, Q2_K 0.33-0.45, IQ1 1.4+. Rough guidance, not a guarantee.
KLD_BANDS = [(0.01, "negligible"), (0.05, "small"), (0.15, "moderate"), (0.5, "large"), (math.inf, "severe")]
# Fallback when only "Same top p" is known: share of positions where the top word differs.
# Same scoreboard: Q8_0 2.3%, Q6_K 4.0%, Q4_K_M 8.1%, Q2_K 28.9%.
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
    """
    result = {key: None for key in RESULT_KEYS}
    result.update(max_kld=None, kld_999=None, ppl_ratio=None, rms_delta_p=None, mean_kld_uncertainty=None,
                  same_top_p_uncertainty=None, final_ppl=None, chunks_done=None)
    last_row = None
    for raw in _ANSI.sub("", text or "").replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line:
            continue
        for key, pattern in _COMPILED:
            match = pattern.search(line)
            if match:
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
    if last_row:
        result["chunks_done"] = int(last_row[0])
        if len(last_row) == 11:
            fill = {"ppl": last_row[1], "mean_kld": last_row[5], "same_top_p": last_row[9]}
        else:
            fill = {"ppl": last_row[1], "mean_kld": last_row[4],
                    "same_top_p": None if last_row[6] is None else last_row[6] * 100}
        for key, value in fill.items():
            if result[key] is None:
                result[key] = value
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
    """Size of the --kl-divergence-base file, following llama.cpp's writer: a small header, the
    tokens, then per chunk the second half of the window at 2 bytes per vocabulary entry (+4)."""
    scored = context - 1 - context // 2
    per_row = (2 * ((vocab_size + 1) // 2) + 4) * 2
    return 16 + context * chunks * 4 + chunks * scored * per_row


def _vocab_size(path):
    """Vocabulary size from the GGUF header when the discover workstream's reader can tell us."""
    try:
        from . import gguf
        info = gguf.read_metadata(path)
        metadata = info.get("metadata") or {}
        for key in [f"{info.get('architecture')}.vocab_size", "tokenizer.ggml.vocab_size"]:
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                return value
    except Exception:  # missing module or unreadable header: fall back to a safe upper bound
        pass
    return None


def _default_chunks(context, corpus_bytes):
    available = int(corpus_bytes / CHARS_PER_TOKEN) // context
    wanted = max(1, round(TARGET_TOKENS / context))
    return max(1, min(wanted, available, 64))


def _common_args(config, context):
    """Flags llama-perplexity shares with llama-server: GPU layers, device, threads, MoE offload.
    Server-only flags (host, port, alias, jinja, parallel, draft) and KV-cache compression are left
    out so the check measures the weights alone. Batch equals the window: one sequence at a time."""
    args = ["-c", str(context), "-b", str(context)]
    env = None
    if config:
        from . import launch
        c = launch.normalize({k: v for k, v in config.items() if k in launch.DEFAULTS})
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
        raise ValueError(f"llama-perplexity took longer than {max(1, timeout // 60)} minutes and was stopped. "
                         "Try fewer chunks or a smaller context.") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=30)
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
    return None


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
            raise ValueError(f"Model file not found: {path}. Download it first.")
    corpus = Path(corpus_path) if corpus_path else CORPUS_PATH
    if not corpus.is_file():
        raise ValueError(f"Sample text not found: {corpus}")
    corpus_bytes = corpus.stat().st_size
    if corpus_bytes / 3 < 2 * context:   # generous 3 characters per token: clearly too short
        raise ValueError("The sample text is too short for this window size. Use a smaller context.")
    chunks = chunks or _default_chunks(context, corpus_bytes)
    reference_label = reference_label or quant_label(reference_path)
    common, env = _common_args(config, context)

    known_vocab = vocab_size or _vocab_size(reference_path)
    estimate = estimate_logits_bytes(known_vocab or ASSUMED_VOCAB, context, chunks)
    temp_root = Path(work_dir) if work_dir else Path(tempfile.gettempdir())
    free = shutil.disk_usage(temp_root).free
    if free < estimate + DISK_MARGIN_BYTES:
        raise ValueError(f"The check needs about {estimate / GIB:.1f} GB of temporary space in {temp_root}, "
                         f"but only {free / GIB:.1f} GB is free. Free up some space or use fewer chunks.")
    notes = [f"Temporary predictions file: about {estimate / GIB:.2f} GB"
             + ("" if known_vocab else " at most (vocabulary size unknown, assumed the largest common one)")
             + f" in {temp_root}; it is deleted afterwards.",
             f"Reads about {chunks * context} tokens of sample text in {chunks} windows of {context}; "
             "a short sample gives a quick, rough answer.",
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
                report(stage, label, f"{label}: window {chunk} of {state['chunks']}", done)
        return on_line

    temp_dir = tempfile.mkdtemp(prefix="llmc-quantcheck-", dir=work_dir)
    logits = Path(temp_dir) / "reference.kld"
    results, reference = {}, {"label": reference_label, "path": str(reference_path), "ppl": None,
                              "logits_bytes": None}
    try:
        check_cancel(cancel)
        report("reference", reference_label, f"Saving {reference_label}'s predictions on the sample text")
        argv = command + ["-m", str(reference_path), "-f", str(corpus), "--kl-divergence-base", str(logits),
                          "--chunks", str(chunks)] + common
        code, output = run(argv, env=env, timeout=timeout, cancel=cancel, on_line=on_line_for("reference", reference_label))
        check_cancel(cancel)
        if code != 0 or not logits.is_file() or logits.stat().st_size <= 16:
            reason = _failure(output) or f"llama-perplexity stopped without saving predictions:\n{_tail(output)}"
            raise ValueError(f"The reference file {reference_label} could not be measured. {reason}")
        parsed = parse_kld_output(output)
        reference.update(ppl=parsed["final_ppl"] or parsed["ppl"], logits_bytes=logits.stat().st_size)
        state["done"] += chunks
        for label, path in paths.items():
            check_cancel(cancel)
            report("candidate", label, f"Comparing {label} with {reference_label}")
            argv = command + ["-m", str(path), "-f", str(corpus), "--kl-divergence-base", str(logits),
                              "--kl-divergence"] + common
            code, output = run(argv, env=env, timeout=timeout, cancel=cancel, on_line=on_line_for("candidate", label))
            check_cancel(cancel)
            parsed = {key: value for key, value in parse_kld_output(output).items() if key != "final_ppl"}
            if parsed["ppl_base"] is None:
                parsed["ppl_base"] = reference["ppl"]
            error = None
            if code != 0 or (parsed["mean_kld"] is None and parsed["same_top_p"] is None):
                error = _failure(output) or f"llama-perplexity gave no comparison:\n{_tail(output)}"
            summary = interpret(parsed, reference_label)
            if error:
                summary = {"plain": f"Could not compare {label} with {reference_label}. {error}", "verdict": None}
            results[label] = {**parsed, **summary, "path": str(path), "error": error}
            state["done"] += chunks
        report("done", None, "Compression check finished")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return {"reference": reference, "results": results,
            "corpus": {"path": str(corpus), "bytes": corpus_bytes, "context": context, "chunks": chunks,
                       "tokens": chunks * context},
            "notes": notes, "estimated_temp_bytes": estimate}
