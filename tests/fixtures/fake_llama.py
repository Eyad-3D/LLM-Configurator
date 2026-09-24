#!/usr/bin/env python3
"""A faithful stand-in for llama.cpp's llama-server, llama-bench, llama-perplexity and llama-cli.

Standard library only, cross-platform. Run as `python fake_llama.py --as server|bench|perplexity|cli <real flags>`,
or copy/symlink it under a name containing llama-server / llama-bench / llama-perplexity / llama-cli.
Output formats, flag names and error messages follow llama.cpp master (checked 2026-09); speeds are
deterministic and depend on the flags, so a tuner has a real optimum to find.

Environment knobs (all optional):
  FAKE_LLAMA_FAIL=oom|arch|port|corrupt|backend   fail like llama.cpp does (backend only warns and runs on CPU)
  FAKE_LLAMA_TPS=<float>            generation speed at the best settings (default 40)
  FAKE_LLAMA_LOAD_SECONDS=<float>   server loading time with /health returning 503 (default 0.2)
  FAKE_LLAMA_MAX_GPU_LAYERS=<int>   more GPU layers than this runs out of GPU memory
  FAKE_LLAMA_BEST_THREADS=<int>     thread count with the best speed (default 8)
  FAKE_LLAMA_SLEEP=<float>          really sleep for this fraction of the reported compute time (default 0)
  FAKE_LLAMA_REPLY=<text>           reply with exactly this text (to test reply checks)
  FAKE_LLAMA_REASONING=1            emit a thinking section (reasoning_content) unless enable_thinking is false
  FAKE_LLAMA_GPU=none               a CPU-only build: no GPU devices, the real "no usable GPU" warnings, CPU speeds
  FAKE_LLAMA_BUILD=<int>, FAKE_LLAMA_COMMIT=<hex>, FAKE_LLAMA_VERSION_STYLE=new|classic (default new, as current
  llama.cpp prints it: "version: 0.1.0-dev (build N, commit HASH)"; classic is "version: N (HASH)")
"""
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import operator
import os
from pathlib import Path
import random
import re
import struct
import sys
import threading
import time
import zlib

BUILD = int(os.environ.get("FAKE_LLAMA_BUILD", "6512"))
COMMIT = os.environ.get("FAKE_LLAMA_COMMIT", "fa4ec0d")
COMPILER = "GNU 13.3.0"
# Current llama.cpp prints "<CMAKE_SYSTEM_NAME> <CMAKE_SYSTEM_PROCESSOR>" as the build target.
TARGET = {"nt": "Windows AMD64"}.get(os.name, "Darwin arm64" if sys.platform == "darwin" else "Linux x86_64")
_STARTED = time.monotonic()
KNOWN_ARCHITECTURES = {"llama", "llama4", "qwen2", "qwen2moe", "qwen3", "qwen3moe", "gemma", "gemma2", "gemma3",
                       "gemma3n", "phi3", "granite", "granitemoe", "olmo2", "gpt-oss", "mistral3", "deepseek2",
                       "command-r", "starcoder2", "falcon", "glm4", "glm4moe", "smollm3"}
CACHE_TYPES = ["f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"]
QUANT_KLD = [("IQ1", 0.9), ("Q2_K", 0.25), ("IQ2", 0.3), ("Q3_K", 0.09), ("IQ3", 0.1), ("IQ4", 0.035), ("Q4_0", 0.05),
             ("Q4_K", 0.03), ("MXFP4", 0.03), ("Q5", 0.012), ("Q6_K", 0.005), ("Q8_0", 0.0015), ("BF16", 0.0002),
             ("F16", 0.0002), ("F32", 0.0)]


def log(line=""):
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def stamp(level, line):
    """A log line with llama.cpp's elapsed-time prefix, for example "0.00.034.765 I srv  llama_server: ..."."""
    log(f"{elapsed()} {level} {line}")


def elapsed():
    """llama.cpp's log prefix: minutes.seconds.milliseconds.microseconds since start."""
    us = int((time.monotonic() - _STARTED) * 1e6)
    return f"{us // 60000000}.{us // 1000000 % 60:02d}.{us // 1000 % 1000:03d}.{us % 1000:03d}"


def cpu_only_build():
    return os.environ.get("FAKE_LLAMA_GPU", "").lower() == "none"


def fail(lines, code=1):
    for line in lines:
        log(line)
    sys.exit(code)


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


# ---------------------------------------------------------------- tiny GGUF reader (enough for the header)

def read_gguf(path):
    """Returns ({"architecture", "layers", "experts", "name"}, error_lines). Missing/short files follow llama.cpp."""
    path = str(path)
    if not os.path.exists(path):
        return None, [f"gguf_init_from_file: failed to open GGUF file '{path}' (No such file or directory)",
                      f"llama_model_load: error loading model: llama_model_loader: failed to load model from {path}"]
    with open(path, "rb") as handle:
        head = handle.read(1 << 16)
    info = {"architecture": "llama", "layers": 32, "experts": 0, "name": Path(path).stem, "context_length": 32768}
    # Leniency for tests: an empty file counts as a small llama model.
    if len(head) == 0:
        return info, []
    if len(head) < 4:
        return None, ["gguf_init_from_reader: failed to read magic",
                      f"llama_model_load: error loading model: llama_model_loader: failed to load model from {path}"]
    if head[:4] != b"GGUF":
        magic = head[:4].decode("latin-1").encode("unicode_escape").decode()
        return None, [f"gguf_init_from_reader: invalid magic characters: '{magic}', expected 'GGUF'",
                      f"llama_model_load: error loading model: llama_model_loader: failed to load model from {path}"]
    try:
        metadata = _gguf_kv(head)
    except (struct.error, ValueError, UnicodeDecodeError):
        return None, ["gguf_init_from_reader: failed to read key-value pairs",
                      f"llama_model_load: error loading model: llama_model_loader: failed to load model from {path}"]
    arch = metadata.get("general.architecture", "llama")
    info.update(architecture=arch, name=metadata.get("general.name", info["name"]),
                layers=int(metadata.get(f"{arch}.block_count", 32)), experts=int(metadata.get(f"{arch}.expert_count", 0)),
                context_length=int(metadata.get(f"{arch}.context_length", 32768)))
    return info, []


_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _gguf_kv(data):
    version, _tensors, count = struct.unpack_from("<IQQ", data, 4)
    if version not in (2, 3):
        raise ValueError("version")
    offset, result = 24, {}

    def string():
        nonlocal offset
        (length,) = struct.unpack_from("<Q", data, offset)
        offset += 8
        value = data[offset:offset + length].decode("utf-8")
        offset += length
        return value

    def value(kind):
        nonlocal offset
        if kind == 8:
            return string()
        if kind == 9:
            (inner, length) = struct.unpack_from("<IQ", data, offset)
            offset += 12
            return [value(inner) for _ in range(length)]
        code = _SCALARS[kind]
        (item,) = struct.unpack_from("<" + code, data, offset)
        offset += struct.calcsize(code)
        return item

    for _ in range(count):
        key = string()
        (kind,) = struct.unpack_from("<I", data, offset)
        offset += 4
        result[key] = value(kind)
    return result


def write_gguf(path, architecture="llama", layers=32, experts=0, name=None, context_length=32768):
    """Minimal valid GGUF v3 header with metadata and no tensors (used by tests via fixtures.write_fake_gguf)."""
    items = [("general.architecture", 8, architecture), ("general.name", 8, name or Path(path).stem),
             (f"{architecture}.block_count", 4, layers), (f"{architecture}.context_length", 4, context_length)]
    if experts:
        items += [(f"{architecture}.expert_count", 4, experts), (f"{architecture}.expert_used_count", 4, min(8, experts))]
    out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, 0, len(items)))
    for key, kind, value in items:
        raw = key.encode()
        out += struct.pack("<Q", len(raw)) + raw + struct.pack("<I", kind)
        if kind == 8:
            encoded = value.encode()
            out += struct.pack("<Q", len(encoded)) + encoded
        else:
            out += struct.pack("<I", value)
    Path(path).write_bytes(bytes(out))
    return Path(path)


def model_error(info_errors, extra=()):
    info, errors = info_errors
    if errors:
        return errors + list(extra)
    if os.environ.get("FAKE_LLAMA_FAIL") == "corrupt":
        return ["llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within the file "
                "bounds, model is corrupted or incomplete"] + list(extra)
    arch = "made-up-arch" if os.environ.get("FAKE_LLAMA_FAIL") == "arch" else info["architecture"]
    if arch not in KNOWN_ARCHITECTURES:
        return [f"llama_model_load: error loading model: unknown model architecture: '{arch}'"] + list(extra)
    return []


# ---------------------------------------------------------------- deterministic speed model

def speeds(s, info, depth=0):
    """(prompt tokens/s, generated tokens/s). Best: threads=BEST, flash attention on, all layers on GPU,
    ubatch 512, batch 2048, f16 KV, no MoE experts on CPU, empty context."""
    best = max(1, int(env_float("FAKE_LLAMA_BEST_THREADS", 8)))
    tg = env_float("FAKE_LLAMA_TPS", 40.0)
    pp = tg * 12
    t = max(1, s["threads"])
    thread = 0.35 + 0.65 * t / best if t <= best else max(0.5, 1 - 0.04 * (t - best))
    layers = max(1, info["layers"])
    offloaded = min(1.0, max(0, s["ngl"]) / layers) if s["ngl"] >= 0 else 1.0
    if s.get("cpu_only"):
        offloaded = 0.0
    gpu_tg = 1 / ((1 - offloaded) / 0.3 + offloaded)
    gpu_pp = 1 / ((1 - offloaded) / 0.1 + offloaded)
    fa_on = s["fa"] != "off"
    ub = max(1, min(s["ubatch"], s["batch"]))
    ubatch = max(0.5, 1 - 0.15 * abs(math.log2(ub / 512)))
    batch = max(0.7, 1 - 0.05 * abs(math.log2(max(1, s["batch"]) / 2048)))
    kv = {"f16": 1.0, "f32": 0.97, "bf16": 1.0, "q8_0": 0.97}.get(s["ctk"], 0.93) * (1.0 if s["ctv"] == "f16" else 0.98)
    moe = 1 / (1 + 0.6 * min(s["ncmoe"], layers) / layers) if info["experts"] else 1.0
    deep = 1 / (1 + depth / (16384 if fa_on else 6144))
    ctx = 1 / (1 + s.get("ctx", 0) / 1048576)
    tg_tps = tg * thread * gpu_tg * (1.0 if fa_on else 0.9) * kv * moe * deep * ctx
    pp_tps = pp * thread * gpu_pp * (1.0 if fa_on else 0.85) * ubatch * batch * moe * deep * ctx
    return round(pp_tps, 4), round(tg_tps, 4)


def gpu_oom(s, layers=32):
    if os.environ.get("FAKE_LLAMA_FAIL") == "oom":
        return True
    limit = os.environ.get("FAKE_LLAMA_MAX_GPU_LAYERS")
    offloaded = layers + 1 if s["ngl"] < 0 else s["ngl"]
    return limit is not None and not s.get("cpu_only") and offloaded > int(limit)


def oom_lines(s):
    if s["ngl"] != 0 and not s.get("cpu_only"):
        return ["ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory",
                "alloc_tensor_range: failed to allocate CUDA0 buffer of size 9663676416",
                "llama_model_load: error loading model: unable to allocate CUDA0 buffer"]
    return ["ggml_aligned_malloc: insufficient memory (attempted to allocate 9216.00 MB)",
            "ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 9663676416",
            "llama_model_load: error loading model: unable to allocate CPU buffer"]


def pause(seconds):
    scale = env_float("FAKE_LLAMA_SLEEP", 0.0)
    if scale > 0:
        time.sleep(min(seconds * scale, 60))


# ---------------------------------------------------------------- tokenizer and answers

TOKEN = re.compile(r"<\|[a-z_]+\|>|\w+|[^\w\s]|\s+")


def tokenize(text):
    pieces = TOKEN.findall(text)
    return [(3 + zlib.crc32(p.encode()) % 151000 if not p.startswith("<|") else 151643 + len(p)) for p in pieces], pieces


def template(messages, thinking=True):
    text = ""
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        text += f"<|im_start|>{m.get('role', 'user')}\n{content or ''}<|im_end|>\n"
    return text + "<|im_start|>assistant\n" + ("" if thinking else "<think>\n\n</think>\n\n")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _calc(node):
    if isinstance(node, ast.Expression):
        return _calc(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.left), _calc(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_calc(node.operand))
    raise ValueError("unsupported")


def arithmetic(text):
    text = text.replace("×", "*").replace("÷", "/")
    text = re.sub(r"(?<=\d)\s*[xX]\s*(?=\d)", "*", text)
    for candidate in sorted(re.findall(r"[\d\s+\-*/().]+", text), key=len, reverse=True):
        candidate = candidate.strip().rstrip(".")
        if re.search(r"\d\s*[-+*/]\s*[\d(]", candidate):
            try:
                value = _calc(ast.parse(candidate, mode="eval"))
            except (SyntaxError, ValueError, ZeroDivisionError):
                continue
            return str(int(value)) if float(value).is_integer() else f"{value:.4g}"
    return None


NEEDLE = re.compile(r"(The (?:secret|magic|special|hidden) (?:code|number|word|password|key|city) is ([^\s.,;!?]+))", re.I)
CAPITALS = {"france": "Paris", "japan": "Tokyo", "germany": "Berlin", "italy": "Rome", "spain": "Madrid",
            "canada": "Ottawa", "australia": "Canberra", "egypt": "Cairo"}
FILLER = ["Local models run best when the whole model fits in fast memory.",
          "A shorter context leaves more room for the model itself.",
          "Smaller downloads trade a little quality for a lot of speed.",
          "The answer depends on how much memory your computer has free.",
          "Testing on your own computer is the only way to be sure."]


def answer(prompt_text, last_user):
    forced = os.environ.get("FAKE_LLAMA_REPLY")
    if forced is not None:
        return forced
    needle = NEEDLE.search(prompt_text)
    if needle:
        return needle.group(1) + "."
    math_answer = arithmetic(last_user)
    if math_answer is not None:
        return math_answer
    capital = re.search(r"capital of (\w+)", last_user, re.I)
    if capital and capital.group(1).lower() in CAPITALS:
        return CAPITALS[capital.group(1).lower()]
    echo = (re.search(r"(?:reply|answer|respond) with (?:only |just |exactly )?the word [\"']?([\w-]+)", last_user, re.I)
            or re.search(r"(?:reply|answer|respond|repeat|say)[^\"']*[\"']([^\"']+)[\"']", last_user, re.I))
    if echo:
        return echo.group(1)
    return FILLER[zlib.crc32(last_user.encode()) % len(FILLER)]


def generate(prompt_text, last_user, n_predict, stop=None, reasoning=False):
    """(content, reasoning_content, stop_type, stopping_word, n_tokens)"""
    text = answer(prompt_text, last_user)
    thinking = ("Let me think about this. " + text) if reasoning else None
    stop = [stop] if isinstance(stop, str) else (stop or [])
    stop_type, word = "eos", ""
    for s in stop:
        if s and s in text:
            text, stop_type, word = text[:text.index(s)], "word", s
    ids, pieces = tokenize(text)
    reasoning_n = len(tokenize(thinking)[0]) if thinking else 0
    if n_predict is not None and n_predict >= 0 and len(ids) + reasoning_n > n_predict:
        keep = max(0, n_predict - reasoning_n)
        text, stop_type = "".join(pieces[:keep]), "limit"
        ids = ids[:keep]
    return text, thinking, stop_type, word, len(ids) + reasoning_n


# ---------------------------------------------------------------- argument parsing

REMOVED = {"--draft": "use --spec-draft-n-max or --spec-ngram-mod-n-max",
           "--draft-n": "use --spec-draft-n-max or --spec-ngram-mod-n-max",
           "--draft-max": "use --spec-draft-n-max or --spec-ngram-mod-n-max",
           "--draft-min": "use --spec-draft-n-min or --spec-ngram-mod-n-min",
           "--draft-n-min": "use --spec-draft-n-min or --spec-ngram-mod-n-min",
           "--spec-ngram-size-n": "use the respective --spec-ngram-*-size-n or --spec-ngram-mod-n-match"}


def parse(args, value_flags, bool_flags, program):
    """{canonical: value}. Unknown flags fail the way llama.cpp's common arg parser does."""
    options, i = {}, 0
    while i < len(args):
        arg = args[i]
        if arg in REMOVED and program != "llama-bench":
            fail([f"error while handling argument \"{arg}\": the argument has been removed. {REMOVED[arg]}", "",
                  "usage:", f"{arg} N    the argument has been removed. {REMOVED[arg]}", "", "",
                  "to show complete usage, run with -h"])
        name = value_flags.get(arg) or bool_flags.get(arg)
        if name is None:
            fail([f"error: invalid argument: {arg}"])
        if arg in bool_flags:
            options[name] = True
            i += 1
            continue
        if i + 1 >= len(args):
            fail([f"error while handling argument \"{arg}\": expected value for argument", "",
                  f"usage: {program} [options]"])
        options[name] = args[i + 1]
        i += 2
    return options


def as_int(options, name, default, flag):
    try:
        return int(options.get(name, default))
    except ValueError:
        fail([f"error while handling argument \"{flag}\": stoi", ""])


def fa_value(text, flag="-fa"):
    value = {"on": "on", "1": "on", "true": "on", "enabled": "on", "off": "off", "0": "off", "false": "off",
             "disabled": "off", "auto": "auto", "-1": "auto"}.get(str(text).lower())
    if value is None:
        fail([f"error while handling argument \"{flag}\": unknown value for --flash-attn: '{text}'", ""])
    return value


def version():
    if os.environ.get("FAKE_LLAMA_VERSION_STYLE", "new") == "classic":
        log(f"version: {BUILD} ({COMMIT})")
        log(f"built with cc (GCC) 13.2.0 for {'x86_64-w64-mingw32' if os.name == 'nt' else 'x86_64-linux-gnu'}")
    else:
        log(f"version: 0.1.0-dev (build {BUILD}, commit {COMMIT})")
        log(f"built with {COMPILER} for {TARGET}")
    sys.exit(0)


def no_gpu():
    return cpu_only_build() or os.environ.get("FAKE_LLAMA_FAIL") == "backend"


def list_devices():
    """`--list-devices` (llama-server, llama-perplexity, llama-cli, llama-bench): stdout, exit 0."""
    print("Available devices:")
    if no_gpu():
        print("  (none)")
    else:
        print("  CUDA0: Fake GPU 24GB (24564 MiB, 23512 MiB free)")
    sys.stdout.flush()
    sys.exit(0)


def gpu_warnings(args):
    """Real llama.cpp prints these while parsing -ngl on a build or machine without a usable GPU, before any
    other argument error, and whatever the -ngl value is."""
    if no_gpu() and any(a in ("-ngl", "--gpu-layers", "--n-gpu-layers") for a in args):
        log("warning: no usable GPU found, --gpu-layers option will be ignored")
        log("warning: one possible reason is that llama.cpp was compiled without GPU support")
        log("warning: consult docs/build.md for compilation instructions")


def backend_banner(s):
    if no_gpu():
        if os.environ.get("FAKE_LLAMA_FAIL") == "backend":
            log("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected")
        s["cpu_only"] = True
        return
    log("ggml_cuda_init: found 1 CUDA devices:")
    log("  Device 0: Fake GPU 24GB, compute capability 8.9, VMM: yes")


def check_device(options, s):
    device = options.get("device")
    if device and device.lower() != "none":
        if no_gpu() or not re.fullmatch(r"(CUDA|Vulkan|ROCm|Metal|SYCL)\d*(,\S+)*", device):
            fail([f"error while handling argument \"-dev\": invalid device: {device}", ""])
    if device and device.lower() == "none":
        s["cpu_only"] = True


COMMON_VALUES = {"-m": "model", "--model": "model", "-c": "ctx", "--ctx-size": "ctx", "-ngl": "ngl",
                 "--gpu-layers": "ngl", "--n-gpu-layers": "ngl", "-t": "threads", "--threads": "threads",
                 "-b": "batch", "--batch-size": "batch", "-ub": "ubatch", "--ubatch-size": "ubatch",
                 "-fa": "fa", "--flash-attn": "fa", "-ctk": "ctk", "--cache-type-k": "ctk", "-ctv": "ctv",
                 "--cache-type-v": "ctv", "--n-cpu-moe": "ncmoe", "-ncmoe": "ncmoe", "-dev": "device",
                 "--device": "device", "-s": "seed", "--seed": "seed", "-md": "draft", "--model-draft": "draft",
                 "--spec-draft-model": "draft", "--spec-draft-n-max": "draft_max", "--spec-draft-n-min": "draft_min",
                 "--spec-type": "spec_type", "-lm": "load_mode", "--load-mode": "load_mode", "-fit": "fit",
                 "--fit": "fit", "-lv": "verbosity", "--verbosity": "verbosity", "--log-verbosity": "verbosity",
                 "-tb": "threads_batch",
                 "--threads-batch": "threads_batch", "-sm": "split_mode", "--split-mode": "split_mode",
                 "-mg": "main_gpu", "--main-gpu": "main_gpu", "-ts": "tensor_split", "--tensor-split": "tensor_split",
                 "--log-file": "log_file", "-ot": "override_tensor", "--override-tensor": "override_tensor",
                 "--temp": "temp", "-n": "n_predict", "--n-predict": "n_predict"}
COMMON_BOOLS = {"--mlock": "mlock", "--no-mmap": "no_mmap", "-v": "verbose", "--verbose": "verbose",
                "--no-warmup": "no_warmup", "--log-disable": "log_disable", "-cb": "cont_batching",
                "--cont-batching": "cont_batching", "--no-kv-offload": "no_kv_offload", "-nkvo": "no_kv_offload"}


def settings(options, default_threads=None):
    s = {"threads": as_int(options, "threads", default_threads or 4, "-t"), "ngl": as_int(options, "ngl", -1, "-ngl"),
         "batch": as_int(options, "batch", 2048, "-b"), "ubatch": as_int(options, "ubatch", 512, "-ub"),
         "fa": fa_value(options.get("fa", "auto")), "ctk": options.get("ctk", "f16"), "ctv": options.get("ctv", "f16"),
         "ncmoe": as_int(options, "ncmoe", 0, "--n-cpu-moe"), "ctx": as_int(options, "ctx", 4096, "-c")}
    for key, flag in [("ctk", "-ctk"), ("ctv", "-ctv")]:
        if s[key] not in CACHE_TYPES:
            fail([f"error while handling argument \"{flag}\": Unsupported cache type: {s[key]}", ""])
    return s


# ---------------------------------------------------------------- llama-server

SERVER_VALUES = {**COMMON_VALUES, "--host": "host", "--port": "port", "-np": "parallel", "--parallel": "parallel",
                 "--alias": "alias", "-a": "alias", "-to": "timeout", "--timeout": "timeout",
                 "--api-key": "api_key", "--reasoning-format": "reasoning_format", "--chat-template": "chat_template",
                 "--cors-origins": "cors_origins",
                 "--threads-http": "threads_http", "--slot-save-path": "slot_save_path", "-kvu": "kv_unified_value"}
SERVER_BOOLS = {**COMMON_BOOLS, "--jinja": "jinja", "--no-jinja": "no_jinja", "--metrics": "metrics",
                "--no-webui": "no_webui", "--slots": "slots", "--no-slots": "no_slots", "--kv-unified": "kv_unified",
                "--embedding": "embedding", "--embeddings": "embedding"}


class State:
    def __init__(self, options, s, info):
        self.options, self.s, self.info = options, s, info
        self.loading = True
        self.lock = threading.Lock()
        self.cached = []
        parallel = max(1, as_int(options, "parallel", 1, "-np"))
        self.n_ctx_slot = max(1, s["ctx"] // parallel)
        self.parallel = parallel
        self.model_path = options["model"]
        self.alias = options.get("alias") or self.model_path
        self.cors = options.get("cors_origins") or os.environ.get("LLAMA_ARG_CORS_ORIGINS") or "*"
        self.api_key = options.get("api_key") or os.environ.get("LLAMA_API_KEY")
        types = [t for t in (options.get("spec_type") or "none").split(",") if t != "none"]
        self.speculative = bool(options.get("draft")) and "draft-simple" in types
        self.draft_max = as_int(options, "draft_max", 3, "--spec-draft-n-max")


def server_log(func, message, level="I"):
    stamp(level, f"srv  {func[-12:]:>12}: {message}")


SPEC_TYPES = {"none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash", "draft-dspark", "ngram-simple",
              "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache"}


def timings(state, prompt_ids, cache_prompt, n_predicted):
    cached = 0
    if cache_prompt:
        for a, b in zip(state.cached, prompt_ids):
            if a != b:
                break
            cached += 1
        cached = min(cached, len(prompt_ids) - 1)
    state.cached = list(prompt_ids)
    prompt_n = len(prompt_ids) - cached
    pp, tg = speeds(state.s, state.info, depth=len(prompt_ids))
    prompt_ms = prompt_n / pp * 1000
    predicted_ms = n_predicted / tg * 1000
    pause((prompt_ms + predicted_ms) / 1000)
    extra = {}
    if state.speculative and n_predicted:
        drafted = min(n_predicted, state.draft_max * max(1, n_predicted // (state.draft_max + 1)))
        extra = {"draft_n": drafted, "draft_n_accepted": drafted}
    return {"cache_n": cached, "prompt_n": prompt_n, "prompt_ms": round(prompt_ms, 3),
            "prompt_per_token_ms": round(prompt_ms / prompt_n, 6) if prompt_n else 0.0,
            "prompt_per_second": round(prompt_n / prompt_ms * 1000, 6) if prompt_ms else 0.0,
            "predicted_n": n_predicted, "predicted_ms": round(predicted_ms, 3),
            "predicted_per_token_ms": round(predicted_ms / n_predicted, 6) if n_predicted else 0.0,
            "predicted_per_second": round(n_predicted / predicted_ms * 1000, 6) if predicted_ms else 0.0, **extra}


def error_body(code, message, kind, **extra):
    return code, {"error": {"code": code, "message": message, "type": kind, **extra}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "llama.cpp"
    state = None

    def log_message(self, fmt, *args):
        pass

    def send_json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def cors_headers(self):
        """--cors-origins / LLAMA_ARG_CORS_ORIGINS: '*' (default) echoes any Origin, 'localhost' only local pages."""
        origin = self.headers.get("Origin")
        if not origin:
            return
        allowed = self.state.cors if self.state else "*"
        host = re.sub(r"^[a-z]+://", "", origin.lower()).split("/")[0].rsplit(":", 1)[0].strip("[]")
        if allowed == "*" or (allowed == "localhost" and host in ("localhost", "127.0.0.1", "::1")) \
                or origin in allowed.split(","):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Access-Control-Allow-Credentials", "true")

    def authorised(self, path):
        key = self.state.api_key if self.state else None
        if not key or path in ("/health", "/v1/health", "/models", "/v1/models"):
            return True
        if self.headers.get("Authorization") == f"Bearer {key}":
            return True
        self.send_json(*error_body(401, "Invalid API Key", "authentication_error"))
        return False

    def body(self):
        length = int(self.headers.get("Content-Length") or 0)
        try:
            data = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(data, dict):
                raise ValueError
            return data
        except ValueError:
            return None

    def gate(self):
        if self.state.loading:
            self.send_json(*error_body(503, "Loading model", "unavailable_error"))
            return False
        return True

    def do_GET(self):
        path = self.path.split("?")[0]
        if not self.gate() or not self.authorised(path):
            return
        state = self.state
        if path in ("/health", "/v1/health"):
            return self.send_json(200, {"status": "ok"})
        if path in ("/v1/models", "/models"):
            name, ftype = state.alias, quant_name(state.model_path)
            return self.send_json(200, {
                "models": [{"name": name, "model": name, "modified_at": "", "size": "", "digest": "", "type": "model",
                            "description": "", "tags": [""], "capabilities": ["completion"], "parameters": "",
                            "details": {"parent_model": "", "format": "gguf", "family": "", "families": [""],
                                        "parameter_size": "", "quantization_level": ""}}],
                "object": "list",
                "data": [{"id": name, "aliases": [name], "tags": [], "object": "model", "created": int(time.time()),
                          "owned_by": "llamacpp",
                          "meta": {"vocab_type": 2, "n_vocab": 151936, "n_ctx": state.s["ctx"],
                                   "n_ctx_train": state.info["context_length"], "n_embd": 4096,
                                   "n_params": 8030261248, "size": 4920733696, "ftype": ftype}}]})
        if path == "/props":
            return self.send_json(200, {
                "default_generation_settings": {"params": {"seed": 4294967295, "temperature": 0.8, "top_k": 40,
                                                           "top_p": 0.95, "min_p": 0.05, "n_predict": -1,
                                                           "speculative.types": state.options.get("spec_type", "none")},
                                                "n_ctx": state.n_ctx_slot},
                "total_slots": state.parallel, "model_alias": state.alias, "model_ftype": quant_name(state.model_path),
                "model_path": state.model_path, "modalities": {"vision": False, "video": False, "audio": False},
                "endpoint_slots": True, "endpoint_props": False, "endpoint_metrics": False,
                "chat_template_caps": {"supports_preserve_reasoning": False, "supports_reasoning_effort": False,
                                       "supports_system_role": True, "supports_tools": False},
                "build_info": f"b{BUILD}-{COMMIT}"})
        self.send_json(*error_body(404, "File Not Found", "not_found_error"))

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        path = self.path.split("?")[0]
        if not self.gate() or not self.authorised(path):
            return
        body = self.body()
        if body is None:
            return self.send_json(*error_body(400, "Failed to parse JSON", "invalid_request_error"))
        routes = {"/v1/chat/completions": self.chat, "/chat/completions": self.chat, "/completion": self.completion,
                  "/completions": self.completion, "/tokenize": self.tokenize, "/detokenize": self.detokenize}
        if path not in routes:
            return self.send_json(*error_body(404, "File Not Found", "not_found_error"))
        try:
            routes[path](body)
        except (ValueError, TypeError, KeyError) as error:
            self.send_json(*error_body(400, f"Invalid request: {error}", "invalid_request_error"))

    def too_long(self, n):
        if n > self.state.n_ctx_slot:
            self.send_json(*error_body(400, f"request ({n} tokens) exceeds the available context size "
                                            f"({self.state.n_ctx_slot} tokens), try increasing it",
                                       "exceed_context_size_error", n_prompt_tokens=n, n_ctx=self.state.n_ctx_slot))
            return True
        return False

    def chat(self, body):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ValueError("'messages' is required")
        kwargs = body.get("chat_template_kwargs") or {}
        thinking_allowed = kwargs.get("enable_thinking", True) is not False
        prompt = template(messages, thinking_allowed)
        ids, _ = tokenize(prompt)
        if self.too_long(len(ids)):
            return
        last_user = next((str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"), "")
        n_predict = body.get("n_predict", body.get("max_completion_tokens", body.get("max_tokens", -1)))
        reasoning = os.environ.get("FAKE_LLAMA_REASONING") == "1" and thinking_allowed
        with self.state.lock:
            text, thinking, stop_type, _, n = generate(prompt, last_user, n_predict, body.get("stop"), reasoning)
            t = timings(self.state, ids, body.get("cache_prompt", True), n)
        finish = "stop" if stop_type in ("eos", "word") else "length"
        message = {"role": "assistant", "content": text}
        if thinking:
            message["reasoning_content"] = thinking
        usage = {"completion_tokens": n, "prompt_tokens": len(ids), "total_tokens": n + len(ids),
                 "prompt_tokens_details": {"cached_tokens": t["cache_n"]}}
        created, ident = int(time.time()), "chatcmpl-" + "%032x" % random.getrandbits(128)
        if body.get("stream"):
            return self.stream([{"choices": [{"finish_reason": None, "index": 0, "delta": {"role": "assistant", "content": None}}]},
                                *({"choices": [{"finish_reason": None, "index": 0, "delta": {"content": piece}}]}
                                  for piece in tokenize(text)[1]),
                                {"choices": [{"finish_reason": finish, "index": 0, "delta": {}}]},
                                {"choices": [], "usage": usage, "timings": t}],
                               {"created": created, "id": ident, "model": self.state.alias,
                                "system_fingerprint": f"b{BUILD}-{COMMIT}", "object": "chat.completion.chunk"}, done=True)
        self.send_json(200, {"choices": [{"finish_reason": finish, "index": 0, "message": message}], "created": created,
                             "model": self.state.alias, "system_fingerprint": f"b{BUILD}-{COMMIT}",
                             "object": "chat.completion", "usage": usage, "id": ident, "timings": t})

    def completion(self, body):
        prompt = body.get("prompt")
        if isinstance(prompt, list):
            prompt = "".join(str(p) for p in prompt)
        if not isinstance(prompt, str):
            raise ValueError("'prompt' is required")
        ids, _ = tokenize(prompt)
        if self.too_long(len(ids)):
            return
        with self.state.lock:
            text, _, stop_type, word, n = generate(prompt, prompt, body.get("n_predict", body.get("max_tokens", -1)),
                                                   body.get("stop"))
            t = timings(self.state, ids, body.get("cache_prompt", True), n)
        result = {"index": 0, "content": text, "tokens": [], "id_slot": 0, "stop": True, "model": self.state.alias,
                  "tokens_predicted": n, "tokens_evaluated": len(ids),
                  "generation_settings": {"n_predict": body.get("n_predict", -1), "seed": body.get("seed", -1),
                                          "temperature": body.get("temperature", 0.8)},
                  "prompt": prompt, "has_new_line": "\n" in text, "truncated": False, "stop_type": stop_type,
                  "stopping_word": word, "tokens_cached": len(ids) + n, "timings": t}
        if body.get("stream"):
            return self.stream([*({"index": 0, "content": piece, "tokens": [], "stop": False, "id_slot": -1,
                                   "tokens_predicted": i + 1, "tokens_evaluated": len(ids)}
                                  for i, piece in enumerate(tokenize(text)[1])), {**result, "content": ""}], {}, done=False)
        self.send_json(200, result)

    def stream(self, chunks, common, done):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(b"data: " + json.dumps({**chunk, **common}).encode() + b"\n\n")
        if done:
            self.wfile.write(b"data: [DONE]\n\n")
        self.close_connection = True

    def tokenize(self, body):
        content = body.get("content", "")
        if not isinstance(content, str):
            raise ValueError("'content' must be a string")
        ids, pieces = tokenize(content)
        if body.get("add_special"):
            ids, pieces = [1] + ids, [""] + pieces
        if body.get("with_pieces"):
            return self.send_json(200, {"tokens": [{"id": i, "piece": p} for i, p in zip(ids, pieces)]})
        self.send_json(200, {"tokens": ids})

    def detokenize(self, body):
        self.send_json(200, {"content": "<detokenized:%d tokens>" % len(body.get("tokens") or [])})


def run_server(args):
    gpu_warnings(args)
    options = parse(args, SERVER_VALUES, SERVER_BOOLS, "llama-server")
    if "model" not in options:
        fail(["error: --model is required (or use -hf to download a model)"])
    s = settings(options)
    check_device(options, s)
    for kind in (options.get("spec_type") or "none").split(","):
        if kind not in SPEC_TYPES:
            fail([f"error while handling argument \"--spec-type\": unknown speculative decoding type: {kind}", ""])
    host, port = options.get("host", "127.0.0.1"), as_int(options, "port", 8080, "--port")
    backend_banner(s)
    stamp("I", "cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)")
    state = State(options, s, {"architecture": "llama", "layers": 32, "experts": 0, "context_length": 32768})
    if state.cors == "*" and not state.api_key:
        for line in ["-----------------", "CORS is set to allow all origins ('*') and no API key is set",
                     "this can be a security risk (cross-origin attacks)",
                     "more info: https://github.com/ggml-org/llama.cpp/pull/25655", "-----------------"]:
            server_log("llama_server", line, "W")
    Handler.state = state
    try:
        httpd = ThreadingHTTPServer((host, port), Handler)
    except OSError:
        httpd = None
    if httpd is None or os.environ.get("FAKE_LLAMA_FAIL") == "port":
        server_log("start", f"couldn't bind HTTP server socket, hostname: {host}, port: {port}", "E")
        server_log("operator()", "operator(): cleaning up before exit...")
        server_log("llama_server", "exiting due to HTTP server error", "E")
        sys.exit(1)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    model = options["model"]
    server_log("load_model", f"loading model '{model}'")
    time.sleep(max(0.0, env_float("FAKE_LLAMA_LOAD_SECONDS", 0.2)))

    def load_failed(lines, what="load model"):
        httpd.server_close()
        for line in lines:
            stamp("E", line)
        verb = "create context with" if what == "context" else "load"
        stamp("E", f"cmn  common_init_: failed to {verb} model '{model}'")
        server_log("load_model", f"failed to {'create_context with model' if what == 'context' else 'load model,'} '{model}'", "E")
        server_log("operator()", "operator(): cleaning up before exit...")
        server_log("llama_server", "exiting due to model loading error", "E")
        sys.exit(1)

    loaded = read_gguf(model)
    errors = model_error(loaded)
    if not errors and options.get("draft"):
        errors = model_error(read_gguf(options["draft"]))
    if errors:
        load_failed(errors + ["llama_model_load_from_file_impl: failed to load model"])
    state.info = loaded[0]
    if gpu_oom(s, state.info["layers"]):
        load_failed(oom_lines(s) + ["llama_model_load_from_file_impl: failed to load model"])
    if s["ctv"] != "f16" and s["fa"] == "off":
        load_failed(["llama_init_from_model: quantized V cache requires flash_attn to be enabled"], "context")
    stamp("I", f"cmn          init: llama threadpool init, n_threads = {s['threads']}")
    server_log("load_model", f"initializing, n_slots = {state.parallel}, n_ctx_slot = {state.n_ctx_slot}, "
                             f"kv_unified = '{'true' if options.get('kv_unified') else 'false'}'")
    server_log("llama_server", "model loaded")
    server_log("llama_server", f"listening on http://{host}:{port}")
    state.loading = False
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        server_log("operator()", "operator(): cleaning up before exit...")
        httpd.shutdown()


# ---------------------------------------------------------------- llama-bench

BENCH_LISTS = {"-m": "model", "--model": "model", "-p": "n_prompt", "--n-prompt": "n_prompt", "-n": "n_gen",
               "--n-gen": "n_gen", "-d": "n_depth", "--n-depth": "n_depth", "-ngl": "ngl", "--n-gpu-layers": "ngl",
               "-t": "threads", "--threads": "threads", "-b": "batch", "--batch-size": "batch", "-ub": "ubatch",
               "--ubatch-size": "ubatch", "-fa": "fa", "--flash-attn": "fa", "-ctk": "ctk", "--cache-type-k": "ctk",
               "-ctv": "ctv", "--cache-type-v": "ctv", "-ncmoe": "ncmoe", "--n-cpu-moe": "ncmoe",
               "-sm": "split_mode", "-mg": "main_gpu", "-nkvo": "nkvo", "-dev": "device", "--device": "device",
               "-mmp": "mmap", "--mmap": "mmap", "-lm": "load_mode", "--load-mode": "load_mode", "-ts": "tensor_split"}
BENCH_SINGLE = {"-r": "reps", "--repetitions": "reps", "-o": "output", "--output": "output", "-oe": "output_err",
                "--delay": "delay", "--prio": "prio"}
BENCH_BOOLS = {"-v": "verbose", "--verbose": "verbose", "--progress": "progress", "--no-warmup": "no_warmup",
               "-h": "help", "--help": "help"}
BENCH_ORDER = ["model", "ngl", "ncmoe", "split_mode", "main_gpu", "device", "load_mode", "batch", "ubatch", "ctk",
               "ctv", "nkvo", "fa", "threads", "n_depth"]
BENCH_DEFAULTS = {"n_prompt": "512", "n_gen": "128", "n_depth": "0", "ngl": "-1", "threads": "4", "batch": "2048",
                  "ubatch": "512", "fa": "auto", "ctk": "f16", "ctv": "f16", "ncmoe": "0", "split_mode": "layer",
                  "main_gpu": "0", "nkvo": "0", "device": "auto", "load_mode": "auto", "tensor_split": "0.00"}
LOAD_MODES = {"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"}
RANGED = {"n_prompt", "n_gen", "n_depth", "batch", "ubatch", "threads", "ngl", "ncmoe", "main_gpu"}


def expand(name, text):
    values = []
    for item in str(text).split(","):
        match = re.fullmatch(r"(-?\d+)-(\d+)(?:([+*])(\d+))?", item) if name in RANGED else None
        if match:
            first, last = int(match.group(1)), int(match.group(2))
            op, step = match.group(3) or "+", int(match.group(4) or 1)
            value = first
            while value <= last:
                values.append(str(value))
                value = value + step if op == "+" else value * step
                if op == "*" and step <= 1:
                    break
        elif item != "":
            values.append(item)
    return values


def bench_usage():
    print(f"usage: {sys.argv[0]} [options]\n\noptions:\n  -h, --help\n"
          "  -r, --repetitions <n>                       number of times to repeat each test (default: 5)\n"
          "  -o, --output <csv|json|jsonl|md|sql>        output format printed to stdout (default: md)\n"
          "  --list-devices                              list available devices and exit\n"
          "  -v, --verbose                               verbose output\n")
    sys.stdout.flush()


def bench_fail(*lines):
    """Argument errors: real llama-bench prints its usage on stdout and the error on stderr, exit 1."""
    bench_usage()
    fail(list(lines))


def run_bench(args):
    lists, single, flags, i = {}, {}, set(), 0
    while i < len(args):
        arg = args[i]
        if arg in BENCH_BOOLS:
            if BENCH_BOOLS[arg] == "help":
                bench_usage()
                sys.exit(0)
            flags.add(BENCH_BOOLS[arg])
            i += 1
            continue
        if arg == "-pg" and i + 1 < len(args):
            lists.setdefault("pg", []).append(args[i + 1])
            i += 2
            continue
        if arg not in BENCH_LISTS and arg not in BENCH_SINGLE:
            bench_fail(f"error: invalid parameter for argument: {arg}")
        if i + 1 >= len(args):
            bench_fail(f"error: invalid parameter for argument: {arg}")
        if arg in ("-mmp", "--mmap"):  # deprecated; real llama-bench maps it onto --load-mode
            lists.setdefault("load_mode", []).extend("none" if v == "0" else "mmap" for v in args[i + 1].split(","))
            i += 2
            continue
        if arg in ("-fa", "--flash-attn") and any(
                v.lower() not in ("on", "off", "auto", "1", "0", "-1", "true", "false", "enabled", "disabled")
                for v in args[i + 1].split(",")):
            bench_fail(f"error: invalid parameter for argument: {arg}")
        if arg in ("-dev", "--device"):
            for device in args[i + 1].split(","):
                if device not in ("none", "auto") and (no_gpu() or not re.fullmatch(
                        r"(CUDA|Vulkan|ROCm|Metal|SYCL)\d*(/\S+)*", device)):
                    bench_fail(f"error: invalid device: {device}", f"error: invalid parameter for argument: {arg}")
        if arg in BENCH_LISTS:
            lists.setdefault(BENCH_LISTS[arg], []).extend(expand(BENCH_LISTS[arg], args[i + 1]))
        else:
            single[BENCH_SINGLE[arg]] = args[i + 1]
        i += 2
    if not lists.get("model"):
        lists["model"] = ["models/7B/ggml-model-q4_0.gguf"]
    for key, value in BENCH_DEFAULTS.items():
        lists.setdefault(key, expand(key, value))
    try:
        reps = int(single.get("reps", 5))
        fa_values = [fa_value(v) for v in lists["fa"]]
        numbers = {k: [int(v) for v in lists[k]] for k in RANGED}
    except ValueError:
        fail(["error: invalid parameter for argument: bad number"])
    output = single.get("output", "md")
    if output not in {"md", "json", "jsonl", "csv", "sql"}:
        fail([f"error: invalid parameter for argument: -o {output}"])
    for ctype in lists["ctk"] + lists["ctv"]:
        if ctype not in CACHE_TYPES:
            bench_fail(f"error: invalid parameter for argument: {'-ctk' if ctype in lists['ctk'] else '-ctv'}")
    if any(mode not in LOAD_MODES for mode in lists["load_mode"]):
        bench_fail("error: invalid parameter for argument: -lm")
    lists.update(numbers)
    lists["fa"] = fa_values
    tests = [(p, 0) for p in lists["n_prompt"] if p] + [(0, n) for n in lists["n_gen"] if n]
    for pair in lists.get("pg", []):
        p, _, n = pair.partition(",")
        tests.append((int(p), int(n)))

    rows = [{}]
    for key in BENCH_ORDER:
        rows = [{**row, key: value} for row in rows for value in lists[key]]
    printer = Printer(output)
    backend = "CPU" if no_gpu() else "CUDA"
    verbose = "verbose" in flags

    def load_error(details, what):
        # Without -v real llama-bench silences llama.cpp's own log, so the cause (out of memory, unknown
        # architecture, ...) is NOT on stderr: only this one line, with the path exactly as given.
        printer.abort()
        fail((details if verbose else []) + [f"llama_bench: error: failed to {what} '{row['model']}'"])

    loaded = {}
    for row in rows:
        if row["model"] not in loaded:
            info_errors = read_gguf(row["model"])
            errors = model_error(info_errors)
            if errors:
                load_error(errors + ["llama_model_load_from_file_impl: failed to load model"], "load model")
            loaded[row["model"]] = info_errors[0]
        info = loaded[row["model"]]
        s = {"threads": row["threads"], "ngl": row["ngl"], "batch": row["batch"], "ubatch": row["ubatch"],
             "fa": row["fa"], "ctk": row["ctk"], "ctv": row["ctv"], "ncmoe": row["ncmoe"],
             "cpu_only": backend == "CPU" or row["device"] == "none"}
        if gpu_oom(s, info["layers"]):
            load_error(oom_lines(s) + ["llama_model_load_from_file_impl: failed to load model"], "load model")
        if s["ctv"] != "f16" and s["fa"] == "off":
            load_error(["llama_init_from_model: quantized V cache requires flash_attn to be enabled"],
                       "create context with model")
        for n_prompt, n_gen in tests:
            pp, tg = speeds(s, info, depth=row["n_depth"] + n_prompt // 2 if n_gen else row["n_depth"])
            if n_prompt and n_gen:
                seconds = n_prompt / pp + n_gen / tg
                tps = (n_prompt + n_gen) / seconds
            else:
                tps = pp if n_prompt else tg
            tokens = n_prompt + n_gen
            samples_ts = [round(tps * (1 + 0.004 * ((k * 7) % 5 - 2)), 6) for k in range(reps)]
            samples_ns = [int(tokens / ts * 1e9) for ts in samples_ts]
            avg_ns = int(sum(samples_ns) / reps)
            avg_ts = sum(samples_ts) / reps
            std = lambda xs, mean: math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
            pause(sum(samples_ns) / 1e9)
            printer.row({
                "build_commit": COMMIT, "build_number": BUILD, "cpu_info": "Fake CPU 16-Core Processor",
                "gpu_info": "" if backend == "CPU" else "Fake GPU 24GB", "backends": backend,
                "model_filename": row["model"], "model_type": f"{info['architecture']} 8B {quant_name(row['model'])}",
                "model_size": 4920733696, "model_n_params": 8030261248, "n_batch": row["batch"],
                "n_ubatch": row["ubatch"], "n_threads": row["threads"], "cpu_mask": "0x0", "cpu_strict": False,
                "poll": 50, "type_k": row["ctk"], "type_v": row["ctv"], "n_gpu_layers": row["ngl"],
                "n_cpu_moe": row["ncmoe"], "split_mode": row["split_mode"], "main_gpu": row["main_gpu"],
                "no_kv_offload": row["nkvo"] not in ("0", "false"), "flash_attn": {"on": 1, "off": 0, "auto": -1}[row["fa"]],
                "devices": row["device"], "tensor_split": "0.00", "tensor_buft_overrides": "none",
                "load_mode": row["load_mode"],
                "embeddings": False, "no_op_offload": 0, "no_host": False, "fit_target": 0, "fit_min_ctx": 0,
                "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": row["n_depth"],
                "test_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "avg_ns": avg_ns,
                "stddev_ns": int(std(samples_ns, avg_ns)), "avg_ts": round(avg_ts, 6),
                "stddev_ts": round(std(samples_ts, avg_ts), 6), "samples_ns": samples_ns, "samples_ts": samples_ts})
    printer.finish()


class Printer:
    def __init__(self, fmt):
        self.fmt, self.count = ("md" if fmt == "sql" else fmt), 0
        fmt = self.fmt
        if fmt == "json":
            sys.stdout.write("[\n")
        elif fmt == "md":
            sys.stdout.write("| model | size | params | backend | ngl | threads | fa | test | t/s |\n"
                             "| ----- | ---: | -----: | ------- | --: | ------: | -: | ---: | --: |\n")
        sys.stdout.flush()

    def row(self, row):
        if self.fmt == "json":
            fields = ",\n".join(f"    {json.dumps(k)}: {json.dumps(v)}" for k, v in row.items())
            sys.stdout.write((",\n" if self.count else "") + "  {\n" + fields + "\n  }")
        elif self.fmt == "jsonl":
            sys.stdout.write(json.dumps(row) + "\n")
        elif self.fmt == "csv":
            if not self.count:
                sys.stdout.write(",".join(row) + "\n")
            sys.stdout.write(",".join(f'"{v}"' for v in row.values()) + "\n")
        elif self.fmt == "md":
            test = f"pp{row['n_prompt']}" if not row["n_gen"] else (f"tg{row['n_gen']}" if not row["n_prompt"]
                                                                    else f"pp{row['n_prompt']}+tg{row['n_gen']}")
            if row["n_depth"]:
                test += f" @ d{row['n_depth']}"
            sys.stdout.write(f"| {row['model_type']} | 4.58 GiB | 8.03 B | {row['backends']} | {row['n_gpu_layers']} | "
                             f"{row['n_threads']} | {row['flash_attn']} | {test} | {row['avg_ts']:.2f} ± {row['stddev_ts']:.2f} |\n")
        self.count += 1
        sys.stdout.flush()

    def abort(self):
        sys.stdout.flush()

    def finish(self):
        if self.fmt == "json":
            sys.stdout.write("\n]\n")
        elif self.fmt == "md":
            sys.stdout.write(f"\nbuild: {COMMIT} ({BUILD})\n")
        sys.stdout.flush()


# ---------------------------------------------------------------- llama-perplexity

PPL_VALUES = {**COMMON_VALUES, "-f": "file", "--file": "file", "--chunks": "chunks",
              "--kl-divergence-base": "kld_base", "-np": "parallel", "--parallel": "parallel"}
PPL_BOOLS = {**COMMON_BOOLS, "--kl-divergence": "kld", "--hellaswag": "hellaswag", "--ppl-stride": "ppl_stride"}


def s_batch(options):
    try:
        return int(options.get("batch", 2048))
    except ValueError:
        return 2048


def quant_name(path):
    match = re.search(r"(I?Q\d_[A-Z0-9_]+|MXFP4|BF16|F16|F32)", Path(path).name.upper())
    return match.group(1) if match else "Q4_K - Medium"


def quant_kld(path):
    name = Path(path).name.upper()
    return next((value for key, value in QUANT_KLD if key in name), 0.02)


def run_perplexity(args):
    options = parse(args, PPL_VALUES, PPL_BOOLS, "llama-perplexity")
    if "model" not in options:
        fail(["error: --model is required"])
    s = settings(options)
    n_ctx = s["ctx"] if "ctx" in options else 512
    check_device(options, s)
    backend_banner(s)
    stamp("I", f"cmn          init: llama threadpool init, n_threads = {s['threads']}")
    loaded = read_gguf(options["model"])
    errors = model_error(loaded)
    if errors:
        fail(errors + ["llama_model_load_from_file_impl: failed to load model",
                       "main: unable to load model"])
    if gpu_oom(s, loaded[0]["layers"]):
        fail(oom_lines(s) + ["main: unable to load model"])
    if options.get("kld"):
        return kl_divergence(options, n_ctx)
    if "file" not in options:
        fail(["main: error: input file not specified (use -f)"])
    try:
        text = Path(options["file"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        fail([f"error: failed to open file '{options['file']}'"])
    ids, _ = tokenize(text)
    if options.get("kld_base"):
        stamp("I", f"perplexity: saving all logits to {options['kld_base']}")
    stamp("I", "perplexity: tokenizing the input ..")
    if len(ids) < 2 * n_ctx:
        fail([f"perplexity: you need at least {2 * n_ctx} tokens to evaluate perplexity with a context of {n_ctx}",
              f"perplexity: the data file you provided tokenizes to only {len(ids)} tokens"])
    n_chunk = len(ids) // n_ctx
    if "chunks" in options:
        n_chunk = max(1, min(n_chunk, as_int(options, "chunks", n_chunk, "--chunks")))
    base = 6.0 + (zlib.crc32(Path(options["model"]).name.encode()) % 300) / 100
    ppl = base * (1 + quant_kld(options["model"]))
    stamp("I", f"perplexity: calculating perplexity over {n_chunk} chunks, n_ctx={n_ctx}, batch_size={min(s['batch'], n_ctx)}, n_seq=1")
    eta_line()
    values = [ppl * (1 + 0.08 / (k + 1)) for k in range(n_chunk)]
    sys.stdout.write(",".join(f"[{k + 1}]{v:.4f}" for k, v in enumerate(values)) + ",\n\n")
    sys.stdout.flush()
    if options.get("kld_base"):
        payload = json.dumps({"ppl": ppl, "model": options["model"], "n_ctx": n_ctx}).encode()
        tokens = ids[:n_chunk * n_ctx]
        with open(options["kld_base"], "wb") as handle:
            handle.write(b"_logits_" + struct.pack("<iii", n_ctx, 151936, n_chunk))
            handle.write(struct.pack(f"<{len(tokens)}i", *tokens))
            handle.write(struct.pack("<I", len(payload)) + payload)
    stamp("I", f"Final estimate: PPL = {ppl:.4f} +/- {ppl * 0.012:.5f}")
    pause(n_chunk * n_ctx / speeds(s, loaded[0])[0])


def eta_line(tool="perplexity"):
    """Real llama-perplexity logs "... - ETA " on stderr without a newline and the minutes on stdout."""
    sys.stderr.write(f"{elapsed()} I {tool}: 0.01 seconds per pass - ETA ")
    sys.stderr.flush()
    sys.stdout.write("0.00 minutes\n")
    sys.stdout.flush()


def kl_divergence(options, n_ctx):
    base_path = options.get("kld_base")
    if not base_path:
        fail(["main: error: --kl-divergence requires --kl-divergence-base"])
    try:
        data = Path(base_path).read_bytes()
        if data[:8] != b"_logits_":
            raise ValueError
        base_ctx, _vocab, n_chunk = struct.unpack_from("<iii", data, 8)
        offset = 20 + 4 * base_ctx * n_chunk
        (length,) = struct.unpack_from("<I", data, offset)
        meta = json.loads(data[offset + 4:offset + 4 + length])
    except (OSError, ValueError, struct.error):
        fail([f"kl_divergence: failed to open {base_path} for reading" if not os.path.exists(base_path)
              else f"kl_divergence: invalid log-likelihood file {base_path}"])
    if "ctx" in options and base_ctx != n_ctx:
        log(f"kl_divergence: warning: n_ctx {n_ctx} differs from the base file ({base_ctx}); using {base_ctx}")
    if "chunks" in options:
        n_chunk = max(1, min(n_chunk, as_int(options, "chunks", n_chunk, "--chunks")))
    same_model = os.path.abspath(meta.get("model", "")) == os.path.abspath(options["model"])
    kld = 0.0 if same_model else quant_kld(options["model"])
    ppl_base = meta["ppl"]
    ppl_q = ppl_base * (1 + kld * 0.9)
    ln_ratio = math.log(ppl_q / ppl_base)
    same_top = 100.0 if same_model else max(50.0, 100 - 60 * math.sqrt(kld))
    rms_dp = 0.0 if same_model else 12 * math.sqrt(kld)
    stamp("I", f"kl_divergence: computing over {n_chunk} chunks, n_ctx={base_ctx}, batch_size={s_batch(options)}, n_seq=1")
    eta_line("kl_divergence")
    print()
    print("chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p")
    for k in range(n_chunk):
        f = 1 + 0.05 / (k + 1)
        print(f"{k + 1:4d}    {ppl_q * f:9.4f} ± {ppl_q * 0.05 / math.sqrt(k + 1):9.4f}"
              f"    {ln_ratio * f:10.5f} ± {abs(ln_ratio) * 0.1:10.5f}"
              f"    {kld * f:10.5f} ± {kld * 0.05:10.5f}"
              f"    {rms_dp * f:6.3f} ± {rms_dp * 0.05:6.3f} %"
              f"    {same_top:6.3f} ± {0.0 if same_model else 0.4:6.3f} %")
    print()
    print("====== Perplexity statistics ======")
    print(f"Mean PPL(Q)                   : {ppl_q:10.6f} ± {ppl_q * 0.012:10.6f}")
    print(f"Mean PPL(base)                : {ppl_base:10.6f} ± {ppl_base * 0.012:10.6f}")
    print(f"Cor(ln(PPL(Q)), ln(PPL(base))): {100 - 50 * kld:6.2f}%")
    print(f"Mean ln(PPL(Q)/PPL(base))     : {ln_ratio:10.6f} ± {abs(ln_ratio) * 0.08:10.6f}")
    print(f"Mean PPL(Q)/PPL(base)         : {ppl_q / ppl_base:10.6f} ± {abs(ln_ratio) * 0.08:10.6f}")
    print(f"Mean PPL(Q)-PPL(base)         : {ppl_q - ppl_base:10.6f} ± {abs(ppl_q - ppl_base) * 0.08:10.6f}")
    print()
    print("====== KL divergence statistics ======")
    print(f"Mean    KLD: {kld:10.6f} ± {kld * 0.01:10.6f}")
    for label, factor in [("Maximum", 60), ("99.9%  ", 25), ("99.0%  ", 8), ("95.0%  ", 3.5), ("90.0%  ", 2.2),
                          ("Median ", 0.35), ("10.0%  ", 0.01), (" 5.0%  ", 0.004), (" 1.0%  ", 0.0005), (" 0.1%  ", 0.00005)]:
        print(f"{label} KLD: {kld * factor:10.6f}")
    print(f"Minimum KLD: {-0.000001 if kld else 0.0:10.6f}")
    print()
    print("====== Token probability statistics ======")
    print(f"Mean    Δp: {-kld * 10:6.3f} ± {kld * 0.5:5.3f} %")
    for label, factor in [("Maximum", 6), ("99.9%  ", 4), ("99.0%  ", 2), ("95.0%  ", 0.8), ("90.0%  ", 0.4),
                          ("75.0%  ", 0.05), ("Median ", 0.0), ("25.0%  ", -0.08), ("10.0%  ", -0.5), (" 5.0%  ", -1),
                          (" 1.0%  ", -2.5), (" 0.1%  ", -5)]:
        print(f"{label} Δp: {rms_dp * factor:6.3f}%")
    print(f"Minimum Δp: {-rms_dp * 7:6.3f}%")
    print(f"RMS Δp    : {rms_dp:6.3f} ± {rms_dp * 0.03:5.3f} %")
    print(f"Same top p: {same_top:6.3f} ± {0.0 if same_model else 0.12:5.3f} %")
    sys.stdout.flush()


# ---------------------------------------------------------------- llama-cli

CLI_VALUES = {**COMMON_VALUES, "-p": "prompt", "--prompt": "prompt", "-f": "file", "--file": "file",
              "-sys": "system", "--system-prompt": "system"}
CLI_BOOLS = {**COMMON_BOOLS, "-no-cnv": "no_cnv", "--no-conversation": "no_cnv", "-cnv": "cnv",
             "--conversation": "cnv", "-st": "single_turn", "--single-turn": "single_turn", "--jinja": "jinja",
             "--no-display-prompt": "no_display_prompt", "-e": "escape"}


def run_cli(args):
    options = parse(args, CLI_VALUES, CLI_BOOLS, "llama-cli")
    s = settings(options)
    check_device(options, s)
    log(f"build: {BUILD} ({COMMIT}) with {COMPILER} for {TARGET}")
    backend_banner(s)
    loaded = read_gguf(options.get("model", "models/7B/ggml-model-f16.gguf"))
    errors = model_error(loaded)
    if errors:
        fail(errors + ["llama_model_load_from_file_impl: failed to load model", "main: error: unable to load model"])
    if gpu_oom(s, loaded[0]["layers"]):
        fail(oom_lines(s) + ["main: error: unable to load model"])
    prompt = options.get("prompt") or ""
    n_predict = as_int(options, "n_predict", -1, "-n")
    text, _, _, _, n = generate(prompt, prompt, n_predict)
    if not options.get("no_display_prompt"):
        sys.stdout.write(prompt)
    sys.stdout.write(text + "\n")
    sys.stdout.flush()
    ids, _ = tokenize(prompt)
    pp, tg = speeds(s, loaded[0], depth=len(ids))
    log("")
    log(f"llama_perf_context_print: prompt eval time = {len(ids) / pp * 1000:10.2f} ms / {len(ids):5d} tokens "
        f"({1000 / pp:8.2f} ms per token, {pp:8.2f} tokens per second)")
    log(f"llama_perf_context_print:        eval time = {n / tg * 1000:10.2f} ms / {n:5d} runs   "
        f"({1000 / tg:8.2f} ms per token, {tg:8.2f} tokens per second)")


MODES = {"server": run_server, "bench": run_bench, "perplexity": run_perplexity, "cli": run_cli}


def main(argv):
    args = argv[1:]
    mode = None
    if len(args) >= 2 and args[0] == "--as":
        mode, args = args[1], args[2:]
    else:
        name = Path(argv[0]).name.lower()
        mode = next((m for m in MODES if f"llama-{m}" in name), None)
    if mode not in MODES:
        fail([f"fake_llama: choose a mode with --as {'|'.join(MODES)}"], code=2)
    if "--version" in args:
        if mode == "bench":  # real llama-bench has no --version: usage on stdout, error on stderr, exit 1
            bench_fail("error: invalid parameter for argument: --version")
        version()
    if "--list-devices" in args:
        list_devices()
    MODES[mode](args)


if __name__ == "__main__":
    main(sys.argv)
