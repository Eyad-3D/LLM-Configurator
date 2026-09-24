"""Read GGUF model headers with the standard library only.

Promises: reads only the header (never the weights), stops after a hard byte cap,
bounds every count so a hostile file cannot exhaust memory, and never invents
numbers: anything the header does not say stays None.
"""
from hashlib import sha256 as _sha256
from pathlib import Path
import re
import stat
import struct

from .domain import ARCHITECTURES, Variant

MAGIC = b"GGUF"
MAX_HEADER_BYTES = 64 * 1024 * 1024  # real headers (with a 256k-token vocabulary) stay well below this
MAX_KV = 100_000
MAX_TENSORS = 100_000  # the largest real models have a few thousand
MAX_ITEMS = 2_000_000  # strings and nested lists stepped over, in total (a 262k vocabulary + merges is ~0.5M)
MAX_KEPT_ITEMS = 262_144  # numbers kept from small arrays, in total (each costs ~40 bytes as a Python int)
MAX_DIMS = 8
MAX_STRING = 16 * 1024 * 1024
MAX_KEPT_STRING = 64 * 1024  # longer strings (for example huge chat templates) are skipped, not stored
MAX_KEPT_ARRAY = 4096  # small numeric arrays (per-layer head counts) are kept for the summary
MAX_NESTING = 4

# GGUF value types -> struct format (None: variable size)
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
_STRING, _ARRAY = 8, 9
_U64 = struct.Struct("<Q")

# llama_ftype (include/llama.h) -> quant name; MXFP4_MOE maps to the MXFP4 name the catalogue uses.
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S",
    12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL",
    26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16",
    36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}

# ggml_type -> (values per block, bytes per block); used only to weigh expert tensors against the rest.
TENSOR_TYPES = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2), 34: (256, 54), 35: (256, 66), 39: (32, 17), 40: (64, 36), 41: (128, 18), 42: (64, 18),
}

# GGUF architecture name -> domain.ARCHITECTURES name. Plain "llama" with experts is Mixtral's layout.
ARCH_MAP = {
    "llama": "llama", "qwen2": "qwen2", "qwen2moe": "qwen2_moe", "qwen3": "qwen3", "qwen3moe": "qwen3_moe",
    "gemma2": "gemma2", "gemma3": "gemma3", "phi3": "phi3", "granite": "granite", "olmo2": "olmo2",
    "gpt-oss": "gpt_oss", "mistral": "mistral",
}
# llama.cpp's built-in sliding-window patterns (src/models/<arch>.cpp): every n-th layer uses full attention.
# phi3 is absent on purpose: llama.cpp disables Phi's sliding window, so every layer keeps full KV.
SWA_PATTERNS = {"gemma2": 2, "gemma3": 6, "gpt-oss": 2, "olmo2": 4}
SWA_DEFAULT_WINDOW = {"gemma2": 4096}  # used by llama.cpp when the header has no window

SHARD = re.compile(r"^(?P<prefix>.+)-(?P<index>[0-9]{5})-of-(?P<count>[0-9]{5})\.gguf$", re.IGNORECASE)
_QUANT_IN_NAME = re.compile(r"(?<![A-Za-z0-9])(" + "|".join(sorted(map(re.escape, set(FILE_TYPES.values()) | {"MXFP4_MOE"}),
                                                                   key=len, reverse=True)) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def _bad(message):
    return ValueError(f"This file is not a readable GGUF model: {message}. Re-download it or pick another file.")


class _Reader:
    def __init__(self, handle, size, limit):
        self.handle, self.size, self.limit, self.pos = handle, size, min(size, limit), 0
        self.last_count = None  # item count of the last top-level list, kept even when its items are skipped
        self.items = self.kept = 0

    def spend(self, count, kept=False):
        """Bounds the work (and memory) a header can cause, not just its bytes: 64 MiB of tiny items is still millions of steps."""
        if kept:
            self.kept += count
            return self.kept <= MAX_KEPT_ITEMS
        self.items += count
        if self.items > MAX_ITEMS:
            raise _bad("the header lists more entries than any real model has")
        return True

    def check(self, count, what="the header is cut off (the file looks truncated)"):
        """Refuses before reading or seeking when `count` more bytes cannot fit in the file or the cap."""
        if count < 0 or self.pos + count > self.limit:
            if self.pos + count > self.size:
                raise _bad(what)
            raise _bad(f"the header is larger than {MAX_HEADER_BYTES // (1024 * 1024)} MiB, which no real model needs")

    def _claim(self, count):
        self.check(count)
        self.pos += count

    def read(self, count):
        self._claim(count)
        data = self.handle.read(count)
        if len(data) != count:
            raise _bad("the header is cut off (the file looks truncated)")
        return data

    def skip(self, count):
        self._claim(count)
        self.handle.seek(self.pos)

    def skip_strings(self, count):
        """Steps over `count` strings, parsing lengths from 1 MiB read-ahead chunks (token lists hold 100k+)."""
        remaining = count
        while remaining:
            self.check(8)
            data = self.handle.read(min(1024 * 1024, self.limit - self.pos))
            offset = 0
            while remaining and offset + 8 <= len(data):
                length = _U64.unpack_from(data, offset)[0]
                if length > MAX_STRING:
                    raise _bad("a text field claims an impossible length")
                if offset + 8 + length > len(data):
                    break
                offset += 8 + length
                remaining -= 1
            self.handle.seek(self.pos)
            if offset:
                self.skip(offset)
            else:  # one string longer than the chunk, or the end of the file: the careful path reports it
                self.string(keep=False)
                remaining -= 1

    def unpack(self, fmt):
        return struct.unpack("<" + fmt, self.read(struct.calcsize(fmt)))[0]

    def string(self, keep=True):
        length = self.unpack("Q")
        if length > MAX_STRING:
            raise _bad("a text field claims an impossible length")
        if not keep or length > MAX_KEPT_STRING:
            self.skip(length)
            return None
        return self.read(length).decode("utf-8", errors="replace")


def _value(reader, kind, depth=0):
    """Returns the value, or None when it was skipped (long string, big or nested array)."""
    if kind in _SCALARS:
        return reader.unpack(_SCALARS[kind])
    if kind == _STRING:
        return reader.string()
    if kind != _ARRAY:
        raise _bad(f"unknown value type {kind}")
    if depth >= MAX_NESTING:
        raise _bad("lists are nested too deeply")
    item_kind, count = reader.unpack("I"), reader.unpack("Q")
    if depth == 0:
        reader.last_count = count
    if item_kind in _SCALARS:
        width = struct.calcsize(_SCALARS[item_kind])
        if count > MAX_KEPT_ARRAY or not reader.spend(count, kept=True):
            reader.skip(count * width)  # the byte cap rejects absurd counts before any seek happens
            return None
        return list(struct.unpack(f"<{count}{_SCALARS[item_kind]}", reader.read(count * width)))
    smallest = 8 if item_kind == _STRING else 12  # a length prefix, or a nested list's type + count
    reader.check(count * smallest, "a list claims more items than the file could hold")
    reader.spend(count)
    if item_kind == _STRING:
        reader.skip_strings(count)  # e.g. the token list: stepped over, never kept
        return None
    if item_kind == _ARRAY:
        for _ in range(count):
            _value(reader, _ARRAY, depth + 1)
        return None
    raise _bad(f"unknown list item type {item_kind}")


def _tensor_infos(reader, count):
    """Totals for the weight table, plus the byte where the last known weight ends (relative to the data start)."""
    totals = {"count": count, "parameters": 0, "expert_parameters": 0, "bytes": 0, "expert_bytes": 0, "unknown_types": 0}
    data_end = 0
    for _ in range(count):
        name = reader.string() or ""
        dims = reader.unpack("I")
        if dims > MAX_DIMS:
            raise _bad("a weight table entry has too many dimensions")
        elements = 1
        for _ in range(dims):
            elements *= reader.unpack("Q")
        kind = reader.unpack("I")
        offset = reader.unpack("Q")
        block, width = TENSOR_TYPES.get(kind, (None, None))
        nbytes = -(-elements // block) * width if block else None
        data_end = max(data_end, offset + (nbytes or 0))
        expert = "_exps" in name  # routed experts; shared experts are "_shexp" and always run
        totals["parameters"] += elements
        totals["expert_parameters"] += elements if expert else 0
        if nbytes is None:
            totals["unknown_types"] += 1
        else:
            totals["bytes"] += nbytes
            totals["expert_bytes"] += nbytes if expert else 0
    return totals, data_end


def _parse(path, tensors=False):
    path = Path(path)
    try:
        info = path.stat()
        if not stat.S_ISREG(info.st_mode):  # a pipe or device would block or never end
            raise ValueError(f"{path.name} is not a regular file. Pick the .gguf model file itself.")
        size = info.st_size
        handle = path.open("rb")
    except OSError as exc:
        raise ValueError(f"Could not open {path.name}: {exc.strerror or exc}. Check the file still exists and is readable.") from None
    try:
        with handle:
            return _parse_open(handle, size, tensors)
    except OSError as exc:
        raise ValueError(f"Could not read {path.name}: {exc.strerror or exc}. Check the disk and try again.") from None


def _parse_open(handle, size, tensors):
    reader = _Reader(handle, size, MAX_HEADER_BYTES)
    if size < 4 or reader.read(4) != MAGIC:
        raise _bad("it does not start with the GGUF marker")
    version = reader.unpack("I")
    if version == 1:
        raise _bad("it uses GGUF version 1, which llama.cpp no longer loads")
    if version not in (2, 3):
        if version & 0xFFFF == 0 and version >> 16:
            raise _bad("it is a big-endian GGUF, which only runs on big-endian machines")
        raise _bad(f"it uses GGUF version {version}, which this app does not know yet")
    tensor_count, kv_count = reader.unpack("Q"), reader.unpack("Q")
    if kv_count > MAX_KV or tensor_count > MAX_TENSORS:
        raise _bad("the header claims an impossible number of entries")
    values, skipped, lengths = {}, [], {}
    for _ in range(kv_count):
        key = reader.string()
        if key is None:
            raise _bad("a setting name is impossibly long")
        kind = reader.unpack("I")
        value = _value(reader, kind)
        if kind == _ARRAY:
            lengths[key] = reader.last_count
        if value is None:
            skipped.append(key)
        else:
            values[key] = value
    table = None
    if tensors:
        table, data_end = _tensor_infos(reader, tensor_count)
        alignment = values.get("general.alignment", 32)
        alignment = alignment if type(alignment) is int and 0 < alignment <= 1 << 20 else 32
        start = -(-reader.pos // alignment) * alignment  # weights begin at the next aligned byte after the table
        if start + data_end > size:  # what llama.cpp reports as "data is not within the file bounds"
            raise _bad("its weights run past the end of the file, so the download is incomplete or cut short")
    return version, tensor_count, values, skipped, table, lengths


def quant_name(file_type=None, filename=""):
    """Quant label from general.file_type, else from the file name; None when neither says."""
    if type(file_type) is int and file_type in FILE_TYPES:
        return FILE_TYPES[file_type]
    match = _QUANT_IN_NAME.search(Path(filename).name)
    if not match:
        return None
    name = match.group(1).upper()
    return "MXFP4" if name == "MXFP4_MOE" else name


def _int(value, ceiling=None):
    """A non-negative int, or None; values above `ceiling` are no real model's and count as unknown."""
    return value if type(value) is int and value >= 0 and (ceiling is None or value <= ceiling) else None


# Far above any real model; a header claiming more is broken or hostile, so the value is treated as unknown.
CEILINGS = {"block_count": 4096, "attention.head_count": 65536, "attention.head_count_kv": 65536,
            "attention.key_length": 65536, "embedding_length": 1 << 20, "context_length": 1 << 30,
            "expert_count": 65536, "expert_used_count": 65536, "attention.sliding_window": 1 << 30, "vocab_size": 1 << 24}


def _peak(value, ceiling):
    """Per-layer lists (e.g. KV heads) are summarised by their maximum: a safe upper bound for memory."""
    if isinstance(value, list):
        numbers = [v for v in value if type(v) is int and v > 0]
        return _int(max(numbers), ceiling) if numbers else None
    return _int(value, ceiling)


def _sliding_layers(arch, values, layers):
    window = _int(values.get(f"{arch}.attention.sliding_window"), CEILINGS["attention.sliding_window"])
    if window is None:
        window = SWA_DEFAULT_WINDOW.get(arch)
    if not window or not layers or arch not in SWA_PATTERNS and f"{arch}.attention.sliding_window_pattern" not in values:
        return window, 0
    pattern = values.get(f"{arch}.attention.sliding_window_pattern", SWA_PATTERNS.get(arch))
    if isinstance(pattern, list):  # one flag per layer
        return window, min(layers, sum(1 for flag in pattern if flag))
    if type(pattern) is int and pattern == 0:
        return window, layers  # llama.cpp's set_swa_pattern(0): every layer slides
    if type(pattern) is int and pattern > 0:
        return window, layers - layers // pattern  # layer % n == n - 1 is the full-attention one
    return window, 0  # unknown pattern: count every layer as full attention (never under-estimates memory)


def _summary(values, filename, lengths=None):
    arch = values.get("general.architecture") if isinstance(values.get("general.architecture"), str) else None
    raw = (lambda key: values.get(f"{arch}.{key}")) if arch else (lambda key: None)
    get = lambda key: _int(raw(key), CEILINGS.get(key))
    layers = get("block_count")
    heads = _peak(raw("attention.head_count"), CEILINGS["attention.head_count"])
    kv_heads = _peak(raw("attention.head_count_kv"), CEILINGS["attention.head_count_kv"]) or heads  # llama.cpp's default
    head_dim = get("attention.key_length")
    first = raw("attention.head_count")
    first = _int(first[0] if isinstance(first, list) and first else first, CEILINGS["attention.head_count"])
    if not head_dim and first and get("embedding_length"):
        head_dim = get("embedding_length") // first  # llama.cpp: n_embd / n_head(layer 0)
    value_dim = _int(raw("attention.value_length"), CEILINGS["attention.key_length"])
    if head_dim and value_dim and value_dim > head_dim:
        head_dim = value_dim  # one size stands for keys and values: take the larger, never under-estimating memory
    window, sliding = _sliding_layers(arch, values, layers) if arch else (None, 0)
    file_type = _int(values.get("general.file_type"))
    if file_type is not None:
        file_type &= ~1024  # LLAMA_FTYPE_GUESSED: a flag, not a type
    split = _int(values.get("split.count"))
    return {
        "name": values.get("general.name") if isinstance(values.get("general.name"), str) else None,
        "layers": layers, "kv_heads": kv_heads, "head_dim": head_dim or None,
        "context_length": get("context_length"),
        "experts": get("expert_count") or 0, "active_experts": get("expert_used_count") or 0,
        "sliding_window": window, "sliding_layers": sliding,
        "file_type": file_type, "quant": quant_name(file_type, filename),
        "split_count": split if split else 1,
        # llama.cpp's n_vocab is the token list's length; the list is stepped over, only its count is kept
        "vocab_size": (lengths or {}).get("tokenizer.ggml.tokens") or get("vocab_size") or None,
    }


def read_metadata(path, tensors=False):
    """Header facts for one GGUF file. Arrays and long strings are stepped over, not loaded.

    `tensors=True` also reads the weight table to count parameters and expert bytes.
    """
    version, tensor_count, values, skipped, table, lengths = _parse(path, tensors)
    scalars = {k: v for k, v in values.items() if not isinstance(v, list)}
    summary = _summary(values, Path(path).name, lengths)
    result = {"version": version, "tensor_count": tensor_count, "metadata": scalars,
              "architecture": scalars.get("general.architecture") if isinstance(scalars.get("general.architecture"), str) else None,
              "summary": summary,
              "skipped_keys": skipped}
    if table is not None:
        result["tensors"] = table
    return result


def shard_paths(path):
    """All parts of a split GGUF in order (first part first); a plain file returns [path].

    Raises ValueError naming the missing parts so the user knows what to fetch.
    """
    path = Path(path)
    match = SHARD.match(path.name)
    if not match:
        return [path]
    count = int(match["count"])
    if not 1 <= count <= 9999 or not 1 <= int(match["index"]) <= count:
        raise ValueError(f"{path.name} has an impossible part number. Rename it to its original name.")
    width = len(match["index"])
    parts = [path.with_name(f"{match['prefix']}-{i:0{width}d}-of-{match['count']}.gguf") for i in range(1, count + 1)]
    missing = [p.name for p in parts if not p.is_file()]
    if missing:
        shown = ", ".join(missing[:3]) + (" and more" if len(missing) > 3 else "")
        raise ValueError(f"This model is split into {count} parts and {len(missing)} are missing ({shown}). "
                         "Put every part in the same folder.")
    return parts


def local_id(path):
    """Stable id for a local file, from where it really lives (links resolved). It does not depend on whether the
    file has been fingerprinted yet, so a scan and a manual `local --add` of the same file agree on it."""
    key = _sha256(str(Path(path).resolve()).encode("utf-8", "surrogateescape")).hexdigest()
    return f"local:{key[:16]}:{Path(path).name}"


def variant_from_file(path, sha256=None):
    """A source="local" Variant from a GGUF on disk, so any local model can be sized and run.

    Split models are read from every part; the first part is the file llama.cpp loads.
    """
    parts = shard_paths(path)
    if parts[0].name != Path(path).name:
        sha256 = None  # the fingerprint given belongs to another part, not to the first one the id is built from
    first = read_metadata(parts[0], tensors=True)
    summary, meta = first["summary"], first["metadata"]
    if summary["split_count"] != len(parts):
        raise ValueError(f"{parts[0].name} says the model is split into {summary['split_count']} parts, but the file names "
                         f"say {len(parts)}. Keep the original part names (…-00001-of-0000N.gguf) in one folder.")
    gguf_arch = first["architecture"]
    arch = ARCH_MAP.get(gguf_arch)
    if gguf_arch == "llama" and summary["experts"]:
        arch = "mixtral"
    if arch not in ARCHITECTURES:
        name = gguf_arch or "unknown"
        raise ValueError(f"This model uses the '{name}' design, which LLM Configurator cannot size yet. "
                         "Pick a model built on Llama, Qwen, Gemma, Mistral, Phi, Granite, OLMo or gpt-oss.")
    missing = [label for label, key in [("layer count", "layers"), ("attention heads", "kv_heads"),
                                        ("head size", "head_dim"), ("context length", "context_length")] if not summary[key]]
    if missing:
        raise ValueError(f"The file's header does not say its {', '.join(missing)}, so its memory needs cannot be worked out. "
                         "Try a GGUF from another publisher.")
    table = dict(first["tensors"])
    try:
        sizes = [p.stat().st_size for p in parts]
    except OSError as exc:
        raise ValueError(f"Could not read {parts[0].name}: {exc.strerror or exc}. Check every part is still there.") from None
    files = []
    if len(parts) > 1:
        for part in parts[1:]:
            extra = read_metadata(part, tensors=True)["tensors"]
            for key in table:
                table[key] += extra[key]
        files = [{"filename": p.name, "size_bytes": sizes[i], "sha256": sha256 if i == 0 else None}
                 for i, p in enumerate(parts)]
    size = sum(sizes)
    if size <= 0:
        raise ValueError(f"{parts[0].name} is empty. Re-download it.")
    experts, active = summary["experts"], min(summary["active_experts"], summary["experts"])
    parameters = table["parameters"] or None
    fraction, active_parameters = 0.0, parameters
    if experts and table["expert_parameters"]:
        if table["bytes"] and not table["unknown_types"]:
            fraction = table["expert_bytes"] / table["bytes"]
        else:  # an unfamiliar weight type: fall back to the share of values, which is close for uniform quants
            fraction = table["expert_parameters"] / table["parameters"]
        fraction = min(fraction, 0.999)
        if active:
            active_parameters = round(parameters - table["expert_parameters"] * (1 - active / experts))
    name = summary["name"] or parts[0].name.removesuffix(".gguf")
    return Variant(
        id=local_id(parts[0]), name=name, base_repo=f"local/{name}", repo="local", revision="", base_revision="",
        filename=parts[0].name, sha256=sha256.lower() if sha256 else None, quant=summary["quant"] or "unknown",
        size_bytes=size, layers=summary["layers"], kv_heads=summary["kv_heads"], head_dim=summary["head_dim"],
        max_context=summary["context_length"], architecture=arch, files=files, parameters=parameters,
        active_parameters=active_parameters, experts=experts, active_experts=active, expert_fraction=fraction,
        sliding_window=summary["sliding_window"] if summary["sliding_layers"] else None,
        sliding_layers=min(summary["sliding_layers"], summary["layers"]),
        family=meta.get("general.basename") if isinstance(meta.get("general.basename"), str) else None,
        license=meta.get("general.license") if isinstance(meta.get("general.license"), str) else None,
        source="local", score_source=None,
    )
