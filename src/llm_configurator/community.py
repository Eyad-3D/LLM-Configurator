"""Opt-in, anonymous sharing of real speed results, and import of the curated list.

Promises:
- Sharing never sends anything. It builds JSON and a prefilled GitHub issue link; the
  user reads both and submits the issue themselves.
- Only allowlisted fields leave the machine: model identity, a coarse hardware class
  (names cleaned of serials/IDs, memory rounded to 4 GiB, OS family), settings and speeds.
  No paths, hostnames, usernames, process lists, GPU UUIDs, fingerprints or exact times.
- Imported rows are untrusted: HTTPS only, size-capped, strictly validated; bad rows are
  dropped and counted, never repaired.
- Community numbers are other people's computers. `evidence` says so and returns None
  rather than a number from fewer than two matching results.
"""
import getpass
import json
import math
import os
import re
import secrets
import socket
import statistics
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener

from .domain import GIB, QUANT_BYTES_PER_PARAMETER, check_cancel, now

SCHEMA = 1
REPO = "Eyad-3D/LLM-Configurator"
DEFAULT_SOURCE = f"https://raw.githubusercontent.com/{REPO}/main/community/results.json"
LABEL = "community-results"
MAX_BYTES = 10 * 1024**2
MAX_ROWS = 50_000
MAX_URL = 8000  # GitHub rejects much longer "new issue" links; fall back to pasting the JSON.
MAX_SHARE = 50
BUCKET_GIB = 4

BACKENDS = {"cuda", "metal", "rocm", "vulkan", "cpu", "unknown"}
OS_FAMILIES = {"Windows", "macOS", "Linux", "other"}
ARCHES = {"x86_64", "arm64", "other"}
PLACEMENTS = {"gpu", "split", "cpu"}
KINDS = {"bench", "speed_test", "tune"}
FLASH = {"on", "off", "auto"}
CACHE = {"f16", "q8_0", "q4_0"}
CONTEXT_BUCKETS = ((4096, "up to 4K"), (16384, "4K–16K"), (65536, "16K–64K"), (math.inf, "over 64K"))
NOTE = "Measured by other people on similar computers, not on yours. Treat it as a hint."

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 .+-]{0,63}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+-]*")
_REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,63}/[A-Za-z0-9][\w.-]{0,95}", re.A)
_FILENAME = re.compile(r"[A-Za-z0-9][\w.+-]{0,199}\.gguf", re.I | re.A)
_SHA = re.compile(r"[0-9a-f]{64}")
_QUANT = re.compile(r"[A-Z0-9_]{1,16}")
_VERSION = re.compile(r"b\d{1,6}|\d{1,4}(?:\.\d{1,4}){0,3}", re.A)  # llama.cpp build tags; nothing free-form
_FILLER = {"cpu", "processor", "@"}
_ARCH_WORDS = {"x86_64", "amd64", "arm64", "aarch64", "i386", "i686", "arm", "unknown", ""}


# ---------- anonymising ----------

def _field(obj, name, default=None):
    if obj is None:
        return default
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def _personal_words():
    """This machine's own names, removed from any hardware string as a second line of defence."""
    values = [os.environ.get(key, "") for key in ("USER", "USERNAME", "LOGNAME", "COMPUTERNAME", "HOSTNAME")]
    for get in (socket.gethostname, getpass.getuser):
        try:
            values.append(get() or "")
        except Exception:  # getuser raises varied errors when no user name is configured
            continue
    words = set()
    for value in values:
        value = value.lower()
        words |= {value} | set(re.split(r"[-._ ]+", value))  # "johns-macbook.local" -> johns, macbook, local
    return {word for word in words if len(word) >= 3 and word not in {"local", "localhost", "lan", "home"}}


def _looks_personal(low, personal):
    return any(low == word or (len(word) >= 4 and word in low) for word in personal)


def _looks_like_id(token):
    """Serial numbers, UUID pieces and asset tags rather than model names."""
    low = token.lower()
    switches = len(re.findall(r"[a-z](?=\d)|\d(?=[a-z])", low))  # "ZX9K2LQ7WP" flips letter/digit constantly
    return bool(re.search(r"[0-9a-f]{8,}", low) or re.search(r"\d{6,}", low) or switches >= 5
                or re.match(r"(gpu-|uuid|s/?n\d|serial)", low) or low.count(".") >= 2)


def clean_name(value, personal=None):
    """A hardware model name with serials, IDs, brackets, clock speeds and personal words removed."""
    if not isinstance(value, str) or value.strip().lower() in _ARCH_WORDS:
        return None  # Linux often reports only the architecture, which is not a CPU name
    personal = _personal_words() if personal is None else personal
    text = re.sub(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}", " ", value)  # (R), (TM), (hostname), [serial]
    text = re.sub(r"@.*$", " ", text)  # "@ 3.60GHz": clock speed, not a model
    kept = []
    for token in text.replace("_", " ").split():
        low = token.lower()
        if (not _TOKEN.fullmatch(token) or len(token) > 24 or low in _FILLER
                or _looks_like_id(token) or _looks_personal(low, personal)):
            continue
        kept.append(token)
    name = " ".join(kept[:8])[:64].strip()
    return name or None


def bucket_gib(value):
    """Memory in bytes rounded to the nearest 4 GiB (never below 4); None when unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return max(BUCKET_GIB, round(value / GIB / BUCKET_GIB) * BUCKET_GIB)


def _os_family(hardware):
    platform = hardware.get("platform") if isinstance(hardware.get("platform"), dict) else {}
    system = str(platform.get("system") or hardware.get("os") or "")
    release = str(platform.get("release") or "")
    text = f"{system} {hardware.get('os') or ''}"
    if "windows" in text.lower():
        build = re.search(r"10\.0\.(\d+)", text)
        version = "11" if build and int(build.group(1)) >= 22000 else (re.match(r"\d{1,2}", release) or [None])[0]
        return "Windows", version if version in {"7", "8", "10", "11"} else None
    if system == "Darwin" or "macos" in text.lower() or "darwin" in text.lower():
        match = re.search(r"macOS-(\d{2})", str(hardware.get("os") or ""))
        return "macOS", match.group(1) if match else None  # Darwin kernel numbers are not macOS versions
    if "linux" in text.lower():
        return "Linux", None  # kernel strings are often custom and identifying
    return "other", None


def _arch(hardware):
    platform = hardware.get("platform") if isinstance(hardware.get("platform"), dict) else {}
    text = f"{platform.get('machine') or ''} {hardware.get('os') or ''} {hardware.get('cpu') or ''}".lower()
    if "arm64" in text or "aarch64" in text:
        return "arm64"
    if "x86_64" in text or "amd64" in text:
        return "x86_64"
    return "other"


def _placement(gpu_layers, total_layers):
    if not gpu_layers:
        return "cpu"
    return "gpu" if total_layers and gpu_layers >= total_layers else "split"


def hardware_class(hardware, gpu_uuid=None, uses_gpu=True, personal=None):
    """The coarse, shareable description of a computer. Also used to match community records."""
    hardware = hardware or {}
    gpus = [g for g in hardware.get("gpus") or [] if isinstance(g, dict)]
    gpu = next((g for g in gpus if gpu_uuid and g.get("uuid") == gpu_uuid), gpus[0] if gpus else None) if uses_gpu else None
    os_family, os_version = _os_family(hardware)
    backend = "cpu"
    if gpu:
        backend = gpu.get("backend") if gpu.get("backend") in BACKENDS else "unknown"
    unified = bool(hardware.get("unified_memory") or (gpu or {}).get("unified"))
    return {"cpu": clean_name(hardware.get("cpu_name") or hardware.get("cpu"), personal),
            "gpu": clean_name(gpu.get("name"), personal) if gpu else None, "backend": backend,
            "vram_gib": bucket_gib(gpu.get("total")) if gpu else None, "ram_gib": bucket_gib(hardware.get("ram_total")),
            "unified": unified, "os": os_family, "os_version": os_version, "arch": _arch(hardware)}


def _number(value, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        return None
    if not low <= value <= high or not math.isfinite(value):  # range first: huge ints overflow isfinite
        return None
    return value if integer else round(float(value), 3)


def _choice(value, allowed):
    return value if value in allowed else None


def _text(value, pattern):
    return value if isinstance(value, str) and pattern.fullmatch(value) else None


def _month(timestamp):
    try:
        moment = datetime.fromisoformat(str(timestamp))
    except ValueError:
        moment = datetime.now(timezone.utc)
    return moment.strftime("%Y-%m")


def anonymize(measurement, hardware, variant):
    """One shareable schema-v1 record built only from allowlisted fields. Raises ValueError if it has no valid speed."""
    tps = _number(measurement.get("tps"), 0.001, 100000)
    context = _number(measurement.get("context"), 1, 1048576, integer=True)
    if tps is None or context is None:
        raise ValueError("This result has no valid speed or context, so there is nothing useful to share.")
    # Local or custom models may carry personal file or repo names; share only the content hash then.
    # A missing source counts as private.
    public = _field(variant, "source") == "catalogue"
    layers = _number(_field(variant, "layers"), 1, 4096, integer=True)
    gpu_layers = _number(measurement.get("gpu_layers"), 0, 4096, integer=True) or 0
    quant = _field(variant, "quant")
    sha = _field(variant, "sha256") or measurement.get("sha256")
    settings = measurement.get("settings") if isinstance(measurement.get("settings"), dict) else {}
    runtime = measurement.get("runtime") if isinstance(measurement.get("runtime"), dict) else {}
    record = {
        "schema": SCHEMA, "id": secrets.token_hex(8), "month": _month(measurement.get("timestamp")),
        "kind": _choice(measurement.get("kind"), KINDS),
        "tps": tps, "pp_tps": _number(measurement.get("pp_tps"), 0.001, 1000000),
        "ttft_s": _number(measurement.get("ttft_s"), 0.0001, 3600),
        "context": context, "depth": _number(measurement.get("depth"), 0, 1048576, integer=True),
        "variant": {"repo": _text(_field(variant, "repo"), _REPO) if public else None,
                    "filename": _text(os.path.basename(str(_field(variant, "filename") or "")), _FILENAME) if public else None,
                    "sha256": _text(sha.lower() if isinstance(sha, str) else None, _SHA),
                    "quant": _text(quant.upper() if isinstance(quant, str) else None, _QUANT), "layers": layers},
        "hardware": hardware_class(hardware, measurement.get("gpu_uuid"), gpu_layers > 0),
        "settings": {"placement": _placement(gpu_layers, layers), "gpu_layers": gpu_layers,
                     "users": _number(measurement.get("users"), 1, 64, integer=True) or 1,
                     "threads": _number(measurement.get("threads"), 1, 1024, integer=True),
                     "flash_attn": _choice(settings.get("flash_attn"), FLASH),
                     "cache_type_k": _choice(settings.get("cache_type_k"), CACHE),
                     "cache_type_v": _choice(settings.get("cache_type_v"), CACHE),
                     "batch": _number(settings.get("batch"), 1, 65536, integer=True),
                     "ubatch": _number(settings.get("ubatch"), 1, 65536, integer=True),
                     "n_cpu_moe": _number(settings.get("n_cpu_moe"), 0, 4096, integer=True)},
        "runtime": {"version": _text(runtime.get("version") or measurement.get("runtime_build"), _VERSION),
                    "backend": _choice(runtime.get("backend"), BACKENDS)},
    }
    return validate_record(record)


# ---------- validation ----------

def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _int(row, name, low, high, optional=False):
    value = row.get(name)
    if value is None and optional:
        return None
    _check(type(value) is int and low <= value <= high, f"{name} must be an integer between {low} and {high}")
    return value


def _float(row, name, low, high, optional=False):
    value = row.get(name)
    if value is None and optional:
        return None
    _check(not isinstance(value, bool) and isinstance(value, (int, float)) and low <= value <= high
           and math.isfinite(value), f"{name} must be a finite number between {low} and {high}")
    return float(value)


def _str(row, name, pattern, optional=True):
    value = row.get(name)
    if value is None and optional:
        return None
    _check(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{name} has an unexpected format")
    return value


def _enum(row, name, allowed, optional=True):
    value = row.get(name)
    if value is None and optional:
        return None
    _check(isinstance(value, str) and value in allowed, f"{name} must be one of {sorted(allowed)}")
    return value


def _memory(row, name, high):
    value = _int(row, name, BUCKET_GIB, high, optional=True)
    _check(value is None or value % BUCKET_GIB == 0, f"{name} must be rounded to {BUCKET_GIB} GiB")
    return value


def _section(row, name):
    value = row.get(name)
    _check(isinstance(value, dict), f"{name} must be an object")
    return value


def validate_record(row):
    """A clean copy of one schema-v1 record, rebuilt from allowlisted keys. Raises ValueError otherwise.

    Unknown keys are dropped rather than copied, so nothing unexpected can ride along.
    """
    _check(isinstance(row, dict), "record must be an object")
    _check(type(row.get("schema")) is int and row["schema"] == SCHEMA, "record schema must be 1")
    variant, hardware = _section(row, "variant"), _section(row, "hardware")
    settings, runtime = _section(row, "settings"), _section(row, "runtime")
    record = {"schema": SCHEMA, "id": _str(row, "id", re.compile(r"[0-9a-f]{16}"), optional=False),
              "month": _str(row, "month", re.compile(r"20\d\d-(0[1-9]|1[0-2])", re.A), optional=False),
              "kind": _enum(row, "kind", KINDS),
              "tps": _float(row, "tps", 0.001, 100000), "pp_tps": _float(row, "pp_tps", 0.001, 1000000, True),
              "ttft_s": _float(row, "ttft_s", 0.0001, 3600, True),
              "context": _int(row, "context", 1, 1048576), "depth": _int(row, "depth", 0, 1048576, True)}
    record["variant"] = {"repo": _str(variant, "repo", _REPO), "filename": _str(variant, "filename", _FILENAME),
                         "sha256": _str(variant, "sha256", _SHA), "quant": _str(variant, "quant", _QUANT),
                         "layers": _int(variant, "layers", 1, 4096, True)}
    _check(record["variant"]["sha256"] or (record["variant"]["repo"] and record["variant"]["quant"]),
           "variant needs a sha256, or a repo and quant")
    for name in ("cpu", "gpu"):
        value = _str(hardware, name, _NAME)
        _check(value is None or value == value.strip(), f"{name} has an unexpected format")
    record["hardware"] = {"cpu": _str(hardware, "cpu", _NAME), "gpu": _str(hardware, "gpu", _NAME),
                          "backend": _enum(hardware, "backend", BACKENDS, optional=False),
                          "vram_gib": _memory(hardware, "vram_gib", 4096), "ram_gib": _memory(hardware, "ram_gib", 65536),
                          "unified": hardware.get("unified", False),
                          "os": _enum(hardware, "os", OS_FAMILIES, optional=False),
                          "os_version": _str(hardware, "os_version", re.compile(r"\d{1,3}", re.A)),
                          "arch": _enum(hardware, "arch", ARCHES, optional=False)}
    _check(type(record["hardware"]["unified"]) is bool, "unified must be true or false")
    record["settings"] = {"placement": _enum(settings, "placement", PLACEMENTS, optional=False),
                          "gpu_layers": _int(settings, "gpu_layers", 0, 4096),
                          "users": _int(settings, "users", 1, 64),
                          "threads": _int(settings, "threads", 1, 1024, True),
                          "flash_attn": _enum(settings, "flash_attn", FLASH),
                          "cache_type_k": _enum(settings, "cache_type_k", CACHE),
                          "cache_type_v": _enum(settings, "cache_type_v", CACHE),
                          "batch": _int(settings, "batch", 1, 65536, True), "ubatch": _int(settings, "ubatch", 1, 65536, True),
                          "n_cpu_moe": _int(settings, "n_cpu_moe", 0, 4096, True)}
    _check((record["settings"]["placement"] == "cpu") == (record["settings"]["gpu_layers"] == 0),
           "placement does not match gpu_layers")
    record["runtime"] = {"version": _str(runtime, "version", _VERSION), "backend": _enum(runtime, "backend", BACKENDS)}
    return record


def _reject_constant(name):
    raise ValueError(f"{name} is not a valid number")


def parse(text):
    """Validated records from JSON text: {"schema": 1, "records": [...]}, a list, or one record.

    Returns (records, rejected_count). Raises ValueError only when the whole file is unusable.
    """
    if isinstance(text, bytes):
        _check(len(text) <= MAX_BYTES, "Community results are larger than 10 MB; refusing to read them.")
        text = text.decode("utf-8", errors="strict")
    _check(isinstance(text, str), "Community results must be JSON text")
    _check(len(text.encode("utf-8")) <= MAX_BYTES, "Community results are larger than 10 MB; refusing to read them.")
    try:
        data = json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"Community results are not valid JSON ({type(error).__name__}). Check the file or link.") from None
    if isinstance(data, dict) and "records" in data:
        schema = data.get("schema")
        _check(type(schema) is int, "Community results need a schema number")
        _check(schema <= SCHEMA, "These community results use a newer format. Update LLM Configurator to read them.")
        _check(schema == SCHEMA, "Unsupported community results format")
        rows = data["records"]
    else:
        rows = data if isinstance(data, list) else [data]
    _check(isinstance(rows, list), "records must be a list")
    _check(len(rows) <= MAX_ROWS, f"Too many community rows (limit {MAX_ROWS}).")
    records, seen, rejected = [], set(), 0
    for row in rows:
        try:
            record = validate_record(row)
        except (ValueError, TypeError, AttributeError, OverflowError):
            rejected += 1
            continue
        if record["id"] in seen:
            rejected += 1
            continue
        seen.add(record["id"])
        records.append(record)
    return records, rejected


# ---------- import ----------

class _HttpsOnlyRedirect(HTTPRedirectHandler):
    """Follow redirects only to other HTTPS addresses; a downgrade to http is refused."""
    max_redirections = 5

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not str(newurl).lower().startswith("https://"):
            raise ValueError("The community results link redirected to an insecure (http) address; refusing it.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _opener():
    return build_opener(HTTPSHandler(), _HttpsOnlyRedirect())


def _fetch(url, progress=None, cancel=None):
    _check(isinstance(url, str) and url.lower().startswith("https://") and len(url) <= 2048,
           "Community results can only be loaded from an https:// link.")
    request = Request(url, headers={"User-Agent": "LLM-Configurator/0.4", "Accept": "application/json"})
    try:
        with _opener().open(request, timeout=20) as response:
            final = response.geturl() if hasattr(response, "geturl") else url
            _check(str(final).lower().startswith("https://"), "The community results link ended on an insecure address.")
            length = response.headers.get("Content-Length") if getattr(response, "headers", None) else None
            if length and length.isdigit() and int(length) > MAX_BYTES:
                raise ValueError("Community results are larger than 10 MB; refusing to download them.")
            total = int(length) if length and length.isdigit() else None
            chunks, size = [], 0
            while True:
                check_cancel(cancel)
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                _check(size <= MAX_BYTES, "Community results are larger than 10 MB; refusing to download them.")
                chunks.append(chunk)
                if progress:
                    progress({"stage": "download", "done": size, "total": total, "message": "Downloading community results"})
            return b"".join(chunks)
    except HTTPError as error:
        raise ValueError(f"Community results returned HTTP {error.code}. Try again later; saved results stay available.") from None
    except (URLError, TimeoutError, OSError) as error:
        raise ValueError(f"Community results are unavailable ({type(error).__name__}). Check your connection; saved results stay available.") from None


def import_records(store, source=None, text=None, progress=None, cancel=None):
    """Load, validate and save community results. Returns {"source", "fetched_at", "records", "rejected"}."""
    if text is None:
        source = source or DEFAULT_SOURCE
        raw = _fetch(source, progress, cancel)
    else:
        source, raw = source or "pasted", text
    check_cancel(cancel)
    if progress:
        progress({"stage": "validate", "done": 0, "total": None, "message": "Checking every row"})
    records, rejected = parse(raw)
    result = {"source": source, "fetched_at": now(), "records": records, "rejected": rejected}
    store.put("community", {"source": source, "fetched_at": result["fetched_at"], "records": records})
    if progress:
        progress({"stage": "done", "done": len(records), "total": len(records),
                  "message": f"Kept {len(records)} results, dropped {rejected} invalid ones"})
    return result


# ---------- sharing ----------

def _issue_body(payload, fits):
    intro = ("These are anonymous speed results from LLM Configurator. Please read the JSON before submitting: "
             "it contains model names, a rough hardware description (names, memory rounded to 4 GB, OS family), "
             "settings and speeds. No file paths, user names, computer names or serial numbers.\n\n")
    if fits:
        return intro + "```json\n" + payload + "\n```\n"
    return intro + "The results were too long for a link. Paste the JSON that LLM Configurator showed you below this line.\n\n"


def issue_url(payload, count):
    """(url, fits): a prefilled 'new issue' link; when too long for GitHub, a link that asks for a paste."""
    title = f"Community results: {count} speed result{'s' if count != 1 else ''}"

    def build(fits):
        query = urlencode({"title": title, "labels": LABEL, "body": _issue_body(payload, fits)}, quote_via=quote)
        return f"https://github.com/{REPO}/issues/new?{query}"

    url = build(True)
    return (url, True) if len(url) <= MAX_URL else (build(False), False)


def share_payload(store, measurement_ids, hardware=None, variants=None):
    """Anonymised JSON plus a prefilled GitHub issue link. Never sends anything.

    Returns {"json", "issue_url", "fits_in_url", "records"}. `hardware` and `variants`
    default to a fresh scan and the cached catalogue.
    """
    _check(isinstance(measurement_ids, list) and 1 <= len(measurement_ids) <= MAX_SHARE
           and all(isinstance(i, str) and re.fullmatch(r"[\w-]{1,64}", i) for i in measurement_ids),
           f"Choose between 1 and {MAX_SHARE} results to share.")
    measurements = {m.get("id"): m for m in store.get("measurements", []) or [] if isinstance(m, dict) and m.get("id")}
    missing = [i for i in measurement_ids if i not in measurements]
    _check(not missing, "Some chosen results no longer exist. Refresh the list and try again.")
    if hardware is None:
        from .hardware import scan
        hardware = scan(include_processes=False)
    variants = {_field(v, "id"): v for v in (store.get("variants", []) if variants is None else variants)}
    records = []
    for measurement_id in dict.fromkeys(measurement_ids):
        measurement = measurements[measurement_id]
        # The hardware description must describe the computer that produced the number.
        _check(measurement.get("fingerprint") == hardware.get("fingerprint"),
               "One result was measured on different hardware than this computer, so it cannot be described honestly.")
        variant = variants.get(measurement.get("variant_id")) or {"sha256": measurement.get("sha256"), "source": "local"}
        records.append(anonymize(measurement, hardware, variant))
    payload = json.dumps({"schema": SCHEMA, "records": records}, indent=1, allow_nan=False)
    url, fits = issue_url(payload, len(records))
    return {"json": payload, "issue_url": url, "fits_in_url": fits, "records": len(records)}


# ---------- evidence ----------

def context_bucket(context):
    return next(label for limit, label in CONTEXT_BUCKETS if context <= limit)


def _same_model(record, variant):
    theirs, sha = record.get("variant") or {}, _field(variant, "sha256")
    if sha and theirs.get("sha256"):
        return theirs["sha256"] == sha.lower()
    quant = _field(variant, "quant")
    return bool(theirs.get("repo") and theirs.get("repo") == _field(variant, "repo")
                and quant and theirs.get("quant") == quant.upper())


def evidence(records, variant, hardware, config):
    """Median speed from other people's similar computers, or None when fewer than 2 match.

    Tiers, most similar first: same GPU model and backend (GPU runs) or same CPU model
    (CPU-only runs), then the same class (memory bucket + backend). Always the same model
    file/quant, the same placement (full GPU / split / CPU) and the same context bucket.
    """
    config = config or {}
    total = config.get("total_layers") or _field(variant, "layers")
    gpu_layers = config.get("gpu_layers") or 0
    placement = _placement(gpu_layers, total)
    mine = hardware_class(hardware, config.get("gpu_uuid"), placement != "cpu")
    if config.get("gpu_backend") in BACKENDS and placement != "cpu":
        mine["backend"] = config["gpu_backend"]
    bucket, users = context_bucket(config.get("context") or 8192), config.get("parallel") or 1
    pool = []
    for record in records or []:
        try:
            settings, theirs = record["settings"], record["hardware"]
            if (settings["placement"] == placement and (settings.get("users") or 1) == users
                    and context_bucket(record["context"]) == bucket and _same_model(record, variant)
                    and math.isfinite(record["tps"]) and record["tps"] > 0):
                pool.append((record["tps"], theirs))
        except (KeyError, TypeError, AttributeError):
            continue
    if placement == "cpu":
        tiers = [("same_cpu", lambda h: mine["cpu"] and h.get("cpu") == mine["cpu"] and h.get("arch") == mine["arch"]),
                 ("same_class", lambda h: h.get("arch") == mine["arch"] and h.get("ram_gib") == mine["ram_gib"])]
    else:
        tiers = [("same_gpu", lambda h: mine["gpu"] and h.get("gpu") == mine["gpu"] and h.get("backend") == mine["backend"]),
                 ("same_class", lambda h: h.get("backend") == mine["backend"] and mine["vram_gib"]
                  and h.get("vram_gib") == mine["vram_gib"] and bool(h.get("unified")) == mine["unified"])]
    for name, similar in tiers:
        speeds = sorted(tps for tps, theirs in pool if similar(theirs))
        if len(speeds) < 2:
            continue
        low, _, high = statistics.quantiles(speeds, n=4, method="inclusive")
        return {"median_tps": round(statistics.median(speeds), 1), "n": len(speeds), "similar": name,
                "range": [round(low, 1), round(high, 1)], "placement": placement, "context_bucket": bucket, "note": NOTE}
    return None
