"""Remote metadata only; refreshing never downloads model weights.

Sizes and checksums come from the Hugging Face file listing, never from guesses. A split (sharded) model
is offered only when every shard is listed. Derived numbers (expert share, active parameters) are
computed from config.json shapes, as described in `moe_shape`.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from http.client import HTTPException
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .domain import ARCHITECTURES, GIB, MOE_ARCHITECTURES, QUANT_BYTES_PER_PARAMETER, Cancelled, Variant, check_cancel, now
from .credentials import resolve, validate_key

HF = "https://huggingface.co"
QUANTS = tuple(QUANT_BYTES_PER_PARAMETER)  # every quant the memory model understands
# The default shortlist: enough steps to trade quality for memory without flooding the results.
DEFAULT_QUANTS = ("Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "MXFP4")
FULL_PRECISION = ("F16", "BF16")  # one of them is offered: F16 when present, it runs on every llama.cpp backend
MXFP4_BYTES_PER_WEIGHT = 17 / 32  # 32 four-bit values plus one shared scale byte
FULL_PRECISION_MAX_PARAMETERS = 4.5e9  # 16-bit files are only a sensible choice for small models
SCORE_FIELDS = {"general": "artificial_analysis_intelligence_index", "coding": "artificial_analysis_coding_index",
                "agentic": "artificial_analysis_agentic_index"}
SHARD = re.compile(r"^(?P<stem>.+)-(?P<index>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
SKIP_FILE = re.compile(r"mmproj|imatrix", re.I)  # vision add-ons and calibration data are not models
SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_WORKERS = 6  # polite to Hugging Face with ~40 entries; the score request shares the pool
MAX_USER_ENTRIES = 200
REMOVED_KEY = "catalogue_removed"
_user_copy_lock = threading.Lock()
# Transformers defaults for fields Gemma 3 configs may leave out (Gemma3TextConfig).
CONFIG_DEFAULTS = {"gemma3_text": {"vocab_size": 262208, "hidden_size": 2304, "intermediate_size": 9216,
                                   "num_hidden_layers": 26, "num_attention_heads": 8, "num_key_value_heads": 4,
                                   "head_dim": 256, "max_position_embeddings": 131072, "sliding_window": 4096,
                                   "sliding_window_pattern": 6}}


# Headers that carry a secret (HF_TOKEN, the Artificial Analysis key); never sent to another host.
_SECRET_HEADERS = {"authorization", "x-api-key"}


def _origin(url):
    parts = urlsplit(url)
    return parts.scheme.lower(), (parts.hostname or "").lower(), parts.port


class _MetadataRedirect(HTTPRedirectHandler):
    """Like downloads.DownloadRedirect: urllib copies every header to the new URL, so a redirect to another
    host (a CDN, or anywhere a compromised mirror points) would receive the token. Drop secrets when the
    host changes, and never follow an HTTPS request down to plain HTTP."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlsplit(req.full_url).scheme.lower() == "https" and urlsplit(newurl).scheme.lower() != "https":
            if fp is not None:
                fp.close()
            raise ValueError("Metadata request was redirected away from HTTPS; refused")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and _origin(req.full_url) != _origin(newurl):
            for name in [n for n in redirected.headers if n.lower() in _SECRET_HEADERS]:
                del redirected.headers[name]
        return redirected


def _open(request, timeout=25):
    """The one place catalogue metadata touches the network."""
    return build_opener(_MetadataRedirect()).open(request, timeout=timeout)


def get_json(url, headers=None):
    request = Request(url, headers={"User-Agent": "LLM-Configurator/0.1", **(headers or {})})
    try:
        with _open(request) as response:
            raw = response.read(12 * 1024**2 + 1)
        if len(raw) > 12 * 1024**2:
            raise ValueError("Metadata response exceeds 12 MiB")
        return json.loads(raw)
    except HTTPError as error:
        raise ValueError(f"Metadata request returned HTTP {error.code}; check access, API key or rate limit") from None
    except (URLError, TimeoutError, OSError, HTTPException, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Metadata unavailable ({type(error).__name__}); cached results remain available") from None


def valid_repo(value):
    return isinstance(value, str) and bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value, re.ASCII)) and ".." not in value \
        and not any(part.startswith(".") for part in value.split("/"))


def validate_entry(entry):
    if not isinstance(entry, dict):
        raise ValueError("Each catalogue entry must be an object")
    for key in ["base_repo", "gguf_repo"] + (["config_repo"] if entry.get("config_repo") is not None else []):
        if not valid_repo(entry.get(key)):
            raise ValueError(f"Invalid Hugging Face repository: {key} (use the form owner/name)")
    if not isinstance(entry.get("tags", []), list) or any(not isinstance(t, str) for t in entry.get("tags", [])):
        raise ValueError(f"{entry['base_repo']}: tags must be a list of words")
    quants = entry.get("quants")
    if quants is not None and (not isinstance(quants, list) or not quants or any(q not in QUANTS for q in quants)):
        raise ValueError(f"{entry['base_repo']}: quants must list known formats: {', '.join(QUANTS)}")
    for key in ["family", "license", "aa_slug"]:
        if entry.get(key) is not None and not isinstance(entry[key], str):
            raise ValueError(f"{entry['base_repo']}: {key} must be text")
    return entry


def packaged_entries():
    return json.loads(Path(__file__).with_name("catalogue.json").read_text(encoding="utf-8"))


def user_copy(store):
    return store.directory / "catalogue.json"


def _saved_entries(store):
    """The user copy's valid entries. A damaged file is set aside (never deleted) so the app still starts."""
    path = user_copy(store)
    if not path.exists():
        return []
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(saved, list):
            raise ValueError("not a list")
    except (OSError, ValueError, UnicodeDecodeError):
        try:
            os.replace(path, path.with_name(f"catalogue.json.damaged-{int(time.time())}"))
        except OSError:
            pass
        return []
    valid = []
    for entry in saved:
        try:
            valid.append(validate_entry(entry))
        except (ValueError, KeyError, TypeError):
            continue  # one bad hand edit must not hide every model
    return valid


def definitions(store):
    """The shipped catalogue plus the user's copy.

    The user copy may hold benchmark mappings (`aa_slug`) for shipped entries and whole user-added
    entries (`"user": true`). Shipped entries the user removed stay hidden, and entries added to the
    shipped list in later versions still appear for people who already have a copy. Shipped entries
    are marked `"shipped": true`, so a later version that drops one does not bring it back as "custom".
    """
    shipped = packaged_entries()
    if not isinstance(shipped, list):
        raise ValueError("catalogue.json must contain an array")
    for entry in shipped:
        validate_entry(entry)
    mine = {e["base_repo"]: e for e in _saved_entries(store)}
    removed = set(store.get(REMOVED_KEY, []) or [])
    entries = []
    for entry in shipped:
        own = mine.pop(entry["base_repo"], None)
        if entry["base_repo"] in removed and not (own and own.get("user")):
            continue
        if own and own.get("user"):
            entries.append(dict(own))
        else:
            entries.append({**entry, "shipped": True,
                            "aa_slug": own.get("aa_slug") if own and "aa_slug" in own else entry.get("aa_slug")})
    # Anything else in the copy was added by the user (including hand-edited v0.3 copies).
    entries.extend({**entry, "user": True} for entry in mine.values() if not entry.get("shipped"))
    return entries


def _write_user_copy(store, entries):
    path = user_copy(store)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".catalogue-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(entries, file, indent=2)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def add_entry(store, base_repo, gguf_repo, config_repo=None, family=None, tags=None):
    """Add a Hugging Face model: `base_repo` holds config.json, `gguf_repo` holds the GGUF files.

    Only names are checked here; the next refresh fetches the files and explains any problem.
    """
    entry = validate_entry({"base_repo": base_repo, "gguf_repo": gguf_repo, "aa_slug": None, "family": family,
                            "tags": list(tags or []) + ["custom"], "user": True,
                            **({"config_repo": config_repo} if config_repo else {})})
    with _user_copy_lock:
        entries = definitions(store)
        # Hugging Face repo names are not case-sensitive.
        if any(e["base_repo"].lower() == base_repo.lower() for e in entries):
            raise ValueError(f"{base_repo} is already in the catalogue")
        if any(e["gguf_repo"].lower() == gguf_repo.lower() for e in entries):
            raise ValueError(f"{gguf_repo} is already used by another catalogue entry")
        if sum(bool(e.get("user")) for e in entries) >= MAX_USER_ENTRIES:
            raise ValueError(f"You can add up to {MAX_USER_ENTRIES} models; remove one first")
        _write_user_copy(store, entries + [entry])
        store.update(REMOVED_KEY, lambda items: [b for b in (items or []) if b != base_repo], [])
    return entry


def set_slug(store, base_repo, slug):
    """Save a benchmark mapping (`aa_slug`) with the same lock and atomic write as add/remove."""
    with _user_copy_lock:
        entries = definitions(store)
        entry = next((e for e in entries if e["base_repo"] == base_repo), None)
        if not entry:
            raise ValueError("Model is not in the configured catalogue")
        entry["aa_slug"] = slug or None
        _write_user_copy(store, entries)
    return entry


def remove_entry(store, base_repo):
    """Remove a model from the catalogue and drop its cached variants; files on disk are untouched."""
    with _user_copy_lock:
        entries = definitions(store)
        entry = next((e for e in entries if e["base_repo"] == base_repo), None)
        if not entry:
            raise ValueError(f"{base_repo} is not in the catalogue")
        _write_user_copy(store, [e for e in entries if e["base_repo"] != base_repo])
        if any(e["base_repo"] == base_repo for e in packaged_entries()):
            store.update(REMOVED_KEY, lambda items: sorted(set(items or []) | {base_repo}), [])
    kept = store.update("variants", lambda items: [v for v in (items or []) if v.get("base_repo") != base_repo], [])
    return {"base_repo": base_repo, "removed": True, "user": bool(entry.get("user")), "variants": len(kept)}


def hf_headers():
    return {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}


def load_config(entry, base_info, headers):
    """config.json from the base repo, or from `config_repo` (an ungated copy) when the base is gated."""
    base, mirror = entry["base_repo"], entry.get("config_repo")
    sources = [(base, base_info["sha"])]
    if mirror and mirror != base:
        # Without a token a gated repo always refuses, so skip the doomed request.
        sources = ([] if base_info.get("gated") and not headers else sources) + [(mirror, None)]
    error = None
    for repo, revision in sources:
        try:
            revision = revision or get_json(f"{HF}/api/models/{repo}", headers)["sha"]
            config = get_json(f"{HF}/{repo}/resolve/{quote(revision)}/config.json", headers)
            if not isinstance(config, dict):
                raise ValueError(f"{repo}: config.json is not a settings object")
            return config
        except (ValueError, KeyError, TypeError) as failure:
            error = failure
    if base_info.get("gated") and not mirror:
        raise ValueError(f"{base} is gated: accept its licence on Hugging Face and set HF_TOKEN, "
                         "or add an ungated config_repo to this catalogue entry")
    raise error


def flatten_config(config):
    """Multimodal configs (Gemma 3, Mistral 3) nest the language model under `text_config`."""
    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    top = config.get("model_type")
    architecture = top if top in ARCHITECTURES else nested.get("model_type", top)
    kind = nested.get("model_type") or top
    flat = {**CONFIG_DEFAULTS.get(kind, {}), **{k: v for k, v in config.items() if k != "text_config"}, **nested}
    flat["model_type"] = kind
    return flat, architecture


def positive_int(config, *names):
    for name in names:
        value = config.get(name)
        if type(value) is int and value > 0:
            return value
    return None


def sliding(config, layers, max_context):
    """(window, number of layers whose KV cache is capped at the window)."""
    window = config.get("sliding_window")
    if type(window) is not int or window <= 0 or window >= max_context:
        return None, 0  # a window as long as the context never saves memory
    kind, types, pattern = config.get("model_type"), config.get("layer_types"), config.get("sliding_window_pattern")
    if isinstance(types, list) and len(types) == layers:
        count = sum(t == "sliding_attention" for t in types)
    elif kind in {"qwen2", "qwen3", "qwen2_moe", "qwen3_moe"}:
        # Qwen quirk: despite the name, layers from max_window_layers upward use the window,
        # and only when use_sliding_window is true (it is false in every released Qwen model).
        start = config.get("max_window_layers")
        count = layers - min(start, layers) if config.get("use_sliding_window") and type(start) is int else 0
    elif type(pattern) is int and pattern > 0:
        count = layers - layers // pattern  # every pattern-th layer keeps full attention
    elif kind == "gemma2":
        count = (layers + 1) // 2  # alternating local/global layers
    else:
        count = 0  # e.g. Mistral 7B v0.1 declares a window but llama.cpp runs full attention
    return (window, count) if count else (None, 0)


def moe_layers(config, layers):
    step = positive_int(config, "decoder_sparse_step") or 1
    dense = set(config.get("mlp_only_layers") or [])
    first = config.get("first_k_dense_replace") if type(config.get("first_k_dense_replace")) is int else 0
    return [i for i in range(layers) if i not in dense and i >= first and (i + 1) % step == 0]


def moe_shape(config, layers):
    """Expert counts and parameters, or None for dense models.

    routed = moe_layers x experts x 3 x hidden_size x expert_ffn   (gate, up and down matrices)
    active = total - routed x (experts - active_experts) / experts
    expert_fraction = routed / total. It is a parameter share used as the byte share: llama.cpp quantises
    expert matrices like the other large matrices, so the two match closely (MXFP4 files keep the small
    non-expert part at higher precision, so the byte share there is a little lower).
    """
    experts = positive_int(config, "num_experts", "num_local_experts", "n_routed_experts")
    active = positive_int(config, "num_experts_per_tok", "experts_per_token", "moe_topk")
    if not experts:
        return None
    hidden = positive_int(config, "hidden_size")
    expert_ffn = positive_int(config, "moe_intermediate_size", "intermediate_size")
    if not (active and hidden and expert_ffn) or active > experts:
        raise ValueError("incomplete mixture-of-experts config (expert count, size or top-k missing)")
    sparse = moe_layers(config, layers)
    shared = (positive_int(config, "shared_expert_intermediate_size") or 0) + \
             (positive_int(config, "n_shared_experts") or 0) * expert_ffn
    return {"experts": experts, "active_experts": active, "moe_layers": len(sparse), "shared_ffn": shared,
            "routed": len(sparse) * experts * 3 * hidden * expert_ffn}


def estimate_parameters(config, layers, moe):
    """Parameter count from config shapes (embeddings, attention, MLP); norms and biases are ignored."""
    hidden, heads, vocab = (positive_int(config, n) for n in ["hidden_size", "num_attention_heads", "vocab_size"])
    if not (hidden and heads and vocab):
        return None
    kv = positive_int(config, "num_key_value_heads") or heads
    head_dim = positive_int(config, "head_dim") or hidden // heads
    attention = hidden * head_dim * (2 * heads + 2 * kv)
    dense_ffn = 3 * hidden * (positive_int(config, "intermediate_size") or 0)
    sparse_layers = moe["moe_layers"] if moe else 0
    if not dense_ffn and sparse_layers < layers:
        return None
    total = layers * attention + (layers - sparse_layers) * dense_ffn
    if moe:
        total += moe["routed"] + sparse_layers * (3 * hidden * moe["shared_ffn"] + moe["experts"] * hidden)
    return total + vocab * hidden * (1 if config.get("tie_word_embeddings", True) else 2)


def reported_parameters(*infos):
    for info in infos:
        for value in [(info.get("safetensors") or {}).get("total"), (info.get("gguf") or {}).get("total")]:
            if type(value) is int and value > 0:
                return value
    return None


def quant_of(path):
    stem = path[:-5] if path.lower().endswith(".gguf") else path
    for quant in sorted(QUANTS, key=len, reverse=True):
        # Trailing '_' is not a boundary: Q6_K_L or Q4_0_4_4 are different formats.
        if re.search(r"(?:^|[-_./])" + re.escape(quant) + r"(?=[-./]|$)", stem, re.I):
            return quant
    return None


def gguf_artifacts(siblings):
    """Single files and complete shard sets as {"filename", "size_bytes", "sha256", "files", "quant"}."""
    singles, shard_sets = [], {}
    for file in siblings:
        name = file.get("rfilename", "")
        if not name.lower().endswith(".gguf") or SKIP_FILE.search(name.rsplit("/", 1)[-1]):
            continue
        lfs = file.get("lfs") or {}
        size = file.get("size") or lfs.get("size")
        # An LFS oid is the file's sha256. Without one a file can be neither downloaded safely nor recognised.
        sha = str(lfs.get("sha256") or lfs.get("oid") or "").lower()
        record = {"filename": name, "size_bytes": size, "sha256": sha if SHA256.fullmatch(sha) else None}
        match = SHARD.match(name)
        if match:
            shard_sets.setdefault((match["stem"], int(match["total"])), {})[int(match["index"])] = record
        elif type(size) is int and size > 0 and record["sha256"]:
            singles.append({**record, "files": [], "quant": quant_of(name)})
    for (stem, total), parts in shard_sets.items():
        # Never count a partial set as a model: every shard must be listed with its size and checksum.
        if total < 1 or sorted(parts) != list(range(1, total + 1)) or \
                any(type(p["size_bytes"]) is not int or p["size_bytes"] <= 0 or not p["sha256"] for p in parts.values()):
            continue
        files = [parts[i] for i in range(1, total + 1)]
        singles.append({"filename": files[0]["filename"], "size_bytes": sum(f["size_bytes"] for f in files),
                        "sha256": files[0]["sha256"], "files": files if total > 1 else [], "quant": quant_of(stem)})
    return singles


def pick_artifacts(artifacts, allowed):
    """One file per quant: plain uploads before 'UD-' dynamic variants, single files before shards."""
    best = {}
    for artifact in artifacts:
        if artifact["quant"] not in allowed:
            continue
        key = ("ud-" in artifact["filename"].lower(), bool(artifact["files"]), len(artifact["filename"]), artifact["filename"])
        if artifact["quant"] not in best or key < best[artifact["quant"]][0]:
            best[artifact["quant"]] = (key, artifact)
    if "F16" in best:
        best.pop("BF16", None)  # the same weights twice; F16 runs everywhere
    return [best[q][1] for q in QUANTS if q in best]


def fetch_variants(entry, cancel=None):
    base, repo = entry["base_repo"], entry["gguf_repo"]
    headers = hf_headers()
    base_info = get_json(f"{HF}/api/models/{base}", headers)
    base_revision = base_info["sha"]
    check_cancel(cancel)
    config, architecture = flatten_config(load_config(entry, base_info, headers))
    check_cancel(cancel)
    info = get_json(f"{HF}/api/models/{repo}?blobs=true", headers)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"{base}: model type '{architecture}' is not supported yet; "
                         f"supported: {', '.join(sorted(ARCHITECTURES))}")
    layers, heads = positive_int(config, "num_hidden_layers"), positive_int(config, "num_attention_heads")
    max_context = positive_int(config, "max_position_embeddings")
    kv_heads = positive_int(config, "num_key_value_heads") or heads
    head_dim = positive_int(config, "head_dim") or (positive_int(config, "hidden_size") or 0) // (heads or 1)
    if not (layers and kv_heads and max_context and head_dim):
        raise ValueError(f"{base}: config.json lacks layer, attention head or context length details")
    moe = moe_shape(config, layers)
    if architecture in MOE_ARCHITECTURES and not moe:
        raise ValueError(f"{base}: mixture-of-experts model without expert counts in config.json")
    estimate = estimate_parameters(config, layers, moe)
    parameters = reported_parameters(base_info, info)
    if estimate and (not parameters or parameters < 0.8 * estimate):
        parameters = estimate  # packed checkpoints (e.g. MXFP4 stored as bytes) under-report their count
    window, sliding_layers = sliding(config, layers, max_context)
    moe_fields = {"experts": 0, "active_experts": 0, "expert_fraction": 0.0, "active_parameters": parameters}
    if moe:
        if not parameters:
            raise ValueError(f"{base}: cannot size the experts without a parameter count")
        idle = moe["routed"] * (moe["experts"] - moe["active_experts"]) / moe["experts"]
        moe_fields = {"experts": moe["experts"], "active_experts": moe["active_experts"],
                      "expert_fraction": round(min(moe["routed"] / parameters, 0.99), 4),
                      "active_parameters": int(round(parameters - idle))}
    allowed = set(entry.get("quants") or DEFAULT_QUANTS)
    if not entry.get("quants") and parameters and parameters <= FULL_PRECISION_MAX_PARAMETERS:
        allowed |= set(FULL_PRECISION)
    license_name = entry.get("license") or (base_info.get("cardData") or {}).get("license")
    variants = []
    for artifact in pick_artifacts(gguf_artifacts(info.get("siblings", [])), allowed):
        if moe and artifact["quant"] == "MXFP4":
            # MXFP4 files keep only the experts at ~4.25 bits; the rest is stored larger, so the expert share of
            # the file's bytes is lower than their share of parameters (gpt-oss-20b: about 0.84 vs 0.91).
            moe_fields = {**moe_fields, "expert_fraction": round(min(
                moe["routed"] * MXFP4_BYTES_PER_WEIGHT / artifact["size_bytes"], moe["routed"] / parameters, 0.99), 4)}
        elif moe:
            moe_fields = {**moe_fields, "expert_fraction": round(min(moe["routed"] / parameters, 0.99), 4)}
        variants.append(Variant(
            id=f"{repo}@{info['sha']}:{artifact['filename']}", name=base.split("/")[-1], base_repo=base, repo=repo,
            revision=info["sha"], base_revision=base_revision, filename=artifact["filename"],
            sha256=artifact["sha256"], quant=artifact["quant"], size_bytes=artifact["size_bytes"], layers=layers,
            kv_heads=kv_heads, head_dim=head_dim, max_context=max_context, architecture=architecture,
            files=artifact["files"], parameters=parameters, sliding_window=window, sliding_layers=sliding_layers,
            family=entry.get("family"), license=license_name if isinstance(license_name, str) else None,
            source="custom" if entry.get("user") else "catalogue", **moe_fields))
    if not variants:
        raise ValueError(f"{repo}: no GGUF files in the supported formats ({', '.join(sorted(allowed))}) "
                         "with complete size information")
    return variants


def fetch_scores(cancel=None):
    key, _ = resolve()
    if not key:
        raise ValueError("Add an API key in Benchmark settings to retrieve Artificial Analysis rankings; quality remains unknown without it")
    items, version = [], None
    for page in range(1, 101):
        check_cancel(cancel)
        payload = get_json(f"https://artificialanalysis.ai/api/v2/language/models/free?page={page}", {"x-api-key": key})
        current = payload.get("intelligence_index_version")
        if page > 1 and current != version:
            raise ValueError("Benchmark version changed during pagination; retry refresh")
        version = current
        items.extend(payload["data"])
        pagination = payload.get("pagination", {})
        if not pagination.get("has_more", page < pagination.get("total_pages", 1)):
            return {"version": str(version) if version is not None else None, "fetched_at": now(), "data": items}
    raise ValueError("Unexpected benchmark pagination; refusing incomplete rankings")


def apply_scores(variant, entry, score_cache):
    # Explicit slug required: reasoning and non-reasoning entries must never be conflated.
    slug = entry.get("aa_slug")
    if not slug or not score_cache:
        return
    matches = [item for item in score_cache["data"] if item.get("slug") == slug]
    if len(matches) != 1:
        return
    item = matches[0]
    variant.scores = {key: float(item["evaluations"][field]) for key, field in SCORE_FIELDS.items()
                      if isinstance(item.get("evaluations", {}).get(field), (int, float))
                      and not isinstance(item["evaluations"][field], bool) and math.isfinite(item["evaluations"][field])}
    variant.score_source = f"https://artificialanalysis.ai/models/{quote(slug)}"
    variant.score_version = score_cache["version"]
    variant.score_settings = item.get("name", slug)


def cached_variants(old, entry):
    # `source` follows the entry, not the cache: a v0.3 cache says "catalogue" even for a user's own repo.
    return [Variant(**{**v, "source": "custom" if entry.get("user") else "catalogue"}) for v in old
            if v["repo"] == entry["gguf_repo"] and v.get("base_repo", entry["base_repo"]) == entry["base_repo"]]


def refresh(store, include_scores=True, include_models=True, progress=None, cancel=None):
    errors = []
    score_cache = store.get("scores")
    old = store.get("variants", [])
    variants = []
    entries = definitions(store)
    state = {"models_done": 0, "models_total": len(entries) if include_models else 0,
             "models_failed": 0, "scores": "running" if include_scores else "skipped"}
    total = state["models_total"] + int(include_scores)

    def notify():
        # Contract keys (stage/done/total/message) next to the older ones the page already reads.
        if progress:
            done = state["models_done"] + int(state["scores"] in {"complete", "failed"})
            message = f"Checked {state['models_done']} of {state['models_total']} model sources"
            progress({**state, "stage": "metadata", "done": done, "total": total, "message": message})
    notify()
    # Fetch rankings and model metadata concurrently; only this thread writes the cache.
    pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    cancelled = False
    try:
        # `cancel` is passed only when given, so simple stand-ins for these functions keep working.
        extra = {"cancel": cancel} if cancel is not None else {}
        score_future = pool.submit(fetch_scores, **extra) if include_scores else None
        futures = [pool.submit(fetch_variants, entry, **extra) for entry in entries] if include_models else [None] * len(entries)
        pending = {future for future in futures if future is not None} | ({score_future} if score_future else set())
        while pending:
            check_cancel(cancel)  # checked often, not only when a slow request finishes
            finished, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            for future in finished:
                failed = future.exception() is not None
                if future is score_future:
                    state["scores"] = "failed" if failed else "complete"
                else:
                    state["models_done"] += 1
                    state["models_failed"] += int(failed)
                notify()
    except Cancelled:
        cancelled = True
        raise
    finally:
        # On cancel, do not wait for requests already on the wire; their results are thrown away.
        pool.shutdown(wait=not cancelled, cancel_futures=True)
    if score_future:
        try:
            score_cache = score_future.result()
            store.put("scores", score_cache)
        except Exception as error:  # noqa: BLE001 - rankings are optional; keep the old ones
            errors.append(str(error) if isinstance(error, ValueError) else f"Benchmark rankings failed ({type(error).__name__})")
    for entry, future in zip(entries, futures):
        try:
            fetched = future.result() if future else cached_variants(old, entry)
        except Exception as error:  # noqa: BLE001 - one odd reply must not lose every other model
            errors.append(f"{entry['base_repo']}: {error}" if isinstance(error, (ValueError, KeyError, TypeError))
                          else f"{entry['base_repo']}: unexpected reply from Hugging Face ({type(error).__name__})")
            fetched = cached_variants(old, entry)
        for variant in fetched:
            # Clear stale mappings if the user removes or changes a slug.
            variant.scores = {}
            variant.score_source = variant.score_version = variant.score_settings = None
            apply_scores(variant, entry, score_cache)
        variants.extend(v.to_dict() for v in fetched)
    store.put("variants", variants)
    status = {"timestamp": now(), "variants": len(variants), "warnings": errors,
              "scores_fetched_at": score_cache.get("fetched_at") if score_cache else None}
    store.put("refresh_status", status)
    return status


def refresh_entry(store, base_repo, cancel=None):
    """Fetch one entry (for example right after add_entry) and replace only its cached variants."""
    entry = next((e for e in definitions(store) if e["base_repo"] == base_repo), None)
    if not entry:
        raise ValueError(f"{base_repo} is not in the catalogue")
    fetched = fetch_variants(entry, **({"cancel": cancel} if cancel is not None else {}))
    for variant in fetched:
        apply_scores(variant, entry, store.get("scores"))
    store.update("variants", lambda items: [v for v in (items or []) if v.get("base_repo") != base_repo]
                 + [v.to_dict() for v in fetched], [])
    return {"base_repo": base_repo, "variants": len(fetched)}


def demo_variants():
    """Fictional fixtures for exploring the interface, never mixed with real results."""
    result = []
    for name, size, layers, kv_heads in [("Demo Small", 4, 32, 4), ("Demo Medium", 8, 32, 8), ("Demo Large", 14, 40, 8)]:
        for quant, bytes_per_param in [("Q4_K_M", 0.62), ("Q6_K", 0.84), ("Q8_0", 1.1)]:
            result.append(Variant(id=f"demo:{name}:{quant}", name=name, base_repo=f"demo/{name.replace(' ', '-')}", repo="demo/fictional",
                                  revision="demo", base_revision="demo", filename=f"{name}-{quant}.gguf", sha256=None,
                                  quant=quant, size_bytes=int(size * bytes_per_param * GIB), layers=layers,
                                  kv_heads=kv_heads, head_dim=128, max_context=32768, architecture="qwen3", demo=True))
    return result


def test_connection(value=None):
    key = validate_key(value) if value else resolve()[0]
    if not key:
        raise ValueError("Enter or save an Artificial Analysis API key first.")
    payload = get_json("https://artificialanalysis.ai/api/v2/language/models/free?page=1", {"x-api-key": key})
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Unexpected response from Artificial Analysis.")
    return {"ok": True, "message": "Connection successful. Your key can retrieve benchmark data."}
