"""Remote metadata only; refreshing never downloads model weights."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
from pathlib import Path
import re
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .domain import GIB, Variant, now
from .credentials import resolve, validate_key

QUANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
SCORE_FIELDS = {"general": "artificial_analysis_intelligence_index", "coding": "artificial_analysis_coding_index",
                "agentic": "artificial_analysis_agentic_index"}


def get_json(url, headers=None):
    request = Request(url, headers={"User-Agent": "LLM-Configurator/0.1", **(headers or {})})
    try:
        with urlopen(request, timeout=25) as response:
            raw = response.read(12 * 1024**2 + 1)
        if len(raw) > 12 * 1024**2:
            raise ValueError("Metadata response exceeds 12 MiB")
        return json.loads(raw)
    except HTTPError as error:
        raise ValueError(f"Metadata request returned HTTP {error.code}; check access, API key or rate limit") from None
    except (URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Metadata unavailable ({type(error).__name__}); cached results remain available") from None


def definitions(store):
    custom = store.directory / "catalogue.json"
    path = custom if custom.exists() else Path(__file__).with_name("catalogue.json")
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError("catalogue.json must contain an array")
    for entry in entries:
        for key in ["base_repo", "gguf_repo"]:
            if not re.fullmatch(r"[\w.-]+/[\w.-]+", entry.get(key, "")):
                raise ValueError(f"Invalid Hugging Face repository: {key}")
    return entries


def fetch_variants(entry):
    base, repo = entry["base_repo"], entry["gguf_repo"]
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    base_info = get_json(f"https://huggingface.co/api/models/{base}", headers)
    base_revision = base_info["sha"]
    config = get_json(f"https://huggingface.co/{base}/resolve/{quote(base_revision)}/config.json", headers)
    info = get_json(f"https://huggingface.co/api/models/{repo}?blobs=true", headers)
    architecture = config.get("model_type")
    if architecture not in {"qwen2", "qwen3", "llama"} or config.get("num_local_experts") or config.get("num_experts"):
        raise ValueError(f"{base}: unsupported architecture; dense qwen2/qwen3/llama only")
    layers, heads = config["num_hidden_layers"], config["num_key_value_heads"]
    head_dim = config.get("head_dim") or config["hidden_size"] // config["num_attention_heads"]
    variants = []
    for file in info.get("siblings", []):
        filename = file["rfilename"]
        # Initial release supports single-file artifacts; never count one shard as a full model.
        if not filename.lower().endswith(".gguf") or re.search(r"-\d{5}-of-\d{5}", filename):
            continue
        quant = next((q for q in QUANTS if re.search(r"(?:^|[-_.])" + q + r"(?:[-_.]|$)", filename, re.I)), None)
        lfs = file.get("lfs") or {}
        size = file.get("size") or lfs.get("size")
        if not quant or not size:
            continue
        variants.append(Variant(id=f"{repo}@{info['sha']}:{filename}", name=base.split("/")[-1], base_repo=base,
                                repo=repo, revision=info["sha"], base_revision=base_revision, filename=filename,
                                sha256=lfs.get("sha256") or lfs.get("oid"), quant=quant, size_bytes=size,
                                layers=layers, kv_heads=heads, head_dim=head_dim,
                                max_context=config["max_position_embeddings"], architecture=architecture))
    if not variants:
        raise ValueError(f"{repo}: no supported single-file GGUF variants with size metadata")
    return variants


def fetch_scores():
    key, _ = resolve()
    if not key:
        raise ValueError("Add an API key in Benchmark settings to retrieve Artificial Analysis rankings; quality remains unknown without it")
    items, version = [], None
    for page in range(1, 101):
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


def refresh(store, include_scores=True, include_models=True, progress=None):
    errors = []
    score_cache = store.get("scores")
    old = store.get("variants", [])
    variants = []
    entries = definitions(store)
    state = {"models_done": 0, "models_total": len(entries) if include_models else 0,
             "models_failed": 0, "scores": "running" if include_scores else "skipped"}
    def notify():
        if progress:
            progress(dict(state))
    notify()
    # Fetch rankings and model metadata concurrently; only this thread writes the cache.
    with ThreadPoolExecutor(max_workers=5) as pool:
        score_future = pool.submit(fetch_scores) if include_scores else None
        futures = [pool.submit(fetch_variants, entry) for entry in entries] if include_models else [None] * len(entries)
        jobs = [future for future in futures if future is not None] + ([score_future] if score_future else [])
        for future in as_completed(jobs):
            failed = future.exception() is not None
            if future is score_future:
                state["scores"] = "failed" if failed else "complete"
            else:
                state["models_done"] += 1
                state["models_failed"] += int(failed)
            notify()
    if score_future:
        try:
            score_cache = score_future.result()
            store.put("scores", score_cache)
        except ValueError as error:
            errors.append(str(error))
    for entry, future in zip(entries, futures):
        try:
            fetched = future.result() if future else [Variant(**v) for v in old if v["repo"] == entry["gguf_repo"]]
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"{entry['base_repo']}: {error}")
            fetched = [Variant(**v) for v in old if v["repo"] == entry["gguf_repo"]]
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
