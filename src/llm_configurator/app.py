"""Application orchestration shared by CLI and local interface."""
import json

from .catalogue import apply_scores, definitions, demo_variants
from .domain import Requirements, Variant
from .engine import recommend
from .hardware import scan


def variants(store, demo=False):
    return demo_variants() if demo else [Variant(**v) for v in store.get("variants", [])]


def evaluate(store, payload, demo=False):
    requirements = Requirements(**payload)
    models = variants(store, demo)
    report = recommend(models, scan(), requirements, store.get("measurements", []))
    report["demo"] = demo
    report["catalogue_status"] = store.get("refresh_status")
    if not models:
        report["notes"].insert(0, "No catalogue data yet. Refresh model metadata or enable the fictional demo.")
    return report


def map_benchmark(store, base_repo, slug):
    entries = definitions(store)
    entry = next((e for e in entries if e["base_repo"] == base_repo), None)
    if not entry:
        raise ValueError("Model is not in the configured catalogue")
    cache = store.get("scores")
    if slug and (not cache or not any(i.get("slug") == slug for i in cache["data"])):
        raise ValueError("Unknown benchmark slug; refresh Artificial Analysis data first")
    entry["aa_slug"] = slug or None
    path = store.directory / "catalogue.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    temporary.replace(path)
    models = variants(store)
    for variant in models:
        if variant.base_repo == base_repo:
            variant.scores = {}
            variant.score_source = variant.score_version = variant.score_settings = None
            apply_scores(variant, entry, cache)
    store.put("variants", [v.to_dict() for v in models])
    return {"base_repo": base_repo, "aa_slug": slug}
