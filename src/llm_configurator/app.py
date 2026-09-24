"""Application orchestration shared by CLI and local interface.

Long operations are job functions `fn(progress, cancel)` for `jobs.JobManager`. Model files
come from the app's models folder, discovered model folders or CLI arguments, and llama.cpp
binaries from `runtime_install.binary`, never from browser input. Modules built by other
v0.4 workstreams are imported inside functions so this file loads without them.
"""
import inspect
import json
from pathlib import Path
import secrets

from .catalogue import apply_scores, definitions, demo_variants
from .domain import Requirements, Variant, check_cancel, now
from .engine import recommend
from .hardware import scan
from .launch import from_candidate
from .storage import models_dir

WORKLOADS = ("general", "coding", "agentic", "documents")
TEST_KINDS = ("smoke", "speed", "full")
GOALS = ("generation", "balanced", "prompt")
# Launch keys a stored tune may change; context, users and the model file always come from the candidate.
TUNABLE = ("gpu_layers", "threads", "batch", "ubatch", "flash_attn", "cache_type_k", "cache_type_v", "n_cpu_moe")


def variants(store, demo=False):
    if demo:
        return demo_variants()
    result = [Variant(**v) for v in store.get("variants", [])]
    known = {v.id for v in result}
    for record in store.get("local_variants", []):
        try:
            variant = Variant(**record["variant"])
        except (KeyError, TypeError, ValueError):
            continue
        if variant.id not in known:
            result.append(variant)
            known.add(variant.id)
    return result


def find_variant(store, variant_id, demo=False):
    variant = next((v for v in variants(store, demo) if v.id == variant_id), None)
    if not variant:
        raise ValueError("Model ID not found. Run 'llm-config models' after refreshing metadata to see the exact IDs.")
    return variant


def engine_extras(store, function=recommend):
    """community/tuned evidence for engines that accept it; older engines keep working."""
    try:
        accepted = inspect.signature(function).parameters
    except (TypeError, ValueError):
        return {}
    extras = {"community": (store.get("community") or {}).get("records") or [], "tuned": store.get("tuned", [])}
    return {k: v for k, v in extras.items() if k in accepted}


def evaluate(store, payload, demo=False):
    requirements = Requirements(**payload)
    models = variants(store, demo)
    hardware = scan(include_processes=bool(requirements.reclaim_pids), process_ids=set(requirements.reclaim_pids))
    report = recommend(models, hardware, requirements, store.get("measurements", []), store.get("calibration"),
                       **({} if demo else engine_extras(store)))
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
    models = [Variant(**v) for v in store.get("variants", [])]
    for variant in models:
        if variant.base_repo == base_repo:
            variant.scores = {}
            variant.score_source = variant.score_version = variant.score_settings = None
            apply_scores(variant, entry, cache)
    store.put("variants", [v.to_dict() for v in models])
    return {"base_repo": base_repo, "aa_slug": slug}


# ---- Files and binaries -------------------------------------------------------------

def binary(store, name):
    from . import runtime_install
    return runtime_install.binary(store, name)


def local_model(store, variant, verify=True):
    """A complete local copy of every file of this variant, or None. Never downloads."""
    for record in store.get("local_variants", []):
        if record.get("variant", {}).get("id") == variant.id and Path(record.get("path", "")).is_file():
            return Path(record["path"])
    if variant.demo:
        return None
    from . import discover
    found = discover.find_for_variant(store, variant, verify=verify)
    if found:
        return Path(found)
    # A finished download in the models folder that discover has not indexed yet.
    from . import downloads
    plan = downloads.plan(variant, models_dir(store))
    if plan["files"] and all(f.get("present") for f in plan["files"]):
        return models_dir(store) / Path(variant.all_files()[0]["filename"]).name
    return None


def require_model(store, variant):
    path = local_model(store, variant)
    if not path:
        raise ValueError(f"{variant.name} {variant.quant} is not on this computer yet. Download it first.")
    return path


def download_plan(store, variant):
    from . import downloads
    plan = downloads.plan(variant, models_dir(store))
    plan.pop("directory", None)
    return {**plan, "local_copy": local_model(store, variant, verify=False) is not None}


def download_job(store, variant):
    def run(progress, cancel):
        existing = local_model(store, variant)
        if existing:
            return {"reused": True, "bytes": 0, "filename": existing.name, "variant_id": variant.id}
        from . import downloads
        path = downloads.download_variant(variant, models_dir(store), progress=progress, cancel=cancel)
        return {"reused": False, "bytes": variant.size_bytes, "filename": Path(path).name, "variant_id": variant.id}
    return run


def remove_download(store, variant):
    """Frees only this variant's files inside the app's models folder, never discovered copies elsewhere."""
    from . import downloads
    return downloads.remove_variant(variant, models_dir(store))


# ---- Candidates and launch settings -------------------------------------------------

def placement(candidate):
    layers, total = candidate.get("gpu_layers") or 0, candidate.get("total_layers")
    return candidate.get("mode") or ("cpu" if not layers else "gpu" if layers == total else "split")


def best_tune(store, variant_id, context, where, fingerprint=None):
    """Highest-speed stored tune for this variant, context and placement on this computer."""
    matches = [r for r in store.get("tuned", []) if r.get("variant_id") == variant_id and r.get("context") == context
               and r.get("placement") == where and (fingerprint is None or r.get("fingerprint") == fingerprint)
               and isinstance(r.get("best"), dict)]
    return max(matches, key=lambda r: ((r.get("best_result") or {}).get("tps") or 0, r.get("timestamp") or ""), default=None)


def launch_config(store, candidate, hardware, model_path=None, tuned=False, **overrides):
    changes = {}
    if tuned:
        record = best_tune(store, candidate["variant_id"], candidate["context"], placement(candidate), hardware.get("fingerprint"))
        if not record:
            raise ValueError("No saved tune for this model, context and placement yet. Run a tune first.")
        changes = {k: record["best"][k] for k in TUNABLE if k in record["best"]}
    return from_candidate(candidate, model_path, hardware, **{**changes, **overrides})


def candidate_for(store, variant, hardware, context=None, gpu_layers=None, kv_cache_type="f16"):
    """CLI: the engine's best-fitting candidate for one model, or a hand-built one for explicit --gpu-layers."""
    context = context or min(8192, variant.max_context)
    if type(context) is not int or not 256 <= context <= variant.max_context:
        raise ValueError(f"Context must be between 256 and {variant.max_context:,} tokens for this model")
    if gpu_layers is not None and not 0 <= gpu_layers <= variant.layers:
        raise ValueError(f"GPU layers must be between 0 and {variant.layers} for this model")
    requirements = Requirements(context=context, min_tps=0, include_rankings=False, kv_cache_type=kv_cache_type)
    report = recommend([variant], hardware, requirements, store.get("measurements", []), store.get("calibration"),
                       **engine_extras(store))
    matches = [c for c in report["candidates"] if c["context"] == context and c["scenario"] == "now"
               and c.get("kv_cache_type", kv_cache_type) == kv_cache_type]
    # Engines without KV compression support still describe placements; the notepad format is a launch setting.
    matches = [{"kv_cache_type": kv_cache_type, **c} for c in matches]
    if gpu_layers is None:
        if not matches:
            raise ValueError(f"{variant.name} {variant.quant} does not fit in free memory at {context:,} tokens. "
                             "Try a smaller --context, or choose --gpu-layers yourself.")
        return max(matches, key=lambda c: (c["gpu_layers"], -(c.get("n_cpu_moe") or 0)))
    chosen = next((c for c in matches if c["gpu_layers"] == gpu_layers), None)
    if chosen:
        return chosen
    gpu = next((g for g in hardware.get("gpus", []) if g.get("index") == requirements.gpu_index), None)
    if gpu_layers and not gpu:
        raise ValueError("No supported GPU was found. Use --gpu-layers 0 to run on the CPU.")
    return {"id": f"{variant.id}|manual|{context}|{gpu_layers}", "variant_id": variant.id, "name": variant.name,
            "quant": variant.quant, "filename": variant.filename, "context": context, "users": 1,
            "gpu_layers": gpu_layers, "total_layers": variant.layers, "gpu_index": requirements.gpu_index if gpu_layers else None,
            "threads": hardware.get("cores") or hardware.get("threads"), "kv_cache_type": kv_cache_type, "n_cpu_moe": 0,
            "mode": "cpu" if not gpu_layers else "gpu" if gpu_layers == variant.layers else "split", "scenario": "manual"}


def _stage(progress, step):
    """Forward progress with a `step` label so multi-part jobs read clearly."""
    return (lambda value: progress({**value, "step": step})) if progress else None


# ---- Test, tune, quality ------------------------------------------------------------

def test_job(store, variant, candidate, hardware, kind="full", tuned=False):
    if kind not in TEST_KINDS:
        raise ValueError("Test kind must be smoke, speed or full")
    def run(progress, cancel):
        from . import testing
        path = require_model(store, variant)
        config = launch_config(store, candidate, hardware, path, tuned)
        server = binary(store, "llama-server")
        bench = binary(store, "llama-bench") if kind != "smoke" else None
        # Both naming styles so the testing module can read either.
        commands = {"server": server, "bench": bench, "llama-server": server, "llama-bench": bench}
        return testing.run_tests(store, variant, config, commands, kind=kind, hardware=hardware, progress=progress, cancel=cancel)
    return run


def tune_job(store, variant, candidate, hardware, budget_seconds=300, goal="generation"):
    if type(budget_seconds) is not int or not 60 <= budget_seconds <= 1800:
        raise ValueError("The time budget must be between 60 and 1800 seconds")
    if goal not in GOALS:
        raise ValueError("Goal must be generation, balanced or prompt")
    def run(progress, cancel):
        from . import tuner
        path = require_model(store, variant)
        base = launch_config(store, candidate, hardware, path)
        result = tuner.tune(binary(store, "llama-bench"), variant, base, hardware, budget_seconds=budget_seconds,
                            goal=goal, progress=progress, cancel=cancel)
        best = {k: v for k, v in (result.get("best") or {}).items() if k != "model_path"}
        record = {"id": secrets.token_hex(6), "variant_id": variant.id, "sha256": variant.sha256, "context": candidate["context"],
                  "users": candidate.get("users", 1), "placement": placement(candidate), "gpu_layers": candidate["gpu_layers"],
                  "fingerprint": hardware.get("fingerprint"), "timestamp": now(), "goal": goal, "budget_seconds": budget_seconds,
                  "best": best, "baseline": result.get("baseline"), "best_result": result.get("best_result"),
                  "improvement": result.get("improvement"), "stopped": result.get("stopped"), "notes": result.get("notes", []),
                  "trials": len(result.get("trials") or [])}
        if result.get("stopped") != "cancelled":
            store.append("tuned", record)
        # Paths stay on this computer: the page gets the tuned settings, not the model's folder.
        return {**result, "best": best, "record_id": record["id"]}
    return run


def start_server(store, variant, candidate, hardware, tuned=False, progress=None, cancel=None):
    from . import llama_server
    path = require_model(store, variant)
    config = launch_config(store, candidate, hardware, path, tuned)
    server = llama_server.LlamaServer(binary(store, "llama-server"), config)
    server.start(progress=progress, cancel=cancel)
    return server, config


def _quality_record(kind, variant, candidate, **values):
    return {"kind": kind, "variant_id": variant.id, "name": variant.name, "quant": variant.quant,
            "context": candidate.get("context"), "placement": placement(candidate), "timestamp": now(), **values}


def quiz_job(store, variant, candidate, hardware, workload="general", include_needle=False):
    if workload not in WORKLOADS:
        raise ValueError("Workload must be general, coding, agentic or documents")
    def run(progress, cancel):
        from . import evals
        server, config = start_server(store, variant, candidate, hardware, progress=_stage(progress, "start"), cancel=cancel)
        try:
            quiz = evals.run_quiz(server.chat, workload, progress=_stage(progress, "quiz"), cancel=cancel)
            store.append("quality_results", _quality_record("quiz", variant, candidate, workload=workload,
                         **{k: quiz.get(k) for k in ["correct", "total", "score", "ci_low", "ci_high", "seconds", "note"]}))
            needle = None
            if include_needle:
                check_cancel(cancel)
                needle = evals.needle_test(server.chat, server.tokenize, config["context"], progress=_stage(progress, "needle"), cancel=cancel)
                store.append("quality_results", _quality_record("needle", variant, candidate, result=needle))
            return {"quiz": quiz, "needle": needle}
        finally:
            server.stop()
    return run


class _Chat:
    """One model's chat session for a blind comparison: callable, and a context manager that stops the server."""
    def __init__(self, open_server):
        self._open, self._server = open_server, None

    def _live(self):
        if self._server is None:
            self._server = self._open()
        return self._server

    def __call__(self, messages, max_tokens=512, **options):
        return self._live().chat(messages, max_tokens=max_tokens, **options)

    def __enter__(self):
        self._live()
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._server is not None:
            self._server.stop()
            self._server = None

    stop = close


def compare_job(store, entries, hardware, prompts, max_tokens=512):
    """entries: [(label, variant, candidate)]. Models run one after another because of memory."""
    def run(progress, cancel):
        from . import evals
        for _, variant, _ in entries:
            require_model(store, variant)  # fail before loading anything
        sessions = []
        def factory(variant, candidate):
            def make():
                session = _Chat(lambda: start_server(store, variant, candidate, hardware, cancel=cancel)[0])
                sessions.append(session)
                return session
            return make
        runners = {label: factory(variant, candidate) for label, variant, candidate in entries}
        try:
            comparison_id = evals.run_comparison(store, prompts, runners, max_tokens=max_tokens, progress=progress, cancel=cancel)
        finally:
            for session in sessions:
                session.close()
        return {"comparison_id": comparison_id}
    return run


def quant_check_job(store, reference, others):
    def run(progress, cancel):
        from . import quantcheck
        reference_path = require_model(store, reference)
        labels = {}
        for variant in others:
            label = variant.quant if variant.quant not in labels else variant.id
            labels[label] = variant
        paths = {label: str(require_model(store, variant)) for label, variant in labels.items()}
        result = quantcheck.kl_check(binary(store, "llama-perplexity"), str(reference_path), paths,
                                     progress=progress, cancel=cancel)
        by_variant = {labels[label].id: value for label, value in (result.get("results") or {}).items() if label in labels}
        store.append("quality_results", {"kind": "quant_check", "variant_id": reference.id, "reference_variant_id": reference.id,
                                         "name": reference.name, "reference_quant": reference.quant,
                                         "variant_ids": [v.id for v in others], "results": by_variant,
                                         "notes": result.get("notes", []), "timestamp": now()})
        return {**result, "reference": reference.quant, "variant_ids": {label: v.id for label, v in labels.items()}}
    return run


# ---- Export, runtime, local files -----------------------------------------------------

def export_config(store, variant, candidate, hardware, fmt, platform="posix", tuned=False):
    from . import export
    if fmt not in {f["id"] for f in export.formats()}:
        raise ValueError("Unknown export format")
    if platform not in {"posix", "windows"}:
        raise ValueError("Platform must be posix or windows")
    path = local_model(store, variant, verify=False)
    config = launch_config(store, candidate, hardware, path or models_dir(store) / Path(variant.all_files()[0]["filename"]).name, tuned)
    try:
        command = binary(store, "llama-server")
    except ValueError:
        command = None
    result = export.export(config, variant, fmt, platform=platform, server_command=command)
    if not path:
        result.setdefault("notes", []).insert(0, "The model is not downloaded yet. The file path shown is where it will be saved after you download it.")
    return result


def runtime_install_job(store, allow_unverified=False):
    def run(progress, cancel):
        from . import runtime_install
        return runtime_install.install(store, scan(False), progress=progress, cancel=cancel, allow_unverified=allow_unverified)
    return run


def page_file(record):
    """A local file record without its folder, for the browser page."""
    path = Path(record.get("path") or "")
    return {"filename": path.name, "size_bytes": record.get("size_bytes"), "sha256": record.get("sha256"),
            "source": record.get("source"), "variant_id": record.get("variant_id"), "gguf": record.get("gguf"),
            "mtime": record.get("mtime"), "verified": bool(record.get("verified"))}


def page_location(location):
    """Folder labels relative to the home folder, never absolute paths."""
    path, home = Path(location.get("path") or ""), Path.home()
    try:
        label = "~/" + path.relative_to(home).as_posix()
    except ValueError:
        label = path.name or "folder"
    return {"source": location.get("source"), "exists": bool(location.get("exists")), "label": label}


def scan_job(store, extra_dirs=()):
    def run(progress, cancel):
        from . import discover
        return discover.scan(store, extra_dirs=tuple(extra_dirs), progress=progress, cancel=cancel)
    return run


def add_local_job(store, path):
    """Register any GGUF on disk so it can be tested, tuned and served like a catalogue model."""
    path = Path(path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError("Choose an existing .gguf file")
    def run(progress, cancel):
        from . import discover, gguf
        sha = discover.hash_cached(store, path, progress=progress, cancel=cancel)
        variant = gguf.variant_from_file(path, sha256=sha)
        record = {"variant": variant.to_dict(), "path": str(path), "added_at": now()}
        store.update("local_variants", lambda items: [r for r in (items or []) if r.get("variant", {}).get("id") != variant.id] + [record], [])
        return record
    return run

