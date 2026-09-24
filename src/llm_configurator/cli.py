"""Command-line entry points. Run `llm-config --help`.

Long commands (download, test, tune, quality checks) show progress on stderr and stop
cleanly on Ctrl+C: the job is cancelled, partial downloads are kept for a resume, and
started llama.cpp processes are stopped.
"""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import time

from . import app
from .app import evaluate, map_benchmark, variants
from .catalogue import definitions, refresh
from .domain import GIB, Requirements
from .hardware import scan
from .calibration import calibrate
from .jobs import JobManager
from .runtime import bench
from .server import serve
from .storage import Store, models_dir

RUN_POLL_SECONDS = 1.0


def parser():
    root = argparse.ArgumentParser(description="Choose local model configurations using available hardware and benchmark evidence, "
                                               "then download, test, tune and run them with llama.cpp")
    root.add_argument("--data-dir", help="Override local cache/settings directory")
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("calibrate", help="Measure synthetic hardware speed and cache it locally; no model download")
    commands.add_parser("scan", help="Print current hardware and process memory as JSON")
    commands.add_parser("refresh", help="Fetch model metadata and optional AA scores; no weight downloads")
    models = commands.add_parser("models", help="List cached variants and their exact IDs, or add/remove your own repos",
                                 description="Without a subcommand, lists every model variant and its exact ID.")
    model_actions = models.add_subparsers(dest="models_action", metavar="ACTION")
    add = model_actions.add_parser("add", help="Add a Hugging Face model and its GGUF repo to your catalogue")
    add.add_argument("base_repo", help="Original model repo, for example Qwen/Qwen3-8B")
    add.add_argument("gguf_repo", help="Repo holding the GGUF files, for example Qwen/Qwen3-8B-GGUF")
    remove = model_actions.add_parser("remove", help="Remove a model you added")
    remove.add_argument("base_repo")
    commands.add_parser("benchmarks", help="List AA names/slugs to explicitly map to local models")
    mapping = commands.add_parser("map", help="Map a base model to an exact AA evaluation entry")
    mapping.add_argument("base_repo")
    mapping.add_argument("slug", help="Exact AA slug, or '-' to clear")
    web = commands.add_parser("serve", help="Open the local browser interface")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--demo", action="store_true", help="Use clearly fictional model fixtures")
    web.add_argument("--no-browser", action="store_true")
    rec = commands.add_parser("recommend", help="Compare deployment configurations")
    rec.add_argument("--workload", choices=list(app.WORKLOADS), default="general")
    rec.add_argument("--context", type=int, default=8192)
    rec.add_argument("--users", type=int, default=1)
    rec.add_argument("--min-tps", type=float, default=15)
    rec.add_argument("--reserve-gib", type=float, default=2)
    rec.add_argument("--gpu-reserve-gib", type=float, default=0.5)
    rec.add_argument("--gpu-index", type=int, default=0)
    rec.add_argument("--reclaim-pids", type=int, nargs="*", default=[])
    rec.add_argument("--strict-speed", action="store_true")
    rec.add_argument("--priority", choices=["balanced", "quality", "speed"], default="balanced")
    rec.add_argument("--no-rankings", dest="include_rankings", action="store_false")
    rec.add_argument("--kv", dest="kv_cache_type", choices=["f16", "q8_0", "q4_0"], default="f16",
                     help="Store the model's short-term notepad (KV cache) compressed to fit longer contexts")
    rec.add_argument("--demo", action="store_true")
    rec.add_argument("--json", action="store_true")
    rec.add_argument("--output", type=Path, help="Save the complete report as JSON")

    runtime = commands.add_parser("runtime", help="Check or install llama.cpp, the engine that runs the models")
    runtime_actions = runtime.add_subparsers(dest="runtime_action", required=True, metavar="ACTION")
    status = runtime_actions.add_parser("status", help="Show whether llama.cpp is installed and where it was found")
    status.add_argument("--json", action="store_true")
    install = runtime_actions.add_parser("install", help="Download the official llama.cpp build for this computer")
    install.add_argument("--allow-unverified", action="store_true", help="Accept a release file without a published checksum (not recommended)")
    archive = runtime_actions.add_parser("install-archive", help="Install llama.cpp from a zip or tar.gz you already downloaded")
    archive.add_argument("path", type=Path)
    use = runtime_actions.add_parser("use", help="Use an existing llama.cpp build folder that contains llama-server")
    use.add_argument("directory", type=Path)

    get = commands.add_parser("download", help="Download a model (resumable, verified, reuses copies already on disk)")
    get.add_argument("variant_id")
    get.add_argument("--directory", type=Path, help="Where to save it (default: the models folder, see 'settings')")
    get.add_argument("--yes", action="store_true", help="Accept the displayed download size without an interactive prompt")

    local = commands.add_parser("local", help="List, find or register model files already on this computer")
    local.add_argument("--scan", action="store_true", help="Look in Hugging Face, LM Studio, Ollama and the models folder")
    local.add_argument("--dir", dest="dirs", type=Path, action="append", default=[], help="Also look in this folder (repeatable)")
    local.add_argument("--add", type=Path, help="Register one GGUF file so you can test and run it")
    local.add_argument("--json", action="store_true")

    def model_options(command, context=True):
        command.add_argument("variant_id", help="Exact model ID from 'llm-config models'")
        if context:
            command.add_argument("--context", type=int, help="Tokens per conversation (default: 8192 or the model's limit)")
        command.add_argument("--gpu-layers", type=int, help="Layers to put on the GPU (default: the most that fit)")
        command.add_argument("--kv", choices=["f16", "q8_0", "q4_0"], default="f16",
                             help="KV cache (short-term notepad) format; q8_0/q4_0 use less memory")
        command.add_argument("--json", action="store_true", help="Print the full result as JSON")

    test = commands.add_parser("test", help="Check that a downloaded model works and measure its real speed")
    model_options(test)
    test.add_argument("--kind", choices=list(app.TEST_KINDS), default="full",
                      help="smoke: one question; speed: reading/writing speed and memory; full: both")
    test.add_argument("--tuned", action="store_true", help="Use the best saved tune")
    tune = commands.add_parser("tune", help="Try settings automatically to find the fastest ones for this computer")
    model_options(tune)
    tune.add_argument("--budget", type=int, default=300, help="Time limit in seconds, 60 to 1800 (default 300)")
    tune.add_argument("--goal", choices=list(app.GOALS), default="generation",
                      help="generation: faster writing; prompt: faster reading of long inputs; balanced: both")
    quiz = commands.add_parser("quiz", help="Ask the model short questions with known answers to check its quality")
    model_options(quiz)
    quiz.add_argument("--workload", choices=list(app.WORKLOADS), default="general")
    quiz.add_argument("--needle", action="store_true", help="Also test recall of a fact hidden in a long document")
    check = commands.add_parser("quant-check", help="Measure how much quality compression loses against a reference file")
    check.add_argument("reference_variant_id", help="Least-compressed downloaded version, for example Q8_0")
    check.add_argument("variant_ids", nargs="+", help="Downloaded versions to compare with it")
    check.add_argument("--json", action="store_true")
    export = commands.add_parser("export", help="Print a ready-to-use launch script or config for other tools")
    model_options(export)
    export.add_argument("--format", required=True, help="llama-server, ollama, docker-compose, openai-python, continue, open-webui or lmstudio")
    export.add_argument("--platform", choices=["posix", "windows"], default="windows" if sys.platform == "win32" else "posix")
    export.add_argument("--tuned", action="store_true", help="Use the best saved tune")
    export.add_argument("--output", type=Path, help="Write the file here instead of printing it")
    run_server = commands.add_parser("run", help="Run a model as an OpenAI-compatible server on this computer until Ctrl+C")
    model_options(run_server)
    run_server.add_argument("--port", type=int, help="Port to listen on (default: a free one)")
    run_server.add_argument("--tuned", action="store_true", help="Use the best saved tune")

    community = commands.add_parser("community", help="Import or share anonymous speed results")
    community_actions = community.add_subparsers(dest="community_action", required=True, metavar="ACTION")
    imported = community_actions.add_parser("import", help="Download the shared results list (read-only)")
    imported.add_argument("--source", help="HTTPS address of a results file (default: the project's list)")
    share = community_actions.add_parser("share", help="Prepare an anonymous result for you to review and post yourself")
    share.add_argument("measurement_ids", nargs="+")
    community_actions.add_parser("status", help="Show how many community results are loaded")

    settings = commands.add_parser("settings", help="Show or change where models are saved")
    settings.add_argument("--models-dir", type=Path, help="Folder for downloaded models")

    benchmark = commands.add_parser("bench", help="Run llama-bench on an existing GGUF; consumes local compute")
    benchmark.add_argument("variant_id")
    benchmark.add_argument("--model", required=True, type=Path)
    benchmark.add_argument("--executable", default="llama-bench")
    benchmark.add_argument("--context", type=int, default=8192)
    benchmark.add_argument("--gpu-layers", type=int, default=0)
    benchmark.add_argument("--gpu-index", type=int, default=0)
    benchmark.add_argument("--timeout", type=int, default=600)
    return root


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False, default=str))


def size(value):
    return f"{value / GIB:.2f} GiB" if value is not None else "unknown size"


def duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60}m" if seconds >= 3600 else f"{seconds // 60}m {seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


class ProgressBar:
    """A one-line bar on terminals; plain lines at each 10% step (or stage change) when output is piped."""

    def __init__(self, stream=None, width=24):
        self.stream = stream or sys.stderr
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self.width, self.last, self.drawn, self.shown = width, None, 0, 0.0

    def describe(self, value):
        done, total = value.get("done") or 0, value.get("total")
        stage = value.get("step") or value.get("stage") or "working"
        parts = [str(stage)]
        in_bytes = "bytes_per_second" in value or (total or 0) > 10 * 1024**2
        if total:
            fraction = max(0.0, min(1.0, done / total))
            parts.append(f"{fraction * 100:3.0f}%")
            parts.append(f"{size(done)} of {size(total)}" if in_bytes else f"{done:g} of {total:g}")
        if value.get("bytes_per_second"):
            parts.append(f"{value['bytes_per_second'] / 1024**2:.1f} MB/s")
        if value.get("eta_seconds") is not None:
            parts.append(f"about {duration(value['eta_seconds'])} left")
        if value.get("message"):
            parts.append(str(value["message"]))
        return parts, (done / total if total else None)

    def update(self, value):
        if not value:
            return
        parts, fraction = self.describe(value)
        if self.tty:
            if time.monotonic() - self.shown < 0.1 and fraction not in (None, 1.0):
                return
            filled = int(round((fraction or 0) * self.width))
            bar = f"[{'#' * filled}{'-' * (self.width - filled)}] " if fraction is not None else ""
            line = (bar + " · ".join(parts))[:max(20, 118)]
            self.stream.write("\r" + line + " " * max(0, self.drawn - len(line)))
            self.stream.flush()
            self.drawn, self.shown = len(line), time.monotonic()
            return
        key = (parts[0], None if fraction is None else int(fraction * 10), value.get("message") if fraction is None else None)
        if key != self.last:
            self.stream.write(" · ".join(parts) + "\n")
            self.stream.flush()
            self.last = key

    def close(self):
        if self.tty and self.drawn:
            self.stream.write("\r" + " " * self.drawn + "\r")
            self.stream.flush()
            self.drawn = 0


def run_job(fn, title):
    """Run a job function with a progress bar. Ctrl+C cancels it cleanly, then exits with 130."""
    print(title, file=sys.stderr, flush=True)
    jobs = JobManager()
    job = jobs.submit("cli", title, fn)
    bar = ProgressBar()
    try:
        while job["state"] not in {"done", "failed", "cancelled"}:
            job = jobs.wait(job["id"], 0.2)
            bar.update(job["progress"])
    except KeyboardInterrupt:
        bar.close()
        print("Stopping… (press Ctrl+C again to quit immediately)", file=sys.stderr, flush=True)
        jobs.cancel(job["id"])
        jobs.wait(job["id"], 120)
        raise
    finally:
        bar.close()
    if job["state"] == "cancelled":
        raise KeyboardInterrupt
    if job["state"] == "failed":
        raise ValueError(job["error"])
    return job["result"]


def model_launch(store, args, hardware=None):
    variant = app.find_variant(store, args.variant_id)
    hardware = hardware or scan(False)
    candidate = app.candidate_for(store, variant, hardware, getattr(args, "context", None), args.gpu_layers, args.kv)
    return variant, candidate, hardware


def show_runtime(info):
    if not info.get("installed"):
        print("llama.cpp is not installed yet. Run: llm-config runtime install")
    else:
        print(f"llama.cpp {info.get('version') or 'version unknown'} ({info.get('backend') or 'unknown'} backend), found via {info.get('source')}")
        if info.get("directory"):
            print(f"  Folder: {info['directory']}")
        for name, command in (info.get("binaries") or {}).items():
            print(f"  {name}: {'ready' if command else 'missing'}")
    for warning in info.get("warnings") or []:
        print(f"  Note: {warning}")


def show_test(result):
    print(result.get("verdict_text") or result.get("verdict") or "Finished.")
    smoke = result.get("smoke") or {}
    if smoke:
        print(f"  Works: {'yes' if smoke.get('ok') else 'no'} — {smoke.get('message') or ''}".rstrip(" —"))
    summary = (result.get("speed") or {}).get("summary") or {}
    labels = [("pp_tps", "Reading speed", "tokens/s"), ("ttft_s", "First-word delay", "s"), ("tps", "Writing speed", "tokens/s")]
    for key, label, unit in labels:
        if summary.get(key) is not None:
            print(f"  {label}: {summary[key]:.1f} {unit}")
    memory = (result.get("speed") or {}).get("memory") or {}
    if memory.get("note"):
        print(f"  Memory: {memory['note']}")


def show_tune(result):
    baseline, best = result.get("baseline") or {}, result.get("best_result") or {}
    if baseline.get("tps") and best.get("tps"):
        print(f"Writing speed: {baseline['tps']:.1f} → {best['tps']:.1f} tokens/s ({(result.get('improvement') or 1):.2f}× as fast)")
    changed = {k: v for k, v in (result.get("best") or {}).items() if k in app.TUNABLE}
    if changed:
        print("Best settings: " + ", ".join(f"{k}={v}" for k, v in changed.items()))
    for note in result.get("notes") or []:
        print(f"  {note}")
    print("Use them with --tuned on test, export or run.")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        store = Store(args.data_dir)
        command = args.command
        if command == "scan":
            print_json(scan())
        elif command == "calibrate":
            result = calibrate(scan(False))
            store.put("calibration", result)
            print_json(result)
        elif command == "refresh":
            print_json(refresh(store))
        elif command == "models":
            from . import catalogue
            if args.models_action == "add":
                print_json(catalogue.add_entry(store, args.base_repo, args.gguf_repo))
            elif args.models_action == "remove":
                print_json(catalogue.remove_entry(store, args.base_repo))
            else:
                print_json([v.to_dict() for v in variants(store)])
        elif command == "benchmarks":
            cache = store.get("scores", {})
            print_json({"source": "Artificial Analysis — https://artificialanalysis.ai", "version": cache.get("version"),
                        "entries": [{"name": i["name"], "slug": i["slug"]} for i in cache.get("data", [])],
                        "mappings": definitions(store)})
        elif command == "map":
            print_json(map_benchmark(store, args.base_repo, None if args.slug == "-" else args.slug))
        elif command == "serve":
            serve(store, args.port, args.demo, not args.no_browser)
        elif command == "recommend":
            payload = {field.name: getattr(args, field.name) for field in fields(Requirements)}
            report = evaluate(store, payload, args.demo)
            if args.output:
                args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.json:
                print_json(report)
            else:
                if report["demo"]:
                    print("DEMO: fictional models, real hardware. No real quality or speed scores.")
                shortlist = set(report["shortlist"])
                for item in report["candidates"]:
                    if item["id"] not in shortlist:
                        continue
                    speed = f"{item['tps']:.1f} tok/s" if item["tps"] is not None else "speed unverified"
                    if item["tps"] is None and item["speed_estimate"].get("available"):
                        estimate = item["speed_estimate"]
                        speed = f"estimated {estimate['low_tps']:.1f}–{estimate['high_tps']:.1f} tok/s (low confidence)"
                    print(f"{item['name']} / {item['quant']} / {item['mode']} / {item['context']:,} tokens per user / {speed}")
                    print(f"  RAM {item['ram_bytes']/GIB:.2f} GiB | VRAM {item['vram_bytes']/GIB:.2f} GiB | {item['scenario']}")
                    if item.get("verdict_text"):
                        print(f"  {item['verdict_text']}")
                    print(f"  {item['quality_evidence']}; reference score: {item['quality_score']}")
                    if item["score_source"]:
                        print(f"  Artificial Analysis: {item['score_source']} (index {item['score_version']})")
                    if not item["demo"]:
                        print(f"  Model ID: {item['variant_id']}")
                if not report["candidates"]:
                    print("No qualifying configurations. Refresh metadata, relax constraints, or include speed-unverified options.")
                print("\n" + "\n".join(report["notes"]))
        elif command == "runtime":
            return runtime_command(store, args)
        elif command == "local":
            return local_command(store, args)
        elif command == "community":
            return community_command(store, args)
        elif command == "settings":
            if args.models_dir:
                directory = args.models_dir.expanduser().resolve()
                directory.mkdir(parents=True, exist_ok=True)
                store.update("settings", lambda saved: {**(saved or {}), "models_dir": str(directory)}, {})
            print_json({**(store.get("settings") or {}), "models_dir": str(models_dir(store))})
        elif command == "quant-check":
            reference = app.find_variant(store, args.reference_variant_id)
            others = [app.find_variant(store, v) for v in dict.fromkeys(args.variant_ids)]
            if reference.id in {v.id for v in others}:
                raise ValueError("Pick a different reference than the models you check")
            result = run_job(app.quant_check_job(store, reference, others), f"Compression check against {reference.name} {reference.quant}")
            if args.json:
                print_json(result)
            else:
                for label, value in (result.get("results") or {}).items():
                    print(f"{label}: {value.get('plain') or value}")
                for note in result.get("notes") or []:
                    print(f"  {note}")
        elif command == "download":
            return download_command(store, args)
        elif command == "bench":
            variant = app.find_variant(store, args.variant_id)
            print("Validating file and memory, then running a local generation benchmark…", file=sys.stderr, flush=True)
            measurement = bench(variant, args.model, args.executable, args.context, args.gpu_layers, args.gpu_index, args.timeout)
            store.append("measurements", measurement)
            print_json(measurement)
        else:
            return model_command(store, args)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


def runtime_command(store, args):
    from . import runtime_install
    if args.runtime_action == "status":
        info = runtime_install.detect(store)
        print_json(info) if args.json else show_runtime(info)
        return 0
    if args.runtime_action == "install":
        info = run_job(app.runtime_install_job(store, args.allow_unverified), "Installing llama.cpp")
    elif args.runtime_action == "install-archive":
        info = run_job(lambda progress, cancel: runtime_install.install_archive(store, args.path, progress=progress, cancel=cancel),
                       "Installing llama.cpp from the archive")
    else:
        info = runtime_install.use_directory(store, args.directory)
    show_runtime(info)
    return 0


def download_command(store, args):
    from . import downloads
    variant = app.find_variant(store, args.variant_id)
    directory = args.directory or models_dir(store)
    if not args.directory:
        existing = app.local_model(store, variant)
        if existing:
            print(f"Already on this computer: {existing}")
            return 0
    plan = downloads.plan(variant, directory)
    print(f"Download {variant.name} {variant.quant}: {size(plan['total_bytes'])} "
          f"({size(plan['remaining_bytes'])} still to fetch) into {directory}", file=sys.stderr, flush=True)
    if not plan.get("enough_space"):
        raise ValueError(f"Not enough disk space: {size(plan['remaining_bytes'])} needed plus 1 GiB spare, "
                         f"{size(plan.get('disk_free'))} free. Free some space or choose --directory.")
    if not args.yes and input("Download this model? [y/N] ").strip().lower() != "y":
        return 0
    path = run_job(lambda progress, cancel: downloads.download_variant(variant, directory, progress=progress, cancel=cancel),
                   f"Downloading {variant.name} {variant.quant}")
    print(path)
    return 0


def local_command(store, args):
    if args.add:
        record = run_job(app.add_local_job(store, args.add), f"Checking {args.add.name}")
        variant = record["variant"]
        if args.json:
            print_json(record)
        else:
            print(f"Registered {variant['name']} {variant['quant']}. Model ID: {variant['id']}")
        return 0
    if args.scan or args.dirs:
        files = run_job(app.scan_job(store, args.dirs), "Looking for model files")
    else:
        files = store.get("local_files", [])
    from . import discover
    if args.json:
        print_json({"files": files, "locations": discover.locations(), "registered": store.get("local_variants", [])})
        return 0
    for location in discover.locations():
        print(f"{'✓' if location.get('exists') else '·'} {location.get('source')}: {location.get('path')}")
    if not files:
        print("No model files found yet. Run: llm-config local --scan")
    for record in files:
        match = record.get("variant_id") or "not in the catalogue"
        print(f"{record.get('path')} ({size(record.get('size_bytes'))}) — {match}{'' if record.get('verified') else ', not verified'}")
    for record in store.get("local_variants", []):
        print(f"Registered: {record['path']} — Model ID: {record['variant']['id']}")
    return 0


def community_command(store, args):
    from . import community
    if args.community_action == "import":
        result = run_job(lambda progress, cancel: community.import_records(store, source=args.source, progress=progress, cancel=cancel),
                         "Importing community results")
        print(f"Imported {len(result.get('records') or [])} results; skipped {result.get('rejected', 0) if not isinstance(result.get('rejected'), list) else len(result['rejected'])} invalid rows.")
    elif args.community_action == "share":
        payload = community.share_payload(store, args.measurement_ids)
        print(payload["json"])
        print("\nNothing has been sent. Review the data above, then open this link to post it yourself:", file=sys.stderr)
        print(payload["issue_url"])
    else:
        saved = store.get("community") or {}
        print(f"{len(saved.get('records') or [])} community results from {saved.get('source') or 'nowhere yet'}"
              f"{', fetched ' + saved['fetched_at'] if saved.get('fetched_at') else ''}. Import with: llm-config community import")
    return 0


def model_command(store, args):
    variant, candidate, hardware = model_launch(store, args)
    if args.command == "test":
        result = run_job(app.test_job(store, variant, candidate, hardware, args.kind, args.tuned), f"Testing {variant.name} {variant.quant}")
        print_json(result) if args.json else show_test(result)
        return 0 if result.get("verdict") != "failed" else 1
    if args.command == "tune":
        result = run_job(app.tune_job(store, variant, candidate, hardware, args.budget, args.goal),
                         f"Tuning {variant.name} {variant.quant} for up to {duration(args.budget)}")
        print_json(result) if args.json else show_tune(result)
        return 0
    if args.command == "quiz":
        result = run_job(app.quiz_job(store, variant, candidate, hardware, args.workload, args.needle), f"Quiz for {variant.name} {variant.quant}")
        if args.json:
            print_json(result)
        else:
            quiz = result["quiz"]
            print(f"{quiz.get('correct')} of {quiz.get('total')} correct ({(quiz.get('score') or 0) * 100:.0f}%; "
                  f"likely between {(quiz.get('ci_low') or 0) * 100:.0f}% and {(quiz.get('ci_high') or 0) * 100:.0f}%)")
            if quiz.get("note"):
                print(f"  {quiz['note']}")
            if result.get("needle"):
                print(f"Long-document recall: {result['needle'].get('plain') or result['needle'].get('summary') or 'see --json'}")
        return 0
    if args.command == "export":
        result = app.export_config(store, variant, candidate, hardware, args.format, args.platform, args.tuned)
        if args.json:
            print_json(result)
            return 0
        if args.output:
            args.output.write_text(result["content"], encoding="utf-8")
            print(f"Saved {args.output}", file=sys.stderr)
        else:
            print(result["content"])
        for line in (result.get("instructions") or []) + (result.get("notes") or []):
            print(f"  {line}", file=sys.stderr)
        return 0
    return run_command(store, args, variant, candidate, hardware)


def run_command(store, args, variant, candidate, hardware):
    """Foreground llama-server: prints the OpenAI address and stops cleanly on Ctrl+C."""
    from . import llama_server
    path = app.require_model(store, variant)
    overrides = {"port": args.port} if args.port else {}
    config = app.launch_config(store, candidate, hardware, path, args.tuned, **overrides)
    server = llama_server.LlamaServer(app.binary(store, "llama-server"), config)
    running = False
    try:
        run_job(lambda progress, cancel: server.start(progress=progress, cancel=cancel), f"Loading {variant.name} {variant.quant}")
        print(f"Running. OpenAI-compatible address: {server.base_url}/v1")
        print("Press Ctrl+C to stop.", file=sys.stderr, flush=True)
        running = True
        misses = 0
        while misses < 3:
            time.sleep(RUN_POLL_SECONDS)
            misses = 0 if server.ready() else misses + 1
        reason = server.failure_reason() or "The server stopped unexpectedly."
        print(f"Error: {reason}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if not running:
            raise
        print("Stopping the server…", file=sys.stderr)
        return 0
    finally:
        server.stop()
