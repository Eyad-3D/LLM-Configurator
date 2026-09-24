"""Command-line entry points. Run `llm-config --help`.

Long commands (download, test, tune, quality checks) show progress on stderr and stop
cleanly on Ctrl+C: the job is cancelled, partial downloads are kept for a resume, and
started llama.cpp processes are stopped.
"""
import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import secrets
import shutil
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
# Stages that report elapsed time against a timeout, not real progress: show seconds, never a percentage.
OPEN_ENDED = {"loading", "starting", "scanning"}
BYTE_STAGES = {"verifying", "checking", "download", "downloading", "extract"}  # done/total count bytes
TIME_STAGES = {"tune"}  # done/total count seconds of the time budget
PAUSED = "Paused. Run the same command again to resume."
CANCEL_WAIT_SECONDS = 120


def parser():
    root = argparse.ArgumentParser(prog="llm-config", description="Choose local model configurations using available hardware and benchmark evidence, "
                                               "then download, test, tune and run them with llama.cpp")
    root.add_argument("--data-dir", help="Override local cache/settings directory")
    commands = root.add_subparsers(dest="command", required=True, metavar="COMMAND")
    commands.add_parser("calibrate", help="Measure synthetic hardware speed and cache it locally; no model download")
    commands.add_parser("scan", help="Print current hardware and process memory as JSON")
    commands.add_parser("refresh", help="Fetch model metadata and optional AA scores; no weight downloads (Ctrl+C stops it)")
    models = commands.add_parser("models", help="List cached variants and their exact IDs, or add/remove your own repos",
                                 description="Without a subcommand, lists every model variant and its exact ID.")
    model_actions = models.add_subparsers(dest="models_action", metavar="ACTION")
    add = model_actions.add_parser("add", help="Add a Hugging Face model and its GGUF repo to your catalogue")
    add.add_argument("base_repo", help="Original model repo, for example Qwen/Qwen3-8B")
    add.add_argument("gguf_repo", help="Repo holding the GGUF files, for example Qwen/Qwen3-8B-GGUF")
    add.add_argument("--config-repo", help="Repo to read the model's config.json from, when the original repo asks you "
                                           "to log in or accept its terms first (for example a public copy of it)")
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
    local.add_argument("--add", type=Path, help="Register one GGUF file (any part of a split model) so you can test and run it")
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
    test.add_argument("--min-tps", type=float, default=15,
                      help="Writing speed you need, in tokens/s; slower results say 'works, but slowly' (default 15, 0 to skip)")
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
    export.add_argument("--port", type=int, help="Port the server should listen on (default 8080)")
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
    benchmark.add_argument("--executable", help="llama-bench to use (default: the one from 'llm-config runtime status')")
    benchmark.add_argument("--context", type=int, default=8192)
    benchmark.add_argument("--gpu-layers", type=int, default=0)
    benchmark.add_argument("--gpu-index", type=int, default=0)
    benchmark.add_argument("--timeout", type=int, default=600)
    return root


def print_json(value):
    print(json.dumps(value, indent=2, allow_nan=False, default=str))


def size(value):
    if value is None:
        return "unknown size"
    return f"{value / GIB:.2f} GiB" if value >= GIB / 2 else f"{value / 1024**2:.1f} MB"


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
        """Plain words first (the job's own message, else its stage), then the numbers."""
        done, total = value.get("done") or 0, value.get("total")
        stage = str(value.get("stage") or value.get("step") or "working")
        parts = [str(value.get("message") or stage.replace("_", " ").capitalize())]
        if stage in OPEN_ENDED:
            total = None  # a load timeout is not a finish line
            if done:
                parts.append(f"{duration(done)} so far")
        in_bytes = ("bytes_per_second" in value or value.get("unit") == "bytes" or stage in BYTE_STAGES
                    or (total or 0) > 1024**2)
        if total:
            fraction = max(0.0, min(1.0, done / total))
            parts.append(f"{fraction * 100:3.0f}%")
            parts.append(f"{size(done)} of {size(total)}" if in_bytes else f"{duration(done)} of {duration(total)}"
                         if stage in TIME_STAGES else f"{done:g} of {total:g}")
        if value.get("bytes_per_second"):
            parts.append(f"{value['bytes_per_second'] / 1024**2:.1f} MB/s")
        if value.get("eta_seconds") is not None:
            parts.append(f"about {duration(value['eta_seconds'])} left")
        return parts, (max(0.0, min(1.0, done / total)) if total else None)

    def update(self, value):
        if not value:
            return
        parts, fraction = self.describe(value)
        if self.tty:
            if time.monotonic() - self.shown < 0.1 and fraction not in (None, 1.0):
                return
            filled = int(round((fraction or 0) * self.width))
            bar = f"[{'#' * filled}{'-' * (self.width - filled)}] " if fraction is not None else ""
            width = shutil.get_terminal_size((80, 20)).columns - 1  # one short of the edge, so it never wraps
            line = (bar + " · ".join(parts))[:max(20, width)]
            self.stream.write("\r" + line + " " * max(0, self.drawn - len(line)))
            self.stream.flush()
            self.drawn, self.shown = len(line), time.monotonic()
            return
        stage = value.get("stage") or value.get("step")
        key = (stage, None if fraction is None else int(fraction * 10), parts[0] if fraction is None else None)
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
        while True:  # always draw at least once, so fast jobs still show their final progress
            job = jobs.wait(job["id"], 0.2)
            bar.update(job["progress"])
            if job["state"] in {"done", "failed", "cancelled"}:
                break
    except (KeyboardInterrupt, SystemExit) as stop:  # Ctrl+C, or a closed terminal / kill (exit handlers)
        bar.close()
        if isinstance(stop, KeyboardInterrupt):
            print("Stopping… (press Ctrl+C again to quit immediately)", file=sys.stderr, flush=True)
        jobs.cancel(job["id"])
        # Poll the state: a Thread.join() interrupted by Ctrl+C can return at once while the job still runs,
        # and exiting then would leave llama-bench or llama-server running on its own.
        deadline = time.monotonic() + CANCEL_WAIT_SECONDS
        while (jobs.get(job["id"]) or {}).get("state") in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.05)
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
    if info.get("reason"):
        print(f"  Why this build: {info['reason']}")
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


SETTING_WORDS = {"gpu_layers": "layers on the GPU", "threads": "CPU threads", "batch": "batch size",
                 "ubatch": "micro-batch size", "flash_attn": "flash attention", "cache_type_k": "notepad format (keys)",
                 "cache_type_v": "notepad format (values)", "n_cpu_moe": "expert layers kept on the CPU"}


def show_tune(result):
    """Both measured speeds with their own ratios; the tuner's `improvement` mixes them for goal=balanced/prompt."""
    baseline, best = result.get("baseline") or {}, result.get("best_result") or {}
    for key, label in (("tps", "Writing speed"), ("pp_tps", "Reading speed")):
        before, after = baseline.get(key), best.get(key)
        if before and after:
            print(f"{label}: {before:.1f} -> {after:.1f} tokens/s ({after / before:.2f}x)")
    changes = result.get("changes")
    if changes is None:  # results from before `changes` existed
        changes = {k: v for k, v in (result.get("best") or {}).items() if k in app.TUNABLE and v is not None}
    changes = {k: v for k, v in changes.items() if v is not None}
    if changes:
        print("Changed settings: " + ", ".join(f"{SETTING_WORDS.get(k, k)} = {v}" for k, v in changes.items()))
    for note in result.get("notes") or []:
        print(f"  {note}")
    if changes and result.get("stopped") != "cancelled":
        print("Use them with --tuned on test, export or run.")


def shown(path):
    """A path as the terminal can print it: a file name that is not valid text shows '?' instead of crashing."""
    text, encoding = str(path), getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return text.encode(encoding, "replace").decode(encoding, "replace")
    except LookupError:
        return text.encode("utf-8", "replace").decode("utf-8")


def _safe_streams():
    """Redirected output on Windows uses the ANSI code page; never crash on a symbol it lacks."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv=None):
    _safe_streams()
    args = parser().parse_args(argv)
    from . import llama_server, runtime_install
    # No blocking Windows "System Error" boxes when a llama.cpp program cannot start.
    runtime_install.quiet_system_errors()
    # A closed terminal or `kill` (SIGTERM/SIGHUP) then still stops every llama-server this command started.
    getattr(llama_server, "install_exit_handlers", lambda: None)()
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
            print_json(run_job(lambda progress, cancel: refresh(store, progress=progress, cancel=cancel),
                               "Fetching model metadata"))
        elif command == "models":
            from . import catalogue
            if args.models_action == "add":
                catalogue.add_entry(store, args.base_repo, args.gguf_repo, config_repo=args.config_repo)
                try:
                    fetched = run_job(lambda progress, cancel: catalogue.refresh_entry(store, args.base_repo, cancel=cancel),
                                      f"Fetching the file list for {args.base_repo}")
                    print(f"Added {args.base_repo}: {fetched.get('variants', 0)} downloadable versions. "
                          "See their IDs with: llm-config models")
                except ValueError as error:
                    print(f"Added {args.base_repo}, but its files could not be fetched yet ({error}). "
                          "Try again later with: llm-config refresh")
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
            result = run_job(app.quant_check_job(store, reference, others, scan(False)),
                             f"Compression check against {reference.name} {reference.quant}")
            if args.json:
                print_json(result)
            else:
                for label, value in (result.get("results") or {}).items():
                    print(f"{label}: {value.get('plain') if isinstance(value, dict) and value.get('plain') else value}")
                for note in result.get("notes") or []:
                    print(f"  {note}")
        elif command == "download":
            return download_command(store, args)
        elif command == "bench":
            variant = app.find_variant(store, args.variant_id)
            print("Validating file and memory, then running a local generation benchmark…", file=sys.stderr, flush=True)
            executable = args.executable or app.binary(store, "llama-bench")[0]
            measurement = bench(variant, args.model, executable, args.context, args.gpu_layers, args.gpu_index, args.timeout)
            measurement.setdefault("id", secrets.token_hex(6))  # so `community share ID` can pick it
            measurement.setdefault("kind", "bench")
            store.append("measurements", measurement)
            print_json(measurement)
        else:
            return model_command(store, args)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except BrokenPipeError:  # `llm-config models | head`: the reader left, which is not an error
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
        return 0
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
        if (store.get("settings") or {}).get("runtime_dir"):
            # `runtime use DIR` wins over the managed install in detect(); the user just asked for this one.
            store.update("settings", lambda saved: {k: v for k, v in (saved or {}).items() if k != "runtime_dir"}, {})
            info = {**runtime_install.detect(store), "reason": info.get("reason"), "warnings": info.get("warnings") or []}
    elif args.runtime_action == "install-archive":
        info = run_job(lambda progress, cancel: runtime_install.install_archive(store, args.path, progress=progress, cancel=cancel),
                       "Installing llama.cpp from the archive")
    else:
        info = runtime_install.use_directory(store, args.directory)
    show_runtime(info)
    return 0


def download_command(store, args):
    from . import discover, downloads
    variant = app.find_variant(store, args.variant_id)
    if variant.source == "local":
        raise ValueError("This model is a file on your computer, so there is nothing to download.")
    directory = (args.directory.expanduser().resolve() if args.directory else models_dir(store))
    existing = run_job(lambda progress, cancel: app.local_model(store, variant, progress=progress, cancel=cancel),
                       "Looking for a copy already on this computer")
    if existing:
        print(f"Already on this computer: {shown(existing)}")
        return 0
    plan = downloads.plan(variant, directory)
    print(f"Download {variant.name} {variant.quant}: {size(plan['total_bytes'])} "
          f"({size(plan['remaining_bytes'])} still to fetch) into {shown(directory)}", file=sys.stderr, flush=True)
    if not plan.get("enough_space"):
        raise ValueError(f"Not enough disk space: {size(plan['remaining_bytes'])} needed plus 1 GiB spare, "
                         f"{size(plan.get('disk_free'))} free. Free some space or choose --directory.")
    if not args.yes:
        if not sys.stdin or not sys.stdin.isatty():
            raise ValueError("Add --yes to download without a question (there is no keyboard input here).")
        try:
            answer = input("Download this model? [y/N] ")
        except EOFError:
            answer = ""
        if answer.strip().lower() != "y":
            return 0
    try:
        path = run_job(lambda progress, cancel: downloads.download_variant(variant, directory, progress=progress, cancel=cancel),
                       f"Downloading {variant.name} {variant.quant}")
    except KeyboardInterrupt:
        print(PAUSED, file=sys.stderr)
        return 130
    app.remember_download(store, variant, directory)
    if args.directory:
        try:  # outside the models folder, other commands only find it once it is registered
            discover.add_file(store, path)
        except ValueError as error:
            print(f"Note: {error} Run: llm-config local --scan --dir \"{shown(directory)}\"", file=sys.stderr)
    print(shown(path))
    return 0


def _model_id(record):
    return record.get("variant_id") or record.get("local_variant_id")


def local_command(store, args):
    from . import discover
    if args.add:
        record = run_job(app.add_local_job(store, args.add), f"Checking {args.add.name}")
        if args.json:
            print_json(record)
        else:
            label = "Matches a catalogue model" if record.get("variant_id") else "Registered as your own model"
            print(f"{label}: {shown(record.get('filename'))}. Model ID: {record.get('model_id') or _model_id(record)}")
            print("Use that ID with recommend, test, tune, quiz, export or run.")
        return 0
    dirs = [d.expanduser().resolve() for d in args.dirs]  # saved paths must work from any folder
    if args.scan or dirs:
        files = run_job(app.scan_job(store, dirs), "Looking for model files")
    else:
        files = store.get("local_files", [])
    locations = discover.locations(store, dirs)
    if args.json:
        print_json({"files": files, "locations": locations, "registered": store.get("local_variants", [])})
        return 0
    for location in locations:
        print(f"{'[x]' if location.get('exists') else '[ ]'} {location.get('source')}: {shown(location.get('path'))}")
    if not files:
        print("No model files found yet. Run: llm-config local --scan")
    for record in files:
        if record.get("variant_id"):
            match = f"Model ID: {record['variant_id']}{'' if record.get('verified') else ' (not verified yet)'}"
        elif record.get("local_variant_id"):
            match = f"not in the catalogue; your own Model ID: {record['local_variant_id']}"
        else:
            match = f"cannot be used: {record.get('gguf_error') or record.get('local_error') or 'incomplete'}"
        print(f"{shown(record.get('path'))} ({size(record.get('size_bytes'))}) - {match}")
    for record in store.get("local_variants", []):
        print(f"Registered: {shown(record['path'])} - Model ID: {record['variant']['id']}")
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
        if payload.get("skipped"):
            count = payload["skipped"]
            print(f"\n{count} result{'s' if count != 1 else ''} could not be shared (measured on different hardware, "
                  "or incomplete) and {} left out.".format("were" if count != 1 else "was"), file=sys.stderr)
        print("\nNothing has been sent. Review the data above, then open this link to post it yourself:", file=sys.stderr)
        print(payload["issue_url"])
        if payload.get("fits_in_url") is False:
            print("The data is too long to fit in the link, so the form opens empty: copy the text above and paste it "
                  "into the form.", file=sys.stderr)
    else:
        saved = store.get("community") or {}
        print(f"{len(saved.get('records') or [])} community results from {saved.get('source') or 'nowhere yet'}"
              f"{', fetched ' + saved['fetched_at'] if saved.get('fetched_at') else ''}. Import with: llm-config community import")
    return 0


def model_command(store, args):
    variant, candidate, hardware = model_launch(store, args)
    if args.command == "test":
        result = run_job(app.test_job(store, variant, candidate, hardware, args.kind, args.tuned, min_tps=args.min_tps or None),
                         f"Testing {variant.name} {variant.quant}")
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
            needle = result.get("needle")
            if needle:
                print(f"Long-document recall: {needle.get('found')} of {needle.get('total')} hidden facts found")
                if needle.get("note"):
                    print(f"  {needle['note']}")
        return 0
    if args.command == "export":
        result = app.export_config(store, variant, candidate, hardware, args.format, args.platform, args.tuned, port=args.port)
        if args.json:
            print_json(result)
            return 0
        notes, instructions = result.get("notes") or [], result.get("instructions") or []
        if args.output:
            executable = write_export(args.output, result)
            # write_export already saved it under the user's name, with a BOM or as runnable.
            notes = [n for n in notes if "UTF-8 with BOM" not in n]
            instructions = [line.replace(result.get("filename") or "\0", args.output.name) for line in instructions
                            if not line.startswith("Save this as") and not (executable and "chmod +x" in line)]
            print(f"Saved {shown(args.output)}", file=sys.stderr)
        else:
            print(result["content"])
        for line in instructions + notes:
            print(f"  {line}", file=sys.stderr)
        return 0
    return run_command(store, args, variant, candidate, hardware)


def write_export(path, result):
    """Bytes, not text mode, so Windows never turns a bash script's \n into \r\n.

    PowerShell scripts get a BOM so Windows PowerShell 5.1 reads non-English paths correctly;
    shell scripts become executable.
    """
    names = {str(path).lower(), str(result.get("filename") or "").lower()}
    powershell = any(n.endswith(".ps1") for n in names)
    path.write_bytes(result["content"].encode("utf-8-sig" if powershell else "utf-8"))
    if os.name != "nt" and (any(n.endswith(".sh") for n in names) or result["content"].startswith("#!")):
        path.chmod(path.stat().st_mode | 0o755)
        return True
    return False


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
        print(f"Running. OpenAI-compatible address: {server.base_url}/v1", flush=True)
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
