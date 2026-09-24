"""Loopback-only interface. No shell execution, arbitrary paths or arbitrary downloads via HTTP.

The page can start downloads, tests, tunes, quality checks and one llama-server, but only for
catalogue model IDs and recommendation candidates the app computed itself. The browser never
supplies file paths, URLs, executables or command-line flags: files live in the app's models
folder or discovered model folders, binaries come from the llama.cpp runtime the user installed,
and launch settings are built by the app. Every request needs the loopback Host header and the
session token; POSTs also need a same-origin Origin (or none) and a small JSON body. Responses
never contain absolute file paths (only file names), except the export text the user asked for.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import functools
import inspect
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import signal
import socket
import sys
import tempfile
import threading
import time
from urllib.parse import urlparse
import webbrowser

from . import app
from .app import evaluate, map_benchmark
from .catalogue import definitions, refresh, test_connection
from . import credentials
from .hardware import scan
from .calibration import calibrate, valid
from .jobs import JobManager
from .storage import models_dir, runtime_dir

STATIC = {"/": ("index.html", "text/html; charset=utf-8")}
STATIC.update({f"/{name}": (name, "text/javascript; charset=utf-8") for name in ["app.js", "wizard.js", "selects.js", "run.js", "jobs.js", "quality.js"]})
STATIC.update({f"/{name}": (name, "text/css; charset=utf-8") for name in ["style.css", "quality.css"]})

JOB_ID = re.compile(r"job-\d{1,9}")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
DEMO_MESSAGE = ("Demo mode uses fictional models, so nothing can be downloaded, tested or run. "
                "Start the app without --demo and refresh the model list to use real models.")
IDLE = {"running": False, "starting": False, "base_url": None, "openai_base_url": None, "pid": None, "config": None,
        "started_at": None, "model": None, "log_tail": "", "error": None, "candidate_id": None, "variant_id": None}
STALE_MESSAGE = "Compare again first. That option is not in the latest comparison."
# Absolute POSIX, Windows or UNC paths inside text; replaced with their last part before reaching the page.
# A folder name may hold single inner spaces ("Program Files", "My Models") as long as a separator follows it.
_SEG = r"[^\s/\\\"'<>|:*?]+(?: [^\s/\\\"'<>|:*?~][^\s/\\\"'<>|:*?]*)*[\\/]"
PATH = re.compile(rf"(?<![\w.:/~\\-])(?:~[\\/](?:{_SEG})*|(?:[A-Za-z]:[\\/]|\\\\|[\\/])(?:{_SEG})+)([^\s/\\\"'<>|:*?]*)")
# Model-written text is shown as written (a quiz answer like "cd /usr/bin" is not a leak), and
# folder labels are built by app.page_location as "~/..." on purpose.
MODEL_TEXT = {"got", "expected", "text", "prompt", "prompts", "reply", "content", "label"}
ANOTHER_PATH = re.compile(r"\s(?:~?[\\/]|[A-Za-z]:[\\/])")  # text holding several paths is not one bare path
MAX_ACTIVE_JOBS = 12
# Long log tails are cut at a byte count, so their first line may start in the middle of a path.
TAILS = {"log_tail"}
SHUTDOWN_WAIT = 15  # seconds in total for cancelled jobs to stop their llama.cpp processes
LOADING_MESSAGES = ("Loading the model into memory", "Starting llama-server")
PREFERRED_PORT = 8080  # what the exported client setups use; a busy port falls back to a free one
REPO = re.compile(r"[A-Za-z0-9][\w.-]{0,95}/[A-Za-z0-9][\w.-]{0,95}")


class Conflict(Exception):
    """Maps to 409: the request is valid but can't run now (demo mode, stale candidate, busy)."""


class NotFound(Exception):
    """Maps to 404."""


def scrub(value, roots=(), keep=frozenset()):
    """JSON-safe copy with absolute paths reduced to file names, so the page never learns folder layouts.

    `roots` (for example the home folder, which may contain spaces or the user's name) are replaced
    literally with "~" first. Keys in `keep` hold text that is passed through unchanged.
    """
    if isinstance(value, str):
        if ("\n" not in value and ". " not in value and ": " not in value and len(value) < 1000
                and not value.startswith(("/api/", "/v1/")) and not ANOTHER_PATH.search(value) and (
                PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute() or value.startswith("\\\\"))):
            return PureWindowsPath(value).name or "…"  # a bare path value, even one with spaces in every folder
        if roots:
            value = _roots_pattern(tuple(roots)).sub("~", value)
        if "/" not in value and "\\" not in value:
            return value
        return PATH.sub(lambda m: m[0] if m[0].startswith(("/api/", "/v1/")) else (m[1] or "…"), value)
    if isinstance(value, dict):
        return {str(k): v if k in keep else scrub(_tail(v) if k in TAILS else v, roots, keep) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [scrub(v, roots, keep) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub(str(value), roots, keep)  # Paths and other objects never break the JSON reply


@functools.lru_cache(maxsize=16)
def _roots_pattern(roots):
    # Whole folders only: "/tmp" must not eat the end of "/home/me/tmp".
    alternatives = "|".join(re.escape(root) for root in sorted(roots, key=len, reverse=True))
    return re.compile(rf"(?<![\w.~-])(?:{alternatives})(?=[\\/\s\"'),;]|$)")


def _tail(value):
    if isinstance(value, str) and len(value) >= 1500 and "\n" in value:
        return value.split("\n", 1)[1]
    return value


def path_roots(store):
    """Folders replaced literally by "~" before the pattern runs, so spaces or odd names in them never leak."""
    candidates = [store.directory, Path.home(), Path(tempfile.gettempdir()), Path(sys.prefix), Path(sys.base_prefix),
                  Path(__file__).parent]
    for find in (lambda: models_dir(store), lambda: runtime_dir(store),
                 lambda: (store.get("settings") or {}).get("runtime_dir"), lambda: (store.get("settings") or {}).get("models_dir")):
        try:
            candidates.append(find())
        except Exception:
            pass
    try:
        from . import discover
        candidates += [l.get("path") for l in discover.locations(store)]
    except Exception:
        pass
    found = set()
    for path in candidates:
        if not path:
            continue
        for variant in {str(path), str(Path(path).expanduser().resolve())}:
            if len(variant) > 3:
                found.add(variant)
    return tuple(sorted(found, key=len, reverse=True))


def fields(body, required=(), optional=()):
    unknown = sorted(set(body) - set(required) - set(optional))
    if unknown:
        raise ValueError(f"Unexpected field: {unknown[0]}")
    missing = [k for k in required if k not in body]
    if missing:
        raise ValueError(f"Missing field: {missing[0]}")


def text(body, name, limit=400):
    value = body.get(name)
    if not isinstance(value, str) or not 0 < len(value) <= limit or any(ord(c) < 32 for c in value):
        raise ValueError(f"{name} must be a short single-line text value")
    return value


def boolean(body, name, default=False):
    value = body.get(name, default)
    if type(value) is not bool:
        raise ValueError(f"{name} must be true or false")
    return value


def choice(body, name, options, default=None):
    value = body.get(name, default)
    if value not in options:
        raise ValueError(f"{name} must be one of: {', '.join(options)}")
    return value


def text_list(body, name, low, high, limit=400):
    values = body.get(name)
    if not isinstance(values, list) or not low <= len(values) <= high:
        raise ValueError(f"{name} must be a list of {low} to {high} items")
    for value in values:
        if not isinstance(value, str) or not 0 < len(value) <= limit or any(ord(c) < 32 for c in value):
            raise ValueError(f"{name} must contain short single-line text values")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not repeat items")
    return values


class AppServer(ThreadingHTTPServer):
    daemon_threads = True

    def server_close(self):
        """Stop the user's llama-server and cancel jobs so nothing outlives the app.

        Job threads are daemons, so we wait (bounded) for each cancelled job to kill its llama-bench or
        llama-perplexity child before the interpreter exits.
        """
        try:
            jobs = getattr(self, "jobs", None)
            cancelled = []
            for job in jobs.list() if jobs else []:
                if job["state"] in {"queued", "running"}:
                    jobs.cancel(job["id"])
                    cancelled.append(job["id"])
            deadline = time.monotonic() + SHUTDOWN_WAIT
            for job_id in cancelled:
                jobs.wait(job_id, max(0.0, deadline - time.monotonic()))
            if getattr(self, "registry", None) is not None:
                with self.registry_lock:
                    self.registry.stop()
        finally:
            super().server_close()


def make_server(store, port=8765, demo=False):
    token = secrets.token_urlsafe(32)
    static = Path(__file__).with_name("static")
    refresh_lock = threading.Lock()
    refresh_state = {"running": False, "result": None}
    calibration_lock = threading.Lock()
    calibration_state = {"running": False, "result": None, "cached": False}
    jobs = JobManager()
    latest_lock = threading.Lock()
    latest = {"candidates": {}, "hardware": None, "workload": "general", "min_tps": None, "generation": 0}
    # Longest first, so the data folder is hidden even when it sits outside the home folder.
    roots_state = {"roots": path_roots(store), "at": time.monotonic()}
    served = {"variant_id": None, "candidate_id": None}
    comparisons = {}  # comparison_id -> {label: candidate_id}, so reveal can name candidates
    catalogue_lock = threading.Lock()

    def roots():
        # Discovered folders and settings can change while the app runs; refresh the list now and then.
        if time.monotonic() - roots_state["at"] > 30:
            roots_state.update(roots=path_roots(store), at=time.monotonic())
        return roots_state["roots"]

    def clean(value, keep=frozenset()):
        return scrub(value, roots(), MODEL_TEXT | keep)

    def generation():
        with latest_lock:
            return latest["generation"]

    def remember(report, seen):
        with latest_lock:
            if seen != latest["generation"]:
                return  # the catalogue changed while this comparison ran; its IDs may be stale
            requirements = report.get("requirements") or {}
            latest.update(candidates={c["id"]: c for c in report.get("candidates", [])}, hardware=report.get("hardware"),
                          workload=requirements.get("workload", "general"), min_tps=requirements.get("min_tps"))

    def forget():
        with latest_lock:
            latest.update(candidates={}, hardware=None, generation=latest["generation"] + 1)

    def no_demo():
        if demo:
            raise Conflict(DEMO_MESSAGE)

    def candidate(candidate_id):
        """(candidate, variant, hardware) from the latest comparison; the page only names an ID."""
        if not isinstance(candidate_id, str) or not 0 < len(candidate_id) <= 600:
            raise ValueError("candidate_id must be text")
        with latest_lock:
            chosen, hardware = latest["candidates"].get(candidate_id), latest["hardware"]
        if chosen is None:
            raise Conflict(STALE_MESSAGE)
        try:
            variant = app.find_variant(store, chosen["variant_id"], demo)
        except ValueError:
            raise Conflict(STALE_MESSAGE) from None
        return dict(chosen), variant, hardware

    def variant(variant_id):
        if not isinstance(variant_id, str) or not 0 < len(variant_id) <= 400:
            raise ValueError("variant_id must be text")
        found = next((v for v in app.variants(store, demo) if v.id == variant_id), None)
        if not found:
            raise NotFound("Unknown model. Refresh the model list and compare again.")
        return found

    def is_local(found):
        return getattr(found, "source", None) == "local"

    def local_file_present(found):
        """A model the user brought from their own disk can only be reused, never fetched from the internet."""
        if is_local(found) and app.local_model(store, found, verify=False) is None:
            raise Conflict("This model came from a file on your computer, and the file is no longer there. "
                           "Put it back, or scan for models again.")

    def use_tune(body, chosen, hardware):
        """An explicit `tuned` wins; otherwise use the saved tune for this candidate when one exists."""
        if "tuned" in body:
            return boolean(body, "tuned")
        try:
            app.launch_config(store, chosen, hardware or {}, None, True)
            return True
        except (ValueError, KeyError, TypeError):
            return False

    def registry():
        with server.registry_lock:
            if server.registry is None:
                from .llama_server import ServerRegistry
                log_dir = store.directory / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                server.registry = ServerRegistry(log_dir=log_dir)
            return server.registry

    def serve_status(status=None):
        """Registry status for the page: which candidate is running, and no folders."""
        if status is None:
            with server.registry_lock:
                current = server.registry
            status = IDLE if current is None else current.status()
        status = {**IDLE, **status}
        if isinstance(status.get("config"), dict):
            status["config"] = {k: (PureWindowsPath(v).name if k.endswith("_path") and isinstance(v, str) else v)
                                for k, v in status["config"].items()}
        active = status.get("running") or status.get("starting")
        return {**status, "candidate_id": served["candidate_id"] if active else None,
                "variant_id": served["variant_id"] if active else None}

    def serving():
        with server.registry_lock:
            return server.registry is not None and bool(server.registry.status().get("running"))

    def compute_free():
        """Loading a second model next to the user's server could run out of memory."""
        if serving():
            raise Conflict("Stop the running model server first. Two models at once may not fit in memory.")

    def submit(kind, title, fn, subject=None, exclusive=None):
        if len(jobs.active()) >= MAX_ACTIVE_JOBS:
            raise Conflict("Too many tasks are waiting. Let some finish or cancel them first.")
        work = fn

        def guarded(progress, cancel):
            def report(value):
                value = dict(value)
                # llama-server loading reports elapsed seconds against its timeout, which is not progress.
                if value.get("stage") == "loading" and str(value.get("message", "")).startswith(LOADING_MESSAGES):
                    value.update(seconds=value.get("done"), done=None, total=None)
                progress(clean(value))
            # Checked again once the compute slot is ours: a server may have started while this job waited.
            if exclusive == "compute" and kind != "serve" and serving():
                raise ValueError("Stop the running model server first. Two models at once may not fit in memory.")
            try:
                result = work(report, cancel)
            except (ValueError, OSError) as error:
                # Job errors are kept in memory and shown on the page: never with folder names in them.
                raise ValueError(clean(str(error) or type(error).__name__)) from None
            return clean(result) if result is not None else None
        return 202, jobs.submit(kind, title, guarded, subject=subject, exclusive=exclusive)

    def busy(variant_id):
        in_jobs = any(variant_id == j["subject"].get("variant_id") or variant_id in (j["subject"].get("variant_ids") or [])
                      for j in jobs.active())
        return in_jobs or (served["variant_id"] == variant_id and serving())

    def preferred_port():
        """8080 when it is free, so the client setups shown next to the server work as shown."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if sys.platform != "win32":  # like llama-server itself; a closed connection's TIME_WAIT is not "in use"
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", PREFERRED_PORT))
            except OSError:
                return None
        return PREFERRED_PORT

    def serve_start(progress, cancel, chosen, found, hardware, tuned):
        server_registry = registry()
        path = app.require_model(store, found)
        port = preferred_port()
        config = app.launch_config(store, chosen, hardware, path, tuned, **({"port": port} if port else {}))
        served.update(variant_id=found.id, candidate_id=chosen["id"])
        status = server_registry.start(app.binary(store, "llama-server"), config, progress=progress, cancel=cancel)
        return serve_status(status)

    def test_job(found, chosen, hardware, kind, tuned):
        with latest_lock:
            min_tps = latest["min_tps"]
        extra = {}
        # app.test_job gains min_tps in the api-cli fix-up; older versions just skip the speed target.
        if min_tps and "min_tps" in inspect.signature(app.test_job).parameters:
            extra["min_tps"] = min_tps
        return app.test_job(store, found, chosen, hardware, kind, tuned, **extra)

    def tune_job(found, chosen, hardware, budget, goal):
        work = app.tune_job(store, found, chosen, hardware, budget, goal)
        def run(progress, cancel):
            result = work(progress, cancel)
            if isinstance(result, dict):
                # A cancelled tune returns what it measured so far but is not saved.
                result = {"saved": result.get("stopped") != "cancelled", **result}
            return result
        return run

    def compare_job(entries, hardware, prompts):
        work = app.compare_job(store, [(label, found, chosen) for label, found, chosen in entries], hardware, prompts)
        ids = {label: chosen["id"] for label, _, chosen in entries}
        def run(progress, cancel):
            result = work(progress, cancel)
            if isinstance(result, dict) and result.get("comparison_id"):
                comparisons[result["comparison_id"]] = ids
            return result
        return run

    def comparison_state(comparison_id):
        from . import evals
        try:
            return evals.blinded(store, comparison_id)
        except ValueError as error:
            raise NotFound(str(error)) from None

    def by_candidate(out, comparison_id):
        """evals keys everything by label; the page wants candidate ids plus friendly names."""
        ids = comparisons.get(comparison_id) or {label: label for label in out.get("tallies") or {}}
        name = lambda label: ids.get(label, label)
        if not isinstance(out.get("mapping"), list):
            return out  # an evals version with another shape: pass it through, the page accepts both
        mapping = []
        for entry in out["mapping"]:
            slots = {slot: name(label) for slot, label in (entry.get("slots") or {}).items()}
            mapping.append({**slots, "slots": slots, "item": entry.get("item"), "vote": entry.get("vote"),
                            "winner": name(entry["winner"]) if entry.get("winner") else None})
        return {**out, "mapping": mapping, "tallies": {name(l): n for l, n in (out.get("tallies") or {}).items()},
                "overall": name(out["overall"]) if out.get("overall") else None,
                "speed": {name(l): v for l, v in (out.get("speed") or {}).items()},
                "labels": {name(l): l for l in ids}}

    def get_route(path):
        if path == "/api/credentials":
            return 200, credentials.status()
        if path == "/api/state":
            cache = store.get("scores", {})
            try:
                entries, problem = definitions(store), None
            except ValueError as error:  # a hand-edited catalogue.json must not stop the whole page from loading
                entries, problem = [], str(error)
            return 200, {"hardware": scan(), "demo": demo, "status": store.get("refresh_status"),
                         "definitions": entries, "scores": [{"slug": x["slug"], "name": x["name"]} for x in cache.get("data", [])],
                         **({"definitions_error": problem} if problem else {})}
        if path == "/api/calibrate":
            return 200, calibration_state
        if path == "/api/refresh":
            return 200, refresh_state
        if path == "/api/jobs":
            return 200, {"jobs": jobs.list()}
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            job = jobs.get(job_id) if JOB_ID.fullmatch(job_id) else None
            if job is None:
                raise NotFound("Unknown job")
            return 200, job
        if path == "/api/runtime":
            from . import runtime_install
            return 200, runtime_install.detect(store)
        if path == "/api/local-models":
            from . import discover
            return 200, {"files": [app.page_file(f) for f in store.get("local_files", [])],
                         "locations": [app.page_location(l) for l in discover.locations(store)]}
        if path == "/api/quality/results":
            return 200, {"results": store.get("quality_results", [])[-200:]}
        if path.startswith("/api/quality/compare/"):
            comparison_id = path.removeprefix("/api/quality/compare/")
            if not SAFE_ID.fullmatch(comparison_id):
                raise NotFound("Unknown comparison")
            return 200, comparison_state(comparison_id)
        if path == "/api/serve":
            return 200, serve_status()
        if path == "/api/export":
            from . import export
            return 200, {"formats": export.formats()}
        if path == "/api/community":
            saved = store.get("community") or {}
            return 200, {"records": (saved.get("records") or [])[-5000:], "source": saved.get("source"), "fetched_at": saved.get("fetched_at")}
        raise NotFound("Not found")

    def post_route(path, body):
        if path == "/api/runtime/install":
            fields(body)
            no_demo()
            active = jobs.active(kind="runtime_install")
            return (202, active[0]) if active else submit("runtime_install", "Install llama.cpp", app.runtime_install_job(store), exclusive="download:runtime")
        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            fields(body)
            job_id = path.removeprefix("/api/jobs/").removesuffix("/cancel")
            if not JOB_ID.fullmatch(job_id) or jobs.get(job_id) is None:
                raise NotFound("Unknown job")
            return 200, jobs.cancel(job_id)
        if path == "/api/local-models/scan":
            fields(body)
            active = jobs.active(kind="local_scan")
            if active:
                return 202, active[0]
            scan_files = app.scan_job(store)
            return submit("local_scan", "Look for models on this computer", lambda p, c: [app.page_file(f) for f in scan_files(p, c)], exclusive="scan")
        if path == "/api/downloads/plan":
            fields(body, ["variant_id"])
            no_demo()
            found = variant(body["variant_id"])
            plan = app.download_plan(store, found)
            return 200, {**plan, "local": is_local(found)} if is_local(found) else plan
        if path == "/api/downloads":
            fields(body, ["variant_id"])
            no_demo()
            found = variant(body["variant_id"])
            local_file_present(found)
            active = jobs.active(kind="download", subject_key="variant_id", subject_value=found.id)
            if active:
                return 202, active[0]
            return submit("download", f"Download {found.name} {found.quant}", app.download_job(store, found),
                          subject={"variant_id": found.id}, exclusive=f"download:{found.id}")
        if path == "/api/downloads/remove":
            fields(body, ["variant_id"])
            no_demo()
            found = variant(body["variant_id"])
            if is_local(found):
                raise Conflict("This file is yours, outside the app's models folder, so the app does not delete it.")
            if busy(found.id):
                raise Conflict("This model is in use. Cancel its tasks or stop the server first.")
            return 200, {"freed_bytes": app.remove_download(store, found)}
        if path == "/api/test":
            fields(body, ["candidate_id"], ["kind", "tuned"])
            no_demo()
            kind = choice(body, "kind", app.TEST_KINDS, "full")
            chosen, found, hardware = candidate(body["candidate_id"])
            tuned = use_tune(body, chosen, hardware)
            compute_free()
            return submit("test", f"Test {found.name} {found.quant}", test_job(found, chosen, hardware, kind, tuned),
                          subject={"variant_id": found.id, "candidate_id": chosen["id"], "kind": kind}, exclusive="compute")
        if path == "/api/tune":
            fields(body, ["candidate_id", "budget_seconds"], ["goal"])
            no_demo()
            budget = body["budget_seconds"]
            if type(budget) is not int or not 60 <= budget <= 1800:
                raise ValueError("budget_seconds must be a whole number from 60 to 1800")
            goal = choice(body, "goal", app.GOALS, "generation")
            chosen, found, hardware = candidate(body["candidate_id"])
            compute_free()
            return submit("tune", f"Tune {found.name} {found.quant}", tune_job(found, chosen, hardware, budget, goal),
                          subject={"variant_id": found.id, "candidate_id": chosen["id"]}, exclusive="compute")
        if path == "/api/quality/quiz":
            fields(body, ["candidate_id"], ["workload", "include_needle"])
            no_demo()
            with latest_lock:
                default_workload = latest["workload"]
            workload = choice(body, "workload", app.WORKLOADS, default_workload)
            needle = boolean(body, "include_needle")
            chosen, found, hardware = candidate(body["candidate_id"])
            compute_free()
            return submit("quiz", f"Quiz {found.name} {found.quant}", app.quiz_job(store, found, chosen, hardware, workload, needle),
                          subject={"variant_id": found.id, "candidate_id": chosen["id"], "workload": workload}, exclusive="compute")
        if path == "/api/quality/compare":
            fields(body, ["candidate_ids", "prompts"])
            no_demo()
            ids = text_list(body, "candidate_ids", 2, 3, 600)
            prompts = body["prompts"]
            if not isinstance(prompts, list) or not 1 <= len(prompts) <= 5 or any(
                    not isinstance(p, str) or not p.strip() or len(p) > 4000 for p in prompts):
                raise ValueError("prompts must be 1 to 5 non-empty texts of up to 4000 characters")
            entries, hardware, labels = [], None, set()
            for candidate_id in ids:
                chosen, found, hardware = candidate(candidate_id)
                label = f"{found.name} {found.quant}"
                label = label if label not in labels else f"{label} ({chosen['context']:,} tokens, {app.placement(chosen)})"
                label = label if label not in labels else f"{label} #{len(labels) + 1}"
                labels.add(label)
                entries.append((label, found, chosen))
            compute_free()
            return submit("compare", "Blind comparison", compare_job(entries, hardware, prompts),
                          subject={"candidate_ids": ids, "variant_ids": [e[1].id for e in entries]}, exclusive="compute")
        if path == "/api/quality/vote":
            fields(body, ["comparison_id", "item", "slot"])
            comparison_id = text(body, "comparison_id", 64)
            if not SAFE_ID.fullmatch(comparison_id):
                raise ValueError("Unknown comparison")
            item = body["item"]
            if type(item) is not int or not 0 <= item < 100:
                raise ValueError("item must be a whole number")
            slot = choice(body, "slot", ("A", "B", "C", "tie"))
            comparison_state(comparison_id)
            from . import evals
            return 200, evals.vote(store, comparison_id, item, slot)
        if path == "/api/quality/reveal":
            fields(body, ["comparison_id"])
            comparison_id = text(body, "comparison_id", 64)
            if not SAFE_ID.fullmatch(comparison_id):
                raise ValueError("Unknown comparison")
            if comparison_state(comparison_id).get("state") == "running":
                raise Conflict("Wait until every model has answered before revealing the names.")
            from . import evals
            return 200, by_candidate(evals.reveal(store, comparison_id), comparison_id)
        if path == "/api/quality/quant-check":
            fields(body, ["reference_variant_id", "variant_ids"])
            no_demo()
            reference = variant(body["reference_variant_id"])
            others = [variant(v) for v in text_list(body, "variant_ids", 1, 3)]
            if reference.id in {v.id for v in others}:
                raise ValueError("Pick a different reference than the models you check")
            compute_free()
            return submit("quant_check", f"Compression check for {reference.name}", app.quant_check_job(store, reference, others),
                          subject={"variant_id": reference.id, "variant_ids": [v.id for v in others]}, exclusive="compute")
        if path == "/api/export":
            fields(body, ["candidate_id", "format"], ["platform", "tuned"])
            no_demo()
            fmt = text(body, "format", 40)
            platform = choice(body, "platform", ("posix", "windows"), "posix")
            chosen, found, hardware = candidate(body["candidate_id"])
            tuned = use_tune(body, chosen, hardware)
            # The export text is what the user copies into a terminal, so it keeps the real model path
            # (instructions and notes can hold commands too, like merging a split model).
            return 200, app.export_config(store, found, chosen, hardware, fmt, platform, tuned), {"content", "instructions", "notes"}
        if path == "/api/serve/start":
            fields(body, ["candidate_id"], ["tuned"])
            no_demo()
            chosen, found, hardware = candidate(body["candidate_id"])
            tuned = use_tune(body, chosen, hardware)
            if jobs.active(kind="serve"):
                raise Conflict("The server is already starting.")
            return submit("serve", f"Start {found.name} {found.quant}",
                          lambda p, c: serve_start(p, c, chosen, found, hardware, tuned),
                          subject={"variant_id": found.id, "candidate_id": chosen["id"]}, exclusive="compute")
        if path == "/api/serve/stop":
            fields(body)
            for job in jobs.active(kind="serve"):
                jobs.cancel(job["id"])
            return 200, serve_status(IDLE if server.registry is None else registry().stop())
        if path == "/api/community/import":
            fields(body)
            active = jobs.active(kind="community_import")
            if active:
                return 202, active[0]
            def run_import(progress, cancel):
                from . import community
                result = community.import_records(store, progress=progress, cancel=cancel)
                return {k: v for k, v in result.items() if k != "records"} | {"count": len(result.get("records") or [])}
            return submit("community_import", "Import community results", run_import, exclusive="download:community")
        if path == "/api/community/share":
            fields(body, ["measurement_ids"])
            ids = text_list(body, "measurement_ids", 1, 50, 64)
            if not all(SAFE_ID.fullmatch(i) for i in ids):
                raise ValueError("measurement_ids must be result IDs")
            from . import community
            # The JSON and issue link are exactly what the user will post, so they are shown unchanged.
            return 200, community.share_payload(store, ids), {"json", "issue_url"}
        if path == "/api/catalogue":
            fields(body, ["base_repo", "gguf_repo"])
            no_demo()
            base_repo, gguf_repo = text(body, "base_repo", 200), text(body, "gguf_repo", 200)
            if not REPO.fullmatch(base_repo) or not REPO.fullmatch(gguf_repo) or ".." in base_repo + gguf_repo:
                raise ValueError("Use Hugging Face repository names like owner/model")
            from . import catalogue
            with catalogue_lock:
                catalogue.add_entry(store, base_repo, gguf_repo)
            forget()
            def fetch(progress, cancel):
                if not refresh_lock.acquire(timeout=600):
                    raise ValueError("A model list refresh is still running. Try again when it finishes.")
                try:
                    progress({"stage": "fetching", "done": 0, "total": None, "message": f"Reading {base_repo} from Hugging Face…"})
                    return catalogue.refresh_entry(store, base_repo)
                finally:
                    forget()
                    refresh_lock.release()
            return submit("catalogue_refresh", f"Add {base_repo}", fetch, subject={"base_repo": base_repo},
                          exclusive="download:catalogue")
        if path == "/api/catalogue/remove":
            fields(body, ["base_repo"])
            no_demo()
            base_repo = text(body, "base_repo", 200)
            if not REPO.fullmatch(base_repo):
                raise ValueError("Use Hugging Face repository names like owner/model")
            from . import catalogue
            with catalogue_lock:
                result = catalogue.remove_entry(store, base_repo)
            forget()
            return 200, result
        raise NotFound("Not found")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, payload, mime="application/json", keep=frozenset()):
            if mime == "application/json":
                data = json.dumps(clean(payload, keep), allow_nan=False).encode()
            else:
                data = payload
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def local_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

        def reply(self, route, *args):
            try:
                status, payload, *keep = route(*args)
                self.send(status, payload, keep=keep[0] if keep else frozenset())
            except Conflict as error:
                self.send(409, {"error": str(error)})
            except NotFound as error:
                self.send(404, {"error": str(error)})
            except ImportError:
                self.send(409, {"error": "This feature is not installed in this version of the app."})
            except (ValueError, TypeError, KeyError, OSError) as error:
                self.send(400, {"error": str(error)})
            except Exception as error:  # A handler bug must not kill the connection silently.
                self.send(500, {"error": f"Unexpected error: {type(error).__name__}"})

        def do_GET(self):
            if not self.local_host():
                return self.send(403, {"error": "Local access only"})
            path = urlparse(self.path).path
            if path in STATIC:
                name, mime = STATIC[path]
                try:
                    data = (static / name).read_bytes().replace(b"__SESSION_TOKEN__", token.encode())
                except OSError:
                    return self.send(404, {"error": "Not found"})
                return self.send(200, data, mime)
            if not secrets.compare_digest(self.headers.get("X-Session-Token") or "", token):
                return self.send(403, {"error": "Reload the application to refresh the session"})
            return self.reply(get_route, path)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self.local_host() or not secrets.compare_digest(self.headers.get("X-Session-Token") or "", token) or
                    origin not in {None, f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}):
                return self.send(403, {"error": "Local session required"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("Expected a JSON object")
                path = urlparse(self.path).path
                if path == "/api/credentials/test":
                    return self.send(200, test_connection(body.get("key")))
                if path == "/api/credentials/save":
                    return self.send(200, credentials.save(body.get("key"), boolean(body, "remember", True)))
                if path == "/api/credentials/remove":
                    return self.send(200, credentials.remove())
                if path == "/api/calibrate":
                    force = body.get("force", False)
                    if type(force) is not bool:
                        raise ValueError("force must be a boolean")
                    if demo:
                        return self.send(200, {"running": False, "result": None, "demo": True})
                    if not calibration_lock.acquire(blocking=False):
                        return self.send(202, calibration_state)
                    calibration_state.update(running=True, result=None, cached=False)
                    def run_calibration():
                        try:
                            hardware = scan(False)
                            cached = store.get("calibration")
                            if not force and valid(cached, hardware) and (cached.get("cpu") or cached.get("gpus")):
                                calibration_state.update(result=cached, cached=True)
                            else:
                                result = calibrate(hardware)
                                store.put("calibration", result)
                                calibration_state["result"] = result
                        except Exception as error:
                            calibration_state["result"] = {"warnings": [f"Calibration unavailable: {type(error).__name__}: {error}"]}
                        finally:
                            calibration_state["running"] = False
                            calibration_lock.release()
                    threading.Thread(target=run_calibration, daemon=True).start()
                    return self.send(202, calibration_state)
                if path == "/api/recommend":
                    seen = generation()
                    report = evaluate(store, body, demo)
                    remember(report, seen)
                    return self.send(200, report)
                if path == "/api/map":
                    if not refresh_lock.acquire(blocking=False):
                        return self.send(409, {"error": "Wait for metadata refresh to finish"})
                    try:
                        slug = body.get("slug")
                        if slug is not None and not isinstance(slug, str):
                            raise ValueError("slug must be text")
                        mapped = map_benchmark(store, text(body, "base_repo", 200), slug)
                    finally:
                        refresh_lock.release()
                    forget()
                    return self.send(200, mapped)
                if path == "/api/refresh":
                    include_scores = body.get("include_scores", True)
                    if type(include_scores) is not bool:
                        raise ValueError("include_scores must be a boolean")
                    include_models = body.get("include_models", True)
                    if type(include_models) is not bool:
                        raise ValueError("include_models must be a boolean")
                    if not refresh_lock.acquire(blocking=False):
                        return self.send(409, {"error": "Metadata refresh already running"})
                    refresh_state.update(running=True, result=None, progress=None)
                    def progress(value):
                        refresh_state["progress"] = value
                    def run():
                        try:
                            refresh_state["result"] = refresh(store, include_scores=include_scores, include_models=include_models, progress=progress)
                        except Exception as error:
                            refresh_state["result"] = {"warnings": [f"Refresh failed: {type(error).__name__}: {error}"]}
                        finally:
                            forget()  # model IDs may have changed; candidates must be recomputed
                            refresh_state["running"] = False
                            refresh_lock.release()
                    threading.Thread(target=run, daemon=True).start()
                    return self.send(202, refresh_state)
            except (ValueError, TypeError, KeyError, OSError) as error:
                return self.send(400, {"error": str(error)})
            except Exception as error:
                return self.send(500, {"error": f"Unexpected error: {type(error).__name__}"})
            return self.reply(post_route, path, body)

    server = AppServer(("127.0.0.1", port), Handler)
    server.jobs, server.registry, server.registry_lock = jobs, None, threading.Lock()
    server.latest = latest
    return server


def serve(store, port=8765, demo=False, open_browser=True):
    server = make_server(store, port, demo)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"LLM Configurator: {url}\nPress Ctrl+C to stop. Runs on this computer only.", flush=True)
    if open_browser:
        webbrowser.open(url)
    def stop(*_):
        raise KeyboardInterrupt
    # Closing the terminal or `kill` must take the same clean path as Ctrl+C: the user's llama-server
    # runs in its own process group and would otherwise keep its memory and port with no owner.
    if threading.current_thread() is threading.main_thread():
        for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
            number = getattr(signal, name, None)
            if number is not None:
                try:
                    signal.signal(number, stop)
                except (OSError, ValueError):
                    pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
