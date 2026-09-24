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
import json
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlparse
import webbrowser

from . import app
from .app import evaluate, map_benchmark
from .catalogue import definitions, refresh, test_connection
from . import credentials
from .hardware import scan
from .calibration import calibrate, valid
from .jobs import JobManager

STATIC = {"/": ("index.html", "text/html; charset=utf-8")}
STATIC.update({f"/{name}": (name, "text/javascript; charset=utf-8") for name in ["app.js", "wizard.js", "selects.js", "run.js", "jobs.js", "quality.js"]})
STATIC.update({f"/{name}": (name, "text/css; charset=utf-8") for name in ["style.css", "quality.css"]})

JOB_ID = re.compile(r"job-\d{1,9}")
SAFE_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
HEX_ID = re.compile(r"[0-9a-f]{4,64}")
DEMO_MESSAGE = ("Demo mode uses fictional models, so nothing can be downloaded, tested or run. "
                "Start the app without --demo and refresh the model list to use real models.")
IDLE = {"running": False, "base_url": None, "openai_base_url": None, "pid": None, "config": None,
        "started_at": None, "model": None, "log_tail": ""}
STALE_MESSAGE = "Compare again first. That option is not in the latest comparison."
# Absolute POSIX or Windows paths inside text; replaced with their last part before reaching the page.
PATH = re.compile(r"(?<![\w.:/~\\-])(?:[A-Za-z]:[\\/]|/)(?:[^\s/\\\"'<>|:*?]+[\\/])+([^\s/\\\"'<>|:*?]*)")


class Conflict(Exception):
    """Maps to 409: the request is valid but can't run now (demo mode, stale candidate, busy)."""


class NotFound(Exception):
    """Maps to 404."""


def scrub(value):
    """Replace absolute paths in every string with their file name; the page never learns folder layouts."""
    if isinstance(value, str):
        return PATH.sub(lambda m: m[0] if m[0].startswith(("/api/", "/v1/")) else (m[1] or "…"), value) if "/" in value or "\\" in value else value
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


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
        """Stop the user's llama-server and cancel jobs so nothing outlives the app."""
        try:
            jobs = getattr(self, "jobs", None)
            for job in jobs.list() if jobs else []:
                if job["state"] in {"queued", "running"}:
                    jobs.cancel(job["id"])
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
    latest = {"candidates": {}, "hardware": None, "workload": "general"}

    def remember(report):
        with latest_lock:
            latest.update(candidates={c["id"]: c for c in report.get("candidates", [])}, hardware=report.get("hardware"),
                          workload=(report.get("requirements") or {}).get("workload", "general"))

    def forget():
        with latest_lock:
            latest.update(candidates={}, hardware=None)

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

    def registry():
        with server.registry_lock:
            if server.registry is None:
                from .llama_server import ServerRegistry
                server.registry = ServerRegistry()
            return server.registry

    def serving():
        with server.registry_lock:
            return server.registry is not None and bool(server.registry.status().get("running"))

    def compute_free():
        """Loading a second model next to the user's server could run out of memory."""
        if serving():
            raise Conflict("Stop the running model server first. Two models at once may not fit in memory.")

    def submit(kind, title, fn, subject=None, exclusive=None):
        return 202, jobs.submit(kind, title, fn, subject=subject, exclusive=exclusive)

    def serve_start(progress, cancel, chosen, found, hardware, tuned):
        server_registry = registry()
        config = app.launch_config(store, chosen, hardware, app.require_model(store, found), tuned)
        return server_registry.start(app.binary(store, "llama-server"), config, progress=progress, cancel=cancel)

    def get_route(path):
        if path == "/api/credentials":
            return 200, credentials.status()
        if path == "/api/state":
            cache = store.get("scores", {})
            return 200, {"hardware": scan(), "demo": demo, "status": store.get("refresh_status"),
                         "definitions": definitions(store), "scores": [{"slug": x["slug"], "name": x["name"]} for x in cache.get("data", [])]}
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
                         "locations": [app.page_location(l) for l in discover.locations()]}
        if path == "/api/quality/results":
            return 200, {"results": store.get("quality_results", [])}
        if path.startswith("/api/quality/compare/"):
            comparison_id = path.removeprefix("/api/quality/compare/")
            if not SAFE_ID.fullmatch(comparison_id):
                raise NotFound("Unknown comparison")
            from . import evals
            return 200, evals.blinded(store, comparison_id)
        if path == "/api/serve":
            return 200, IDLE if server.registry is None else registry().status()
        if path == "/api/community":
            saved = store.get("community") or {}
            return 200, {"records": saved.get("records", []), "source": saved.get("source"), "fetched_at": saved.get("fetched_at")}
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
            active = jobs.active(kind="scan")
            if active:
                return 202, active[0]
            scan_files = app.scan_job(store)
            return submit("scan", "Look for models on this computer", lambda p, c: [app.page_file(f) for f in scan_files(p, c)], exclusive="scan")
        if path == "/api/downloads/plan":
            fields(body, ["variant_id"])
            no_demo()
            return 200, app.download_plan(store, variant(body["variant_id"]))
        if path == "/api/downloads":
            fields(body, ["variant_id"])
            no_demo()
            found = variant(body["variant_id"])
            active = jobs.active(kind="download", subject_key="variant_id", subject_value=found.id)
            if active:
                return 202, active[0]
            return submit("download", f"Download {found.name} {found.quant}", app.download_job(store, found),
                          subject={"variant_id": found.id}, exclusive=f"download:{found.id}")
        if path == "/api/downloads/remove":
            fields(body, ["variant_id"])
            no_demo()
            found = variant(body["variant_id"])
            if jobs.active(subject_key="variant_id", subject_value=found.id):
                raise Conflict("This model is busy. Cancel its download or test first.")
            return 200, {"freed_bytes": app.remove_download(store, found)}
        if path == "/api/test":
            fields(body, ["candidate_id"], ["kind", "tuned"])
            no_demo()
            kind = choice(body, "kind", app.TEST_KINDS, "full")
            tuned = boolean(body, "tuned")
            chosen, found, hardware = candidate(body["candidate_id"])
            compute_free()
            return submit("test", f"Test {found.name} {found.quant}", app.test_job(store, found, chosen, hardware, kind, tuned),
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
            return submit("tune", f"Tune {found.name} {found.quant}", app.tune_job(store, found, chosen, hardware, budget, goal),
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
            return submit("compare", "Blind comparison", app.compare_job(store, entries, hardware, prompts),
                          subject={"candidate_ids": ids}, exclusive="compute")
        if path == "/api/quality/vote":
            fields(body, ["comparison_id", "item", "slot"])
            comparison_id = text(body, "comparison_id", 64)
            if not SAFE_ID.fullmatch(comparison_id):
                raise ValueError("Unknown comparison")
            item = body["item"]
            if type(item) is not int or not 0 <= item < 100:
                raise ValueError("item must be a whole number")
            slot = choice(body, "slot", ("A", "B", "C"))
            from . import evals
            return 200, evals.vote(store, comparison_id, item, slot)
        if path == "/api/quality/reveal":
            fields(body, ["comparison_id"])
            comparison_id = text(body, "comparison_id", 64)
            if not SAFE_ID.fullmatch(comparison_id):
                raise ValueError("Unknown comparison")
            from . import evals
            return 200, evals.reveal(store, comparison_id)
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
            tuned = boolean(body, "tuned")
            chosen, found, hardware = candidate(body["candidate_id"])
            # The export text is what the user copies into a terminal, so it keeps the real model path.
            return 200, app.export_config(store, found, chosen, hardware, fmt, platform, tuned), False
        if path == "/api/serve/start":
            fields(body, ["candidate_id"], ["tuned"])
            no_demo()
            tuned = boolean(body, "tuned")
            chosen, found, hardware = candidate(body["candidate_id"])
            if jobs.active(kind="serve"):
                raise Conflict("The server is already starting.")
            return submit("serve", f"Start {found.name} {found.quant}",
                          lambda p, c: serve_start(p, c, chosen, found, hardware, tuned),
                          subject={"variant_id": found.id, "candidate_id": chosen["id"]}, exclusive="compute")
        if path == "/api/serve/stop":
            fields(body)
            for job in jobs.active(kind="serve"):
                jobs.cancel(job["id"])
            return 200, IDLE if server.registry is None else registry().stop()
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
            ids = text_list(body, "measurement_ids", 1, 20, 64)
            if not all(HEX_ID.fullmatch(i) for i in ids):
                raise ValueError("measurement_ids must be result IDs")
            from . import community
            return 200, community.share_payload(store, ids)
        raise NotFound("Not found")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, payload, mime="application/json", private=True):
            if mime == "application/json":
                data = json.dumps(scrub(payload) if private else payload, allow_nan=False).encode()
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
                status, payload, *private = route(*args)
                self.send(status, payload, private=private[0] if private else True)
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
            if self.headers.get("X-Session-Token") != token:
                return self.send(403, {"error": "Reload the application to refresh the session"})
            return self.reply(get_route, path)

        def do_POST(self):
            origin = self.headers.get("Origin")
            if (not self.local_host() or self.headers.get("X-Session-Token") != token or
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
                    return self.send(200, credentials.save(body.get("key"), body.get("remember", True)))
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
                    report = evaluate(store, body, demo)
                    remember(report)
                    return self.send(200, report)
                if path == "/api/map":
                    if refresh_state["running"]:
                        return self.send(409, {"error": "Wait for metadata refresh to finish"})
                    return self.send(200, map_benchmark(store, body["base_repo"], body.get("slug")))
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
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
