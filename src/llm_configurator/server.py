"""Loopback-only interface. No shell execution, process termination or downloads via HTTP."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
from urllib.parse import urlparse
import webbrowser

from .app import evaluate, map_benchmark
from .catalogue import definitions, refresh, test_connection
from . import credentials
from .hardware import scan


def make_server(store, port=8765, demo=False):
    token = secrets.token_urlsafe(32)
    static = Path(__file__).with_name("static")
    refresh_lock = threading.Lock()
    refresh_state = {"running": False, "result": None}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send(self, status, payload, mime="application/json"):
            data = json.dumps(payload, allow_nan=False).encode() if mime == "application/json" else payload
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

        def do_GET(self):
            if not self.local_host():
                return self.send(403, {"error": "Local access only"})
            path = urlparse(self.path).path
            try:
                if path in {"/", "/app.js", "/wizard.js", "/selects.js", "/style.css"}:
                    name = {"/": "index.html", "/app.js": "app.js", "/wizard.js": "wizard.js", "/selects.js": "selects.js", "/style.css": "style.css"}[path]
                    data = (static / name).read_bytes().replace(b"__SESSION_TOKEN__", token.encode())
                    mime = {"/": "text/html; charset=utf-8", "/app.js": "text/javascript; charset=utf-8", "/wizard.js": "text/javascript; charset=utf-8", "/selects.js": "text/javascript; charset=utf-8", "/style.css": "text/css; charset=utf-8"}[path]
                    return self.send(200, data, mime)
                if self.headers.get("X-Session-Token") != token:
                    return self.send(403, {"error": "Reload the application to refresh the session"})
                if path == "/api/credentials":
                    return self.send(200, credentials.status())
                if path == "/api/state":
                    cache = store.get("scores", {})
                    return self.send(200, {"hardware": scan(), "demo": demo, "status": store.get("refresh_status"),
                                           "definitions": definitions(store), "scores": [{"slug": x["slug"], "name": x["name"]} for x in cache.get("data", [])]})
                if path == "/api/refresh":
                    return self.send(200, refresh_state)
                return self.send(404, {"error": "Not found"})
            except (ValueError, OSError) as error:
                return self.send(400, {"error": str(error)})

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
                if path == "/api/recommend":
                    return self.send(200, evaluate(store, body, demo))
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
                            refresh_state["running"] = False
                            refresh_lock.release()
                    threading.Thread(target=run, daemon=True).start()
                    return self.send(202, refresh_state)
                return self.send(404, {"error": "Not found"})
            except (ValueError, TypeError, KeyError, OSError) as error:
                return self.send(400, {"error": str(error)})

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
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
