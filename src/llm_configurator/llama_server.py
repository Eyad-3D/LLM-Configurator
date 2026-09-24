"""Start, talk to and stop a local llama-server safely.

Promises: argv lists only (never a shell), output goes to a log file (never an unread pipe),
servers listen on this computer only, the whole process tree is stopped on stop, error or cancel,
and failures are reported as plain reasons. Timings come from llama.cpp itself; missing values
stay None rather than being guessed.
"""
import atexit
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import weakref

import psutil

from . import launch
from .domain import Cancelled, check_cancel, now

TIMING_KEYS = ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"]
# Loopback only: a system proxy must never see local traffic.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LIVE = weakref.WeakSet()

# Ordered: the first match wins, so specific causes come before generic ones.
FAILURES = [
    (re.compile(r"out of memory|failed to allocate|unable to allocate|insufficient memory|cudaMalloc failed"
                r"|ErrorOutOfDeviceMemory|OutOfMemory|std::bad_alloc", re.I),
     "Not enough memory to load this model with these settings. Try a smaller download, a shorter context "
     "or fewer GPU layers, and close other programs."),
    (re.compile(r"unknown model architecture: '([^']*)'", re.I),
     "This llama.cpp version does not know this model's design ({0}). Update llama.cpp or pick another model."),
    (re.compile(r"No such file or directory|failed to open GGUF file", re.I),
     "The model file is missing. Download it again."),
    (re.compile(r"invalid magic|failed to read magic|failed to read key-value|corrupt|not within the file bounds"
                r"|failed to read tensor|invalid split|gguf_init_from_\w+: failed", re.I),
     "The model file looks damaged or incomplete. Delete it and download it again."),
    (re.compile(r"couldn't bind|address already in use|bind\(\) failed", re.I),
     "The network port is already used by another program. Pick another port or close that program."),
    (re.compile(r"V cache quantization requires flash.?attn", re.I),
     "Compressed notes (the KV cache, the model's short-term notepad) need flash attention turned on."),
    (re.compile(r"invalid device|failed to initialize CUDA|no usable GPU|No devices found|driver version is insufficient"
                r"|failed to load backend|vk::|ggml_metal_init: error", re.I),
     "llama.cpp could not use the graphics card. Update the GPU driver, or run on the processor with 0 GPU layers."),
    (re.compile(r"error while handling argument|invalid argument|unknown argument|unknown value for", re.I),
     "This llama.cpp version does not understand one of the settings. Update llama.cpp."),
]
WARNINGS = [
    (re.compile(r"no usable GPU found", re.I),
     "llama.cpp found no usable graphics card, so the model runs on the processor only (slower)."),
]


def free_port():
    """A currently free loopback port. Another program could still take it before we bind, so callers retry once."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def failure_from_log(text, returncode=None):
    """Plain reason for a failed start, or None when the log gives no recognisable cause."""
    for pattern, message in FAILURES:
        match = pattern.search(text or "")
        if match:
            return message.format(*(match.groups() or ("unknown",)))
    if returncode is not None and returncode in (-9, 137):
        return "The system stopped the model server, most likely because memory ran out."
    return None


def argv(command):
    """Normalise an argv prefix; a string is one executable path, never a shell command line."""
    if isinstance(command, (str, os.PathLike)):
        return [os.fspath(command)]
    if not command or not all(isinstance(part, (str, os.PathLike)) for part in command):
        raise ValueError("llama-server command must be a path or a list of arguments")
    return [os.fspath(part) for part in command]


def kill_tree(pid, timeout=10):
    """Terminate a process and all its children, then force-kill whatever is left. Returns nothing."""
    try:
        parent = psutil.Process(pid)
        processes = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=timeout)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=5)


def _normalize_timings(raw):
    raw = raw if isinstance(raw, dict) else {}
    timings = {key: value for key, value in raw.items() if isinstance(value, (int, float)) and not isinstance(value, bool)}
    for key in TIMING_KEYS:
        timings.setdefault(key, None)
    return timings


class LlamaServer:
    """One llama-server process. Use start()/stop() or `with LlamaServer(...) as server:` (starts on enter)."""

    def __init__(self, command, config, log_path=None):
        self.command = argv(command)
        self.config = launch.normalize(config)
        if not self.config["model_path"]:
            raise ValueError("A downloaded model file is required")
        self.log_path = Path(log_path) if log_path else None
        self._own_log = log_path is None
        self.process = None
        self.port = None
        self.started_at = None
        self.load_seconds = None
        self._log = None
        self._final_tail = ""
        self._stopped = False

    @property
    def base_url(self):
        return f"http://{self.config['host']}:{self.port}" if self.port else None

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, timeout=300, progress=None, cancel=None):
        """Spawn and wait until /health answers 200. Raises ValueError with a plain reason; Cancelled on cancel."""
        if self.process is not None:
            raise ValueError("This server was already started")
        attempts = 1 if self.config["port"] else 2
        for attempt in range(attempts):
            try:
                self._spawn(self.config["port"] or free_port())
                self._wait_ready(timeout, progress, cancel)
                return self
            except ValueError as error:
                port_taken = "port is already used" in str(error)
                self.stop()
                if not (port_taken and attempt + 1 < attempts):
                    raise
                self.process, self._stopped = None, False
            except BaseException:
                self.stop()
                raise

    def _spawn(self, port):
        self.port = port
        config = {**self.config, "port": port}
        args = self.command + launch.server_args(config)
        if self.log_path is None:
            handle, name = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
            os.close(handle)
            self.log_path = Path(name)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "wb")
        options = {"stdin": subprocess.DEVNULL, "stdout": self._log, "stderr": subprocess.STDOUT,
                   "env": launch.server_env(config)}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        try:
            self.process = subprocess.Popen(args, **options)
        except OSError as error:
            self._log.close()
            raise ValueError(f"Could not run llama-server ({error.strerror or error}). Check the llama.cpp install.") from None
        self.started_at = now()
        _LIVE.add(self)

    def _wait_ready(self, timeout, progress, cancel):
        begin = time.monotonic()
        while True:
            check_cancel(cancel)
            elapsed = time.monotonic() - begin
            code = self.process.poll()
            if code is not None:
                reason = self.failure_reason() or "llama-server stopped while loading the model."
                raise ValueError(f"{reason} (exit code {code})")
            state = self._health()
            if state == "ok":
                self.load_seconds = round(elapsed, 3)
                if progress:
                    progress({"stage": "ready", "done": 1, "total": 1, "message": "The model is loaded and ready."})
                return
            if elapsed > timeout:
                raise ValueError(f"The model did not finish loading within {int(timeout)} seconds. "
                                 "It may be too large for this computer, or the disk may be slow.")
            if progress:
                progress({"stage": "loading", "done": round(elapsed, 1), "total": timeout,
                          "message": "Loading the model into memory…" if state == "loading" else "Starting llama-server…"})
            time.sleep(0.1 if elapsed < 2 else 0.25)

    def _health(self):
        try:
            with _OPENER.open(f"{self.base_url}/health", timeout=2) as response:
                return "ok" if response.status == 200 else "loading"
        except urllib.error.HTTPError as error:
            error.close()
            return "loading" if error.code == 503 else "error"
        except (OSError, ValueError):
            return "starting"

    def ready(self):
        return self.running() and self._health() == "ok"

    def request(self, path, body=None, timeout=300):
        """JSON request to the local server; errors become plain ValueErrors."""
        if not self.running():
            raise ValueError(self._dead_message())
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method="GET" if body is None else "POST",
                                         headers={"Content-Type": "application/json"})
        try:
            with _OPENER.open(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ValueError(self._http_error(error)) from None
        except (socket.timeout, TimeoutError):
            raise ValueError(f"The model server took longer than {int(timeout)} seconds to answer.") from None
        except (OSError, urllib.error.URLError):
            raise ValueError(self._dead_message() if not self.running() else
                             "The model server is not responding. Try stopping and starting it again.") from None
        except ValueError:
            raise ValueError("The model server sent an unreadable answer.") from None

    def _http_error(self, error):
        try:
            detail = json.loads(error.read().decode("utf-8")).get("error") or {}
        except (ValueError, AttributeError, OSError):
            detail = {}
        finally:
            error.close()
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        kind = detail.get("type") if isinstance(detail, dict) else None
        if kind == "exceed_context_size_error" or "exceeds the available context size" in (message or ""):
            return ("The text is longer than the model's context window (its short-term notepad). "
                    "Use a longer context or a shorter text.")
        if error.code == 503:
            return "The model is still loading. Try again in a moment."
        return f"The model server refused the request ({error.code}): {message or 'no details given'}"

    def _dead_message(self):
        if self.process is None:
            return "The model server has not been started."
        reason = self.failure_reason()
        return f"The model server stopped. {reason}" if reason else "The model server stopped unexpectedly."

    def chat(self, messages, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300,
             cache_prompt=False, enable_thinking=None, extra=None):
        """OpenAI-style chat. cache_prompt=False makes every call re-read the whole prompt, so timings are honest.
        enable_thinking=False asks reasoning models (for example Qwen3) to answer without a thinking section."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                "cache_prompt": cache_prompt, "stream": False}
        if stop:
            body["stop"] = stop
        if enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        body.update(extra or {})
        begin = time.monotonic()
        raw = self.request("/v1/chat/completions", body, timeout)
        try:
            choice = raw["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError):
            raise ValueError("The model server sent an answer without any text.") from None
        return {"text": message.get("content") or "", "reasoning": message.get("reasoning_content"),
                "finish_reason": choice.get("finish_reason"), "timings": _normalize_timings(raw.get("timings")),
                "usage": raw.get("usage") or {}, "seconds": round(time.monotonic() - begin, 4)}

    def complete(self, prompt, max_tokens=256, temperature=0.0, seed=1, stop=None, timeout=300, cache_prompt=False, extra=None):
        """Raw /completion (no chat template). Same return shape as chat()."""
        body = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature, "seed": seed,
                "cache_prompt": cache_prompt, "stream": False}
        if stop:
            body["stop"] = stop
        body.update(extra or {})
        begin = time.monotonic()
        raw = self.request("/completion", body, timeout)
        if not isinstance(raw, dict):
            raise ValueError("The model server sent an unreadable answer.")
        stop_type = raw.get("stop_type")
        finish = {"eos": "stop", "word": "stop", "limit": "length"}.get(stop_type)
        evaluated, predicted = raw.get("tokens_evaluated"), raw.get("tokens_predicted")
        usage = {"prompt_tokens": evaluated, "completion_tokens": predicted,
                 "total_tokens": evaluated + predicted if isinstance(evaluated, int) and isinstance(predicted, int) else None}
        return {"text": raw.get("content") or "", "reasoning": None, "finish_reason": finish,
                "timings": _normalize_timings(raw.get("timings")), "usage": usage,
                "seconds": round(time.monotonic() - begin, 4)}

    def tokenize(self, text, add_special=False, timeout=60):
        tokens = self.request("/tokenize", {"content": text, "add_special": add_special}, timeout).get("tokens")
        if not isinstance(tokens, list):
            raise ValueError("The model server sent an unreadable token list.")
        return [t["id"] if isinstance(t, dict) else t for t in tokens]

    def log_tail(self, chars=4000):
        if self.log_path is None or not self.log_path.exists():
            return self._final_tail[-chars:]
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - chars * 4))
                return handle.read().decode("utf-8", errors="replace")[-chars:]
        except OSError:
            return self._final_tail[-chars:]

    def failure_reason(self):
        """Plain reason read from llama-server's own messages, or None if the log shows no known failure."""
        code = self.process.poll() if self.process else None
        return failure_from_log(self.log_tail(20000), code)

    def warnings(self):
        text = self.log_tail(20000)
        return [message for pattern, message in WARNINGS if pattern.search(text)]

    def stop(self, timeout=10):
        """Stop the whole process tree. Safe to call more than once. Returns the exit code (None if never started)."""
        code = None
        if self.process is not None and not self._stopped:
            if self.process.poll() is None:
                kill_tree(self.process.pid, timeout)
            try:
                code = self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                code = self.process.wait(timeout=5)
            self._stopped = True
        elif self.process is not None:
            code = self.process.returncode
        if self._log and not self._log.closed:
            self._log.close()
        if self._own_log and self.log_path and self.log_path.exists():
            self._final_tail = self.log_tail(20000)
            try:
                self.log_path.unlink()
            except OSError:
                pass
        _LIVE.discard(self)
        return code

    def __enter__(self):
        if self.process is None:
            self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


@atexit.register
def _stop_all():
    for server in list(_LIVE):
        try:
            server.stop(timeout=3)
        except Exception:
            pass


def _vram_by_pid(timeout=5):
    """{pid: bytes} from nvidia-smi, or None when it is unavailable or reports no numbers (for example on WDDM)."""
    try:
        completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return None
    usage = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            usage[int(parts[0])] = usage.get(int(parts[0]), 0) + int(parts[1]) * 1024**2
    return usage


class PeakMemory:
    """Samples the process tree's RAM (RSS) and, for CUDA, its VRAM until stop(). Unified memory counts as RAM."""

    def __init__(self, pid, gpu_backend=None, interval=0.25):
        self.pid = pid
        self.gpu_backend = gpu_backend
        self.interval = interval
        self.peak_ram = 0
        self.peak_vram = None
        self.samples = 0
        self._vram_ok = gpu_backend == "cuda"
        self._last_vram = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _tree(self):
        try:
            parent = psutil.Process(self.pid)
            return [parent] + parent.children(recursive=True)
        except psutil.NoSuchProcess:
            return []

    def sample(self):
        processes = self._tree()
        ram = 0
        for process in processes:
            try:
                ram += process.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        if processes:
            self.samples += 1
            self.peak_ram = max(self.peak_ram, ram)
        # nvidia-smi is slow to start, so VRAM is sampled at most once a second.
        if self._vram_ok and processes and time.monotonic() - self._last_vram >= max(self.interval, 1.0):
            self._last_vram = time.monotonic()
            usage = _vram_by_pid()
            if usage is None:
                self._vram_ok = False
            else:
                pids = {p.pid for p in processes}
                total = sum(value for pid, value in usage.items() if pid in pids)
                self.peak_vram = max(self.peak_vram or 0, total)

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, name="peak-memory", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self._last_vram = 0.0
        self.sample()
        return {"peak_ram_bytes": self.peak_ram or None, "peak_vram_bytes": self.peak_vram, "samples": self.samples}


class ServerRegistry:
    """The user's one long-running server. Starting a new one stops the old one first."""

    def __init__(self, log_dir=None):
        self.log_dir = Path(log_dir) if log_dir else None
        self._server = None
        self._starting = None
        self._error = None
        self._lock = threading.Lock()      # serialises start/stop
        self._state = threading.Lock()     # guards the fields read by status()

    def start(self, command, config, progress=None, cancel=None, timeout=300):
        with self._lock:
            self._stop_current()
            log_path = self.log_dir / "llama-server.log" if self.log_dir else None
            server = LlamaServer(command, config, log_path=log_path)
            with self._state:
                self._starting, self._error = server, None
            try:
                server.start(timeout=timeout, progress=progress, cancel=cancel)
            except Cancelled:
                with self._state:
                    self._starting = None
                raise
            except ValueError as error:
                with self._state:
                    self._starting, self._error = None, str(error)
                raise
            with self._state:
                self._server, self._starting = server, None
            return self.status()

    def _stop_current(self):
        with self._state:
            server, self._server = self._server, None
        if server:
            server.stop()

    def stop(self):
        with self._lock:
            self._stop_current()
            with self._state:
                self._error = None
        return self.status()

    def status(self):
        with self._state:
            server, starting, error = self._server, self._starting, self._error
        if server and not server.running():
            error = server.failure_reason() or "The server stopped unexpectedly."
        running = bool(server and server.running())
        shown = server or starting
        base = server.base_url if running else None
        return {"running": running, "starting": starting is not None, "base_url": base,
                "openai_base_url": f"{base}/v1" if base else None, "pid": server.pid if running else None,
                "config": dict(shown.config) if shown else None, "started_at": server.started_at if server else None,
                "model": (shown.config.get("alias") or Path(shown.config["model_path"]).name) if shown else None,
                "log_tail": shown.log_tail(2000) if shown else "", "error": error}
