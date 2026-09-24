"""Start, talk to and stop a local llama-server safely.

Promises: argv lists only (never a shell), output goes to a log file (never an unread pipe),
servers listen on this computer only, the whole process tree is stopped on stop, error or cancel,
and failures are reported as plain reasons. Timings come from llama.cpp itself; missing values
stay None rather than being guessed.
"""
import atexit
import http.client
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

import psutil

from . import launch
from .domain import Cancelled, check_cancel, now

TIMING_KEYS = ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"]
# Loopback only: a system proxy must never see local traffic.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_LIVE = set()                 # every started, not yet stopped server (strong refs: a dropped server still gets stopped)
_LIVE_LOCK = threading.Lock()
_SHUTTING_DOWN = threading.Event()

# Ordered: the first match wins, so specific causes come before generic ones. Lines come from real llama.cpp
# logs (tests/integration/samples/error-*.txt); a CPU-only build prints "no usable GPU found" on every start, so
# that line is only a warning, never a failure reason.
PORT_BUSY = "The network port is already used by another program. Pick another port or close that program."
CRASHED = "llama.cpp crashed with these settings. Try different settings, or update llama.cpp."
FAILURES = [
    (re.compile(r"out of memory|failed to allocate|unable to allocate|insufficient memory|cudaMalloc failed"
                r"|ErrorOutOfDeviceMemory|OutOfMemory|std::bad_alloc", re.I),
     "Not enough memory to load this model with these settings. Try a smaller download, a shorter context "
     "or fewer GPU layers, and close other programs."),
    (re.compile(r"unknown model architecture: '([^']*)'", re.I),
     "This llama.cpp version does not know this model's design ({0}). Update llama.cpp or pick another model."),
    (re.compile(r"failed to open GGUF file '([^']*)' \((?:Permission denied|Access is denied)", re.I),
     "The app is not allowed to read the model file ({0}). Check the file's permissions."),
    (re.compile(r"failed to open GGUF file '([^']*)'", re.I),
     "The model file is missing ({0}). Download it again."),
    (re.compile(r"illegal split file idx|must be loaded with the first split", re.I),
     "This is not the first part of a split model. Choose the file whose name ends in -00001-of-…"),
    (re.compile(r"invalid magic|failed to read magic|failed to read key-value|corrupt|not within the file bounds"
                r"|failed to read tensor|invalid split|gguf_init_from_\w+: failed", re.I),
     "The model file looks damaged or incomplete. Delete it and download it again."),
    (re.compile(r"couldn't bind|address already in use|bind\(\) failed", re.I), PORT_BUSY),
    (re.compile(r"V cache quantization requires flash.?attn|quantized V cache requires flash.?attn", re.I),
     "Compressed notes (the KV cache, the model's short-term notepad) need flash attention turned on."),
    (re.compile(r"invalid device", re.I),
     "This llama.cpp cannot use the graphics card it was asked to use (it may be built without support for it). "
     "Install the llama.cpp build for your graphics card, or run on the processor with 0 GPU layers."),
    (re.compile(r"the argument has been removed", re.I),
     "This llama.cpp version no longer accepts one of the settings the app sent. Update LLM Configurator, "
     "or use an older llama.cpp."),
    (re.compile(r"error while handling argument|invalid argument|unknown argument|unknown value for", re.I),
     "This llama.cpp version does not understand one of the settings. Update llama.cpp."),
    (re.compile(r"GGML_ASSERT|GGML_ABORT|Segmentation fault|core dumped", re.I), CRASHED),
]
_CRASH_CODES = {-6, -11, 134, 139, 3221225477, 3221226505}  # SIGABRT, SIGSEGV; Windows access violation, stack guard
# Checked after the exit code: these lines can appear next to other failures, so they are only a last guess.
GPU_FAILURE = (re.compile(r"failed to initialize CUDA|No devices found|driver version is insufficient"
                          r"|failed to load backend|vk::|ggml_metal_init: error", re.I),
               "llama.cpp could not use the graphics card. Update the GPU driver, or run on the processor with 0 GPU layers.")
WARNINGS = [
    (re.compile(r"no usable GPU found", re.I),
     "llama.cpp found no usable graphics card, so the model runs on the processor only (slower)."),
]
# Settings the user's environment must not change behind the app's back (llama.cpp reads LLAMA_ARG_* as flags).
_ENV_BLOCKED = re.compile(r"^(LLAMA_ARG_|LLAMA_API_KEY$)", re.I)


def free_port():
    """A currently free loopback port. Another program could still take it before we bind, so callers retry once."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def port_is_free(port, host="127.0.0.1"):
    """True when nothing listens on host:port. SO_REUSEADDR on POSIX, like llama-server, so TIME_WAIT is not "busy"."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if os.name != "nt":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def server_env(config):
    """launch.server_env without the user's LLAMA_ARG_* / LLAMA_API_KEY (they would silently change flags the app
    leaves at llama.cpp defaults, or lock the app out). CORS is limited to localhost pages; builds that don't know
    the variable ignore it."""
    env = {k: v for k, v in launch.server_env(config).items() if not _ENV_BLOCKED.match(k)}
    env["LLAMA_ARG_CORS_ORIGINS"] = "localhost"
    return env


def failure_from_log(text, returncode=None):
    """Plain reason for a failed start, or None when the log gives no recognisable cause."""
    for pattern, message in FAILURES:
        match = pattern.search(text or "")
        if match:
            # File names only: a reason is shown to users and must not carry absolute paths.
            names = [re.split(r"[\\/]", group)[-1] if group else "unknown" for group in match.groups()]
            return message.format(*(names or ["unknown"]))
    if returncode is not None and returncode in (-9, 137):
        return "The system stopped the model server, most likely because memory ran out."
    if returncode in _CRASH_CODES:
        return CRASHED
    pattern, message = GPU_FAILURE
    return message if pattern.search(text or "") else None


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
        # "localhost" may resolve to ::1 first, where another program could answer; the server and our requests
        # both use the IPv4 loopback address instead.
        self.config["host"] = "127.0.0.1"
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
        if self.config["port"] and not port_is_free(self.config["port"], self.config["host"]):
            # Otherwise the program already on that port would answer /health and look like our model.
            raise ValueError(PORT_BUSY)
        for attempt in range(attempts):
            try:
                self._spawn(self.config["port"] or free_port())
                self._wait_ready(timeout, progress, cancel)
                return self
            except ValueError as error:
                port_taken = str(error).startswith(PORT_BUSY)
                self.stop()
                if not (port_taken and attempt + 1 < attempts):
                    raise
                self.process, self._stopped = None, False
            except BaseException:
                self.stop()
                raise

    def _spawn(self, port):
        if _SHUTTING_DOWN.is_set():  # the app is exiting: a server started now would outlive it
            raise Cancelled()
        self.port = port
        config = {**self.config, "port": port}
        args = self.command + launch.server_args(config)
        if self.log_path is None:
            handle, name = tempfile.mkstemp(prefix="llama-server-", suffix=".log")
            os.close(handle)
            self.log_path = Path(name)
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "wb")
        except OSError:
            raise ValueError("Could not write the llama-server log file. Check free disk space and folder "
                             "permissions.") from None
        options = {"stdin": subprocess.DEVNULL, "stdout": self._log, "stderr": subprocess.STDOUT,
                   "env": server_env(config)}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        try:
            self.process = subprocess.Popen(args, **options)
        except OSError as error:
            self._log.close()
            raise ValueError(f"Could not run llama-server ({error.strerror or error}). Check the llama.cpp install.") from None
        _kill_with_parent(self.process)
        self.started_at = now()
        with _LIVE_LOCK:
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
            if state == "ok" and self.process.poll() is None:
                self.load_seconds = round(elapsed, 3)
                if progress:
                    progress({"stage": "ready", "done": 1, "total": 1, "message": "The model is loaded and ready."})
                return
            if elapsed > timeout:
                raise ValueError(f"The model did not finish loading within {int(timeout)} seconds. "
                                 "It may be too large for this computer, or the disk may be slow.")
            if progress:
                # The load time is unknown in advance, so there is no total (no invented fraction).
                message = "Loading the model into memory…" if state == "loading" else "Starting llama-server…"
                progress({"stage": "loading", "done": round(elapsed, 1), "total": None, "elapsed_seconds": round(elapsed, 1),
                          "message": f"{message} ({int(elapsed)} s so far)" if elapsed >= 2 else message})
            time.sleep(0.1 if elapsed < 2 else 0.25)

    def _health(self):
        try:
            with _OPENER.open(f"{self.base_url}/health", timeout=2) as response:
                return "ok" if response.status == 200 else "loading"
        except urllib.error.HTTPError as error:
            error.close()
            return "loading" if error.code == 503 else "error"
        except (OSError, ValueError, http.client.HTTPException):
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
                answer = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ValueError(self._http_error(error)) from None
        except (socket.timeout, TimeoutError):
            raise ValueError(f"The model server took longer than {int(timeout)} seconds to answer.") from None
        except (OSError, urllib.error.URLError, http.client.HTTPException):
            raise ValueError(self._dead_message() if not self.running() else
                             "The model server is not responding. Try stopping and starting it again.") from None
        except ValueError:
            raise ValueError("The model server sent an unreadable answer.") from None
        if not isinstance(answer, dict):
            raise ValueError("The model server sent an unreadable answer.")
        return answer

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
             cache_prompt=False, enable_thinking=None, extra=None, chat_template_kwargs=None):
        """OpenAI-style chat. cache_prompt=False makes every call re-read the whole prompt, so timings are honest.
        enable_thinking=False asks reasoning models (for example Qwen3) to answer without a thinking section;
        chat_template_kwargs are passed to the model's chat template as they are (enable_thinking wins)."""
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "seed": seed,
                "cache_prompt": cache_prompt, "stream": False}
        if stop:
            body["stop"] = stop
        template_kwargs = dict(chat_template_kwargs or {})
        if enable_thinking is not None:
            template_kwargs["enable_thinking"] = bool(enable_thinking)
        if template_kwargs:
            body["chat_template_kwargs"] = template_kwargs
        body.update(extra or {})
        begin = time.monotonic()
        raw = self.request("/v1/chat/completions", body, timeout)
        choices = raw.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise ValueError("The model server sent an answer without any text.")
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
                code = self._stop_tree(timeout)
            else:
                code = self.process.returncode
            self._stopped = True
        elif self.process is not None:
            code = self.process.returncode
        if self._log and not self._log.closed:
            self._log.close()
        if self._own_log and self.log_path and self.log_path.exists():
            self._final_tail = self.log_tail(20000)
            for attempt in range(5):  # Windows: a status() reading the log at this moment blocks the delete
                try:
                    self.log_path.unlink()
                    break
                except FileNotFoundError:
                    break
                except OSError:
                    time.sleep(0.1)
        with _LIVE_LOCK:
            _LIVE.discard(self)
        return code

    def _stop_tree(self, timeout):
        """Terminate the children and our process, then force-kill what is left. Our own process is waited on with
        Popen (not psutil), so Popen keeps the real exit code and the PID is never reaped behind its back."""
        try:
            children = psutil.Process(self.process.pid).children(recursive=True)
        except psutil.Error:
            children = []
        for process in children:
            try:
                process.terminate()
            except psutil.Error:
                pass
        try:
            self.process.terminate()
        except OSError:
            pass
        try:
            code = self.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.process.kill()
            code = self.process.wait(timeout=5)
        _, alive = psutil.wait_procs(children, timeout=min(timeout, 5))
        for process in alive:
            try:
                process.kill()
            except psutil.Error:
                pass
        psutil.wait_procs(alive, timeout=5)
        return code

    def __enter__(self):
        if self.process is None:
            self.start()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False


def _kill_with_parent(process):
    """Windows: put the server in a Job Object that closes with this app, so ending the app in Task Manager or
    closing its console also ends llama-server. Best effort; POSIX relies on stop(), atexit and exit handlers."""
    global _JOB
    if os.name != "nt":
        return
    try:
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
        kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
        kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        if _JOB is None:
            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return

            class _Limits(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class _ExtendedLimits(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", _Limits), ("IoInfo", ctypes.c_uint64 * 6),
                            ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

            info = _ExtendedLimits()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
                kernel32.CloseHandle(job)
                return
            _JOB = job  # kept open for the app's lifetime; Windows closes it (and the servers) when the app ends
        kernel32.AssignProcessToJobObject(_JOB, int(process._handle))  # Popen's handle has PROCESS_ALL_ACCESS
    except Exception:
        pass


_JOB = None


def install_exit_handlers():
    """Turn SIGTERM/SIGHUP (and SIGBREAK on Windows) into a normal exit so atexit stops every server. Call once from
    the main thread of a program (the CLI's main and the app's serve); from another thread it does nothing."""
    import signal
    if threading.current_thread() is not threading.main_thread():
        return

    def handler(signum, _frame):
        signal.signal(signum, signal.SIG_IGN)  # a second signal must not interrupt the clean-up below
        sys.exit(128 + signum)

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None and signal.getsignal(number) in (signal.SIG_DFL, None):
            try:
                signal.signal(number, handler)
            except (OSError, ValueError):
                pass


@atexit.register
def _stop_all():
    _SHUTTING_DOWN.set()
    with _LIVE_LOCK:
        servers = list(_LIVE)
    for server in servers:
        try:
            server.stop(timeout=3)
        except BaseException:
            pass


def _vram_by_pid(timeout=5):
    """{pid: bytes, or None when nvidia-smi gives no number for that process ("[N/A]" on Windows WDDM,
    "[Insufficient Permissions]")}. None when nvidia-smi is missing or fails; {} after a time-out (try later)."""
    try:
        completed = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        return {}
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return None
    usage = {}
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        if not parts[1].isdigit() or usage.get(pid, 0) is None:
            usage[pid] = None
        else:
            usage[pid] = usage.get(pid, 0) + int(parts[1]) * 1024**2
    return usage


def _peak_rss(process):
    """The OS's own high-water mark of resident memory for one process (catches the model-load peak even when
    sampling started late), or None where the OS doesn't keep one."""
    try:
        if os.name == "nt":
            return getattr(process.memory_info(), "peak_wset", None)
        if sys.platform.startswith("linux"):
            with open(f"/proc/{process.pid}/status", encoding="ascii", errors="replace") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError, psutil.Error):
        pass
    return None


class PeakMemory:
    """Samples the process tree's RAM (RSS) and, for CUDA, its VRAM until stop(). Unified memory counts as RAM."""

    def __init__(self, pid, gpu_backend=None, interval=0.25):
        self.pid = pid
        self._parent = None
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
            if self._parent is None:  # kept, so a reused PID is never measured after our process ends
                self._parent = psutil.Process(self.pid)
            if not self._parent.is_running():
                return []
            return [self._parent] + self._parent.children(recursive=True)
        except psutil.Error:
            return []

    def sample(self):
        processes = self._tree()
        ram, high = 0, 0
        for process in processes:
            try:
                rss = process.memory_info().rss
            except psutil.Error:
                continue
            ram += rss
            high += max(rss, _peak_rss(process) or 0)
        if processes:
            self.samples += 1
            self.peak_ram = max(self.peak_ram, ram, high)
        # nvidia-smi is slow to start, so VRAM is sampled at most once a second.
        if self._vram_ok and processes and time.monotonic() - self._last_vram >= max(self.interval, 1.0):
            self._last_vram = time.monotonic()
            usage = _vram_by_pid()
            if usage is None:
                self._vram_ok = False
            else:
                pids = {p.pid for p in processes}
                ours = [value for pid, value in usage.items() if pid in pids]
                # Not listed (yet, or at all, e.g. inside a container) or no number: unknown, never 0.
                if ours and None not in ours:
                    self.peak_vram = max(self.peak_vram or 0, sum(ours))

    def _run(self):
        while not self._stop.is_set():
            self.sample()
            self._stop.wait(self.interval)

    def start(self):
        self._stop.clear()
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
            log_path = self.log_dir / "llama-server.log" if self.log_dir else None
            server = LlamaServer(command, config, log_path=log_path)  # validate before stopping the old one
            self._stop_current()
            with self._state:
                self._starting, self._error = server, None
            try:
                server.start(timeout=timeout, progress=progress, cancel=cancel)
            except Cancelled:
                with self._state:
                    self._starting = None
                raise
            except BaseException as error:  # never leave "starting" set, whatever went wrong
                message = str(error) if isinstance(error, ValueError) else "The model server could not be started."
                with self._state:
                    self._starting, self._error = None, message
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
