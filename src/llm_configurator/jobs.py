"""Background jobs with progress and cancellation for downloads, tests, tuning and serving.

Heavy jobs share an exclusive key (for example "compute") so two model loads never
compete for the same memory. Job records hold no secrets and are kept in memory only.
"""
import itertools
import threading
import traceback

from .domain import Cancelled, now

STATES = {"queued", "running", "done", "failed", "cancelled"}


class JobManager:
    def __init__(self, keep=50):
        self.keep = keep
        self._jobs = {}
        self._cancel = {}
        self._locks = {}
        self._guard = threading.Lock()
        self._ids = itertools.count(1)

    def submit(self, kind, title, fn, subject=None, exclusive=None):
        """Run fn(progress, cancel) in a thread. `exclusive` jobs with the same key run one at a time."""
        with self._guard:
            job_id = f"job-{next(self._ids)}"
            job = {"id": job_id, "kind": kind, "title": title, "state": "queued", "progress": None,
                   "result": None, "error": None, "created_at": now(), "finished_at": None,
                   "subject": dict(subject or {}), "exclusive": exclusive}
            self._jobs[job_id] = job
            self._cancel[job_id] = threading.Event()
            lock = self._locks.setdefault(exclusive, threading.Lock()) if exclusive else None
            self._prune()
        threading.Thread(target=self._run, args=(job_id, fn, lock), daemon=True, name=job_id).start()
        return self.get(job_id)

    def _run(self, job_id, fn, lock):
        cancel = self._cancel[job_id]

        def progress(value):
            with self._guard:
                self._jobs[job_id]["progress"] = dict(value)

        if lock:
            # Poll so a queued job can still be cancelled before it starts.
            while not lock.acquire(timeout=0.2):
                if cancel.is_set():
                    return self._finish(job_id, "cancelled", error="Cancelled before starting")
        try:
            if cancel.is_set():
                return self._finish(job_id, "cancelled", error="Cancelled before starting")
            with self._guard:
                self._jobs[job_id]["state"] = "running"
            result = fn(progress, cancel)
            self._finish(job_id, "cancelled" if cancel.is_set() and result is None else "done", result=result)
        except Cancelled:
            self._finish(job_id, "cancelled", error="Cancelled by user")
        except Exception as error:  # Every failure becomes a readable job error, never a dead thread.
            message = str(error) or type(error).__name__
            if not isinstance(error, (ValueError, OSError)):
                message = f"{type(error).__name__}: {message}"
                traceback.print_exc()
            self._finish(job_id, "failed", error=message)
        finally:
            if lock:
                lock.release()

    def _finish(self, job_id, state, result=None, error=None):
        with self._guard:
            self._jobs[job_id].update(state=state, result=result, error=error, finished_at=now())

    def _prune(self):
        finished = [j for j in self._jobs.values() if j["state"] in {"done", "failed", "cancelled"}]
        for job in sorted(finished, key=lambda j: j["created_at"])[:max(0, len(self._jobs) - self.keep)]:
            self._jobs.pop(job["id"], None)
            self._cancel.pop(job["id"], None)

    def get(self, job_id):
        with self._guard:
            job = self._jobs.get(job_id)
            return None if job is None else {**job, "progress": dict(job["progress"]) if job["progress"] else None}

    def list(self):
        with self._guard:
            ids = list(self._jobs)
        return [job for job in (self.get(i) for i in reversed(ids)) if job]

    def active(self, kind=None, subject_key=None, subject_value=None):
        """Running or queued jobs, optionally filtered, so callers can avoid duplicate work."""
        return [j for j in self.list() if j["state"] in {"queued", "running"} and (kind is None or j["kind"] == kind)
                and (subject_key is None or j["subject"].get(subject_key) == subject_value)]

    def cancel(self, job_id):
        event = self._cancel.get(job_id)
        if event is None:
            raise ValueError("Unknown job")
        event.set()
        return self.get(job_id)

    def wait(self, job_id, timeout=None):
        """Block until a job finishes (CLI and tests)."""
        thread = next((t for t in threading.enumerate() if t.name == job_id), None)
        if thread:
            thread.join(timeout)
        return self.get(job_id)
