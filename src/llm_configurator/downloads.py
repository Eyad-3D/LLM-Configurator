"""Resumable, verified model downloads. Only runs when the user asks for it.

Promises: HTTPS only (redirects included), the token never leaves the Hub host, nothing larger than
the catalogue size is written, every file is SHA256-checked before it gets its final name, an existing
different file is never overwritten, and cancelling keeps the `.part` file so the next run resumes.
"""
import hashlib
import http.client
import os
from pathlib import Path
import re
import shutil
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .domain import GIB, Cancelled, check_cancel

DEFAULT_ENDPOINT = "https://huggingface.co"
CHUNK = 1024**2
TIMEOUT = 60
RETRIES = 4  # attempts in a row without any new bytes before giving up
BACKOFF = (2, 4, 8, 16)
PROGRESS_INTERVAL = 0.2  # at most five progress callbacks per second
_clock = time.monotonic
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)$")


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


class DownloadRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("Model downloads require HTTPS redirects")
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None and urlparse(req.full_url).netloc != urlparse(newurl).netloc:
            redirected.remove_header("Authorization")
        return redirected


def _open(request, timeout=TIMEOUT):
    """The one place that touches the network; tests replace it."""
    return build_opener(DownloadRedirect()).open(request, timeout=timeout)


def _sleep(seconds, cancel):
    if cancel is not None:
        cancel.wait(seconds)
    else:
        time.sleep(seconds)


def endpoint():
    value = (os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("HF_ENDPOINT must be an https:// address (for example https://huggingface.co)")
    return value


def file_url(variant, filename):
    return f"{endpoint()}/{quote(variant.repo)}/resolve/{quote(variant.revision, safe='')}/{quote(filename)}"


def _files(variant, directory):
    """[(entry, final path, part path)] with local names that stay inside `directory`."""
    directory = Path(directory)
    result, seen = [], set()
    for entry in variant.all_files():
        name = Path(entry["filename"]).name
        # ":" would name a Windows drive or alternate data stream; "x" and "x.part" would share a file.
        if name in {"", ".", ".."} or "\\" in name or ":" in name or name in seen or name + ".part" in seen:
            raise ValueError(f"Unsafe or duplicate model file name: {entry['filename']}")
        seen.update((name, name + ".part"))
        result.append((entry, directory / name, directory / (name + ".part")))
    return result


def _size(path, follow=False):
    """Size of a regular file; symlinks count only when `follow` (reading through one is safe, deleting is not)."""
    return path.stat().st_size if path.is_file() and (follow or not path.is_symlink()) else None


def _disk_free(directory):
    path = Path(directory).absolute()
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free


def plan(variant, directory):
    """Cheap overview for the UI/CLI: no hashing, no network. `present` means the right size is on disk."""
    files, total, remaining = [], 0, 0
    for entry, target, partial in _files(variant, directory):
        size = entry["size_bytes"]
        present = _size(target, follow=True) == size
        partial_bytes = 0 if present else min(_size(partial) or 0, size)
        files.append({"filename": target.name, "size_bytes": size, "present": present, "partial_bytes": partial_bytes})
        total += size
        remaining += 0 if present else size - partial_bytes
    free = _disk_free(directory)
    return {"files": files, "total_bytes": total, "remaining_bytes": remaining, "disk_free": free,
            "enough_space": free >= remaining + GIB}


class _Progress:
    """Totals across every shard, a smoothed speed and throttled callbacks."""

    def __init__(self, callback, total, count):
        self.callback, self.total, self.count = callback, total, count
        self.done = 0
        self.speed = None
        self.last_emit = None
        self.sample = (_clock(), 0)  # (time, bytes transferred this session)
        self.transferred = 0
        self.info = {}

    def file(self, index, name, size, file_done=0):
        self.info = {"file": name, "file_index": index, "file_count": self.count, "file_done": file_done, "file_total": size}

    def add(self, count):
        self.done += count
        self.info["file_done"] = self.info.get("file_done", 0) + count

    def received(self, count):
        self.add(count)
        self.transferred += count
        now = _clock()
        elapsed = now - self.sample[0]
        if elapsed >= 0.5:
            rate = (self.transferred - self.sample[1]) / elapsed
            self.speed = rate if self.speed is None else 0.3 * rate + 0.7 * self.speed
            self.sample = (now, self.transferred)
        self.emit("downloading")

    def emit(self, stage, message=None, force=False):
        if self.callback is None:
            return
        now = _clock()
        if not force and self.last_emit is not None and now - self.last_emit < PROGRESS_INTERVAL:
            return
        self.last_emit = now
        remaining = self.total - self.done
        speed = round(self.speed) if self.speed else None
        eta = round(remaining / self.speed) if self.speed and stage == "downloading" else None
        name = self.info.get("file", "")
        default = {"checking": f"Checking {name} already on disk", "downloading": f"Downloading {name}",
                   "verifying": f"Checking {name} is complete and undamaged", "done": "Download complete"}
        self.callback({"stage": stage, "done": self.done, "total": self.total, "message": message or default.get(stage, name),
                       "bytes_per_second": speed, "eta_seconds": eta, **self.info})


def _hash_existing(path, hasher, tracker, cancel):
    with path.open("rb") as stream:
        while chunk := stream.read(4 * CHUNK):
            check_cancel(cancel)
            hasher.update(chunk)
            tracker.add(len(chunk))
            tracker.emit("checking")


def _http_error(error, name):
    if error.code in (401, 403):
        return ValueError(f"Hugging Face refused access to {name} (HTTP {error.code}). If the model is gated, accept its "
                          "licence on huggingface.co and set HF_TOKEN to an access token, then try again.")
    if error.code == 404:
        return ValueError(f"{name} was not found at the pinned revision (HTTP 404). Refresh the catalogue and try again.")
    return ValueError(f"The server refused the download of {name} (HTTP {error.code}). Try again later.")


def _transient(error):
    if isinstance(error, HTTPError):
        return error.code == 429 or error.code >= 500
    return isinstance(error, (OSError, http.client.HTTPException))


def _fetch(url, entry, target, partial, token, tracker, cancel):
    size, expected = entry["size_bytes"], entry["sha256"]
    name = target.name
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    have = _size(partial) or 0
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise ValueError(f"{partial} is not a regular file; remove it and try again")
    if have > size:
        partial.unlink()
        have = 0
    hasher = hashlib.sha256()
    if have:
        tracker.emit("checking", f"Checking the {have} bytes of {name} downloaded earlier", force=True)
        _hash_existing(partial, hasher, tracker, cancel)
    failures = 0
    best = have  # most bytes ever on disk; re-fetching bytes we already had is not progress
    while have < size:
        check_cancel(cancel)
        request = Request(url, headers={**headers, **({"Range": f"bytes={have}-"} if have else {})})
        before = best
        try:
            with _open(request, timeout=TIMEOUT) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status == 206 and have:
                    match = _CONTENT_RANGE.match(response.headers.get("Content-Range", "").strip())
                    if not match or match.group(3) not in ("*", str(size)):
                        partial.unlink(missing_ok=True)
                        raise ValueError(f"The server reports a different size for {name} than the catalogue. "
                                         "Refresh the catalogue and try again.")
                    if int(match.group(1)) != have or int(match.group(2)) > size - 1:
                        raise _Restart()   # a shorter range (some CDNs cap them) is fine: the loop asks for the rest
                    mode = "ab"
                elif status == 200:
                    if have:  # the server ignored Range: start this file again from zero
                        tracker.add(-have)
                        have, hasher = 0, hashlib.sha256()
                    mode = "wb"
                else:
                    raise ValueError(f"Unexpected reply from the server for {name} (HTTP {status}). Try again later.")
                length = response.headers.get("Content-Length")
                if length and length.isdigit() and have + int(length) > size:
                    raise _TooLarge()
                output = _open_partial(partial, mode, name)
                try:
                    while chunk := response.read(CHUNK):
                        check_cancel(cancel)
                        if have + len(chunk) > size:
                            raise _TooLarge()
                        _save(output.write, chunk, name)
                        hasher.update(chunk)
                        have += len(chunk)
                        tracker.received(len(chunk))
                finally:
                    # The last chunk usually sits in the write buffer until close; a full disk shows up here.
                    _save(output.close, None, name)
            if have < size:
                raise ConnectionError("connection closed early")
        except (Cancelled, ValueError):
            raise
        except _TooLarge:
            partial.unlink(missing_ok=True)
            tracker.add(-have)
            raise ValueError(f"The server sent more data for {name} than the catalogue size, so the download was "
                             "stopped and the partial file deleted. Refresh the catalogue and try again.") from None
        except _Restart:
            partial.unlink(missing_ok=True)
            tracker.add(-have)
            have, hasher = 0, hashlib.sha256()
            failures += 1
        except HTTPError as error:
            if error.code == 416:  # the partial no longer lines up with the server's file
                partial.unlink(missing_ok=True)
                tracker.add(-have)
                have, hasher = 0, hashlib.sha256()
                failures += 1
            elif not _transient(error):
                raise _http_error(error, name) from None
            else:
                failures = 0 if have > before else failures + 1
        except Exception as error:  # network trouble: resume from what is on disk
            if not _transient(error):
                raise
            failures = 0 if have > before else failures + 1
        best = max(best, have)
        if have < size:
            if failures >= RETRIES:
                raise ValueError(f"The network kept failing while downloading {name}. Check your connection and run "
                                 "the download again; it will continue where it stopped.")
            if failures:
                tracker.emit("retrying", f"Connection problem; trying {name} again", force=True)
                _sleep(BACKOFF[min(failures, len(BACKOFF)) - 1], cancel)
    tracker.emit("verifying", force=True)
    if _size(partial) != size:  # the hash covers what we wrote; this checks it all reached the disk
        raise ValueError(f"{name} was not saved completely. Free up disk space and run the download again; it will "
                         "resume where it stopped.")
    if hasher.hexdigest() != expected:
        partial.unlink(missing_ok=True)
        tracker.add(-size)
        raise ValueError(f"{name} did not match its published SHA256 (it was damaged or changed), so the partial file "
                         "was deleted. Run the download again.")
    os.replace(partial, target)


def _open_partial(partial, mode, name):
    try:
        return partial.open(mode)
    except OSError as error:
        raise ValueError(f"Could not save {name}: {error.strerror or error}. Check the folder is writable, or choose "
                         "another folder.") from None


def _save(action, chunk, name):
    try:
        return action(chunk) if chunk is not None else action()
    except OSError as error:
        raise ValueError(f"Could not save {name}: {error.strerror or error}. Free up disk space or choose another "
                         "folder; the download will resume where it stopped.") from None


class _TooLarge(Exception):
    pass


class _Restart(Exception):
    pass


def download_variant(variant, directory, progress=None, cancel=None, token=None):
    """Download, resume and verify every file of `variant`; returns the first file (what llama.cpp loads)."""
    if variant.demo or not variant.sha256:
        raise ValueError("A real model with a published SHA256 is required")
    files = _files(variant, directory)
    if any(not entry.get("sha256") for entry, _, _ in files):
        raise ValueError("Every file of this model needs a published SHA256 before it can be downloaded")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    token = token or os.environ.get("HF_TOKEN") or None
    tracker = _Progress(progress, sum(entry["size_bytes"] for entry, _, _ in files), len(files))
    existing = []
    for index, (entry, target, _) in enumerate(files):
        if target.is_symlink() or target.exists():
            if _size(target, follow=True) != entry["size_bytes"]:
                raise ValueError(f"{target.name} already exists in {directory} with a different size; "
                                 "choose a different folder or move that file")
            existing.append(index)
    summary = plan(variant, directory)
    needed = summary["remaining_bytes"]
    if not summary["enough_space"]:
        raise ValueError(f"Not enough free disk space: this download needs {needed / GIB:.1f} GiB plus 1 GiB spare, "
                         f"but only {summary['disk_free'] / GIB:.1f} GiB is free. Free up space or choose another folder.")
    for index, (entry, target, partial) in enumerate(files):
        tracker.file(index, target.name, entry["size_bytes"])
        if index in existing:
            tracker.emit("checking", force=True)
            hasher = hashlib.sha256()
            _hash_existing(target, hasher, tracker, cancel)
            if hasher.hexdigest() != entry["sha256"]:
                raise ValueError(f"{target.name} already exists in {directory} with different contents; "
                                 "choose a different folder or move that file")
            continue
        _fetch(file_url(variant, entry["filename"]), entry, target, partial, token, tracker, cancel)
    tracker.emit("done", force=True)
    return files[0][1]


def remove_variant(variant, directory):
    """Delete this variant's finished and partial files in `directory`; returns bytes freed.

    Only regular files with this variant's names are touched, and a finished file is removed only when its size
    matches, so a different model that happens to share a name is left alone. Symlinks are never followed.
    """
    freed = 0
    for entry, target, partial in _files(variant, directory):
        size = entry["size_bytes"]
        for path, keep in ((target, lambda s: s == size), (partial, lambda s: s <= size)):
            actual = _size(path)
            if actual is not None and keep(actual):
                path.unlink()
                freed += actual
    return freed
