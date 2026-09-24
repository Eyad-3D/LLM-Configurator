"""Find or install llama.cpp. Downloads only official ggml-org release assets over HTTPS,
refuses anything without a published SHA256 (unless the user opts out on the CLI), never
extracts outside the target folder and never touches PATH or system folders."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile
import zlib

from .domain import GIB, check_cancel, now
from .storage import runtime_dir

RELEASE_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
DOWNLOAD_PREFIX = "https://github.com/ggml-org/llama.cpp/releases/download/"
BINARIES = ("llama-server", "llama-bench", "llama-perplexity", "llama-cli")
NOT_INSTALLED = "llama.cpp is not installed yet. Click Install runtime or run: llm-config runtime install"
MAX_ASSET_BYTES = 4 * GIB          # the biggest real asset (CUDA runtime) is ~0.6 GiB
MAX_EXTRACTED_BYTES = 8 * GIB      # stops "zip bombs" (tiny archives that unpack to huge sizes)
MAX_MEMBERS = 20000
VERSION_TIMEOUT = 15
CHUNK = 1024**2
PROGRESS_INTERVAL = 0.2            # at most five progress updates a second, like model downloads
_clock = time.monotonic
HOMEBREW_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/home/linuxbrew/.linuxbrew/bin", "~/.linuxbrew/bin")

# Oldest NVIDIA driver that fully supports each CUDA toolkit (Linux, Windows). Source: NVIDIA CUDA
# Toolkit release notes, "CUDA Toolkit and Corresponding Driver Versions" table (GA releases).
CUDA_DRIVERS = {(12, 0): ("525.60.13", "527.41"), (12, 1): ("530.30.02", "531.14"), (12, 2): ("535.54.03", "536.25"),
                (12, 3): ("545.23.06", "545.84"), (12, 4): ("550.54.14", "551.61"), (12, 5): ("555.42.02", "555.85"),
                (12, 6): ("560.28.03", "560.76"), (12, 8): ("570.26", "570.65"), (12, 9): ("575.51.03", "576.02"),
                (13, 0): ("580.65.06", "580.88")}
# CUDA "minor version compatibility": any 12.x/13.x build runs on a driver of that major's first release.
CUDA_MAJOR_DRIVERS = {12: ("525.60.13", "528.33"), 13: ("580.65.06", "580.88")}

_OS = {"win": "windows", "windows": "windows", "macos": "macos", "osx": "macos", "darwin": "macos",
       "ubuntu": "linux", "linux": "linux"}
_ARCH = {"x64": "x64", "x86_64": "x64", "amd64": "x64", "arm64": "arm64", "aarch64": "arm64"}
_BACKENDS = {"cpu": "cpu", "cuda": "cuda", "vulkan": "vulkan", "metal": "metal", "hip": "rocm", "rocm": "rocm"}
# CPU instruction-set flavours used by older Windows builds; harmless for our choice.
_CPU_FLAVOURS = {"avx", "avx2", "avx512", "noavx", "openblas"}
_EXTENSIONS = (".tar.gz", ".tgz", ".zip")


# ---------- choosing a release asset ----------

def _version(text):
    return tuple(int(part) for part in re.findall(r"\d+", str(text or "")))


def classify(name):
    """Read an asset file name like llama-b6000-bin-win-cuda-12.4-x64.zip. Unknown flavours
    (SYCL, OpenVINO, Snapdragon, ...) come back with backend None so they are never picked."""
    lower = name.lower()
    ext = next((e for e in _EXTENSIONS if lower.endswith(e)), None)
    match = re.match(r"^(cudart-)?llama-(?:(b\d+)-)?bin-(.+)$", lower[:-len(ext)] if ext else "")
    if not match:
        return None
    rest = re.sub(r"(?:cuda-)?cu-?(\d+(?:\.\d+)+)", r"cuda-\1", match.group(3))
    info = {"name": name, "companion": bool(match.group(1)), "build": int(match.group(2)[1:]) if match.group(2) else None,
            "os": None, "arch": None, "backend": None, "cuda": None, "format": ext}
    extras = []
    for token in re.findall(r"\d+(?:\.\d+)+|[a-z0-9_]+", rest):
        if token in _OS and not info["os"]:
            info["os"] = _OS[token]
        elif token in _ARCH:
            info["arch"] = _ARCH[token]
        elif token in _BACKENDS:
            info["backend"] = _BACKENDS[token]
        elif re.fullmatch(r"\d+(?:\.\d+)+", token) and info["backend"] == "cuda":
            info["cuda"] = _version(token)
        elif token in _CPU_FLAVOURS:
            info["backend"] = info["backend"] or "cpu"
        else:
            extras.append(token)
    if extras or not info["os"] or not info["arch"] or (info["backend"] == "cuda" and not info["cuda"]):
        info["backend"] = None
    elif info["backend"] is None:
        info["backend"] = "metal" if (info["os"], info["arch"]) == ("macos", "arm64") else "cpu"
    return info


def _normal_system(system, machine):
    system = {"darwin": "macos"}.get(system.lower(), system.lower())
    return system, _ARCH.get(machine.lower(), machine.lower())


def _gpu_kinds(hardware):
    gpus = [g for g in (hardware or {}).get("gpus") or [] if isinstance(g, dict)]
    nvidia = [g for g in gpus if (g.get("vendor") or "").lower() == "nvidia" or g.get("backend") == "cuda"
              or "nvidia" in (g.get("name") or "").lower()]
    others = [g for g in gpus if g not in nvidia and g.get("backend") != "metal"]
    return nvidia, others


def max_cuda_for_driver(driver, system):
    """Highest CUDA toolkit this driver fully supports, or None when the driver is unknown or too old."""
    have = _version(driver)
    if not have:
        return None
    column = 1 if system == "windows" else 0
    known = [cuda for cuda, drivers in CUDA_DRIVERS.items() if have >= _version(drivers[column])]
    return max(known) if known else None


def _cuda_ok(cuda, driver, system):
    """True if a build made with CUDA `cuda` should run on this driver, with a plain explanation."""
    have, column = _version(driver), 1 if system == "windows" else 0
    exact = CUDA_DRIVERS.get(cuda[:2])
    if exact:
        return have >= _version(exact[column]), "fully supported by your driver"
    major = CUDA_MAJOR_DRIVERS.get(cuda[0])
    if major and have >= _version(major[column]):
        return True, f"runs through NVIDIA's CUDA {cuda[0]} compatibility rules"
    return False, ""


def _asset_record(asset, info, allow_unverified):
    url, digest = asset.get("browser_download_url") or "", asset.get("digest") or ""
    if not url.startswith(DOWNLOAD_PREFIX):
        raise ValueError(f"Release file {asset.get('name')} does not come from the official llama.cpp downloads page.")
    sha256 = digest[7:].lower() if digest.lower().startswith("sha256:") else None
    if sha256 and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        sha256 = None
    if not sha256 and not allow_unverified:
        raise ValueError(f"GitHub did not publish a checksum (a digital fingerprint) for {asset.get('name')}, so it "
                         "cannot be checked for damage or tampering. Try again later, install from a file with: "
                         "llm-config runtime install-archive PATH, or accept the risk with: "
                         "llm-config runtime install --allow-unverified")
    size = asset.get("size")
    if type(size) is not int or not 0 < size <= MAX_ASSET_BYTES:
        raise ValueError(f"Release file {asset.get('name')} has a missing or unexpected size; refusing to download it.")
    return {"name": asset["name"], "url": url, "size": size, "sha256": sha256}


def choose_asset(release, hardware, system=None, machine=None, allow_unverified=False, backend=None):
    """Pick the official build that suits this computer. `backend` forces one kind (used to fall
    back to the CPU build when a GPU build cannot start)."""
    system, machine = _normal_system(system or platform.system(), machine or platform.machine())
    assets = {a.get("name"): a for a in (release or {}).get("assets") or [] if isinstance(a, dict) and a.get("name")}
    infos = [i for i in map(classify, assets) if i and i["backend"]]
    mine = [i for i in infos if i["os"] == system and i["arch"] == machine and not i["companion"]]
    if not mine:
        raise ValueError(f"This llama.cpp release has no ready-made build for {system} on {machine}. Build llama.cpp "
                         "yourself and run: llm-config runtime use DIR")
    preferred = ".zip" if system == "windows" else ".tar.gz"
    by_backend = {}
    for info in sorted(mine, key=lambda i: (i["cuda"] or (), i["format"] == preferred), reverse=True):
        by_backend.setdefault(info["backend"], []).append(info)
    nvidia, others = _gpu_kinds(hardware)
    notes, order = [], []
    if system == "macos":
        order = ["metal", "cpu"] if machine == "arm64" else ["cpu", "metal"]
    elif nvidia:
        order = ["cuda", "vulkan", "cpu"]
    elif others:
        order = ["vulkan", "cpu"]
    else:
        order = ["cpu", "vulkan"]
    if backend:
        order = [backend]
    for kind in order:
        for info in by_backend.get(kind, []):
            companions = []
            if kind == "cuda":
                driver = next((g.get("driver") for g in nvidia if g.get("driver")), None)
                if not driver:
                    notes.append("your NVIDIA driver version is unknown, so the CUDA build was skipped")
                    break
                ok, why = _cuda_ok(info["cuda"], driver, system)
                if not ok:
                    notes.append(f"CUDA {'.'.join(map(str, info['cuda']))} needs a newer NVIDIA driver than {driver}")
                    continue
                companion = next((c for c in infos if c["companion"] and c["os"] == system and c["arch"] == machine
                                  and c["cuda"][:2] == info["cuda"][:2] and c["format"] == info["format"]), None)
                if not companion:
                    notes.append(f"the CUDA {'.'.join(map(str, info['cuda']))} runtime files are missing from this release")
                    continue
                companions.append(_asset_record(assets[companion["name"]], companion, allow_unverified))
            record = _asset_record(assets[info["name"]], info, allow_unverified)
            record.update(backend=kind, companions=companions, tag=(release or {}).get("tag_name"),
                          build=info["build"], cuda=".".join(map(str, info["cuda"])) if info["cuda"] else None,
                          reason=_reason(kind, system, machine, info, nvidia, others, notes,
                                         why if kind == "cuda" else ""))
            record["unverified"] = record["sha256"] is None or any(c["sha256"] is None for c in companions)
            return record
    raise ValueError(f"This llama.cpp release has no suitable build for {system} on {machine}"
                     + (f" ({'; '.join(notes)})" if notes else "") + ". Update your graphics driver, or build llama.cpp "
                     "yourself and run: llm-config runtime use DIR")


def _reason(kind, system, machine, info, nvidia, others, notes, cuda_why):
    where = {"windows": "Windows", "macos": "macOS", "linux": "Linux"}.get(system, system)
    chip = "Apple Silicon" if machine == "arm64" and system == "macos" else machine
    if kind == "cuda":
        text = (f"Your {nvidia[0].get('name') or 'NVIDIA graphics card'} can use CUDA, NVIDIA's own fast route to the "
                f"graphics card. Picked the newest CUDA build ({'.'.join(map(str, info['cuda']))}) that {cuda_why}.")
    elif kind == "metal":
        text = f"Picked the {where} {chip} build, which uses Metal to run models on the Mac's built-in graphics chip."
    elif kind == "vulkan":
        gpu = (nvidia or others or [{}])[0].get("name") or "graphics card"
        text = (f"Picked the Vulkan build: Vulkan is a common graphics language most graphics cards speak, "
                f"so your {gpu} can help run models.")
    else:
        text = (f"Picked the plain {where} {machine} build that runs on the main processor (CPU)"
                + (", because no usable graphics card was found." if not (nvidia or others) else "."))
    return text + (f" Note: {'; '.join(notes)}." if notes else "")


# ---------- network ----------

class _HttpsOnly(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urlparse(newurl).scheme != "https":
            raise ValueError("The llama.cpp download tried to switch to an insecure (non-HTTPS) address; stopped.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open(request, timeout=60):
    """The only network entry point; tests replace it."""
    return build_opener(_HttpsOnly()).open(request, timeout=timeout)


def _get_json(url, limit=16 * 1024**2):
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "llm-configurator"})
    try:
        with _open(request, timeout=30) as response:
            data = response.read(limit + 1)
    except (HTTPError, URLError, OSError) as error:
        raise ValueError(f"Could not reach GitHub to look up llama.cpp releases ({error}). Check your internet "
                         "connection, or install from a downloaded file with: llm-config runtime install-archive PATH") from None
    if len(data) > limit:
        raise ValueError("GitHub's release list was unexpectedly large; stopped reading it.")
    try:
        return json.loads(data)
    except ValueError:
        raise ValueError("GitHub sent a release list that could not be read. Try again in a few minutes.") from None


def _has_builds(release):
    return any((classify(a.get("name") or "") or {}).get("backend") for a in release.get("assets") or [])


def fetch_release():
    """The newest release that actually carries ready-made builds. In 2026 `releases/latest` can point
    at a version tag (e.g. v0.5.0) with no binaries while builds ship as b<N> prereleases."""
    latest = _get_json(RELEASE_API + "/latest")
    if isinstance(latest, dict) and _has_builds(latest):
        return latest
    releases = _get_json(RELEASE_API + "?per_page=10")
    for release in releases if isinstance(releases, list) else []:
        if isinstance(release, dict) and not release.get("draft") and _has_builds(release):
            return release
    raise ValueError("Could not find a llama.cpp release with ready-made builds on GitHub. Try again later.")


def _download(record, folder, progress=None, cancel=None, done_before=0, grand_total=None):
    """Stream to <folder>/<sha>-<name>.part, hashing as we go; resume with an HTTP Range request.
    Keeps the partial file on cancel so the next attempt continues where it stopped."""
    folder.mkdir(parents=True, exist_ok=True)
    size, expected = record["size"], record["sha256"]
    key = expected[:16] if expected else "unverified"
    final = folder / f"{key}-{Path(record['name']).name}"
    part = final.with_name(final.name + ".part")
    if not expected:
        part.unlink(missing_ok=True)
        final.unlink(missing_ok=True)     # nothing to check a leftover against: always fetch it fresh
    if final.exists() and final.stat().st_size == size and (not expected or _sha256(final) == expected):
        return final
    if shutil.disk_usage(folder).free < size - (part.stat().st_size if part.exists() else 0) + GIB:
        raise ValueError(f"Not enough free disk space for {record['name']} plus 1 GiB spare. Free some space and try again.")
    for attempt in range(2):
        have = part.stat().st_size if part.exists() else 0
        if have > size:
            part.unlink()
            have = 0
        hasher = hashlib.sha256()
        if have:
            with part.open("rb") as stream:
                for chunk in iter(lambda: stream.read(CHUNK), b""):
                    check_cancel(cancel)
                    hasher.update(chunk)
        if have == size:
            break              # already complete (e.g. cancelled while verifying); checked below, no network needed
        headers = {"User-Agent": "llm-configurator", "Accept": "application/octet-stream"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            response = _open(Request(record["url"], headers=headers), timeout=60)
        except HTTPError as error:
            if error.code == 416 and have == size:
                break          # already complete; verified below
            if error.code == 416 and attempt == 0:
                part.unlink(missing_ok=True)
                continue
            raise ValueError(f"The download of {record['name']} failed (HTTP {error.code}). Try again later.") from None
        except (URLError, OSError) as error:
            raise ValueError(f"The download of {record['name']} failed ({error}). Check your connection and try "
                             "again; it will continue where it stopped.") from None
        with response:
            status = getattr(response, "status", None) or response.getcode()
            content_range = (getattr(response, "headers", None) or {}).get("Content-Range") or f"bytes {have}-"
            if have and status == 206 and not content_range.startswith(f"bytes {have}-"):
                part.unlink(missing_ok=True)      # server resumed from the wrong place: start over
                if attempt == 0:
                    continue
                raise ValueError(f"The download of {record['name']} could not be resumed. Try again.")
            if have and status != 206:            # server ignored Range: start again from zero
                have, hasher = 0, hashlib.sha256()
            try:
                _stream(response, part, have, size, hasher, record, progress, cancel, done_before, grand_total)
            except (OSError, HTTPException) as error:
                raise ValueError(f"The download of {record['name']} was interrupted ({error}). Try again; it will "
                                 "continue where it stopped.") from None
        break
    if part.stat().st_size != size:
        raise ValueError(f"The download of {record['name']} stopped early. Try again; it will continue where it stopped.")
    # Check what is on disk, not only what came over the wire (a failed flush or a second writer would differ).
    if expected and (hasher.hexdigest() != expected or _sha256(part) != expected):
        part.unlink(missing_ok=True)
        raise ValueError(f"{record['name']} failed its checksum check (it may be damaged or tampered with) and was "
                         "deleted. Try again.")
    os.replace(part, final)
    return final


def _stream(response, part, have, size, hasher, record, progress, cancel, done_before, grand_total):
    started, last, received = _clock(), None, 0
    with part.open("ab" if have else "wb") as output:
        written = have
        while True:
            check_cancel(cancel)
            chunk = response.read(CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > size:
                output.close()
                part.unlink(missing_ok=True)
                raise ValueError(f"{record['name']} is bigger than GitHub said it would be; stopped and deleted it.")
            output.write(chunk)
            hasher.update(chunk)
            received += len(chunk)
            moment = _clock()
            if progress and (last is None or moment - last >= PROGRESS_INTERVAL or written == size):
                last = moment
                elapsed = moment - started
                speed = round(received / elapsed) if elapsed >= 0.5 else None
                remaining = (grand_total or size) - (done_before + written)
                progress({"stage": "download", "done": done_before + written, "total": grand_total or size,
                          "message": f"Downloading {record['name']}", "bytes_per_second": speed,
                          "eta_seconds": round(remaining / speed) if speed else None})


def _sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            result.update(chunk)
    return result.hexdigest()


# ---------- safe extraction ----------

def _safe_parts(name):
    """Split an archive path; reject absolute paths, drive letters and `..` (path-traversal attacks)."""
    name = name.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[a-zA-Z]:", name):
        raise ValueError(f"The archive contains an unsafe absolute path ({name}); refusing to unpack it.")
    parts = [p for p in PurePosixPath(name).parts if p not in ("", ".")]
    if any(":" in p for p in parts):
        raise ValueError(f"The archive contains an unsafe drive-letter path ({name}); refusing to unpack it.")
    if ".." in parts:
        raise ValueError(f"The archive contains an unsafe path that climbs out of its folder ({name}); refusing to unpack it.")
    return parts


def _within(root, path):
    try:
        Path(os.path.normpath(path)).relative_to(root)
        return True
    except ValueError:
        return False


def safe_extract(archive, target, cancel=None):
    """Unpack a .zip or .tar.gz into `target`. Symlinks are allowed only if they stay inside `target`;
    hard links only to files already unpacked; device files never. Every failure is a plain ValueError."""
    try:
        return _extract(archive, target, cancel)
    except (OSError, EOFError, zlib.error, zipfile.BadZipFile, zipfile.LargeZipFile, tarfile.TarError) as error:
        raise ValueError(f"The llama.cpp archive is damaged or unusual and could not be unpacked ({error}).") from None


def _extract(archive, target, cancel):
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    name = str(archive).lower()
    total = 0
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            if len(members) > MAX_MEMBERS:
                raise ValueError("The archive has far too many files; refusing to unpack it.")
            for member in members:
                check_cancel(cancel)
                parts = _safe_parts(member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise ValueError(f"The archive contains a shortcut link ({member.filename}); refusing to unpack it.")
                if not parts:
                    continue
                path = target.joinpath(*parts)
                if not _within(target, path.parent.resolve()):
                    raise ValueError(f"The archive contains an unsafe path ({member.filename}); refusing to unpack it.")
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                _clear(path, member.filename)
                total += member.file_size
                if total > MAX_EXTRACTED_BYTES:
                    raise ValueError("The archive unpacks to an unreasonable size; refusing to unpack it.")
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
                if os.name != "nt" and mode & 0o111:
                    path.chmod(0o755)
        return target
    if not name.endswith((".tar.gz", ".tgz", ".tar")):
        raise ValueError("Only .zip and .tar.gz llama.cpp archives are supported.")
    with tarfile.open(archive, "r:*") as bundle:
        count = 0
        for member in bundle:
            check_cancel(cancel)
            count += 1
            if count > MAX_MEMBERS:
                raise ValueError("The archive has far too many files; refusing to unpack it.")
            parts = _safe_parts(member.name)
            if not parts:
                continue
            path = target.joinpath(*parts)
            if not _within(target, path.parent.resolve()):
                raise ValueError(f"The archive tries to write through a link ({member.name}); refusing to unpack it.")
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            elif member.issym():
                link = member.linkname.replace("\\", "/")
                if (link.startswith("/") or re.match(r"^[a-zA-Z]:", link) or not _within(target, path.parent / link)
                        or not _within(target, (path.parent.resolve() / link).resolve())):
                    raise ValueError(f"The archive contains a link pointing outside its folder ({member.name}); refusing to unpack it.")
                path.parent.mkdir(parents=True, exist_ok=True)
                _clear(path, member.name)
                try:
                    os.symlink(link, path)
                except OSError:
                    raise ValueError("This computer does not allow creating the links inside this archive. "
                                     "Use the .zip build instead.") from None
            elif member.islnk():
                source = target.joinpath(*_safe_parts(member.linkname))
                if not source.is_file() or source.is_symlink() or not _within(target, source.resolve()):
                    raise ValueError(f"The archive contains a bad hard link ({member.name}); refusing to unpack it.")
                path.parent.mkdir(parents=True, exist_ok=True)
                _clear(path, member.name)
                shutil.copyfile(source, path)
            elif member.isfile():
                total += member.size
                if total > MAX_EXTRACTED_BYTES:
                    raise ValueError("The archive unpacks to an unreasonable size; refusing to unpack it.")
                _clear(path, member.name)
                path.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, path.open("wb") as output:
                    shutil.copyfileobj(source, output, CHUNK)
                path.chmod(0o755 if member.mode & 0o111 else 0o644)
            else:
                raise ValueError(f"The archive contains a special device file ({member.name}); refusing to unpack it.")
    # Final guard: every link must still point inside, whatever order the archive used.
    for folder, dirs, files in os.walk(target):
        for entry in dirs + files:
            path = Path(folder) / entry
            if path.is_symlink() and not _within(target, path.resolve()):
                raise ValueError(f"The archive contains a link pointing outside its folder ({entry}); refusing to unpack it.")
    return target


def _clear(path, name):
    """Never write through an existing link or over a folder."""
    if path.is_symlink():
        path.unlink()
    elif path.is_dir():
        raise ValueError(f"The archive lists {name} twice (as a folder and a file); refusing to unpack it.")


# ---------- finding and checking binaries ----------

def _exe(name):
    return name + ".exe" if os.name == "nt" else name


def find_bin_dir(directory, depth=4):
    """The folder holding llama-server: the folder itself, bin/, build/bin/, or the shallowest match below."""
    root = Path(directory).expanduser()
    for sub in ("", "bin", "build/bin", "build/bin/Release"):
        if (root / sub / _exe("llama-server")).is_file():
            return root / sub if sub else root
    if not root.is_dir():
        return None
    level = [root]
    for _ in range(depth):
        nxt = []
        for folder in level:
            try:
                children = sorted(p for p in folder.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                continue
            for child in children:
                if (child / _exe("llama-server")).is_file():
                    return child
                nxt.append(child)
        level = nxt
    return None


def _run_tool(argv, flag, env=None):
    """Run a llama.cpp tool with one info flag; returns (ok, stdout + stderr). llama.cpp prints
    `--version` on stderr and `--list-devices` on stdout."""
    try:
        done = subprocess.run(list(argv) + [flag], capture_output=True, text=True, timeout=VERSION_TIMEOUT,
                              errors="replace", env=env, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except subprocess.TimeoutExpired:
        return False, f"llama-server did not answer {flag} in time"
    except OSError as error:
        return False, str(error)
    return done.returncode == 0, (getattr(done, "stdout", "") or "") + "\n" + (getattr(done, "stderr", "") or "")


def _run_version(argv):
    """Run `llama-server --version`; returns (ok, combined output)."""
    return _run_tool(argv, "--version")


def _run_devices(argv, env=None):
    """Run `llama-server --list-devices` (also works for llama-bench); returns (ok, combined output)."""
    return _run_tool(argv, "--list-devices", env)


# Current builds: `version: 0.1.0-dev (build 1234, commit abc1234)`; older ones: `version: 1234 (abc1234)`.
_NEW_VERSION = re.compile(r"^\s*version:\s*(\S+)\s+\(build\s+(\d+),\s*commit\s+([0-9a-fA-F]+)\)", re.M)
_OLD_VERSION = re.compile(r"^\s*version:\s*(\S+)(?:\s*\(([0-9a-fA-F]+)\))?", re.M)


def parse_version(text):
    """Read the `--version` output plus backend hints from load/init lines. `version` is `b<build>`
    whenever the build number is known: current builds print a constant `0.1.0-dev` semver, so the
    build number is the only version that tells builds apart (the raw semver is kept in `semver`)."""
    text = text or ""
    result = {"version": None, "build": None, "commit": None, "backend": None}
    new = _NEW_VERSION.search(text)
    old = None if new else _OLD_VERSION.search(text)
    if new:
        result.update(build=int(new.group(2)), version=f"b{int(new.group(2))}", commit=new.group(3), semver=new.group(1))
    elif old:
        raw = old.group(1)
        result["commit"] = old.group(2)
        if raw.isdigit():
            result["build"], result["version"] = int(raw), f"b{raw}"
        else:
            result["version"] = raw
            build = re.fullmatch(r"b?(\d+)", raw) or re.search(r"\bb(\d{3,})\b", text)
            result["build"] = int(build.group(1)) if build else None
    lower = text.lower()
    for backend, hints in [("cuda", ("ggml_cuda_init", "loaded cuda backend", "cuda devices")),
                           ("rocm", ("ggml_hip", "loaded rocm backend", "loaded hip backend", "rocm devices")),
                           ("vulkan", ("ggml_vulkan", "loaded vulkan backend", "vulkan devices")),
                           ("metal", ("ggml_metal", "loaded metal backend", "mtl"))]:
        if any(h in lower for h in hints):
            result["backend"] = backend
            break
    else:
        if "loaded cpu backend" in lower:
            result["backend"] = "cpu"
        elif "apple-darwin" in lower and "arm64" in lower:
            result["backend"] = "metal"     # Homebrew/Apple builds embed Metal and print no load line
    return result


# ggml device-name prefixes (`--list-devices`) -> our backend names. CUDA builds compiled for AMD
# (HIP) name their devices ROCm0...; Metal is `Metal` on older builds and `MTL0` on newer ones.
DEVICE_BACKENDS = (("cuda", "CUDA"), ("rocm", "ROCm"), ("rocm", "HIP"), ("vulkan", "Vulkan"), ("metal", "Metal"),
                   ("metal", "MTL"))
# Accelerators that run on the main processor; they are not graphics cards.
_CPU_DEVICES = ("BLAS", "CPU", "AMX", "KLEIDIAI", "ACCELERATE", "RPC")
_DEVICE_LINE = re.compile(r"^\s+([A-Za-z][\w.-]*):\s*(.*?)(?:\s+\((\d+) MiB, (\d+) MiB free\))?\s*$")


def device_backend(name):
    """The backend of a ggml device name (`CUDA0` -> cuda), "cpu" for CPU-side accelerators, else "unknown"."""
    for backend, prefix in DEVICE_BACKENDS:
        if re.fullmatch(re.escape(prefix) + r"\d*", name or ""):
            return backend
    return "cpu" if (name or "").upper().startswith(_CPU_DEVICES) else "unknown"


def parse_devices(text):
    """Read `--list-devices` output: `Available devices:` then `  CUDA0: NVIDIA ... (24080 MiB, 23000 MiB free)`
    lines, or `  (none)`. Returns None when the text has no device list at all (old build or failure)."""
    lines = (text or "").splitlines()
    start = next((i for i, line in enumerate(lines) if line.strip().lower().startswith("available devices")), None)
    if start is None:
        return None
    devices = []
    for line in lines[start + 1:]:
        match = _DEVICE_LINE.match(line)
        if not match:
            if line.strip() and not line.startswith((" ", "\t")):
                break           # the list is over
            continue
        name, description, total, free = match.groups()
        devices.append({"name": name, "description": description, "backend": device_backend(name),
                        "total": int(total) * 1024**2 if total else None, "free": int(free) * 1024**2 if free else None})
    return devices


def list_devices(argv, env=None):
    """GPU devices a llama.cpp tool can offload to (CPU-side accelerators left out), or None if it could not say.
    `env` matters: CUDA_VISIBLE_DEVICES and friends change what is listed."""
    ok, output = _run_devices(argv, env) if env is not None else _run_devices(argv)
    devices = parse_devices(output) if ok else None
    return None if devices is None else [d for d in devices if d["backend"] != "cpu"]


def _backend_from_files(folder):
    names = " ".join(p.name.lower() for p in folder.iterdir()) if folder.is_dir() else ""
    for backend, marker in [("cuda", "ggml-cuda"), ("rocm", "ggml-hip"), ("vulkan", "ggml-vulkan"), ("metal", "ggml-metal")]:
        if marker in names:
            return backend
    return None


def _quarantined(path):
    """macOS marks files downloaded by browsers with com.apple.quarantine; Gatekeeper may then block them."""
    if platform.system() != "Darwin" or not shutil.which("xattr"):
        return False
    try:
        done = subprocess.run(["xattr", "-p", "com.apple.quarantine", str(path)], capture_output=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return False
    return done.returncode == 0


def _inspect(folder, source, manifest=None):
    folder = Path(folder)
    server = folder / _exe("llama-server")
    result = empty_result()
    result.update(source=source, directory=str(folder))
    binaries = {}
    for name in BINARIES:
        path = folder / _exe(name)
        found = str(path) if path.is_file() else (shutil.which(name) if source == "path" else None)
        binaries[name] = [found] if found else None
    result["binaries"] = binaries
    ok, output = _run_version([str(server)])
    info = parse_version(output)
    if not ok or not info["version"]:
        result["warnings"].append(f"Found llama-server in {folder}, but it would not start or report its version. "
                                  "It may be damaged or need extra system files. Reinstall with: llm-config runtime install")
        result["installed"] = False
        return result
    manifest = manifest or {}
    result.update(installed=True, version=info["version"], build=info["build"] or manifest.get("build"),
                  commit=info["commit"], tag=manifest.get("tag"))
    devices = list_devices([str(server)])
    result["devices"] = [{k: d[k] for k in ("name", "description", "backend")} for d in devices or []]
    installed_as = manifest.get("backend")
    if devices:
        # What the build can actually use right now beats what it was installed as.
        result["backend"] = devices[0]["backend"]
    elif devices is not None:
        result["backend"] = "cpu"
        if installed_as in ("cuda", "rocm", "vulkan", "metal"):
            result["warnings"].append(f"This {_BACKEND_WORDS.get(installed_as, installed_as)} build of llama.cpp found no "
                                      "usable graphics card, so it will run on the CPU only. Updating your graphics "
                                      "driver may fix this.")
    else:
        result["backend"] = installed_as or info["backend"] or _backend_from_files(folder) or "unknown"
    if _quarantined(server):
        result["warnings"].append("macOS has marked this llama.cpp as downloaded from the internet, so it may refuse "
                                  f"to run it. If it does, run: xattr -dr com.apple.quarantine \"{folder}\"")
    return result


def _plain(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def pick_device(devices, gpu):
    """The ggml device name (`CUDA0`, `Vulkan1`, `MTL0`, ...) that is the hardware-scan GPU `gpu`, chosen
    from a `list_devices` result. None when it cannot be told apart safely: never guess a name, because
    llama.cpp exits on an unknown one (`invalid device: CUDA0`)."""
    devices = [d for d in devices or [] if d.get("backend") != "cpu"]
    if not devices or not gpu:
        return None
    same = [d for d in devices if d["backend"] == gpu.get("backend")] or devices
    if len(same) == 1:
        return same[0]["name"]
    name = _plain(gpu.get("name"))
    named = [d for d in same if name and (name in _plain(d["description"]) or _plain(d["description"]) in name)]
    return named[0]["name"] if len(named) == 1 else None


_BACKEND_WORDS = {"cuda": "CUDA (NVIDIA)", "rocm": "ROCm (AMD)", "vulkan": "Vulkan", "metal": "Metal"}


def empty_result():
    return {"installed": False, "source": None, "directory": None, "version": None, "build": None, "backend": None,
            "binaries": {name: None for name in BINARIES}, "warnings": []}


def _manifests(store):
    root = runtime_dir(store)
    found = []
    for path in root.glob("*/manifest.json"):
        if path.parent.name.startswith("."):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(data, dict):
            found.append((path.parent, data))
    return found


def _managed(store):
    """The install `current.json` points at, else the newest managed install."""
    root = runtime_dir(store)
    try:
        current = json.loads((root / "current.json").read_text(encoding="utf-8")).get("directory")
    except (OSError, ValueError, AttributeError):
        current = None
    items = _manifests(store)
    chosen = next((item for item in items if item[0].name == current), None)
    if not chosen and items:
        chosen = max(items, key=lambda item: (item[1].get("build") or 0, item[1].get("installed_at") or ""))
    return chosen


def _path_candidates():
    found = shutil.which("llama-server")
    if found:
        yield Path(found).parent
    if os.name != "nt":
        for folder in HOMEBREW_DIRS:
            folder = Path(folder).expanduser()
            if (folder / "llama-server").is_file():
                yield folder


def detect(store):
    """Configured folder first, then the app-managed install, then PATH (Homebrew included)."""
    warnings = []
    configured = (store.get("settings") or {}).get("runtime_dir")
    if configured:
        folder = find_bin_dir(configured)
        if folder:
            result = _inspect(folder, "configured")
            if result["installed"]:
                return _cache(store, result, warnings)
            warnings += result["warnings"]
        else:
            warnings.append(f"The llama.cpp folder you chose ({configured}) no longer contains llama-server. "
                            "Choose another with: llm-config runtime use DIR")
    managed = _managed(store)
    if managed:
        folder = managed[0] / managed[1].get("bin_dir", ".")
        result = _inspect(folder, "managed", managed[1])
        if result["installed"]:
            return _cache(store, result, warnings)
        warnings += result["warnings"]
    for folder in _path_candidates():
        result = _inspect(folder, "path")
        if result["installed"]:
            return _cache(store, result, warnings)
        warnings += result["warnings"]
    return _cache(store, empty_result(), warnings)


def _cache(store, result, warnings):
    result["warnings"] = list(dict.fromkeys(warnings + result["warnings"]))
    store.put("runtime", dict(result, checked_at=now()))
    return result


def binary(store, name):
    """argv prefix for a llama.cpp tool, e.g. ["/…/llama-server"]. Uses the cached detect() result
    when that file still exists, so callers don't pay for a --version run every time."""
    name = name[:-4] if name.endswith(".exe") else name
    if name not in BINARIES:
        raise ValueError(f"Unknown llama.cpp tool: {name}")
    cached = (store.get("runtime") or {})
    argv = (cached.get("binaries") or {}).get(name) if cached.get("installed") else None
    if not (argv and Path(argv[0]).is_file()):
        result = detect(store)
        if not result["installed"]:
            raise ValueError(NOT_INSTALLED)
        argv = result["binaries"].get(name)
    if not argv:
        raise ValueError(f"This llama.cpp build has no {name}. Reinstall with: llm-config runtime install")
    return list(argv)


# ---------- installing ----------

def _make_executable(folder):
    if os.name == "nt":
        return
    for path in Path(folder).iterdir():
        if path.is_file() and not path.is_symlink() and (path.name.startswith("llama-") or ".so" in path.name
                                                          or path.suffix == ".dylib"):
            path.chmod(path.stat().st_mode | 0o755)


def _merge_companion(archive, bin_dir, cancel):
    """CUDA runtime files (cudart) ship separately; they must sit next to llama-server."""
    with tempfile.TemporaryDirectory(dir=bin_dir.parent) as temp:
        safe_extract(archive, temp, cancel)
        folders = [Path(root) for root, _, _ in os.walk(temp)]
        libs = lambda f: sum(1 for p in f.iterdir() if re.search(r"\.(dll|so(\.\d+)*|dylib)$", p.name.lower()))
        source = max(folders, key=libs)
        for path in source.iterdir():
            # Only plain files: a relative link checked where it was unpacked would point somewhere else
            # once moved to bin_dir, so links are never carried over.
            if path.is_symlink() or not path.is_file() or (bin_dir / path.name).exists():
                continue
            os.replace(path, bin_dir / path.name)


def _tag_from(name):
    match = re.search(r"\b(b\d+)\b", name.lower())
    return match.group(1) if match else "local"


def _finish(store, archives, info, progress, cancel):
    """Unpack into a hidden staging folder, check it runs, then move it into place in one step."""
    root = runtime_dir(store)
    folder_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{info['tag']}-{info['backend']}")
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
    try:
        if progress:
            progress({"stage": "extract", "done": 0, "total": None, "message": "Unpacking llama.cpp"})
        safe_extract(archives[0], staging, cancel)
        bin_dir = find_bin_dir(staging)
        if not bin_dir:
            raise ValueError("The archive does not contain llama-server, so it is not a usable llama.cpp build.")
        for companion in archives[1:]:
            _merge_companion(companion, bin_dir, cancel)
        _make_executable(bin_dir)
        check_cancel(cancel)
        if progress:
            progress({"stage": "check", "done": 0, "total": None, "message": "Checking that llama.cpp starts"})
        ok, output = _run_version([str(bin_dir / _exe("llama-server"))])
        parsed = parse_version(output)
        if not ok or not parsed["version"]:
            raise ValueError("The downloaded llama.cpp would not start on this computer"
                             + (f" ({output.strip().splitlines()[-1][:200]})" if output.strip() else "") + ".")
        devices = None if info["backend"] else list_devices([str(bin_dir / _exe("llama-server"))])
        backend = (info["backend"] or (devices[0]["backend"] if devices else "cpu" if devices is not None else None)
                   or parsed["backend"] or _backend_from_files(bin_dir) or "unknown")
        if info["backend"] is None:
            folder_name = re.sub(r"[^A-Za-z0-9._-]", "_", f"{info['tag']}-{backend}")
        manifest = dict(info, backend=backend, build=info.get("build") or parsed["build"],
                        version=parsed["version"], bin_dir=bin_dir.relative_to(staging).as_posix(), installed_at=now())
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        final = root / folder_name
        old = None
        if final.exists():
            old = root / f".old-{folder_name}-{os.getpid()}"
            try:
                os.replace(final, old)
            except OSError:
                raise ValueError(f"Could not replace the existing install in {final}. Close any running llama.cpp "
                                 "and try again.") from None
        try:
            os.replace(staging, final)
        except OSError:
            if old:
                os.replace(old, final)
            raise ValueError(f"Could not move the new llama.cpp into {final}. Close any running llama.cpp (or wait "
                             "for antivirus to finish scanning) and try again.") from None
        if old:
            shutil.rmtree(old, ignore_errors=True)
        (root / "current.json").write_text(json.dumps({"directory": folder_name}), encoding="utf-8")
        return final
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def install(store, hardware, progress=None, cancel=None, release=None, allow_unverified=False):
    """Download the right official build, verify its SHA256, unpack it safely and check it runs.
    If a graphics-card build cannot start (e.g. missing Vulkan system files) fall back to the CPU build."""
    if progress:
        progress({"stage": "release", "done": 0, "total": None, "message": "Looking up the newest llama.cpp release"})
    release = release or fetch_release()
    choice = choose_asset(release, hardware, allow_unverified=allow_unverified)
    try:
        _install_choice(store, choice, progress, cancel)
        notes = []
    except ValueError as error:
        if choice["backend"] in ("cpu", "metal") or "would not start" not in str(error):
            raise
        cpu = choose_asset(release, hardware, allow_unverified=allow_unverified, backend="cpu")
        _install_choice(store, cpu, progress, cancel)
        notes = [f"The {choice['backend']} build would not start here, so the CPU-only build was installed instead. "
                 "Updating your graphics driver may let the faster build work."]
        choice = cpu
    result = detect(store)
    if result["source"] == "configured":
        notes.append("The new llama.cpp was installed, but the app keeps using the llama.cpp folder you chose earlier "
                     "with 'llm-config runtime use'. Remove that setting to switch to the new install.")
    result["warnings"] = notes + result["warnings"]
    result["reason"] = choice["reason"]
    if choice.get("unverified"):
        result["warnings"].append("This llama.cpp was installed without a checksum check, at your request.")
    return result


def _install_choice(store, choice, progress, cancel):
    folder = runtime_dir(store) / "downloads"
    records = [choice] + list(choice.get("companions") or [])
    total = sum(r["size"] for r in records)
    archives, done = [], 0
    for record in records:
        archives.append(_download(record, folder, progress, cancel, done_before=done, grand_total=total))
        done += record["size"]
    info = {"tag": choice.get("tag") or _tag_from(choice["name"]), "build": choice.get("build"),
            "backend": choice["backend"], "cuda": choice.get("cuda"), "source": "release",
            "asset": choice["name"], "sha256": choice["sha256"], "reason": choice.get("reason"),
            "companions": [{"name": r["name"], "sha256": r["sha256"]} for r in records[1:]]}
    try:
        _finish(store, archives, info, progress, cancel)
    except ValueError:
        if "would not start" in str(sys.exc_info()[1]):
            for archive in archives:
                archive.unlink(missing_ok=True)
        raise
    for archive in archives:
        archive.unlink(missing_ok=True)


def install_archive(store, archive_path, progress=None, cancel=None):
    """Offline install from a llama.cpp zip/tar.gz the user already has (no checksum to compare with)."""
    archive = Path(archive_path).expanduser()
    if not archive.is_file():
        raise ValueError(f"No file at {archive}. Give the path to a llama.cpp .zip or .tar.gz.")
    if not archive.name.lower().endswith((".zip", ".tar.gz", ".tgz")):
        raise ValueError("Only .zip and .tar.gz llama.cpp archives are supported.")
    guess = classify(archive.name) or {}
    info = {"tag": _tag_from(archive.name), "build": guess.get("build"), "backend": guess.get("backend"),
            "cuda": ".".join(map(str, guess["cuda"])) if guess.get("cuda") else None, "source": "archive",
            "asset": archive.name, "sha256": _sha256(archive), "companions": []}
    _finish(store, [archive], info, progress, cancel)
    return detect(store)


def use_directory(store, directory):
    """Point the app at a llama.cpp the user built or installed themselves (CLI only)."""
    folder = Path(directory).expanduser()
    bin_dir = find_bin_dir(folder)
    if not bin_dir:
        raise ValueError(f"No llama-server found in {folder} (also looked in bin/ and build/bin/). "
                         "Pick the folder that contains llama-server.")
    result = _inspect(bin_dir, "configured")
    if not result["installed"]:
        raise ValueError(result["warnings"][0] if result["warnings"] else "That llama-server did not start.")
    store.update("settings", lambda s: dict(s or {}, runtime_dir=str(folder.resolve())), {})
    return detect(store)
