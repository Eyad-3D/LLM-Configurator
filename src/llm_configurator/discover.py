"""Find model files already on this computer so nobody downloads the same 10 GB twice.

Promises: only reads model files (never moves, renames or deletes them); never follows a link
out of the folder being scanned, except Hugging Face's own snapshot -> blob links; never hashes
gigabytes during a scan; a name-and-size match is reported as "likely", never as verified.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import re

from . import gguf
from .domain import Variant, check_cancel

HEX64 = re.compile(r"^[0-9a-f]{64}$")
OLLAMA_MODEL_LAYER = "application/vnd.ollama.image.model"
MAX_DEPTH = 8  # model folders are shallow; stops a scan of a huge custom folder running away
MAX_ENTRIES = 200_000
MAX_SMALL_FILE = 1024 * 1024  # manifests and settings files
HASH_CHUNK = 8 * 1024 * 1024
HASH_CACHE_LIMIT = 2000
LABELS = {"hf_cache": "Hugging Face downloads", "lmstudio": "LM Studio models", "ollama": "Ollama models",
          "models_dir": "this app's models folder", "custom": "a folder you added"}
NOTES = {"sha256": "Same file as the catalogue model (its fingerprint matches).",
         "name_size": "Likely the same file as the catalogue model (same name and size), not yet verified."}


def _home():
    return Path.home()


def _system():
    return platform.system()


def _expand(text):
    """Like expanduser, but relative to _home() so every OS branch can be tested."""
    if text == "~" or text.startswith(("~/", "~\\")):
        return _home() / text[2:]
    return Path(text)


def _env_path(name):
    value = os.environ.get(name)
    return _expand(value) if value else None


def _read_small(path):
    try:
        if path.is_file() and path.stat().st_size <= MAX_SMALL_FILE:
            return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    return None


def _hf_roots():
    home = _home()
    default = home / ".cache" / "huggingface" / "hub"
    hf_home = _env_path("HF_HOME")
    if not hf_home and _env_path("XDG_CACHE_HOME"):
        hf_home = _env_path("XDG_CACHE_HOME") / "huggingface"
    effective = _env_path("HF_HUB_CACHE") or _env_path("HUGGINGFACE_HUB_CACHE") or (hf_home / "hub" if hf_home else default)
    return [effective, default]  # the default too: files downloaded before the variable was set stay reusable


def _ollama_roots():
    roots = [_env_path("OLLAMA_MODELS"), _home() / ".ollama" / "models"]
    if _system() == "Linux":
        roots.append(Path("/usr/share/ollama/.ollama/models"))  # the Linux install script runs Ollama as its own user
    return [r for r in roots if r]


def _lmstudio_roots():
    home = _home()
    pointer = _read_small(home / ".lmstudio-home-pointer")  # LM Studio records a moved home folder here
    lm_home = _expand(pointer.strip()) if pointer and pointer.strip() else home / ".lmstudio"
    roots = []
    for settings in [lm_home / "settings.json", home / ".cache" / "lm-studio" / "settings.json"]:
        try:
            folder = json.loads(_read_small(settings) or "{}").get("downloadsFolder")
        except (ValueError, AttributeError):
            folder = None
        if isinstance(folder, str) and folder.strip():
            roots.append(_expand(folder.strip()))  # LM Studio itself does not expand "~"; we do
    return roots + [lm_home / "models", home / ".cache" / "lm-studio" / "models"]


def _key(path):
    text = os.path.abspath(str(path))
    return text.lower() if _system() == "Windows" else text


def locations(store=None, extra_dirs=()):
    """Every folder a scan looks in, for Windows, macOS and Linux. `exists` says whether it is there now."""
    found = [("hf_cache", p) for p in _hf_roots()] + [("lmstudio", p) for p in _lmstudio_roots()]
    found += [("ollama", p) for p in _ollama_roots()]
    if store is not None:
        from .storage import models_dir
        found.append(("models_dir", models_dir(store)))
    found += [("custom", _expand(str(p))) for p in extra_dirs]
    result, seen = [], set()
    for source, path in found:
        if _key(path) not in seen:
            seen.add(_key(path))
            result.append({"source": source, "path": str(path), "exists": path.is_dir()})
    return result


def _inside(path, root):
    try:
        return path.is_relative_to(root)
    except (OSError, ValueError):
        return False


def _stat_entry(path, real, **extra):
    try:
        info = real.stat()
    except OSError:
        return None
    if not real.is_file() or info.st_size <= 0:
        return None
    return {"path": path, "real": real, "size": info.st_size, "mtime": info.st_mtime, "sha256": None, "hash_source": None, **extra}


def _walk(root, cancel, counter, pattern=lambda name: name.lower().endswith(".gguf")):
    """(path, is_link) for matching files; skips hidden folders and never descends through linked folders."""
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        check_cancel(cancel)
        here = Path(dirpath)
        dirnames[:] = [] if len(here.parts) - base_depth >= MAX_DEPTH else [d for d in dirnames if not d.startswith(".")]
        counter[0] += len(filenames) + len(dirnames)
        if counter[0] > MAX_ENTRIES:
            dirnames[:] = []
            return
        for name in filenames:
            if pattern(name):
                yield here / name


def _scan_plain(root, source, cancel, counter):
    real_root = root.resolve()
    for path in _walk(root, cancel, counter):
        real = path.resolve() if path.is_symlink() else path
        if path.is_symlink() and not _inside(real, real_root):
            continue  # a link pointing outside the scanned folder is never followed
        entry = _stat_entry(path, real, source=source, filename=path.name)
        if entry:
            yield entry


def _scan_hf(root, cancel, counter):
    try:
        repos = sorted(p for p in root.iterdir() if p.name.startswith("models--") and p.is_dir() and not p.is_symlink())
    except OSError:
        return
    for repo_dir in repos:
        repo = repo_dir.name[len("models--"):].replace("--", "/")
        blobs = (repo_dir / "blobs").resolve()
        snapshots = repo_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for path in _walk(snapshots, cancel, counter):
            snapshot = path.relative_to(snapshots).parts[0]
            filename = path.relative_to(snapshots / snapshot).as_posix()
            if path.is_symlink():
                real = path.resolve()
                if real.parent != blobs:
                    continue  # only the snapshot -> own blobs/ pattern is trusted
                entry = _stat_entry(path, real, source="hf_cache", filename=filename, repo=repo)
                if entry and HEX64.match(real.name):
                    entry.update(sha256=real.name, hash_source="blob_name")  # the cache names blobs by content hash
            else:  # Windows without symlink rights stores plain copies; no hash to borrow
                entry = _stat_entry(path, path, source="hf_cache", filename=filename, repo=repo)
            if entry:
                yield entry


def _ollama_name(parts):
    """manifests/<registry>/<namespace>/<model>/<tag> -> "model:tag" (or "namespace/model:tag")."""
    registry, namespace, model, tag = ([""] * 4 + list(parts))[-4:]
    short = model if namespace == "library" and registry == "registry.ollama.ai" else f"{namespace}/{model}".strip("/")
    return f"{short}:{tag}" if tag else short


def _scan_ollama(root, cancel, counter):
    manifests, blobs = root / "manifests", root / "blobs"
    if not manifests.is_dir() or not blobs.is_dir():
        return
    real_blobs = blobs.resolve()
    found = {}
    for manifest in _walk(manifests, cancel, counter, pattern=lambda name: True):
        if manifest.is_symlink():
            continue
        try:
            layers = json.loads(_read_small(manifest) or "{}").get("layers") or []
        except (ValueError, AttributeError):
            continue
        for layer in layers if isinstance(layers, list) else []:
            if not isinstance(layer, dict) or layer.get("mediaType") != OLLAMA_MODEL_LAYER:
                continue
            digest = layer.get("digest")
            hexdigest = digest[7:] if isinstance(digest, str) and digest.startswith("sha256:") else ""
            if not HEX64.match(hexdigest):
                continue
            name = _ollama_name(manifest.relative_to(manifests).parts)
            if hexdigest in found:
                found[hexdigest]["names"].append(name)
                continue
            path = blobs / f"sha256-{hexdigest}"
            real = path.resolve() if path.is_symlink() else path
            if path.is_symlink() and not _inside(real, real_blobs):
                continue
            entry = _stat_entry(path, real, source="ollama", filename=path.name, names=[name])
            if entry:
                entry.update(sha256=hexdigest, hash_source="blob_name")
                found[hexdigest] = entry
    yield from found.values()


def _group(entries):
    """One record per model: split files become one entry listing every part, first part first."""
    groups, order = {}, []
    for entry in entries:
        match = gguf.SHARD.match(entry["path"].name)
        key = (_key(entry["path"].parent), match["prefix"].lower(), match["count"]) if match else (_key(entry["path"]), "", "")
        if key not in groups:
            groups[key] = {"count": int(match["count"]) if match else 1, "parts": {}}
            order.append(key)
        groups[key]["parts"][int(match["index"]) if match else 1] = entry
    for key in order:
        group = groups[key]
        if 1 not in group["parts"]:
            continue  # without the first part llama.cpp cannot load the model
        parts = [group["parts"][i] for i in sorted(group["parts"])]
        yield parts, len(parts) == group["count"]


def _files(record):
    return record.get("files") or [{"path": record["path"], "filename": record.get("filename") or Path(record["path"]).name,
                                    "size_bytes": record["size_bytes"], "sha256": record.get("sha256")}]


def _compare(files, wanted):
    """"sha256" when every part's fingerprint matches, "name_size" when names and sizes match, else None."""
    if len(files) != len(wanted):
        return None
    how = "sha256"
    for have, want in zip(files, wanted):
        if have.get("size_bytes") != want.get("size_bytes"):
            return None
        mine, theirs = (have.get("sha256") or "").lower(), (want.get("sha256") or "").lower()
        if mine and theirs:
            if mine != theirs:
                return None
        elif Path(have.get("filename") or "").name.lower() == Path(want.get("filename") or "").name.lower():
            how = "name_size"
        else:
            return None
    return how


def _catalogue_index(store):
    by_sha, by_name = {}, {}
    for item in store.get("variants") or []:
        try:
            variant = Variant(**item)
        except (TypeError, ValueError):
            continue
        wanted = variant.all_files()
        if wanted[0].get("sha256"):
            by_sha.setdefault(wanted[0]["sha256"].lower(), []).append((variant.id, wanted))
        by_name.setdefault((Path(wanted[0]["filename"]).name.lower(), wanted[0]["size_bytes"]), []).append((variant.id, wanted))
    return by_sha, by_name


def _match(record, index):
    by_sha, by_name = index
    files = _files(record)
    first = files[0]
    candidates = by_sha.get((first.get("sha256") or "").lower(), []) + by_name.get((Path(first["filename"]).name.lower(), first["size_bytes"]), [])
    best = (None, None)
    for variant_id, wanted in candidates:
        how = _compare(files, wanted)
        if how == "sha256":
            return variant_id, how
        if how and not best[0]:
            best = (variant_id, how)
    return best


def _record(parts, complete, cache, index):
    first = parts[0]
    for part in parts:
        cached = cache.get(str(part["real"].resolve()))
        if not part["sha256"] and cached and cached.get("size") == part["size"] and cached.get("mtime") == part["mtime"]:
            part.update(sha256=cached["sha256"], hash_source="hashed")
    record = {
        "path": str(first["path"]), "filename": first["filename"], "size_bytes": sum(p["size"] for p in parts),
        "sha256": first["sha256"], "hash_source": first["hash_source"], "source": first["source"],
        "repo": first.get("repo"), "names": first.get("names"), "mtime": first["mtime"], "complete": complete,
        "files": [{"path": str(p["path"]), "filename": Path(p["filename"]).name, "size_bytes": p["size"],
                   "sha256": p["sha256"], "hash_source": p["hash_source"]} for p in parts],
        "gguf": None, "gguf_error": None, "variant_id": None, "match": None, "match_note": None,
        "local_variant_id": None, "local_variant": None, "local_error": None, "verified": False,
    }
    try:
        record["gguf"] = gguf.read_metadata(first["path"])["summary"]
    except ValueError as exc:
        record["gguf_error"] = str(exc)
    if complete:
        record["variant_id"], record["match"] = _match(record, index)
        record["match_note"] = NOTES.get(record["match"])
        record["verified"] = record["match"] == "sha256"
    if complete and not record["variant_id"] and record["gguf"]:
        try:  # an unknown GGUF can still be sized and run as a "local" model
            variant = gguf.variant_from_file(first["path"], sha256=first["sha256"])
            record["local_variant_id"], record["local_variant"] = variant.id, variant.to_dict()
        except ValueError as exc:
            record["local_error"] = str(exc)
    return record


def _report(progress, done, message):
    if progress:
        progress({"stage": "scanning", "done": done, "total": None, "message": message})


def scan(store, extra_dirs=(), progress=None, cancel=None):
    """Looks for GGUF models in every known folder, matches them to catalogue models cheaply and saves the list.

    Hashes are only borrowed (Hugging Face / Ollama blob names, earlier checks), never computed here.
    """
    places = locations(store, extra_dirs)
    cache = store.get("hash_cache") or {}
    index = _catalogue_index(store)
    entries, seen, counter = [], set(), [0]
    scanners = {"hf_cache": _scan_hf, "ollama": _scan_ollama}
    for place in places:
        if not place["exists"]:
            continue
        _report(progress, len(entries), f"Looking in {LABELS[place['source']]}…")
        root = Path(place["path"])
        found = scanners[place["source"]](root, cancel, counter) if place["source"] in scanners else \
            _scan_plain(root, place["source"], cancel, counter)
        for entry in found:
            check_cancel(cancel)
            if _key(entry["real"]) not in seen:  # the same blob reached from two snapshots or folders counts once
                seen.add(_key(entry["real"]))
                entries.append(entry)
    records = []
    for number, (parts, complete) in enumerate(_group(entries), 1):
        check_cancel(cancel)
        _report(progress, number, f"Reading model details ({number})…")
        records.append(_record(parts, complete, cache, index))
    known = {_key(r["path"]) for r in records}

    def merge(previous):  # files added one by one survive a rescan while they exist
        return records + [r for r in previous or [] if r.get("added") and _key(r["path"]) not in known and Path(r["path"]).is_file()]
    records = store.update("local_files", merge, [])
    _report(progress, len(records), f"Found {len(records)} model file{'s' if len(records) != 1 else ''} on this computer.")
    return records


def add_file(store, path):
    """Registers one GGUF the user points at (CLI `local --add`). Raises ValueError with a plain reason."""
    path = Path(path).expanduser().absolute()
    if not path.is_file():
        raise ValueError(f"{path.name} was not found. Check the path and try again.")
    parts = gguf.shard_paths(path)
    entries = [_stat_entry(p, p, source="custom", filename=p.name) for p in parts]
    if any(e is None for e in entries):
        raise ValueError(f"{path.name} could not be read or is empty.")
    record = _record(entries, True, store.get("hash_cache") or {}, _catalogue_index(store))
    if record["gguf_error"]:
        raise ValueError(record["gguf_error"])
    if not record["variant_id"] and record["local_error"]:
        raise ValueError(record["local_error"])
    record["added"] = True
    store.update("local_files", lambda items: [r for r in items or [] if _key(r["path"]) != _key(record["path"])] + [record], [])
    return record


def hash_cached(store, path, progress=None, cancel=None):
    """SHA-256 of a file, reusing an earlier result while its size and modification time are unchanged."""
    path = Path(path)
    key = str(path.resolve())
    try:
        before = path.stat()
    except OSError as exc:
        raise ValueError(f"Could not read {path.name}: {exc.strerror or exc}.") from None
    entry = (store.get("hash_cache") or {}).get(key)
    if entry and entry.get("size") == before.st_size and entry.get("mtime") == before.st_mtime:
        return entry["sha256"]
    digest, done, reported = hashlib.sha256(), 0, 0
    with path.open("rb") as handle:
        while chunk := handle.read(HASH_CHUNK):
            check_cancel(cancel)
            digest.update(chunk)
            done += len(chunk)
            if progress and (done - reported >= 64 * 1024 * 1024 or done == before.st_size):
                reported = done
                progress({"stage": "verifying", "done": done, "total": before.st_size,
                          "message": f"Checking {path.name} is complete and unchanged…"})
    after = path.stat()
    if (after.st_size, after.st_mtime) != (before.st_size, before.st_mtime):
        raise ValueError(f"{path.name} changed while it was being checked. Wait for any copy or download to finish, then try again.")
    result = digest.hexdigest()

    def save(cache):
        cache = {k: v for k, v in (cache or {}).items() if k != key}
        cache[key] = {"size": before.st_size, "mtime": before.st_mtime, "sha256": result}
        return dict(list(cache.items())[-HASH_CACHE_LIMIT:])
    store.update("hash_cache", save, {})
    return result


def _candidates(store, variant, wanted):
    from .storage import models_dir
    base = models_dir(store)
    real_base = base.resolve()
    direct = [base / f["filename"] for f in wanted]
    if all(_inside(p.resolve(), real_base) for p in direct):  # catalogue file names never lead out of the folder
        yield [{"path": str(p), "filename": Path(f["filename"]).name, "size_bytes": f["size_bytes"], "sha256": None}
               for p, f in zip(direct, wanted)], None
    for record in store.get("local_files") or []:
        if not record.get("complete", True):
            continue
        files = _files(record)
        if variant.id in {record.get("variant_id"), record.get("local_variant_id")} or _compare(files, wanted):
            yield files, record


def _blob_hash_holds(item, expected):
    """A borrowed blob-name hash counts only while the file still resolves to the blob named by that hash."""
    if item.get("hash_source") != "blob_name" or (item.get("sha256") or "").lower() != expected:
        return False
    name = Path(item["path"]).resolve().name
    return name in {expected, f"sha256-{expected}"}


def find_for_variant(store, variant, verify=True, progress=None, cancel=None):
    """Path of a complete local copy (first part for split models), or None.

    verify=True checks each file's SHA-256 against the catalogue (cached in hash_cache); a file
    without a published hash can only be matched on name and size.
    """
    variant = variant if isinstance(variant, Variant) else Variant(**variant)
    wanted = variant.all_files()
    for files, record in _candidates(store, variant, wanted):
        check_cancel(cancel)
        paths = [Path(f["path"]) for f in files]
        try:
            if len(paths) != len(wanted) or any(not p.is_file() or p.stat().st_size != w["size_bytes"] for p, w in zip(paths, wanted)):
                continue
        except OSError:
            continue
        if not verify:
            return paths[0]
        ok = True
        for item, path, want in zip(files, paths, wanted):
            expected = (want.get("sha256") or "").lower()
            if expected and not _blob_hash_holds(item, expected) and hash_cached(store, path, progress, cancel) != expected:
                ok = False
                break
        if ok:
            if record is not None and all(w.get("sha256") for w in wanted):
                _mark_verified(store, record["path"], None if variant.source == "local" else variant.id)
            return paths[0]
    return None


def _mark_verified(store, path, variant_id):
    def change(items):
        for item in items or []:
            if item.get("path") == path and item.get("variant_id") in {None, variant_id}:
                item["verified"] = True
                if variant_id:
                    item.update(variant_id=variant_id, match="sha256", match_note="Checked: this is exactly the catalogue model's file.")
        return items or []
    store.update("local_files", change, [])


def local_variants(store):
    """Variants built from local GGUFs that match no catalogue model, so they can be sized and run too."""
    result = []
    for record in store.get("local_files") or []:
        try:
            if record.get("local_variant"):
                result.append(Variant(**record["local_variant"]))
        except (TypeError, ValueError):
            continue
    return result
