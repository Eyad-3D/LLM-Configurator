# Handoff: `downloads`

Branch `claude/v04-downloads`. Files: `src/llm_configurator/downloads.py` (new), `src/llm_configurator/runtime.py` (download now delegates), `tests/test_downloads.py` (new).

## What I built

Resumable, verified, multi-file (sharded) model downloads with progress and cancel.

- Every file in `variant.all_files()` is fetched to `<name>.part`, SHA256-checked, then renamed into place.
- **Resume:** an existing `.part` is re-hashed once, then `Range: bytes=<n>-` is sent. A `206` is accepted only when `Content-Range` starts at `n`, ends at `size-1` and its total is the catalogue size (or `*`). A `200` (Range ignored) truncates and restarts from zero. A mismatched start or a `416` discards the partial and restarts. A partial larger than the expected size is deleted.
- **Retries:** network errors (timeouts, resets, early close, HTTP 429/5xx) are retried with 2/4/8/16 s back-off, resuming each time. The counter resets whenever an attempt made progress, so a long, flaky download still finishes. After 4 attempts in a row with no progress: a plain "network kept failing, run again to resume" error, and the `.part` is kept.
- **Security kept from v0.3:** HTTPS-only redirects, `Authorization` removed on cross-host redirects, never writes past the catalogue size (checked against `Content-Length` up front and per chunk), SHA256 check, never overwrites a different existing file. New: `HF_ENDPOINT` must be `https://`, local names are the basename only (no path escapes, no duplicates), and `remove_variant` never follows symlinks.
- **Cancel** (`domain.check_cancel` per chunk and while re-hashing) raises `Cancelled` and keeps `.part` files.
- **Progress:** totals across all shards, smoothed speed, throttled to at most 5 callbacks per second (stage changes are always sent).

## Public API as built

```python
downloads.plan(variant, directory) -> {
    "files": [{"filename", "size_bytes", "present": bool, "partial_bytes": int}],
    "total_bytes", "remaining_bytes", "disk_free", "enough_space": bool}   # needs remaining + 1 GiB

downloads.download_variant(variant, directory, progress=None, cancel=None, token=None) -> Path  # first shard
downloads.remove_variant(variant, directory) -> int   # bytes freed

runtime.download(variant, directory, progress=None, cancel=None, token=None) -> Path  # thin wrapper
```

Progress dict: `{"stage": "checking"|"downloading"|"retrying"|"verifying"|"done", "done", "total", "message",
"bytes_per_second": int|None, "eta_seconds": int|None, "file", "file_index", "file_count", "file_done", "file_total"}`.
`done`/`total` are bytes across all shards; bytes already on disk count towards `done`. Speed and ETA are `None` until measured (after about 0.5 s of transfer), never guessed.

Helpers others may use: `downloads.digest(path)`, `downloads.endpoint()`, `downloads.file_url(variant, filename)`, `downloads.DownloadRedirect` (also re-exported from `runtime`, together with `digest`, so `runtime.bench` and existing test patches keep working).

## Deviations and decisions

- `token=None` falls back to `HF_TOKEN`. The token is sent only as `Authorization: Bearer …` and stripped on cross-host redirects.
- Local file name = basename of the repo path (`Q4_K_M/m-00001-of-00003.gguf` → `m-00001-of-00003.gguf`), so all shards sit side by side, as llama.cpp's split loader expects.
- `plan()` is cheap: `present` means "a file with that name and exact size exists" (symlinks followed). The full SHA256 check happens in `download_variant`, which re-hashes files already present before reusing them. That is slow for very large files; `discover.find_for_variant` + `hash_cache` is the fast reuse path (the api workstream calls that first).
- `remove_variant` deletes a finished file only if its size matches, and a `.part` only if it is not larger than expected. It does not hash (would take minutes for big models) and does not delete through symlinks (they are skipped and count as 0 bytes).
- Every shard needs its own `sha256`; if the catalogue lacks one, the download is refused with a plain message.
- A leftover `.part` used to be an error ("inspect/remove it"); it is now resumed. That is the intended v0.4 behaviour.

## Known gaps

- No early check against `X-Linked-Etag` / `X-Linked-Size` (the SHA256 and size the Hub advertises on its first reply). urllib follows the redirect internally, so those headers are not visible on the final response. The final SHA256 check covers it, just later.
- No lock file: two processes downloading the same variant into the same folder would collide. Inside the app, `exclusive="download:<variant_id>"` prevents it.
- Re-hashing a large `.part` on resume takes time (about 1–2 GB/s); it is reported as stage `checking`.

## Requests to other owners

- **api (`cli.py`):** `runtime.download` now accepts `progress=` and `cancel=`, so the CLI can show a progress bar. Consider calling `downloads.plan` first to show sizes and free space, and `downloads.download_variant` directly. Please catch `Cancelled` on Ctrl+C and say "Paused. Run the same command again to resume."
- **api (`server.py`):** `/api/downloads/plan` can return `plan()` as is: it has only file *names*, never absolute paths.
- **catalogue:** please fill `sha256` for every shard in `files`, or those variants can't be downloaded.

## How I tested

`tests/test_downloads.py` (25 tests, no network). `downloads._open` is swapped for a fake Hub that honours Range. Redirect tests run the **real** urllib redirect chain through a fake `HTTPSHandler`. Covered: full download; `HF_ENDPOINT`/`HF_TOKEN`; http endpoint refused; 3-shard download (first shard returned, totals, monotonic progress); resume with 206; server ignoring Range; mismatched Content-Range; oversize partial; hash mismatch (partial deleted); too-large body (with and without Content-Length); wrong total in Content-Range; transient errors retried and resumed; giving up after repeated failures (partial kept); 401 message; cancel mid-file then resume; speed/ETA/throttling with a fake clock; insufficient disk (mocked `shutil.disk_usage`); refusing to overwrite a different file; `plan` shapes; `remove_variant` only removing matching files (not symlinks, not other files); `runtime.download` delegating; cross-host redirect strips auth and keeps Range; same-host keeps auth; http redirect refused (partial kept).
I also checked that breaking each safety check (auth strip, size cap, HTTPS-only, hash check) makes a test fail. Full suite: `python3 -m unittest discover -s tests` OK, `npm test` OK.

## Facts assumed (for the integration harness to confirm)

No llama.cpp facts, except that split GGUFs load from the first shard with all shards in the same folder under their original names.

Hugging Face behaviour. I read it from `huggingface_hub/file_download.py` on GitHub; huggingface.co itself is unreachable from this container.

- `GET {endpoint}/{repo}/resolve/{revision}/{path}` returns a 302 to a CDN host (signed URL). The official client drops auth when the redirect leaves the Hub host, as we do.
- The official client resumes with `Range` and truncates if it gets a `200` back, as we do.
- `HF_ENDPOINT` replaces the `https://huggingface.co` prefix.
- Unverified here: that the CDN answers Range with `206` + `Content-Range: bytes a-b/total`. The code falls back safely if not.
