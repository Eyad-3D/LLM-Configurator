# Handoff: `discover` workstream

Branch `claude/v04-discover`. Files: `src/llm_configurator/gguf.py`, `src/llm_configurator/discover.py`,
`tests/test_gguf.py`, `tests/test_discover.py`, this file. No other file was touched.

## What I built

- **`gguf.py`**: a GGUF header reader that uses only the standard library. It reads GGUF v2 and v3,
  little-endian, and every value type (uint8 to float64, bool, string, arrays, nested arrays). It never
  reads the weights. Token lists and merges are stepped over, not loaded: string lengths are parsed from
  1 MiB read-ahead chunks, so a 262k-token vocabulary takes about 0.1–0.25 s. Safety limits:
  - at most 64 MiB of header is read (`MAX_HEADER_BYTES`)
  - at most 100k settings and 1M tensors
  - each string is at most 16 MiB
  - array counts are checked against the bytes left in the file before any loop runs
  - arrays nest at most 4 levels deep
  - a tensor has at most 8 dimensions

  Bad files get a plain `ValueError`: wrong magic, v1, an unknown version, big-endian, truncated, or
  impossible counts. Optionally (`tensors=True`) it reads the tensor table to count parameters and the
  bytes held in routed-expert tensors (`*_exps`; shared experts `*_shexp` are not counted).
- **`variant_from_file`** builds a `source="local"` `Variant` from any supported GGUF, including split
  files (`-00001-of-00003.gguf`). You can pass any part. It finds the other parts, adds up their sizes,
  reads every part's tensor table, and fills `files`. If parts are missing, it names them.
- **`discover.py`**: finds models already on disk. It covers:
  - the Hugging Face cache
  - Ollama
  - LM Studio, including the `downloadsFolder` in its settings and the `~/.lmstudio-home-pointer` file
  - the app's `models_dir`
  - folders the user adds

  It matches files to catalogue variants cheaply, keeps hashes in `hash_cache`, and finds a verified
  local copy for a variant.

## Public API as built

### `gguf.py`

```python
read_metadata(path, tensors=False) -> {
  "version": 2|3, "tensor_count": int, "metadata": {scalar keys only; strings > 64 KiB skipped},
  "architecture": str|None,
  "summary": {"name", "layers", "kv_heads", "head_dim", "context_length", "experts", "active_experts",
              "sliding_window", "sliding_layers", "file_type", "quant", "split_count"},
  "skipped_keys": [keys whose value was stepped over (arrays > 4096 items, string lists, long strings)],
  "tensors": {"count", "parameters", "expert_parameters", "bytes", "expert_bytes", "unknown_types"}  # only when tensors=True
}
variant_from_file(path, sha256=None) -> Variant       # ValueError with a plain reason for unsupported/incomplete files
shard_paths(path) -> [Path]                           # all parts in order, or [path]; ValueError lists missing parts
quant_name(file_type=None, filename="") -> str|None   # llama_ftype first, then the file name
local_id(path, sha256=None) -> "local:<16 hex of sha256, or of the resolved path>:<filename>"
FILE_TYPES, TENSOR_TYPES, ARCH_MAP, SHARD              # constants
```

How the summary is built:

- `kv_heads`: the largest value in a per-layer list, which is a safe upper bound. If missing, it falls back
  to `head_count`, as llama.cpp does.
- `head_dim`: `key_length`, otherwise `embedding_length // head_count`.
- `experts` and `active_experts`: 0 when absent.
- `split_count`: 1 when absent.
- `sliding_layers` (extra key): comes from `<arch>.attention.sliding_window_pattern` (an int or a per-layer
  bool list). Otherwise it uses llama.cpp's built-in pattern for `gemma2` (2), `gemma3` (6) and
  `gpt-oss` (2). Otherwise it is 0, which counts full KV and never underestimates memory.

The local `Variant` is filled like this:

- `id` = `local_id`.
- `repo="local"` and `base_repo="local/<general.name>"`.
- `quant` is the name above, or `"unknown"`.
- `parameters` is the sum of tensor elements.
- `expert_fraction` = expert bytes ÷ all tensor bytes. If a tensor type is unknown, it falls back to the
  share of elements. It is capped at 0.999.
- `active_parameters` = parameters − expert params × (1 − active/experts).
- `family` comes from `general.basename` and `license` from `general.license`.
- `sliding_window` is set only when `sliding_layers > 0`.

### `discover.py`

```python
locations(store=None, extra_dirs=()) -> [{"source": "hf_cache"|"lmstudio"|"ollama"|"models_dir"|"custom", "path": str, "exists": bool}]
scan(store, extra_dirs=(), progress=None, cancel=None) -> [record]      # also saved under store "local_files"
find_for_variant(store, variant, verify=True, progress=None, cancel=None) -> Path | None   # variant: Variant or dict
hash_cached(store, path, progress=None, cancel=None) -> str             # sha256 hex
add_file(store, path) -> record          # extra: register one GGUF (CLI `local --add`); survives rescans
local_variants(store) -> [Variant]       # extra: local GGUFs that match no catalogue model
```

Record (the contract keys plus extras):

```
{"path", "size_bytes" (all parts), "sha256"|None (first part), "source", "variant_id"|None (catalogue match),
 "gguf": summary|None, "mtime", "verified": bool,
 # extras
 "filename" (HF: path inside the snapshot, e.g. "Q4_K_M/x.gguf"), "hash_source": "blob_name"|"hashed"|None,
 "repo" (HF "org/name")|None, "names" (Ollama "model:tag" list)|None, "complete": bool (all split parts present),
 "files": [{"path", "filename", "size_bytes", "sha256", "hash_source"}],
 "match": "sha256"|"name_size"|None, "match_note": plain sentence|None, "gguf_error": str|None,
 "local_variant_id", "local_variant" (Variant dict)|None, "local_error": str|None, "added"?: True}
```

Progress events:

- `{"stage": "scanning", "done", "total": None, "message"}` while scanning
- `{"stage": "verifying", "done", "total", "message"}` while hashing

Messages never contain folder paths.

## Deviations from the contract and why

- `locations()` takes optional `store` and `extra_dirs`. Without a store it cannot know `models_dir`.
  `scan()` passes both.
- `read_metadata` adds `summary.sliding_layers`, `skipped_keys`, and an optional `tensors` block. All of
  these are extra keys, which the contract allows.
- **What `verified` means:** it is `True` only when every part's SHA-256 equals the catalogue's. The
  hash can come from our own earlier hashing (`hash_cache`) or from a Hugging Face or Ollama blob name.
  I trust blob names without re-hashing, as the task asked. Both tools name a blob by its content hash,
  and `find_for_variant` checks again that the path still resolves to the blob with that name. A
  name-and-size match is `match="name_size"` with the note "Likely the same file … not yet verified".
- **Files with no published hash:** `find_for_variant(verify=True)` can only match these on name and
  size, because there is nothing to check against. `verified` stays `False` for them.
- `find_for_variant` looks only in `models_dir` and in stored `local_files` records. It does not scan
  on its own, so call `scan()` first (or at startup) to find copies in other apps' folders.
- **Where a local model's file lives:** a local `Variant` has no path field. Its file is found through
  `local_files[*].local_variant_id`, which `find_for_variant` handles.

## Known gaps

- **Architectures:** only the names in `ARCH_MAP` are mapped. Llama plus `expert_count` becomes
  `mixtral`. Anything else (for example `mistral3`, `gemma3n`, `granitemoe`, `deepseek2`, `llama4`)
  raises a plain "cannot size yet" error until `domain.ARCHITECTURES` grows.
- **Split files that are not GGUF:** we do not handle these (for example Ollama models stored in other
  formats). They show up with `gguf_error`.
- **Symlinked copies:** if two copies of one file are reachable through a symlink inside the same root,
  only one record is kept. Which path it reports depends on walk order.
- **Scan limits:** scans stop at 8 folders deep and 200k directory entries. Hidden folders are skipped.
  A very large custom folder may therefore be scanned only in part, and no message says so.
- **LM Studio's home pointer:** I don't have first-hand confirmation of the `~/.lmstudio-home-pointer`
  format (a plain text path). If it is missing or unreadable, we fall back to `~/.lmstudio`.
- **Hashing uses Python's `hashlib`:** it runs at roughly disk speed, 1–2 GB/s, so a 10 GB file takes
  several seconds or more. It supports progress and cancel.

## Requests to other owners

- **api**: expose `discover.locations(store)` and `discover.scan` at `/api/local-models` and
  `/api/local-models/scan`. When sending records to the page, strip `path`, `files[*].path` and
  `local_variant` (paths must not reach the browser).
- **api**: CLI `llm-config local --add PATH` should call `discover.add_file(store, path)`.
  `--dir DIR` should pass `extra_dirs`.
- **api/app**: merge `discover.local_variants(store)` into the variant list so users can size and run
  their own GGUFs. Their `source` is `"local"`, and they must never be downloaded.
- **downloads/api**: before downloading, call `discover.find_for_variant(store, variant)`. If it returns
  a path, reuse it (`{"reused": true}`). Downloads that land in `models_dir/<filename>` are found
  automatically.
- **ui-quality (`LocalModels`)**: `match_note`, `names`, `repo`, `gguf.quant` and `size_bytes` are good
  labels. The two sentences to show are "Likely the same file … not yet verified" and "Same file as the
  catalogue model".
- **lead**: consider adding `mistral3`/`llama4`/`deepseek2` to `domain.ARCHITECTURES` later. Adding them
  to `gguf.ARCH_MAP` is then a one-line change.

## How I tested

- `python3 -m unittest discover -s tests`: the full suite passes.
  - `tests/test_gguf.py` has 17 tests. They use a small pure-Python GGUF writer inside the test file.
  - `tests/test_discover.py` has 19 tests.
- **GGUF reader, checked both ways against the official `gguf` package** (installed only in a scratch
  folder, not a dependency):
  - Files from our test writer load in the official `GGUFReader`. There is an optional test that runs
    only when that package is installed.
  - A file written by the official `GGUFWriter` (qwen3moe with a per-layer head list, a token list and a
    nested array) reads correctly with our reader. Its expert fraction matches a hand calculation.
- **GGUF reader, other checks:**
  - Covers every value type, v2 and v3, and per-layer KV heads.
  - Covers sliding-window patterns (Gemma 3 and gpt-oss defaults, and the pattern stored in the header).
  - Covers MoE expert bytes and parameters, split files (missing part, passing any part), and arch mapping.
  - Rejects bad input: corrupt magic, v1, v9, big-endian, truncated files, unknown types, and hostile
    counts (2^40 settings, 2^62-byte strings, 2^60-item lists, deep nesting, 2^50 tensors, 1000 dims). All
    fail in under 2 s.
  - Enforces the header byte cap, and handles string lists that cross read chunks.
- **Discovery:** tests use fake HF, Ollama and LM Studio folders under a temporary `HOME` (`_home`
  patched) and patched env variables.
  - **Hugging Face:** snapshot→blob links, the same blob in two snapshots, and Windows-style copies with
    no symlinks.
  - **Ollama:** two tags sharing one blob, plus malformed, huge, traversal and missing-blob manifests.
  - **LM Studio:** a custom `downloadsFolder` with `~`, and the home pointer.
  - **Symlinks:** links out of the root are refused, both as file links and linked folders. A HF link
    that points anywhere other than its own `blobs/` is refused.
  - **Split files:** complete, incomplete and orphaned groups.
  - **Matching:** by sha, by name and size, and "same name but different hash is not a match".
  - **Scanning:** a scan never calls `hashlib.sha256`. Covered: cancel, the depth limit, and rescans
    reusing `hash_cache`.
  - **`hash_cached`:** cache hits, mtime changes, cancel, a file that changes while being hashed, and the
    size bound.
  - **`find_for_variant`:** sharded verification (a corrupted part with the same size is rejected),
    `verify=False`, no rehash for HF blobs, likely matches promoted to verified, a `../` filename that
    stays inside `models_dir`, and local-variant round trips.
  - **OS branches:** Windows and macOS are checked through `_system` mocks (no `/usr/share/ollama`,
    case-insensitive dedupe).

## Facts assumed

These should be confirmed by the harness:

- **`llama_ftype` values** match current `include/llama.h` (checked 2026-09 on master, including 38
  MXFP4_MOE, 39 NVFP4, 40 Q1_0, 41 Q2_0). ggml type block sizes match `gguf-py` `GGML_QUANT_SIZES`.
- **GGUF layout:** magic `GGUF`, then a uint32 version, then uint64 tensor count and uint64 KV count (v2+).
  Strings are a uint64 length followed by bytes. An array is a uint32 item type, a uint64 count, then
  the items. A tensor info entry is name, uint32 n_dims, uint64 dims, uint32 type, uint64 offset.
- **llama.cpp's sliding-window patterns:**
  - Gemma 2: every 2nd layer uses full attention.
  - Gemma 3: every 6th layer uses full attention.
  - gpt-oss: every 2nd layer uses full attention; the others use the 128-token window.
  - A layer is sliding when `layer % n < n - 1`.
- **Split GGUF:** the first part holds the model metadata, every part has its own tensor table, and
  llama.cpp loads the first part (`-00001-of-0000N.gguf`).
- **Hugging Face cache:** `HF_HUB_CACHE` wins, then `HF_HOME/hub`, then `$XDG_CACHE_HOME/huggingface/hub`,
  then `~/.cache/huggingface/hub` (Windows too). LFS files are `snapshots/<rev>/<file>` →
  `blobs/<sha256>`. Windows without symlink rights stores plain copies.
- **Ollama:** models live in `OLLAMA_MODELS`, or `~/.ollama/models`, or on Linux service installs
  `/usr/share/ollama/.ollama/models`. Manifests are `manifests/<registry>/<namespace>/<model>/<tag>`.
  The layer with `mediaType` `application/vnd.ollama.image.model` points to `blobs/sha256-<hex>`, which
  is a plain GGUF file llama.cpp can load.
- **LM Studio:** models live in `~/.lmstudio/models/<publisher>/<repo>/*.gguf` (older installs use
  `~/.cache/lm-studio/models`). A custom folder is the `downloadsFolder` key in `~/.lmstudio/settings.json`.
  LM Studio does not expand `~` in that key; we do.
