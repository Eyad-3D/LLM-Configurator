# Fix-up: `discover` (`gguf.py`, `discover.py`)

Branch `claude/v04-fix-discover`. Files touched: `src/llm_configurator/gguf.py`, `src/llm_configurator/discover.py`,
`tests/test_gguf.py`, `tests/test_discover.py`, this file. Nothing else.

## Fixed

### `gguf.py`

- **`summary["vocab_size"]`** (asked for by quantcheck). This is the length of `tokenizer.ggml.tokens`, read from the list's header while the tokens themselves are skipped. If the file has no token list, it falls back to `<arch>.vocab_size`, otherwise `None`. It matches llama.cpp's `n_vocab` on every real file (see the table below).
- **Weights past the end of the file.** `read_metadata(tensors=True)` and `variant_from_file` now reject a file whose weight table points past its end. llama.cpp reports this as "data is not within the file bounds". Before the fix, `corrupt.gguf` (a download cut in half) was accepted as a working local model. The header-only `read_metadata()` still reads it, so a scan can still show what the file is.
- **Head size when head counts differ per layer.** The fallback `embedding_length / head_count` now uses layer 0's head count, the same as llama.cpp's `n_head()`. It used the largest count, which gives a head size that is too small.
- **Head size when `value_length` > `key_length`.** The larger of the two is used, because `head_dim` stands for both. Memory is never under-counted.
- **Sliding-window defaults now match llama.cpp's `src/models/*.cpp`:**
  - `olmo2` uses a pattern of 4.
  - `gemma2` uses a window of 4096 when the header has none.
  - A pattern of `0` means every layer slides (`set_swa_pattern(0)`).
  - Architectures with no built-in pattern (for example `phi3`, where llama.cpp turns sliding windows off) count every layer as full attention.
  - The count is now closed-form. A hostile `block_count` of 2^62 used to hang the `range()` loop.
- **Tensor-type table checked against `ggml.h` / `ggml-common.h`.** Added `Q2_0` (type 42: 64 values in 18 bytes). Q8_1's block is 36 bytes, not 40. gguf-py still says 40, but ggml now stores two halves. `FILE_TYPES` matches `llama.h` exactly (0–41, with 33–35 removed). The `LLAMA_FTYPE_GUESSED` flag (1024) is stripped.
- **Split files:**
  - The header's `split.count` must agree with the file names. A renamed shard can't load, so it now gets a plain error.
  - A `sha256` passed with part 2+ is no longer stored as part 1's fingerprint.
  - Shard numbers must be ASCII digits (`\d` also matched Arabic-Indic digits).
- **Local ids no longer depend on the hash.** `local_id(path)` now comes from the resolved path only. Before, a scan (no hash yet) and `local --add` (hashed) gave the same file two different ids, so the model appeared twice once both lists were merged.
- **Robustness (from the adversarial review):**
  - A FIFO or device named `*.gguf` used to block forever on open. It is now refused.
  - OS read errors become `ValueError`.
  - A work budget of 2M skipped list items stops a header of millions of empty nested lists; one 64 MiB example took 11 s before.
  - A budget of 262k kept array numbers stops about 600 MiB of memory use from small arrays.
  - `MAX_TENSORS` drops from 1M to 100k (real models have a few thousand).
  - Values no real model has are treated as unknown (`None`), never as numbers: more than 4096 layers, 65536 heads, head size 65536, context 2^30, window 2^30, or 65536 experts.

### `discover.py`

- **Symlink loops.** `a.gguf → b.gguf → a.gguf` used to crash the whole scan with `RuntimeError` on Python 3.10–3.12. All `resolve()` calls now go through `_resolve()`, which returns `None` for a loop.
- **Linked cache folders are no longer followed.** A Hugging Face `snapshots/` or `blobs/`, or an Ollama `manifests/` or `blobs/`, that is itself a symlink is skipped. Neither app ever creates such links. Before the fix, `snapshots -> /` walked the whole disk (30 s, 20k files reported as `hf_cache`). `blobs -> /etc` also got past the "own blobs only" check.
- **`hash_cached`** refuses non-regular files (a pipe never ends) and turns OS read errors into a plain `ValueError`.
- **A local variant resolves only to its own file.** Before, a same-named, same-sized file in the models folder counted as the user's local model.
- **Flat downloads are found.** `downloads` saves `Q4_K_M/x.gguf` as `models/x.gguf`, but discover only looked in `models/Q4_K_M/x.gguf` (api-cli review #1). It now tries the flat name first, then the nested one.
- **New `remember_hash(store, path, sha256)`.** Callers that have just verified a file (a finished download) can seed `hash_cache`, so the first test or run doesn't re-read gigabytes (api-http review #8). The cached hash is valid only while the file's size and mtime stay the same.
- **`$VAR` inside `HF_HOME` / `HF_HUB_CACHE` / `OLLAMA_MODELS` is expanded,** as `huggingface_hub` does.

## Checked and left as is (rejected, with reasons)

- **`kv_heads` for per-layer lists is the largest value.** llama.cpp prints `[4, 2, 1, 2]`, and we report 4. A single number has to stand for all layers, and the largest never under-counts memory.
- **Blob names are trusted as SHA-256 without re-hashing.** This is by design. Hugging Face and Ollama both check a blob's content against its name when they download it, and `find_for_variant` checks the link still resolves to the blob with that name. Re-hashing every blob would break "never hash gigabytes during a scan". A blob someone edits by hand afterwards is not detected. That is written down here, not hidden.
- **Ollama digests in uppercase hex, and blob files named `sha256:<hex>`, are not found.** Ollama writes lowercase digests, and it renamed `sha256:` files to `sha256-` years ago on startup. Too rare to be worth the extra code.
- **Hugging Face blob names with 40 hex characters (git SHA-1, for non-LFS files)** are never taken as SHA-256, because `HEX64` needs exactly 64. Confirmed by the reviewer.
- **Folder layouts were checked against upstream sources:**
  - `huggingface_hub/constants.py` for the order `HF_HUB_CACHE` > `HUGGINGFACE_HUB_CACHE` > `$HF_HOME/hub` > `$XDG_CACHE_HOME/huggingface/hub` > `~/.cache/huggingface/hub`, on all OSes.
  - Ollama `docs/faq.mdx` and `manifest/paths.go` for the defaults per OS, `OLLAMA_MODELS`, and `manifests/<host>/<ns>/<model>/<tag>` pointing to `blobs/sha256-<hex>`.
  - lmstudio-js `findLMStudioHome.ts` for the home pointer, else `~/.cache/lm-studio` if it exists, else `~/.lmstudio`, plus `downloadsFolder` in `<home>/settings.json`.
  - The code already matched all of these.
- **Scan speed:**
  - 100k non-GGUF files take 0.14 s.
  - 20k small `.gguf` files take 3.8 s.
  - A sparse 20 GB `.gguf` is scanned in 0.03 s, because only the header is read.
  - No hashing happens during a scan (tested).

## Requests

- **api-cli (`app.py`)**
  1. `variants()`: also merge `discover.local_variants(store)`, and skip ids already known. Scanned GGUFs that match no catalogue model can't be used at all right now (reviews: api-cli #2, api-http #9). Ids now agree between a scan and `local --add`, so deduping by id is enough.
  2. `add_local_job`: start with `path = gguf.shard_paths(path)[0]` so that llama-server gets the first shard; llama.cpp refuses any other part with "illegal split file idx". Also drop the `.gguf`-suffix check, because Ollama blobs have no suffix and `gguf` rejects non-GGUF files anyway. The simplest fix is to delegate to `discover.add_file`.
  3. `local_model` / `require_model`: pass `progress=` and `cancel=` through to `discover.find_for_variant`. Also drop the size-only `downloads.plan` fallback when `verify=True`, since discover now finds flat downloads itself.
  4. `download_job`: after `download_variant` returns, call `discover.remember_hash(store, path, f["sha256"])` for each file that had a published hash. The first test then doesn't re-read the model.
- **api-cli (`cli.py`)**
  1. Use `discover.locations(store, dirs)` for both listings (lines 450 and 452), and resolve `--dir` paths first.
  2. Print the `local_variant_id` as the Model ID when `variant_id` is None.
  3. Print paths with `errors="replace"`. A non-UTF-8 file name can raise `UnicodeEncodeError` on a strict stdout.
- **api-http (`server.py:259`)**
  1. Call `discover.locations(store)`, so "this app's models folder" is listed.
  2. Consider adding `local_variant_id` (an id, not a path) to `page_file`.
- **quantcheck:** `_vocab_size` can read `gguf.read_metadata(path)["summary"]["vocab_size"]`. It matches llama.cpp's `n_vocab` exactly.
- **downloads:** unchanged. You can call `discover.remember_hash` yourself instead of api doing it, if you prefer.
- **lead:** as before, adding `mistral3`, `llama4`, `deepseek2`, `gemma3n` and `cohere2` to `domain.ARCHITECTURES` is then a one-line `ARCH_MAP` change. Their sliding-window patterns are in `src/models/*.cpp`.

## API changes (all additive except `local_id`)

- `summary["vocab_size"]: int | None` (new key).
- `gguf.local_id(path)`: the `sha256` argument was removed, and ids are always path-based. Nothing outside `gguf.py` called it.
- `discover.remember_hash(store, path, sha256) -> None` (new).
- `variant_from_file` raises `ValueError` for a file cut short, a split-count mismatch, or a non-regular file.

## Evidence

### Real files: `gguf.py` compared with llama.cpp

llama.cpp was built from the llama-cpp-python 0.3.35 sdist (build 4df29be). Its values come from `llama-server -m FILE -lv 4 -fit off` (`print_info:` lines). The files are the tiny models from `scripts/make_tiny_models.py`, plus four header-only files written with gguf-py for architectures the tiny set lacks.

| file | ours: arch / layers / kv_heads / head_dim / ctx / experts (active) / quant / vocab / window | llama.cpp: n_layer / n_head_kv / n_embd_head_k / n_ctx_train / n_expert (used) / file type / n_vocab / n_swa | params ours vs llama.cpp |
|---|---|---|---|
| tiny-llama-F16 | llama / 4 / 4 / 64 / 16384 / 0 / F16 / 422 / – | 4 / 4 / 64 / 16384 / 0 / F16 / 422 / 0 | 6,728,192 vs 6.73 M |
| tiny-llama-Q8_0 | llama / 4 / 4 / 64 / 16384 / 0 / Q8_0 / 422 / – | 4 / 4 / 64 / 16384 / 0 / Q8_0 / 422 / 0 | 6,728,192 vs 6.73 M |
| tiny-llama-Q4_K_M | llama / 4 / 4 / 64 / 16384 / 0 / Q4_K_M / 422 / – | 4 / 4 / 64 / 16384 / 0 / Q4_K - Medium / 422 / 0 | 6,728,192 vs 6.73 M |
| tiny-qwen3moe-F16 | qwen3moe / 4 / 4 / 64 / 16384 / 8 (2) / F16 / 422 / – | 4 / 4 / 64 / 16384 / 8 (2) / F16 / 422 / 0 | 16,182,272 vs 16.18 M |
| tiny-qwen3moe-Q4_K_M | qwen3moe / 4 / 4 / 64 / 16384 / 8 (2) / Q4_K_M / 422 / – | 4 / 4 / 64 / 16384 / 8 (2) / Q4_K - Medium / 422 / 0 | 16,182,272 vs 16.18 M |
| split …-00001-of-00004 | llama / 4 / 4 / 64 / 16384 / 0 / Q8_0 / 422 / split 4 | 4 / 4 / 64 / 16384 / 0 / Q8_0 / 422 / 0 | 6,728,192 (all 4 parts) vs 6.73 M |
| split …-00002..4-of-00004 | header has no model keys; `variant_from_file` resolves to part 1 | "illegal split file idx: 1/2/3" (must load part 1) | – |
| hdr-gemma3 | gemma3 / 12 / 2 / 128 / 8192 / 0 / Q4_K_M / 300 / 512 (10 of 12 layers sliding) | 12 / 2 / 128 / 8192 / 0 / Q4_K - Medium / 300 / 512, is_swa_any=1 | 76,800 vs 76.80 K |
| hdr-gpt-oss | gpt-oss / 6 / 1 / 64 / 8192 / 32 (4) / MXFP4 / 300 / 128 (3 of 6) | 6 / 1 / 64 / 8192 / 32 (4) / MXFP4 MoE / 300 / 128, is_swa_any=1 | 76,800 vs 76.80 K |
| hdr-qwen3 (key_length 128 ≠ 256/4) | qwen3 / 3 / 2 / 128 / 8192 / 0 / F16 / 300 / – | 3 / 2 / 128 / 8192 / 0 / F16 / 300 / 0 | 76,800 vs 76.80 K |
| hdr-llama-perlayer | llama / 4 / **4** / 64 / 8192 / 0 / Q8_0 / 300 / – | 4 / **[4, 2, 1, 2]** / 64 / 8192 / 0 / Q8_0 / 300 / 0 | 76,800 vs 76.80 K |
| corrupt.gguf | header reads; `variant_from_file`: "weights run past the end of the file … incomplete" | "tensor 'blk.1.ffn_down.weight' data is not within the file bounds" | – |
| unknown-arch.gguf | `variant_from_file`: "uses the 'notarealarch' design, which LLM Configurator cannot size yet" | "unknown model architecture: 'notarealarch'" | – |

Other checks:

- **MoE expert share, checked against the official `gguf.GGUFReader`** (bytes in `*_exps` tensors ÷ all tensor bytes): F16 0.7765 vs 0.7765, Q4_K_M 0.7306 vs 0.7306. Parameter counts are identical.
- **Sliding layers checked against llama.cpp's `set_swa_pattern`** (`il % n < n - 1`): gemma3 (n=6, 12 layers) → 10, gpt-oss (n=2, 6 layers) → 3.
- **`discover.scan(store, extra_dirs=("/opt/llama-work/models",))`** ran in 0.021 s:
  - 8 records, and the 4-part split is one complete record with 4 files.
  - `corrupt.gguf` has a `local_error` (cut short) and `unknown-arch.gguf` has a `local_error` (design).
  - Every other file gets a `local_variant_id`.
- **`tests.integration.test_real_runtime` with the real binaries:** 23 of 24 pass. The one failure is `runtime_install` build parsing, which is already known and not in this area.

### Adversarial reviews (two subagents)

- **GGUF robustness** covered:
  - every truncation prefix of a valid file
  - counts and lengths of 2^63 and 2^64−1
  - nesting depths of 4, 5 and 50000
  - non-UTF-8 keys, and wrong types for `general.architecture`
  - NaN and negative values
  - `/dev/zero` behind a symlink
  - Unicode digits in shard names
  - a 262k-token vocabulary with 150k merges (0.22–0.26 s)

  It found 4 bugs: the FIFO hang, the nested-list slowness, the memory blow-up from kept arrays, and the unbounded tensor count. All are fixed above.
- **Scan safety and layouts** covered link loops, links to `/`, cross-repo HF links, Ollama path traversal, FIFOs, folders named `x.gguf`, depth, 100k files and a sparse 20 GB file. It found the symlink-loop crash and the linked `snapshots`/`blobs`/`manifests` escapes, and both are fixed. Layouts were checked against the upstream sources listed above.

### Tests

- `tests/test_gguf.py`: 23 tests, 7 of them new or extended:
  - vocab size
  - split fingerprint and count
  - Q2_0 and the GUESSED flag
  - a huge `block_count`
  - a named pipe
  - weights cut short
  - an id that does not depend on the hash
- `tests/test_discover.py`: 25 tests, 6 of them new:
  - a local variant only resolves to its own file
  - scan and add agree on the id
  - link loops and linked cache folders
  - a pipe in `hash_cached`
  - flat downloads
  - `remember_hash`

  The link-loop and linked-folder test hangs or crashes on the old code.
- **Full suite:** `python3 -m unittest discover -s tests` runs 616 tests. The only failures are the 2 already known in `integration.test_fake_matches_real`: the fake accepts `--draft-max` and answers `llama-bench --version`. Both are owned by server-fake.
