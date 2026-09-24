# Fix-up: `community-export`

Branch `claude/v04-fix-community-export`. Files: `community.py`, `community/README.md`, `export.py`,
`tests/test_community.py`, `tests/test_export.py`, `.github/ISSUE_TEMPLATE/community-results.yml` (new).

Three review agents attacked the code (privacy, import and evidence, export). Their findings were checked against the real shapes from
`hardware.py`, `testing.py`, `runtime.py` and `engine.py`, and against a real llama.cpp build (commit 4df29be) and real
PowerShell 7.6.6.

## Fixed

### Sharing (issue form)
- New `.github/ISSUE_TEMPLATE/community-results.yml` (issue form, `labels: [community-results]`, a `results` JSON
  textarea and a "nothing personal" checkbox). Issue forms apply their label for everyone, including people without triage rights.
- `issue_url` now links `issues/new?template=community-results.yml&title=…&labels=…&results=<json>`. Issue forms
  fill fields from query parameters named after the field `id`. They ignore `body`, so `body` is no longer sent. When the JSON is
  too long for a link, `results` is left out and the form asks for a paste. A test keeps the template's field id and label in step
  with the code.

### Privacy (`anonymize`, `clean_name`, `share_payload`)
- **Old custom models could publish a private repo name.** `Variant.source` defaults to `"catalogue"`, and `app.py`
  re-saves old variants through `Variant(**v)`, so pre-v0.4 user-added repos became "catalogue". Now a repo or file name is
  shared only if the source is `catalogue` **and** the repo is in the packaged `catalogue.json`.
- **Names people typed stayed in hardware names** when they did not match this machine's name (VMs, custom strings). All-lowercase
  words (`zorblax-laptop`, `qwertyuser`) are now dropped. Vendors don't use them, except `rev` and `with`. Serial detection is
  stricter: 4 or more letter/digit switches in a token of 6+ characters (`ZX9K2LQ`), which real model numbers such as `7950X3D` and
  `i9-14900K` don't reach.
- **Over-scrubbing**: a default Mac host name `Johns-MacBook-Pro` turned `Apple M3 Pro` into `Apple M3`, and `16-Core`
  lost `Core`. Hardware words (`pro`, `max`, `air`, `core`, `laptop`, …) are no longer treated as personal words, and substring
  matching needs at least 5 characters.
- **Runtime fields were never right for real runs.** `testing.py` stores `str(build_number)` and llama-bench's `backends`
  (`"CUDA"`, `"Metal,BLAS"`). Versions are now shared as `b<N>`. Build 1 is dropped because it is the sdist/no-history artefact
  (llama-cpp-facts §2), and builds of 10000 or more now fit. Backends are lowercased (`BLAS` → `cpu`, `HIP` → `rocm`). Commit hashes
  are still never shared.
- The month is taken in UTC (`2026-10-01T00:30+05:30` → `2026-09`).
- `users > 64` no longer silently becomes 1. That result is refused.
- `share_payload` now **skips and counts** (new key `skipped`) results whose fingerprint differs from this computer's, for
  example measured before a driver or kernel update. Before, one old result failed the whole batch. It still raises
  if nothing is left. A missing fingerprint on either side counts as a mismatch.

### Import (`parse`, `import_records`, `_fetch`)
- **A bad download wiped saved results.** A 200 page like `{"message":"Not Found"}`, a file in a newer format
  (`{"schema":2,…}` with no `records`), or a file where every row fails used to save an empty list. Now `schema > 1` always
  gives the "newer format, update" message, and an import that keeps 0 of more than 0 rows raises and keeps the saved results. An
  intentionally empty file still imports.
- A UTF-8 BOM (pasted from Windows editors) is accepted. Non-UTF-8 bytes and `http.client` protocol errors
  (`BadStatusLine`, `IncompleteRead`) now give the plain-language message instead of a raw exception.

### Evidence (`evidence` as called by `engine.community_speed`)
- **Hand-edited store rows could crash `recommend`.** `hardware: "x"` raised `AttributeError`, which `community_speed` does
  not catch. `tps: true` counted as 1 tok/s. Rows now need dict sections, a real int/float `tps` and an int `context`.
- **MoE with experts on the CPU matched full-GPU results.** `gpu_layers == total` with `n_cpu_moe > 0` is now `split` (the engine
  labels it `split` too) in both `anonymize` and `evidence`, and `n_cpu_moe` must match exactly.
- **Wrong GPU without a UUID.** The engine only sets `gpu_uuid` when the GPU has one, and matching fell back to `gpus[0]` (e.g.
  the Intel iGPU instead of the RTX 4090). The GPU is now chosen by UUID, else as the largest GPU of the config's `gpu_backend`.
  `anonymize` does the same using the run's backend.
- Repo matching ignores case (Hugging Face repo names are case-insensitive).

### Export
- **A non-first shard path** (`x-00002-of-00003.gguf`) made llama-server fail ("model must be loaded with the first split")
  and broke the merge command. It is now rewritten to `-00001-of-N`, with a note.
- A split `variant` with an already-merged `model_path` no longer says "merge first". When a real path is given, the file name decides.
- The Ollama merge command uses `llama-gguf-split` from the same folder as the known `llama-server` (quoted; `& '…'` in
  PowerShell), and otherwise the bare name.
- `alias`, `device` and `gpu_uuid` with control characters (`\r` slipped past `launch.normalize`, which only checks `\n`) are
  refused before they reach text files. A non-string path gets a "must be text" error instead of "contains line breaks".
- PowerShell scripts refuse any argument containing `"`. Windows PowerShell 5.1's legacy argument passing splits such an
  argument (Windows paths cannot contain `"`, but an alias could). bash scripts are unaffected.
- Docker refuses relative paths (`m.gguf`, `C:m.gguf`). They would have mounted the compose folder or `C:` instead of the
  model's folder. A UNC path gets a note that Docker Desktop cannot mount network shares.
- The K≠V Ollama note now says quality and memory can differ from the tested run.

## Rejected (with reasons)
- **Stop sharing the sha256 of local or custom models.** It is in the contract allowlist. It only matches people who have the identical
  file, and without it private-model results are useless. The README already says it is shared.
- **Match evidence on KV cache type.** Tried it. It splits a sparse community pool for a speed effect that sits inside the
  reported quartile range. `n_cpu_moe` (a big effect) is matched instead.
- **Replace `--mlock`/`--no-mmap` in exports with `--load-mode`.** Export mirrors `launch.server_args` exactly, so the exported
  command is the tested one. Both flags still work (only a `DEPRECATED` warning, seen in the real logs below). See Requests.
- **`--%` as a whole model path in PowerShell.** It is PowerShell's stop-parsing token, but only a relative path named exactly
  `--%` hits it. The app always passes absolute paths.
- **A full vendor allowlist for hardware names.** It would drop new models the list doesn't know yet, and the user and a
  maintainer both review the JSON.

## Requests
- **lead (`launch.py`)**:
  - Emit `-lm/--load-mode` (`mlock`, `none`, `mmap+mlock`) instead of the deprecated `--mlock`/`--no-mmap`. Export follows
    automatically.
  - In `normalize`, reject every control character (not just `\n`) in `alias`, `device`, `gpu_uuid` and `draft_model_path`.
- **lead (`domain.py`)**: consider defaulting `Variant.source` to `None` (or migrating old user entries to `custom`).
  `community` no longer relies on the flag alone, but other code might.
- **runtime-downloads (`runtime.py`)**: `runtime.bench` measurements have no `id` or `kind`, so `cli bench` results can never be
  shared. Add `"id": secrets.token_hex(6), "kind": "bench"`, plus `"runtime": {"version": str(build_number), "backend": backends}`.
- **engine**:
  - Add `AttributeError` to the `except` in `community_speed` (belt and braces; `evidence` no longer raises it).
  - Pass `n_cpu_moe` in the config you give `evidence`. `_launch` already does.
- **api-http / ui**: `POST /api/community/share` returns a new `skipped` count. Show "N results were measured on
  different hardware and were left out" when it is above 0. With `fits_in_url: false`, the issue form opens empty and asks for a paste.
- **api-cli**: print `skipped` after `community share`. Write `.ps1` exports with `encoding="utf-8-sig"` and `chmod 755` `.sh`
  (from the export handoff, still open if not done).
- **docs**: mention the issue form in the contributing docs if they describe sharing.

## Evidence

### Unit tests
- `python3 -m pip install -e .`, then `python3 -m unittest discover -s tests` (with `pwsh` on PATH): **626 tests, 2 failures,
  24 skipped**. The 2 failures are the known fake-llama.cpp gaps from FIXUPS.md (`test_server_rejects_removed_draft_max_like_real`,
  `test_bench_has_no_version_flag_like_real`), owned by `server-fake`. `test_community` (54) and `test_export` (42) all pass.
- `tests.integration.test_real_runtime` against the real build: 24 tests, 1 failure. It is the known `runtime_install` version parsing
  (owned by `runtime-downloads`).
- `tests/test_export.py` now **runs the exported scripts** in real bash and, when `pwsh` is on PATH, in real PowerShell, with a
  stand-in server that prints its argv. Paths tested: spaces, `'`, `"`, `$HOME`, backticks, `;&|>!*{,}~`, é, 模型, UNC, `@(1)`,
  `#`, and a leading `-`. The printed argv equals `launch.server_args(config)`.

### Real llama.cpp
Real llama.cpp build 4df29be (`scripts/build_llama_cpp.sh`) with the tiny models, copied into a folder named
`My Models/it's $HOME \`x\` é 模型`. Each exported `llama-server` script was run with `server_command=[…/llama-server]`,
first in bash and then in PowerShell 7.6.6. For each run, the script polled `/health` and sent one chat:

| Case | bash | PowerShell |
|---|---|---|
| plain Q8_0 | answered `42` | answered `42` |
| draft model + q8_0 KV + mlock + no-mmap + 2 users | `42`, `draft_n: 4` (speculative decoding really on) | same |
| split model, given the **last** shard | `42` | `42` |
| qwen3moe, `--n-cpu-moe 2`, CPU | `42` | `42` |

The only warnings in the logs were the known `DEPRECATED: --mlock …` and `--mmap and --no-mmap …` lines.
The Ollama instructions' exact merge command (`…/llama-gguf-split --merge '<first shard>' '<merged>'`) ran through bash with
exit 0 and produced the merged file named in `FROM`.
The docker-compose `command` list (parsed with PyYAML, container mounts mapped back to host folders, `$$` → `$`) started the
real llama-server with `--host 0.0.0.0` and answered, with `draft_n: 4`. Ports were `127.0.0.1:P:P`. There is no Docker daemon in
this container, so the image itself was not run. Its entrypoint (`/app/llama-server`) was checked in the upstream
`.devops/cuda.Dockerfile`.

### External facts checked by the export reviewer
- **Ollama:** `num_ctx`, `num_gpu`, `num_thread` and `num_batch` are fields of `api/types.go` `Runner` options.
  `parser/parser.go` strips `"…"` in `FROM` without escapes, so rejecting `"` is right. The FAQ says context is multiplied by
  parallel requests, so `num_ctx` is per user.
- **llama.cpp:** a non-first shard is refused by `src/llama-model-loader.cpp`.
- **GitHub:** issue forms are prefilled by field id. Found by web search, because docs.github.com is blocked here.
