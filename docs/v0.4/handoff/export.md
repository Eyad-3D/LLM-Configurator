# Handoff: `export`

Branch `claude/v04-export`. Files: `src/llm_configurator/export.py`, `tests/test_export.py`, this file.

## What I built

`export.py` turns a launch config (§2.2) into ready-to-use files for other tools:

| id | filename | what it is |
|---|---|---|
| `llama-server` | `start-llama-server.sh` / `.ps1` | Start script. The argv is exactly `launch.server_args(config)`. |
| `ollama` | `Modelfile` | `FROM` + `num_ctx`, `num_gpu`, `num_thread`, `num_batch`; server env vars and limits go in `notes`. |
| `docker-compose` | `docker-compose.yml` | Official `ghcr.io/ggml-org/llama.cpp` server image, model folder mounted read-only, port on `127.0.0.1` only. |
| `openai-python` | `chat_local.py` | OpenAI client snippet. |
| `continue` | `continue-config.yaml` | A complete, valid Continue `config.yaml` (`schema: v1`) with one model. |
| `open-webui` | `open-webui.txt` | Connection details and steps. |
| `lmstudio` | `lmstudio-settings.txt` | Settings for the LM Studio load panel, using its exact labels. |

Safety and honesty:
- **bash:** every token goes through `shlex.quote`.
- **PowerShell:** tokens are single-quoted. All four quote characters PowerShell treats as single quotes (`' ‘ ’ ‚ ‛`) are doubled. The script runs through the call operator `&`. Only plain flags (`--x-y`) and `[A-Za-z0-9_]+` words stay unquoted. Anything else is quoted, because PowerShell splits words like `-x.y` and `127.0.0.1` in odd ways.
- **Compose:** YAML scalars are JSON double-quoted strings, with `$` doubled to `$$` so compose does not substitute variables. Mounts use the long `type: bind` syntax, so Windows drive letters and colons are safe.
- **Line breaks:** model names are reduced to one line before going into comments. Paths with control characters or line breaks raise a plain `ValueError`.
- **Deterministic:** no timestamps or randomness.

## Public API as built

```python
formats() -> [{"id", "label", "description"}]          # 7 entries, order as in the table above
export(config, variant, fmt, platform="posix"|"windows", server_command=None)
    -> {"format", "filename", "content", "instructions": [2–5 plain str], "notes": [str]}
ps_quote(text) -> str                                    # PowerShell single-quoted literal (helper, public for reuse)
PLACEHOLDER = "<path-to-model.gguf>"
```

- `config`: a launch config. It may have `model_path=None` (not downloaded yet). In that case the output uses `<path-to-model.gguf>` and a note explains it. For docker the placeholders are `<folder-containing-the-model>` and `/models/<variant.filename>`. For Ollama with a split model it is `<path-to-merged-model.gguf>`.
- `variant`: a `domain.Variant`, or `None`. It supplies the title, the alias fallback, the filename and split detection.
- `server_command`: an argv prefix as `str` or `list`, e.g. from `runtime_install.binary`. It defaults to `llama-server` on `PATH`, with a note saying so.
- Raises `ValueError` for: an unknown format, an unknown platform, a path with control characters, and a Modelfile path containing `"`.

## Deviations and decisions

- **Port and alias get filled in when missing.**
  - If `port` is `None`, the export uses `8080` (llama-server's default) and adds a note (only for formats that use a port).
  - If `alias` is `None`, it uses a slug of `"<name>-<quant>"`, so client snippets have a stable model name.
  - These two flags are the only difference from the tested args. They don't affect speed or memory.
- **Docker post-processing.**
  - Only for docker, the `--host` value is changed to `0.0.0.0`, and model/draft paths are rewritten to `/models/<file>` and `/draft/<file>`.
  - Reason: the port mapping `127.0.0.1:P:P` means only this computer can reach it, but inside the container the server must accept traffic coming from Docker's bridge. This is explained in `notes`.
  - The image picks backend by `gpu_backend` when `gpu_layers > 0`: CUDA → `server-cuda` with `deploy.resources.reservations.devices`, using `device_ids: [uuid]` when a UUID is known and otherwise `count: all`. Vulkan → `server-vulkan` + `/dev/dri`. ROCm → `server-rocm` + `/dev/kfd`, `/dev/dri`, `group_add: video`. Metal / CPU → plain `server`, and Metal gets a note that Docker cannot use Metal and the native script is better.
- **Ollama and split GGUFs.**
  - The Modelfile always points at a merged file and includes the exact `llama-gguf-split --merge <first shard> <merged>` command, quoted for the platform.
  - Current Ollama docs say newer versions accept one `FROM` line per shard (or a wildcard). We still merge, because that works on every version; a note mentions the alternative.
- **Ollama settings mapping.**
  - `num_ctx` = the context per user. `parallel > 1` becomes `OLLAMA_NUM_PARALLEL`.
  - `num_gpu` = `launch.runtime_gpu_layers(...)`, the same count as `-ngl`.
  - The KV cache type is global in Ollama (`OLLAMA_KV_CACHE_TYPE`). If K ≠ V we use `q8_0` and say so.
  - Flash attention becomes `OLLAMA_FLASH_ATTENTION` (forced to 1 when the KV cache is compressed).
  - These go into instructions as `KEY=VAL ollama serve` (posix) or `setx` (Windows).
  - `n_cpu_moe`, the draft model, `ubatch`, mlock/mmap and GPU pinning are listed in `notes` as things Ollama can't express.
  - Model names are sanitised to `[a-z0-9._-]`, at most 80 characters, starting and ending with a letter or digit.
- **Continue** uses `provider: openai` with `apiBase …/v1`, not `provider: llama.cpp`. That keeps the same base URL as every other client snippet. Continue documents both.
- **Open WebUI:** on Linux, `host.docker.internal` cannot reach a server bound to `127.0.0.1`. The posix note therefore tells users to run Open WebUI with `--network=host`.
- **PowerShell and non-ASCII:** a script containing non-ASCII characters gets a note to save it as UTF-8 with BOM, because Windows PowerShell 5.1 reads BOM-less files as ANSI.

## Known gaps

- Split models in docker/llama-server rely on llama.cpp finding the other shards next to the first. The whole folder is mounted, so this works.
- LM Studio's GPU Offload is shown as `gpu_layers of total_layers` transformer layers. LM Studio's slider counts are assumed to match.
- No Open WebUI "Direct Connections" (per-user) variant, only the admin connection.
- The Windows docker output assumes Docker Desktop; Vulkan/ROCm only work on Linux, and the notes say so.

## Requests

- **lead (`launch.py`):** current llama.cpp **removed `--draft-max`** (the server README says to use `--spec-draft-n-max N`). `-md` still works as an alias of `--spec-draft-model`. `server_args` should emit `--spec-draft-n-max` (or whichever the harness confirms), or fall back by runtime version. Export follows automatically.
- **api:** when writing a `.ps1` to disk (CLI `export`), write it with `encoding="utf-8-sig"` so Windows PowerShell 5.1 reads non-ASCII paths correctly. Set `.sh` files to mode 0755. For `/api/export`, pass `model_path=None` when the model isn't downloaded (export handles it), and pass `server_command=runtime_install.binary(store, "llama-server")` when installed.
- **ui-run:** show `instructions` as a numbered list and `notes` below the copy box. `filename` is a good suggestion for a "Download" button.

## How I tested

`tests/test_export.py` has 33 tests; the full suite (`python3 -m unittest discover -s tests`) passes with 105 tests. Coverage:

- Exact snapshots: bash and PowerShell scripts, the Modelfile and the CPU compose file.
- Every format × both platforms: deterministic, complete keys, 2–5 instructions, no dates.
- Round-trips on hostile paths (spaces, `'`, `"`, `$`, backticks, `$(...)`, `;`, `&`, typographic quotes, é and 模型):
  - bash through `shlex.split`.
  - PowerShell through a small lexer in the test that only accepts safe bare words and single-quoted strings.
  - In both cases the parsed argv must equal `launch.server_args(config)`.
- PowerShell quoting unit tests.
- MoE, KV and draft configs across formats.
- Placeholder path for every format and platform, and `variant=None`.
- Split models, detected by filename and by `variant.files`.
- Compose parsed with PyYAML when available (skipped otherwise): ports, read-only mounts, the `0.0.0.0` rewrite, `$$`, CUDA device reservation, Windows paths.
- The Python snippet parsed with `ast`, and the Continue YAML parsed.
- Model names containing line breaks can't inject script lines.

## Facts assumed (please confirm in the harness)

Checked against upstream docs and sources in Sept 2026 unless marked otherwise:

- **llama-server flags:** `-fa on|off|auto`, `--n-cpu-moe N`, `-md`, `--jinja` (now on by default, so passing it is harmless), and `-dev none`. `--draft-max` is **removed** in current builds (see Requests).
- **Docker image:** `ghcr.io/ggml-org/llama.cpp:server|server-cuda|server-vulkan|server-rocm` has entrypoint `/app/llama-server`, so args are appended directly. The image also sets `LLAMA_ARG_HOST=0.0.0.0`; we pass `--host 0.0.0.0` explicitly anyway.
- **Docker devices:** compose GPU reservations need `capabilities: [gpu]`, and `device_ids` accepts NVIDIA GPU UUIDs. **Unverified:** that the Vulkan image needs `/dev/dri`, and that ROCm needs `/dev/kfd` + `/dev/dri` (llama.cpp docs don't say; Ollama's docs do for its own images).
- **`llama-gguf-split --merge IN_FIRST_SHARD OUT`:** verified.
- **Ollama:**
  - `FROM` accepts absolute or relative GGUF paths, and `num_ctx`, `num_gpu`, `num_thread`, `num_batch` are accepted PARAMETERs (verified in source).
  - `OLLAMA_KV_CACHE_TYPE` is global, and `OLLAMA_FLASH_ATTENTION` accepts 0/1 (verified).
  - There is no `n_cpu_moe` equivalent (verified).
  - **Unverified:** that `FROM "path with spaces"` (plain double quotes, no escapes) is parsed correctly.
- **Continue:** the `config.yaml` top-level `name`, `version`, `schema: v1` and model fields `provider`, `model`, `apiBase`, `apiKey`, `roles` are verified. **Unverified:** `defaultCompletionOptions.contextLength`.
- **Open WebUI:** the Admin → Settings → Connections → OpenAI → Add Connection path and `host.docker.internal` are verified. **Unverified:** the Linux `--network=host` advice (standard Docker behaviour).
- **LM Studio:** labels are verified from LM Studio's localization `en/config.json`. **Unverified:** the "Manually choose model load parameters" toggle wording.
