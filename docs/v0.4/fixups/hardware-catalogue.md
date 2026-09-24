# Fix-up: `hardware-catalogue`

Branch `claude/v04-fix-hardware-catalogue`, based on `claude/keen-volta-7mnrm5` (076bab8).
Files changed: `hardware.py`, `catalogue.py`, `tests/test_hardware.py`, `tests/test_catalogue.py`, `tests/test_adapters.py`, and this file. `catalogue.json` is unchanged.

Method: three review subagents, each looking from a different angle:
1. everyone who reads `scan()` output (engine, speed, learning, runtime_install, launch, runtime, calibration, community, tuner, testing, app, server, cli, export, static JS);
2. platform facts (Apple, AMD sysfs, Windows, fingerprint, speed), with sources;
3. the catalogue (repos, MoE arithmetic, sliding window, add/remove, refresh).

Findings were checked against the code before fixing.

## Fixed

### hardware.py

| # | Problem | Fix |
|---|---|---|
| H1 | **Apple default GPU limit was wrong for 36 GiB Macs.** The code gave 2/3 of RAM up to 36 GiB. Metal's `recommendedMaxWorkingSetSize` is 2/3 **up to 32 GiB** and 3/4 above. llama.cpp logs show 16 GiB → 10922.67 MiB, 32 GiB → 21845.34 MiB, 36 GiB → 27648 MiB (exactly 0.75). | Threshold is now `> 32 GiB`. There is a test for each logged value. |
| H2 | macOS 13 uses `debug.iogpu.wired_limit` (bytes), not `iogpu.wired_limit_mb`. | Both keys are read in one `sysctl` call. |
| H3 | If `sysctl` failed, an Apple Silicon Mac fell into the Intel-Mac branch ("cannot be read on Intel Macs"). | Falls back to `platform.machine() == "arm64"`. The limit is then the labelled `estimated_default`. |
| H4 | Under Rosetta the brand string is not the chip, so the GPU name and uuid were wrong. | If the name doesn't start with "Apple", the name comes from the `system_profiler` model instead. |
| H5 | **One nvidia-smi row with `[N/A]` memory made every NVIDIA GPU vanish.** This happens on GB10/DGX Spark. | Each row is read on its own. A row without memory goes to `other_gpus`. Good rows and the fingerprint are unchanged. |
| H6 | A slow or failing nvidia-smi was reported as "nvidia-smi was not found". | `nvidia_gpus()` now also returns whether the tool exists. Two different reasons: "did not answer" vs "is missing". |
| H7 | **The Windows adapter list was cached with the first scan's NVIDIA state.** One nvidia-smi timeout pinned the card in `other_gpus` for the whole run. | Only the raw registry list is cached. The NVIDIA filter runs on every scan. |
| H8 | AMD tool names `"N/A"` were accepted as names, and a tool that timed out once was cached. That gave a generic name and a different fingerprint for the whole run. | Placeholder names are ignored. A failed tool answer is not cached. |
| H9 | **AMD `backend` said `rocm` when ROCm tools existed.** But `runtime_install` never picks a ROCm build: it installs Vulkan. So the export wrote a ROCm docker image with `/dev/kfd`, and community records said `rocm`. | `backend` is now always `vulkan`, the build the app installs. A new `rocm: bool` says whether ROCm tools are present. |
| H10 | Windows `IsProcessorFeaturePresent` returns 0 when Windows *cannot detect* a feature: AVX flags before build 19041, DotProd before 22000. Old Windows reported "no AVX2". | A missing feature on those builds is `None` (unknown). |

Kept as they were, and checked:
- The fingerprint is byte-identical to v0.3 for NVIDIA-only and CPU-only machines, including the new `[N/A]` case.
- Every subprocess call uses a list, a timeout of 5 s or less, and `CREATE_NO_WINDOW`.
- The `\d{4}` subkey filter never opens the protected `Properties` key.
- `qwMemorySize` avoids WMI's 4 GiB `AdapterRAM` cap.
- The PF_* constants are correct (39/40/41/29/43).

### catalogue.py

| # | Problem | Fix |
|---|---|---|
| C1 | **Variants whose files had no sha256 were offered.** Downloads refuse them and discover can't verify them. This covers non-LFS files and shard sets with one checksum missing. | A file or shard set is offered only when every part has a 64-hex sha256. An LFS `oid` counts. |
| C2 | **gpt-oss `expert_fraction` was a parameter share (0.914), but contract §2.1 says byte share.** In MXFP4 files the experts take 17/32 bytes per weight, and the rest is stored larger. The engine would move about 0.9 GB (20b) or 1.3 GB (120b) too much off the GPU at full `--n-cpu-moe`, enough to run out of memory. | For MXFP4 files: `routed × 17/32 / file size`, never above the parameter share. gpt-oss-20b is now 0.838. K-quants keep the parameter share, which is within about 1 point. |
| C3 | **A damaged user `catalogue.json` broke `/api/state`, refresh and mapping** (`definitions()` raised). | The damaged file is renamed to `catalogue.json.damaged-<time>` (kept, never deleted). Invalid saved entries are skipped one by one. |
| C4 | The user copy stores the whole merged list. So a shipped model dropped in a later version would come back as "custom". | `definitions()` marks shipped entries `"shipped": true`. Saved entries with that mark are never turned into user entries. This works with `app.map_benchmark` unchanged. |
| C5 | Cancel was slow. It was checked only when a request finished, then `shutdown(wait=True)` waited for 25 s requests. | Cancel is checked every 0.25 s, and requests still running are not waited for. `cancel=` reaches `fetch_variants` / `fetch_scores` / `refresh_entry` between requests. It is passed only when given, so other people's mocks still work. |
| C6 | Progress dicts lacked the contract §1.9 keys. | Adds `stage`, `done`, `total`, `message`. The old keys that `app.js` reads are kept. |
| C7 | One odd reply (`IncompleteRead`, a config.json that is a list, an AttributeError) aborted the whole refresh and nothing was stored. | `get_json` catches `HTTPException`/decode errors. config.json must be an object. Refresh catches any error per entry and keeps that entry's cache. |
| C8 | Cached (fallback) variants kept their old `source`. A v0.3 cache said `catalogue` for a user's own repo, which breaks community's privacy rule. | `cached_variants` sets `source` from the entry. |
| C9 | Repo names accepted Unicode `\w`. Duplicate checks were case-sensitive, but HF names are not. | `re.ASCII`; duplicates compared lowercased. |
| C10 | Models of 4.5B or fewer got both F16 and BF16 (the same weights twice). | F16 only, or BF16 when there is no F16. F16 runs on every backend. |
| C11 | `app.map_benchmark` writes the user copy without the lock, through a fixed temp name. | New `catalogue.set_slug(store, base_repo, slug)` with the same lock and atomic write (see Requests). |

## Catalogue verification

huggingface.co is blocked from this container: curl got proxy 403, and WebFetch got `EGRESS_BLOCKED`. So `fetch_variants` could not run live, and no file listing or `lfs.sha256` could be read. Repos were checked with WebSearch, where the exact HF URL appeared in results. "search" means verified that way. Nothing below was confirmed through the HF API.

| base_repo → gguf_repo (config_repo) | gated | seen in search | notes |
|---|---|---|---|
| Qwen/Qwen3-0.6B, 1.7B → Qwen/…-GGUF | no | yes | Only Q8_0 published, so 1 variant each |
| Qwen/Qwen3-4B, 8B, 14B, 32B → Qwen/…-GGUF | no | yes | Q4_K_M and up, no Q3_K_M (32B: Q5_K_M not seen) |
| Qwen/Qwen3-4B-Instruct-2507 → unsloth/…-GGUF | no | **yes (now confirmed)** | was "partial" |
| Qwen/Qwen3-30B-A3B → Qwen/…-GGUF | no | yes | |
| Qwen/Qwen3-30B-A3B-Instruct-2507, Coder-30B-A3B-Instruct → unsloth/…-GGUF | no | yes | |
| Qwen/Qwen3-30B-A3B-Thinking-2507 → unsloth/…-GGUF | no | **yes (now confirmed)** | was "partial" |
| Qwen/Qwen3-235B-A22B-Instruct-2507 → unsloth/…-GGUF | no | yes | `Q4_K_M/…-0000k-of-00003` (about 142 GB) |
| Qwen/Qwen2.5-Coder-7B/14B/32B-Instruct → Qwen/…-GGUF | no | yes | lowercase names, single and split copies (single preferred) |
| deepseek-ai/DeepSeek-R1-Distill-{Qwen-7B,14B,32B,Llama-8B} → unsloth/…-GGUF | no | per build handoff | not re-searched |
| meta-llama/Llama-3.2-1B/3B, 3.1-8B, 3.3-70B → unsloth/…-GGUF (unsloth/… mirror) | **yes** | yes; the 70B mirror's config.json was found | all have `config_repo` |
| google/gemma-3-1b/4b/12b/27b-it → unsloth/…-GGUF (mirror or GGUF repo) | **yes** | yes | all have `config_repo`; the GGUF repos for 1b and 12b hold config.json |
| mistralai/Mistral-7B-Instruct-v0.3 → bartowski/… (unsloth/mistral-7b-instruct-v0.3) | **yes** | yes | |
| mistralai/Mistral-Nemo-2407, Small-24B-2501 → unsloth/… (unsloth mirror) | **yes** | per build handoff | |
| microsoft/Phi-4-mini-instruct, phi-4 → unsloth/…-GGUF | no | yes | |
| openai/gpt-oss-20b, 120b → ggml-org/…-GGUF | no | yes | 20b is a single `mxfp4` file (12.1 GB). 120b is 3 shards, and shard 1 holds only about 13 MB of metadata |
| HuggingFaceTB/SmolLM2-1.7B-Instruct → bartowski/… | no | yes | |
| ibm-granite/granite-3.3-2b/8b-instruct → ibm-granite/…-GGUF | no | yes | |
| allenai/OLMo-2-1124-7B/13B-Instruct → allenai/…-GGUF | no | yes | |

Every gated base has a `config_repo`. Nothing was removed: every entry is at least search-verified or carried over from the build handoff's search verification.

**Before release:** run one `llm-config refresh` on a machine that can reach HF, and read `refresh_status.warnings`.

### MoE arithmetic (recomputed by hand; the code matches exactly)

The columns come from the `moe_shape` formula:
- **Routed**: `layers × experts × 3 × hidden × expert_ffn`
- **Expert share by params**: `routed / total`
- **Active**: `total − routed × (E−k)/E`

| Model | Routed | Total | Expert share by params | By bytes (MXFP4) | Active |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | 48·128·3·2048·768 = 28.99B | 30.53B | 0.9495 | – | 3.353B (advertised 3.3B) |
| gpt-oss-20b | 24·32·3·2880·2880 = 19.11B | 20.91B | 0.914 | **0.838** | 4.19B (OpenAI's 3.61B leaves out the 0.58B input embedding) |
| gpt-oss-120b | – | 116.8B | 0.982 | about 0.96 | 5.71B (OpenAI's 5.13B also leaves out the input embedding) |
| Mixtral 8x7B | 32·8·3·4096·14336 = 45.10B | 46.70B | 0.966 | – | 12.88B (advertised 12.9B) |

gpt-oss fuses `gate_up_proj` as [E, H, 2F], so 3·H·F still holds. The expert biases (6.6M) are negligible.

### Sliding window (checked)

| Model | Window | Sliding layers |
|---|---|---|
| gemma-3-1b | 512 | 22 of 26 |
| gemma-3-4b | 1024 | 29 of 34 |
| gemma-3-12b | 1024 | 40 of 48 |
| gemma-3-27b | 1024 | 52 of 62 |
| gpt-oss | 128 | half the layers |

The real configs carry `sliding_window`. The 4096 default in `CONFIG_DEFAULTS` applies only when the key is missing, and it overestimates the KV cache (the safe direction), so it was kept.

## Rejected or deferred (with reasons)

- **Switch the Qwen3 dense models to unsloth repos** (to get Q3_K_M, and more than Q8_0 for 0.6B/1.7B): not done. The contract prefers official repos, and the unsloth file lists can't be checked live here. It is a one-line change per entry once someone checks the file lists.
- **Match amd-smi names to cards by PCI address** (`--bus`): the JSON shape can't be verified without the tool. The existing rule (use tool names only when the counts match) stays.
- **AMD APUs with a large carve-out or GTT (Strix Halo)**: a real gap. The right budget (VRAM plus GTT, shared with RAM) needs a unified-memory model for AMD in the engine, so it is deferred. A carve-out of 2 GiB or more already shows as a dedicated GPU of that size, which is honest.
- **A per-backend device number (`device`) on GPU dicts**: nothing pins non-CUDA devices yet (see Requests). Adding a key nobody reads would just be another seam.
- **A cross-process lock for the user copy** (CLI and server at once): rare. The atomic write prevents a corrupt file; the worst case is a lost slug.
- **`active_parameters` without the input embedding** (OpenAI's convention): left out because it would change engine speed numbers. The current value slightly over-counts the bytes read per token, which is the cautious direction.
- **A race between `refresh` and a concurrent `remove_entry` / `refresh_entry`**: minor; the next refresh fixes it.

## Requests

- **runtime-downloads (`runtime.py:31-45`)**: `bench` always passes `-dev CUDA0` and `CUDA_VISIBLE_DEVICES`. Branch on `gpu["backend"]`: `metal` needs neither; `vulkan` uses no `-dev` flag (or `VulkanN`, matched by name from `--list-devices`, never by our `index`). Also say "GPU", not "NVIDIA GPU".
- **runtime-downloads (`runtime_install._gpu_kinds`)**: on Windows, every AMD/Intel GPU (and NVIDIA without nvidia-smi) is in `other_gpus`. Each entry has `vendor` and `backend`, so the CPU build is chosen today. If you start picking Vulkan from `other_gpus`, CPU runs must also get `-dev none` (launch config `gpu_backend="vulkan"`, `gpu_layers=0`), or llama.cpp will use the GPU outside the memory budget. AMD `gpus` entries now say `backend: "vulkan"` and `rocm: bool`.
- **engine**:
  - `expert_fraction` for MXFP4 is now a byte share.
  - llama.cpp's iSWA cache gives each sliding layer about `min(ctx, pad256(window·(n_seq if unified else 1) + ubatch))` cells, not `window`. With the default ubatch of 512, that is about 1536 for Gemma 3 and 640 for gpt-oss. Cap at window + ubatch, not at the window.
  - `other_gpus` entries always have `available: None`, so ignore them in memory maths.
- **engine / launch (lead)**: the launch config's `gpu_backend` comes from the hardware dict. It should come from the installed runtime's backend (`runtime_install.detect()["backend"]`). On Linux NVIDIA with a Vulkan build, `CUDA_VISIBLE_DEVICES` is ignored and llama.cpp spreads layers over every Vulkan device.
- **community-export**: use the runtime backend (not the hardware one) for the docker image and the community `backend` field. User entries and their cached variants are `source="custom"` again.
- **api-cli (`app.map_benchmark`)**: call `catalogue.set_slug(store, base_repo, slug)` instead of writing `catalogue.json` directly (it currently skips the lock and uses a fixed `catalogue.tmp`). Pass `cancel=` to `catalogue.refresh`. After `models add`, call `catalogue.refresh_entry`. Consider a `--config-repo` flag for gated custom models.
- **api-http (`server.py:559`)**: pass `cancel=` to `refresh()`. Progress now has `stage/done/total/message`.
- **ui**:
  - `app.js:59` says "No supported NVIDIA telemetry"; show `hw.cpu_name || hw.cpu`.
  - List `other_gpus` (name plus `reason`) and `hw.warnings`.
  - Label a unified GPU as "Shared with system RAM".
  - Escape `gpu.name` at `app.js:58`.
- **calibration** (lead): GPU probes are CUDA-only, so Apple and AMD GPU placements never get memory-speed calibration.

## Evidence

- `python3 -m unittest discover -s tests`: 619 tests, 25 skipped, 2 failures. Both failures are the known fake-llama tests listed in FIXUPS.md (`test_fake_matches_real`: `--draft-max`, `llama-bench --version`). They belong to `server-fake` and fail on the base branch too. `test_hardware` (32), `test_catalogue` (36) and `test_adapters` (12) all pass. The keyring panic was fixed with `pip install --ignore-installed cryptography`.
- Sources:
  - Apple working set 2/3 up to 32 GiB, 3/4 above: [llama.cpp #2182](https://github.com/ggml-org/llama.cpp/discussions/2182), [meta-llama/llama #964 (36 GB → 27648 MiB)](https://github.com/meta-llama/llama/issues/964), [llama-cpp-python #687 (16/32 GB)](https://github.com/abetlen/llama-cpp-python/issues/687), [devnote: override macOS Metal VRAM cap](https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap)
  - `[N/A]` memory on GB10: [llama-swap #782](https://github.com/mostlygeek/llama-swap/issues/782)
  - IsProcessorFeaturePresent constants and OS limits: [MicrosoftDocs sdk-api](https://github.com/MicrosoftDocs/sdk-api/blob/docs/sdk-api-src/content/processthreadsapi/nf-processthreadsapi-isprocessorfeaturepresent.md)
  - rocm-smi `Card series` key: [rocm_smi.py](https://github.com/ROCm/rocm_smi_lib/blob/master/python_smi_tools/rocm_smi.py)
  - amdgpu sysfs files: [linux drivers/gpu/drm/amd](https://github.com/torvalds/linux/tree/master/drivers/gpu/drm/amd)
  - Qwen 2507 repos: [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3-30B-A3B-Thinking-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Thinking-2507)
