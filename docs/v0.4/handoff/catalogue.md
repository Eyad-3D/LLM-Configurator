# Handoff: `catalogue` workstream

Branch `claude/v04-catalogue`. Files touched: `src/llm_configurator/catalogue.py`, `src/llm_configurator/catalogue.json`, `tests/test_adapters.py`, `tests/test_catalogue.py` (new), and this file.

## What I built

- **Catalogue with 39 models** (it was 6). It covers:
  - Qwen3 dense, plus Qwen3-4B-2507
  - Qwen3 MoE: 30B-A3B, the 2507 Instruct/Thinking versions, Coder-30B-A3B and 235B-A22B-2507 (split into several files)
  - Qwen2.5-Coder 7/14/32B
  - DeepSeek-R1 distills (Qwen 7/14/32B, Llama 8B)
  - Llama 3.2 1B/3B, 3.1 8B and 3.3 70B
  - Gemma 3 1B/4B/12B/27B
  - Mistral 7B v0.3, Nemo and Small 24B
  - Phi-4 and Phi-4-mini
  - gpt-oss 20B and 120B
  - SmolLM2 1.7B, Granite 3.3 2B/8B and OLMo 2 7B/13B

  Every entry has `base_repo`, `gguf_repo`, `aa_slug: null`, `family`, `architecture` (informational only; the code reads the real `model_type` from config.json), `tags` (from `general|coding|reasoning|small|moe|large`) and `license`. Gated models also have `config_repo`.
- **`config_repo`, for gated base repos (Llama, Gemma, Mistral).** A gated repo is one where you must accept a licence before downloading. The Hugging Face model *info* API still answers for these repos, so `base_revision`, the parameter count and the licence come from the real base repo. Only `config.json` is read from `config_repo`, an ungated copy (unsloth mirrors, or unsloth's GGUF repo, which ships `config.json`). The order is:
  - **No `HF_TOKEN` and the base is gated:** go straight to the mirror, which avoids a request that is certain to fail.
  - **Otherwise:** try the base repo first and fall back to the mirror.
  - **Gated with no mirror:** the error message says "accept its licence and set HF_TOKEN, or add an ungated config_repo".
- **Split (sharded) GGUFs.** Files named `-0000k-of-0000n.gguf` are grouped, at the root or inside quant subfolders such as `Q4_K_M/…`.
  - Sizes are summed, and `files` lists the shards in order with each shard's sha256.
  - `filename` and `sha256` are the first shard's, which is what llama.cpp loads and what `runtime.bench` checks.
  - A set is skipped if any shard is missing or has no size.
  - A 1-of-1 "set" becomes a plain single file.
- **Mixture of experts (MoE).** In an MoE model, each word uses only a few of many "expert" blocks. These fields are read:
  - Expert count: `num_experts` / `num_local_experts` / `n_routed_experts`
  - Experts used per token: `num_experts_per_tok` / `experts_per_token` / `moe_topk`
  - Expert size: `moe_intermediate_size`, falling back to `intermediate_size`
  - Which layers have experts: `decoder_sparse_step`, `mlp_only_layers`, `first_k_dense_replace`
  - Shared experts: `shared_expert_intermediate_size`, `n_shared_experts`

  From these, the code fills `experts`, `active_experts`, `expert_fraction`, `parameters` and `active_parameters`.
- **Formula (see the `moe_shape` docstring).**
  - `routed = moe_layers × experts × 3 × hidden_size × expert_ffn` (the gate, up and down matrices)
  - `active_parameters = parameters − routed × (experts − active_experts) / experts`
  - `expert_fraction = routed / parameters`, capped at 0.99

  A layer has experts if `i ∉ mlp_only_layers`, `i ≥ first_k_dense_replace` and `(i+1) % decoder_sparse_step == 0`, which matches transformers. Shared experts count toward the total but not toward `routed`, because they are not offloadable by `--n-cpu-moe`.

  Sanity checks from the tests, using real configs:

  | Model | Expert share | Active parameters | Advertised |
  |---|---|---|---|
  | Qwen3-30B-A3B | 0.95 | 3.35B | 3.3B |
  | Mixtral 8x7B | – | 12.9B | 12.9B |
  | gpt-oss-20b | 0.91 | – | – |
- **`parameters`.** The count reported by Hugging Face (`safetensors.total` on the base repo, else `gguf.total` on the GGUF repo) is used if present. It is replaced by a count computed from config.json shapes when there is no report, or when the report is below 80 % of the computed count. Packed MXFP4 checkpoints under-report.
- **Sliding window.** In these layers the model's short-term notepad (the KV cache) only keeps the last N tokens. `sliding(config, layers, max_context)` returns `(window, sliding_layers)` using these rules, in priority order:
  1. `layer_types` (count of `"sliding_attention"`), used by gpt-oss and newer Gemma 3.
  2. Qwen family: `use_sliding_window` must be true. Then the layers **from** `max_window_layers` upward slide. This is the transformers quirk; every released Qwen ships `use_sliding_window: false`, so the count is 0.
  3. `sliding_window_pattern` (every Nth layer is global), used by Gemma 3.
  4. `gemma2`: alternating layers.
  5. Anything else counts as 0. Mistral 7B v0.1 declares a window, but llama.cpp runs full attention there.

  A window at least as long as `max_position_embeddings` counts as 0 (Phi-4-mini's 262144 and Qwen2.5's 131072 never cap anything).
- **Nested configs.** `text_config` (Gemma 3, Mistral 3) is merged over the top level. The architecture is the top-level `model_type` if it is supported (`gemma3`), otherwise the nested one (`mistral3` → `mistral`). Gemma 3 `text_config` leaves most fields out, so the `Gemma3TextConfig` defaults are filled in (heads 8, kv 4, head_dim 256, 131072 context, window 4096, pattern 6). These were confirmed from the transformers source.
- **Quants.** Every key of `domain.QUANT_BYTES_PER_PARAMETER` is recognised. The default shortlist is `Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0, MXFP4`, plus `F16`/`BF16` only for models of 4.5B parameters or fewer. An entry may set `"quants": [...]` to override the shortlist (then F16 is not added automatically).
  - **Name matching.** It ignores case, and `-`, `_`, `.` and `/` count as boundaries before the quant name. A trailing `_` is *not* a boundary, so `Q6_K_L`, `Q4_0_4_4` and `UD-Q4_K_XL` are never mistaken for Q6_K, Q4_0 or Q4_K.
  - **Skipped files.** `mmproj*` (vision add-ons) and `imatrix*` files are skipped.
  - **One file per quant.** Plain uploads win over `UD-` ones, single files win over split copies (the official Qwen2.5-Coder repos have both), and after that the shorter name wins.
  - **gpt-oss.** `…-mxfp4.gguf` becomes MXFP4.
- **Architecture check.** The model type must be in `domain.ARCHITECTURES`; otherwise the error is "model type 'x' is not supported yet; supported: …". An MoE architecture without expert fields, or an MoE config missing top-k or expert size, is rejected with a plain message. A missing layers, heads or context field is a `ValueError`, not a `KeyError`.
- **User repos.** `add_entry` and `remove_entry` store user repos in the user copy `<data dir>/catalogue.json`, with atomic writes (temp file plus `os.replace`) and a lock. `definitions()` now *merges* the shipped list with that copy:
  - Shipped entries keep the user's `aa_slug`.
  - Entries with `"user": true`, and any unknown entries (for example hand-edited v0.3 copies), are kept in full and marked `user`.
  - Shipped entries the user removed are hidden through the store key `catalogue_removed`.
  - **Why:** in v0.3, anyone who mapped a benchmark got a frozen copy of the 6-model list and would never have seen the new models. `app.map_benchmark` still works unchanged; it writes the merged list back, and the merge is stable (tested).
- **Refresh.** The worker pool is `MAX_WORKERS = 6` threads, shared with the score request, which is submitted first, so both still run at once. Progress reporting is unchanged. A failure in one entry still falls back to that entry's cached variants and adds a warning. The cache fallback now also matches on `base_repo`, so two entries sharing a GGUF repo cannot swap variants. There is a new optional `cancel=` (a `threading.Event`): queued fetches are cancelled and `domain.Cancelled` is raised.

## Public API as built

```python
fetch_variants(entry: dict) -> list[Variant]
    # entry: {"base_repo", "gguf_repo", "config_repo"?, "family"?, "license"?, "tags"?, "quants"?, "user"?}
    # Variant: all §2.1 fields filled; source = "custom" if entry["user"] else "catalogue"
definitions(store) -> list[dict]                     # merged shipped + user entries (validated)
add_entry(store, base_repo, gguf_repo, config_repo=None, family=None, tags=None) -> dict
    # returns the saved entry: {"base_repo", "gguf_repo", "aa_slug": None, "family", "tags": [..., "custom"], "user": True, "config_repo"?}
    # ValueError: invalid name, duplicate base_repo, gguf_repo already used, more than 200 user entries
remove_entry(store, base_repo) -> {"base_repo", "removed": True, "user": bool, "variants": int}
    # "variants" = number of cached variants left in the store after pruning this model's
refresh_entry(store, base_repo) -> {"base_repo", "variants": int}   # new: fetch one entry right after add_entry
refresh(store, include_scores=True, include_models=True, progress=None, cancel=None) -> status  # unchanged shape
demo_variants(), apply_scores(variant, entry, score_cache), test_connection(value=None), fetch_scores()  # unchanged
# Helpers used by tests and other code: gguf_artifacts(siblings), pick_artifacts(artifacts, allowed), quant_of(path),
# flatten_config(config), sliding(config, layers, max_context), moe_shape(config, layers), validate_entry(entry), valid_repo(name)
```

Repo names must match `owner/name` using `[\w.-]`. `..` and a segment starting with `.` are rejected.

## Deviations from the contract

- `add_entry` has extra optional keyword arguments (`config_repo`, `family`, `tags`). This is allowed by §4.
- `refresh()` has an extra `cancel=` argument. `refresh_entry()` is new.
- `definitions()` merges instead of "user copy replaces shipped list". The reason is given above.
- `expert_fraction` is the expert **parameter** share, used as the byte share. llama.cpp quantises expert matrices at the same type as the other big matrices, so the two agree closely. In MXFP4 files (gpt-oss), the small non-expert part is stored at higher precision, so the true byte share is a few points lower (gpt-oss-20b: about 0.91 by parameters vs about 0.85 by bytes). The engine should treat this as an estimate.
- Mixtral is not in the catalogue. No publisher from the approved list has a GGUF repo for it; only TheBloke does. The code and tests still cover Mixtral-style configs, and users can add it with `add_entry`.
- SmolLM3 is not in the catalogue because its `model_type` is `smollm3`, which is not in `domain.ARCHITECTURES`. SmolLM2 (llama) is used instead. Granite 4 (`granitemoehybrid`) is left out for the same reason.

## Known gaps

- **Nothing was checked live against Hugging Face.** This container blocks huggingface.co for both curl and WebFetch. Repos were verified with WebSearch (the exact HF URL appeared in results), and configs through GitHub copies of the config.json files plus the transformers source. Run one real `llm-config refresh` on a machine with HF access before release, and check `refresh_status.warnings`.
- **Official Qwen3 GGUF repos have only Q4_K_M and up** (Qwen3-0.6B/1.7B have only Q8_0, plus F16 if present). Those models get fewer variants. Switching them to unsloth would add Q3_K_M. I kept the official repos as the contract prefers.
- **The HF model-info API for gated repos** (meta-llama, google, mistralai) is assumed to answer without a token and to include `sha`, `gated`, `safetensors.total` and `cardData.license`. This is standard HF behaviour, but I could not check it here. If it fails, the whole entry fails with a plain warning (as before).
- **The 235B and 120B models** are large (about 142 GB and 63 GB). They are tagged `large`, and the engine is expected to rule them out on normal hardware.
- **Parameter counts from config shapes** ignore norms and biases (a difference under 0.1 %).

## Verification status of each catalogue entry

In the table below, **yes** means the exact repo URLs for the base, GGUF and config repos were seen in search results or fetched GitHub copies of HF files. **partial** means the GGUF repo and the model are confirmed, but one id was taken from the publisher's naming pattern.

| base_repo | gguf_repo | config_repo | status |
|---|---|---|---|
| Qwen/Qwen3-0.6B | Qwen/Qwen3-0.6B-GGUF | – | yes (only Q8_0 seen in the GGUF repo) |
| Qwen/Qwen3-1.7B | Qwen/Qwen3-1.7B-GGUF | – | yes (Q8_0 seen) |
| Qwen/Qwen3-4B | Qwen/Qwen3-4B-GGUF | – | yes |
| Qwen/Qwen3-8B | Qwen/Qwen3-8B-GGUF | – | yes (Q4_K_M…Q8_0) |
| Qwen/Qwen3-14B | Qwen/Qwen3-14B-GGUF | – | yes |
| Qwen/Qwen3-32B | Qwen/Qwen3-32B-GGUF | – | yes |
| Qwen/Qwen3-4B-Instruct-2507 | unsloth/Qwen3-4B-Instruct-2507-GGUF | – | partial (GGUF repo seen; base id follows the 2507 naming) |
| Qwen/Qwen3-30B-A3B | Qwen/Qwen3-30B-A3B-GGUF | – | yes (config values confirmed) |
| Qwen/Qwen3-30B-A3B-Instruct-2507 | unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF | – | yes |
| Qwen/Qwen3-30B-A3B-Thinking-2507 | unsloth/Qwen3-30B-A3B-Thinking-2507-GGUF | – | partial (GGUF repo seen; base id not seen directly) |
| Qwen/Qwen3-Coder-30B-A3B-Instruct | unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF | – | yes |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | unsloth/Qwen3-235B-A22B-Instruct-2507-GGUF | – | yes (`Q4_K_M/…-00001-of-00003.gguf` layout seen) |
| Qwen/Qwen2.5-Coder-7B-Instruct | Qwen/Qwen2.5-Coder-7B-Instruct-GGUF | – | yes (lowercase names; single and split copies) |
| Qwen/Qwen2.5-Coder-14B-Instruct | Qwen/Qwen2.5-Coder-14B-Instruct-GGUF | – | yes |
| Qwen/Qwen2.5-Coder-32B-Instruct | Qwen/Qwen2.5-Coder-32B-Instruct-GGUF | – | partial (repo seen; file list from third-party tools) |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF | – | yes |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-14B | unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF | – | yes |
| deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF | – | yes |
| deepseek-ai/DeepSeek-R1-Distill-Llama-8B | unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF | – | yes |
| meta-llama/Llama-3.2-1B-Instruct | unsloth/Llama-3.2-1B-Instruct-GGUF | unsloth/Llama-3.2-1B-Instruct | yes |
| meta-llama/Llama-3.2-3B-Instruct | unsloth/Llama-3.2-3B-Instruct-GGUF | unsloth/Llama-3.2-3B-Instruct | yes |
| meta-llama/Llama-3.1-8B-Instruct | unsloth/Llama-3.1-8B-Instruct-GGUF | unsloth/Llama-3.1-8B-Instruct | yes |
| meta-llama/Llama-3.3-70B-Instruct | unsloth/Llama-3.3-70B-Instruct-GGUF | unsloth/Llama-3.3-70B-Instruct | partial (GGUF and Q8_0 subfolder seen; mirror id follows unsloth naming) |
| google/gemma-3-1b-it | unsloth/gemma-3-1b-it-GGUF | unsloth/gemma-3-1b-it-GGUF | yes (config.json in the GGUF repo seen) |
| google/gemma-3-4b-it | unsloth/gemma-3-4b-it-GGUF | unsloth/gemma-3-4b-it | yes |
| google/gemma-3-12b-it | unsloth/gemma-3-12b-it-GGUF | unsloth/gemma-3-12b-it-GGUF | yes (config.json in the GGUF repo seen) |
| google/gemma-3-27b-it | unsloth/gemma-3-27b-it-GGUF | unsloth/gemma-3-27b-it | yes |
| mistralai/Mistral-7B-Instruct-v0.3 | bartowski/Mistral-7B-Instruct-v0.3-GGUF | unsloth/mistral-7b-instruct-v0.3 | yes |
| mistralai/Mistral-Nemo-Instruct-2407 | unsloth/Mistral-Nemo-Instruct-2407-GGUF | unsloth/Mistral-Nemo-Instruct-2407 | yes |
| mistralai/Mistral-Small-24B-Instruct-2501 | unsloth/Mistral-Small-24B-Instruct-2501-GGUF | unsloth/Mistral-Small-24B-Instruct-2501 | yes |
| microsoft/Phi-4-mini-instruct | unsloth/Phi-4-mini-instruct-GGUF | – | yes (names like `Phi-4-mini-instruct.Q8_0.gguf`, which are handled) |
| microsoft/phi-4 | unsloth/phi-4-GGUF | – | yes |
| openai/gpt-oss-20b | ggml-org/gpt-oss-20b-GGUF | – | yes (`gpt-oss-20b-mxfp4.gguf`) |
| openai/gpt-oss-120b | ggml-org/gpt-oss-120b-GGUF | – | yes (3 shards, `…-mxfp4-00001-of-00003.gguf`) |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | bartowski/SmolLM2-1.7B-Instruct-GGUF | – | yes |
| ibm-granite/granite-3.3-2b-instruct | ibm-granite/granite-3.3-2b-instruct-GGUF | – | yes |
| ibm-granite/granite-3.3-8b-instruct | ibm-granite/granite-3.3-8b-instruct-GGUF | – | yes (`-f16.gguf` lowercase, handled) |
| allenai/OLMo-2-1124-7B-Instruct | allenai/OLMo-2-1124-7B-Instruct-GGUF | – | yes |
| allenai/OLMo-2-1124-13B-Instruct | allenai/OLMo-2-1124-13B-Instruct-GGUF | – | yes |

- **Licence strings** come from each family's known licence (`apache-2.0`, `mit`, `llama3.x`, `gemma`). At refresh time, `cardData.license` from HF is used only when an entry has none.
- **OLMo-2 32B was left out** on purpose. There are reports of broken official GGUFs (llama.cpp issue #12376).

## Requests to other owners

- **engine:** treat `expert_fraction` as an estimate (see Deviations). `sliding_layers` counts only layers that llama.cpp caps. `files` is non-empty only for split models.
- **api:** add `POST /api/catalogue` `{base_repo, gguf_repo}` → `add_entry`, then a job running `refresh_entry`, plus `DELETE`/`remove` → `remove_entry`. The CLI command `llm-config models add|remove` from §6 can use them directly. Pass `cancel=` to `refresh()` when it runs as a job.
- **discover / downloads:** shard filenames may include a folder (`Q4_K_M/x-00001-of-00003.gguf`). Keep the relative path when you save files, or at least put all shards in the same directory; llama.cpp finds the other shards next to the first.
- **lead / docs:** document `HF_TOKEN` for gated models, and `config_repo` for people who edit their catalogue.

## How I tested

- `python3 -m unittest discover -s tests`: 100 tests, OK. `npm test`: 18 tests, OK after `npm ci`.
- `tests/test_catalogue.py` (new, 28 tests). A fake Hugging Face hub answers by URL. The fixtures copy real config values and cover:
  - dense Qwen3-8B
  - Qwen3-30B-A3B (MoE)
  - gpt-oss-20b single file and gpt-oss-120b in 3 shards listed out of order
  - Mixtral
  - Gemma 3 4B with a sparse `text_config` and a gated base using a mirror
  - Gemma 3 `layer_types`
  - the Qwen2 `max_window_layers` quirk
  - Mistral 3 nesting
  - quant name edge cases
  - quant subfolders with incomplete and size-less shard sets
  - the F16 rule and `quants` override
  - gated repos with and without a token
  - catalogue.json structure: 30–50 entries, unique repos, valid names, family/licence/tags, `moe` tag matching the architecture, gated families having `config_repo`
  - `add_entry` / `remove_entry`: validation, duplicates, no temp files left behind, concurrent adds, removed shipped entries, re-adding
  - old v0.3 user copies
  - compatibility with `app.map_benchmark`
  - `refresh_entry`
  - refresh cancel, and a per-entry failure with the cache fallback
- `tests/test_adapters.py`: the unsupported-architecture test now uses `mamba`, because `qwen3_moe` is supported now. Its intent is kept.

## llama.cpp facts assumed (for the harness to confirm)

- For a split GGUF, llama.cpp loads the first shard (`-00001-of-0000N.gguf`) and finds the rest in the same directory.
- `--n-cpu-moe` keeps the routed-expert tensors (`ffn_*_exps`) in RAM, and shared experts stay on the GPU.
- llama.cpp caps the KV cache at the window for Gemma 2/3 and gpt-oss sliding layers (the iSWA cache, unless `--swa-full` is set). It does **not** do this for Mistral-v0.1-style or Qwen2 windows (these count as 0 here).
- MXFP4 GGUFs (gpt-oss) keep the non-expert tensors at higher precision than the experts.
- `mmproj-*.gguf` files are vision projectors, not language models.
