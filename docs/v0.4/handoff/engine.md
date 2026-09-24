# Handoff: `engine` workstream

Branch `claude/v04-engine`. Files: `engine.py`, `speed.py`, `learning.py` (new), `tests/test_engine.py`,
`tests/test_speed.py` (new), `tests/test_learning.py` (new). No other files touched.

## What I built

- **Memory model** (`engine.allocations`): compressed conversation memory (KV cache f16 / q8_0 / q4_0),
  sliding-window layers, MoE expert offload (`--n-cpu-moe`), and Apple unified memory as one pool.
- **Placement search** (`engine.placements`): CPU only, the largest fitting layer split, full GPU, and for MoE
  models that don't fit fully on the GPU, "experts on CPU" with the *fewest* expert layers moved to RAM.
  It uses binary search, not a scan of every layer. An independent reviewer checked it against a brute-force
  scan on 20,000 random cases: 0 mismatches. A test repeats this on 300 cases.
- **Evidence ladder** per candidate: measured > tuned > interpolated > community > estimated > none.
- **Plain verdicts** (`runs_well` / `runs_slowly` / `too_slow` / `unknown`), each with one `verdict_text` sentence.
- **Speed model** (`speed.estimate`): MoE active bytes, KV type, `n_cpu_moe` split and unified memory.
- **Learning** (`learning.fit_efficiency`): corrects estimate ranges using this machine's own speed tests.
- **Launch fragment** per candidate for `launch.from_candidate`.
- **Backwards compatible:** with default arguments, `recommend` gives *identical* ids, memory, max-context,
  speed ranges and ordering to v0.3. I checked this on 2,153 candidates across random hardware and requirements.
  The one addition is the extra `mid_tps` key in the estimate.

## Public API as built

```python
engine.allocations(variant, context, users, gpu_layers, kv_cache_type="f16", n_cpu_moe=0, unified=False)
  -> {"ram", "vram", "kv_total", "kv_vram", "weights_ram", "weights_vram", "expert_ram", "total", "unified"}
engine.placements(variant, context, users, budget, gpu_budget, has_gpu, kv_cache_type="f16", unified=False)
  -> [(gpu_layers, n_cpu_moe, memory)]
engine.maximum_context(variant, users, layers, ram_budget, vram_budget, kv_cache_type="f16", n_cpu_moe=0, unified=False) -> int
engine.matching_speed(records, variant, hardware, context, layers, gpu_uuid, threads, kv_cache_type="f16", n_cpu_moe=0) -> record | None
engine.interpolated_speed(records, variant, hardware, context, layers, gpu_uuid, threads, kv_cache_type="f16", n_cpu_moe=0)
  -> {"tps", "contexts": [int], "method": str, "measurement_ids": [str]} | None
engine.matching_tuned(tuned, variant, hardware, context, layers, kv_cache_type="f16", n_cpu_moe=0, gpu_uuid=None)
  -> {"tps", "pp_tps", "improvement", "timestamp", "stopped", "settings": {threads, batch, ubatch, flash_attn}} | None
engine.community_speed(records, variant, hardware, config, evidence=None) -> {"median_tps", "n", "similar", "range"} | None
engine.verdict(evidence, target, tps=None, interpolated=None, community=None, speed_estimate=None, users=1) -> (verdict, sentence)
engine.recommend(variants, hardware, requirements, measurements=(), calibration=None, community=(), tuned=(),
                 community_evidence=None) -> report
speed.kv_bytes(variant, context, users, kv_cache_type="f16") -> int
speed.moved_expert_layers(variant, gpu_layers, n_cpu_moe) -> int
speed.active_fraction(variant) -> float
speed.estimate(variant, hardware, calibration, context, layers, users, gpu, target,
               kv_cache_type="f16", n_cpu_moe=0, unified=False, efficiency=None) -> dict
learning.fit_efficiency(measurements, hardware, calibration, variants=())
  -> {"factor", "spread", "n", "label", "by_mode": {mode: {"factor", "spread", "n"}}} | None
learning.ratios(measurements, hardware, calibration, variants=()) -> [(mode, ratio)]
```

`community` may be a list of records or the stored `{"source", "fetched_at", "records"}` dict.
`community_evidence` is an injectable replacement for `community.evidence`. Without it, `community.evidence`
is imported inside the function, and an `ImportError` means "no community evidence".

**New candidate keys** (all existing keys kept):
- `kv_cache_type`, `n_cpu_moe`, `unified_memory`, `launch`
- `verdict`, `verdict_text`, `evidence`
- `community` (`{median_tps, n, similar, range}` | None), `tuned` (summary | None), `speed_interpolated` (dict | None)
- `memory_total_bytes`

`threads` equals `launch.threads`, which may come from a tune result.

**New report keys:** `speed_adjustment` (`{factor, spread, n, label}` | None).

**Candidate `id`:** `"{variant}|{scenario}|{context}|{layers}"` exactly as in v0.3, plus `|kv:q8_0` when the
cache type isn't f16 and `|moe:K` when K > 0.

**Launch fragment:**
- Always: `{context, parallel, gpu_layers, total_layers, threads, cache_type_k, cache_type_v, n_cpu_moe, flash_attn}`.
- `flash_attn` is `"on"` for compressed caches, because llama.cpp needs flash attention for a quantized V cache.
  Otherwise it is `"auto"`.
- `gpu_backend` is added when the GPU reports cuda/metal/rocm/vulkan. It is added for CPU candidates too, so
  `launch.server_args` adds `-dev none` and a CPU run really is CPU only.
- `gpu_uuid` is added when layers > 0.
- A matching tune result's `threads/batch/ubatch/flash_attn` are merged in. A tuned `flash_attn: off` is
  dropped when the cache is compressed.

## Formulas

Symbols: L = layers, g = GPU layers, u = users, c = context, W = file bytes × 1.10 (placement margin),
e = `expert_fraction`, E / A = experts / active experts.

**Conversation memory (KV):**
- `per_token_layer = 2 × kv_heads × head_dim × KV_BYTES_PER_ELEMENT[type]` (f16 = 2, q8_0 = 34/32, q4_0 = 18/32).
- `kv = ceil(per_token_layer × ((L − S) × c × u + S × min(c × u, window × u + 512)))`, where S = `sliding_layers`
  (counted only when `sliding_window` is set).
- The extra 512 is llama.cpp's own sliding cache size, `n_swa × n_seq + n_ubatch`.
- With no sliding layers and f16, this is exactly the v0.3 formula.

**Expert offload:** `moved = clamp(n_cpu_moe − (L − g), 0, g)`.
- llama.cpp offloads the *last* g blocks. `--n-cpu-moe K` pins the expert tensors of blocks 0..K−1.
- Only the overlap moves. At full offload (the only case `recommend` creates), moved = K.
- Dense models: moved = 0.

**Weights:**
- `expert_ram = W × e × moved / L`
- `weights_ram = W × (1 − g/L) + expert_ram`
- `weights_vram = W × g/L − expert_ram`

**Pools:** `buffer = (0.75 + 0.20·u) GiB`.
- `ram = weights_ram + kv × (1 − g/L) + buffer`
- `vram = weights_vram + kv × g/L + buffer` if g > 0, else 0.
- The KV of GPU layers stays on the GPU even with expert offload, because attention runs there.
- **Discrete GPU:** `ram += 0.5 GiB` staging if g > 0. `total = ram + vram`.
- **Unified (Apple):** `ram += vram`, meaning `ram` is the whole shared pool. There is no staging copy.
  `vram` is the part inside that pool that the GPU uses, and `total = ram`. Two separate compute buffers are
  counted on purpose, because llama.cpp allocates a CPU one and a Metal one.

**Budgets:**
- RAM: `ram_available − reserve` (or the after-closing scenario).
- Discrete GPU: `available − gpu_reserve`.
- Unified: `min(available − gpu_reserve, scenario RAM budget)`. In the after-closing scenario the working-set cap
  may rise up to `gpu.total` (the wired limit from hardware §4.10), still capped by the RAM budget.
  A candidate fits when `ram ≤ RAM budget` and `vram ≤ GPU budget`. Each byte is counted once against the pool;
  the GPU check is only a limit on the slice.
- **GPU with unknown free memory** (`available` None, missing, NaN or ≤ 0): the GPU is skipped, only CPU
  placements are shown, and a note explains why.

**Placement search:** for g ≥ 1, VRAM never falls as g grows and RAM never rises; at full offload, VRAM never
rises as K grows.
- The largest split is `max g ∈ [1, L−1]` with vram ≤ budget, kept if its RAM also fits.
- MoE offload is `min K ∈ [1, L]` with vram(L, K) ≤ budget, kept if RAM fits. It is tried only when full GPU
  doesn't fit.
- `maximum_context` binary-searches whole 256-token steps. This gives the same answer as v0.3's search and is
  cached per (variant, scenario, layers, K).

**Speed estimate (bandwidth model):**
- `share = (1 − e) + e × A/E` (1 for dense). `layer_bytes = size / L`.
- `expert_active = layer_bytes × e × A/E`. `kv_layer = kv(c, u=1, type) / L`.
- `gpu_traffic = g × (layer_bytes × share + kv_layer) − moved × expert_active`
- `cpu_traffic = (L − g) × (layer_bytes × share + kv_layer) + moved × expert_active`
- `parameters = active_parameters or (parameters or size / QUANT_BYTES_PER_PARAMETER[quant]) × share`
- **CPU part** (if cpu_traffic > 0):
  - slow += `max(cpu_traffic/ram_bw / 0.35, parameters × cpu_share / q4_rate)`
  - fast += `cpu_traffic/ram_bw / 0.8`
- **GPU part** (if g > 0):
  - slow += `gpu_traffic/vram_bw / 0.10`
  - fast += `gpu_traffic/vram_bw / 0.55`
- **Link crossings** (discrete GPU only): `2 × moved + (2 if g < L)`.
  - `overhead = crossings × (max(4096, kv_heads × head_dim × 4)/link + latency)`
  - slow += 4 × overhead, fast += overhead
- `low = 1/slow`, `high = 1/fast`, `mid_tps = sqrt(low × high)` (unrounded).
- **Unified memory:** uses the GPU's own `vram_bytes_s` from calibration (the Metal bandwidth). If it is missing,
  the estimate is unavailable ("The graphics chip has no memory-speed calibration yet.").
- With defaults this equals v0.3 exactly.

**Learning:**
- Which tests count: same fingerprint, users = 1, age < 90 days, finite tps > 0, a known variant, and the sha256
  matches when both sides have one.
- For each such test: `ratio = measured tps / mid_tps` of an unadjusted estimate for that test's own context,
  layers, cache type and K.
- Fit: `factor = exp(median(log ratio))`, `spread = max(0.10, 1.5 × MAD(log ratio))`. At least 2 tests are
  needed. A per-mode fit (cpu / gpu / split) is used when that mode has ≥ 2 tests.
- Adjusted range: `mid × factor × e^(±spread)`.
- For a mode *without* its own fit, only the machine-wide factor shifts the range, and the width stays at least
  the raw width: `spread = max(spread, ln(high/low)/2)`.
- The label reads "adjusted from N local measurements". Confidence stays "low". The raw range is kept in
  `raw_low_tps` / `raw_high_tps`.

**Interpolation:** needs the same variant + sha256, fingerprint, gpu_layers, gpu_uuid, threads, users = 1,
cache type and K, from tests less than 30 days old at other contexts.
- Two tests on either side: linear in context.
- Only shorter tests: `tps × (A_w + kv(c₀))/(A_w + kv(c)) × 0.9`, where `A_w = size × share`. Allowed up to 4× the
  tested length.
- Only longer tests: the nearest one's tps, because a shorter conversation isn't slower.
- Labelled `interpolated`. `tps` stays None.

**Verdict thresholds** (only `measured` / `tuned` evidence; T = `min_tps`):
- `runs_well` when tps ≥ T ("comfortably above" when ≥ 1.5 T).
- `runs_slowly` when 0.5 T ≤ tps < T.
- `too_slow` when tps < 0.5 T.
- Every other kind of evidence is `unknown`, with wording like "likely fast enough", "might reach your 15" or
  "probably slower than your 15". An estimate can never produce "runs well".

## Deviations and decisions

- **Too-slow verified configurations are now kept** in the list with verdict `too_slow` / `runs_slowly`, so the
  UI can show the badge. They are still removed when `strict_speed` is on, and only removed ones count in
  `rejected.speed`. In balanced ordering they rank below untested options (`(meets is True, meets is not False, quality)`).
- **Sentences say "tokens per second", not "words".** This matches the UI's existing "tok/s" and the minimum-speed
  field. Tokens aren't words, and saying "words" would overstate the speed by about 30%.
- **The sliding-window cache adds 512 tokens** (one micro-batch) above `min(c, window)`, as llama.cpp does. The
  brief said `min(context, sliding_window)`; this is slightly more conservative.
- **`--n-cpu-moe` counts from layer 0**, as in llama.cpp, not from "the first GPU layer". Both readings agree at
  full offload, which is the only case `recommend` creates. `allocations` models the real overlap for any g.
- **When the user picks f16, only f16 is considered.** If a model then fits nowhere but would fit at q8_0 or
  q4_0 (at the requested context, in any scenario), a note names up to 3 such models.
- **Tune matching:**
  - The record needs `variant_id`, `fingerprint`, `timestamp`, `best` and `best_result.tps`.
  - `best` must have the same context, gpu_layers, cache type, n_cpu_moe and parallel = 1.
  - `best.gpu_uuid` must equal the GPU when layers > 0, if present.
  - `sha256` must match if the record has one.
  - Threads/batch aren't placement, so they are adopted from the tune result instead of being matched.
- `learning.fit_efficiency` takes an extra optional `variants=()`. It is needed to recompute an estimate for each
  test. Without it the function returns None.
- `speed.BYTES_PER_PARAMETER` is now an alias of `domain.QUANT_BYTES_PER_PARAMETER`, which covers more quants
  with the same values.

## Known gaps

- MoE expert offload is tried only at full layer offload. If attention plus shared weights plus the cache don't
  fit in VRAM even with every expert layer on the CPU, a partial-layer + `n_cpu_moe` mix isn't offered.
- Multi-user speed (users > 1) has no estimate, interpolation or community evidence. Only a matching
  measurement could verify it, and measurements match users = 1 only, as before.
- Community evidence is taken on trust from `community.evidence`. I only check its shape (median_tps > 0, n ≥ 1).
- The perf test allows 1.5 s for CI. Locally, 250 variants × 5 contexts × 2 scenarios with calibration and
  measurements take about 0.3 s.

## Requests

- **api (`app.py` / `server.py`):**
  - `app.evaluate` should pass `community=store.get("community")` (the dict is accepted as is) and
    `tuned=store.get("tuned", [])` to `recommend`.
  - `/api/recommend` should accept `kv_cache_type` (already in `Requirements`).
- **tuner / api:** tune records appended to `"tuned"` should include `variant_id`, `sha256`, `fingerprint`
  (`hardware["fingerprint"]`) and `timestamp` (`domain.now()`) next to the §4.6 result. Without them a tune result
  can't be matched to a candidate.
- **testing / tuner:** measurement records should always include `settings.cache_type_k` and `settings.n_cpu_moe`.
  Missing settings are read as f16 / 0, which is right for v0.3 records but wrong for a compressed-cache run.
- **hardware:** a unified GPU needs `unified: True`, `backend: "metal"`, `available` (the working-set limit capped
  by free RAM) and `total` (the wired limit). Unknown free memory must be `None`, never 0. That is exactly what
  §4.10 says.
- **calibration (lead):** unified memory needs `calibration["gpus"][<apple uuid>]["vram_bytes_s"]` from a Metal
  bandwidth probe. Until it exists, Apple GPU estimates say they're unavailable instead of guessing.
- **ui-run:** use `verdict` for the badges and `verdict_text` as the sentence under them. For MoE candidates,
  `mode` is `"split"` and `n_cpu_moe > 0`; show "Experts on CPU". On unified machines, `ram_bytes` is the whole
  pool; don't add `vram_bytes` to it (`memory_total_bytes` is always the right total).

## How I tested

- `python3 -m unittest discover -s tests` passes, and so does `npm test` (18/18).
- New tests cover:
  - KV types, including a bad type
  - sliding-window arithmetic, including several users
  - MoE offload memory, following llama.cpp's layer order
  - dense models ignoring `n_cpu_moe`
  - the smallest fitting K in `recommend`, and `from_candidate`/`server_args` producing `--n-cpu-moe`
  - binary search vs brute force on 300 random cases
  - unified single pool, working-set-forced split and the metal backend
  - skipping a GPU with unknown or missing free memory
  - interpolation (two points, one point, the 4× cap, same-placement rules)
  - old records without `settings`
  - evidence precedence: measured > tuned > interpolated > community > estimated > none
  - an optional or malformed community module, and the launch config passed to `community.evidence`
  - verdict thresholds and wording, and estimates never saying "runs well"
  - slow verified configurations shown unless strict, and their ordering
  - unique ids and valid launch fragments across cache types
  - performance
  - speed model: MoE active bytes, KV type, offload between GPU and CPU, transfer calibration, unified, and the
    efficiency adjustment
  - learning: minimum count, median robustness, filters, and the `recommend` integration
- A one-off script compared the new engine with the v0.3 engine (loaded from git) on 2,153 candidates: identical.
- An independent reviewer subagent checked the formulas and ran 20,000 random brute-force comparisons of
  `placements()` (0 mismatches). Its fixes are included: the unrounded midpoint for learning, full width for
  untested placements, and the GPU uuid check for tune results.

## llama.cpp facts assumed (please confirm in the harness)

1. `--n-cpu-moe N` (`-ncmoe`) adds buffer overrides `blk.{i}.ffn_(up|down|gate|gate_up)_(ch|)exps` → CPU for
   i in 0..N−1. I checked this in `common/arg.cpp` and `common/common.h` on master, Sept 2026.
2. `-ngl N` offloads the **last** N transformer blocks, and N = L + 1 also offloads the output layer (matches
   `launch.runtime_gpu_layers`).
3. A layer's KV cache lives on that layer's device (default `--kv-offload`), so expert offload leaves KV on the GPU.
4. q8_0 / q4_0 cache blocks are 34 / 18 bytes per 32 values, and a quantized V cache needs flash attention.
5. The sliding-window cache holds about `n_swa × n_seq + n_ubatch` tokens per SWA layer when `--swa-full` is off
   (the llama-server default).
6. On Metal, model weights are memory-mapped and shared with the GPU (no copy). The GPU working set is limited by
   `recommendedMaxWorkingSetSize` / `iogpu.wired_limit_mb` (from hardware ws).
