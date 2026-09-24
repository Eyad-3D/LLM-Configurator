# Handoff: `tuner`

Branch `claude/v04-tuner`. Files: `src/llm_configurator/tuner.py`, `tests/test_tuner.py`, this file.

## What I built

An auto-tuner that tries llama.cpp settings with `llama-bench` on the user's own machine within a time budget and returns the fastest configuration that passed the memory check.

- **Baseline:** it measures `base_config` first. If the starting settings fail to run, or the memory check calls them unsafe, it raises a plain `ValueError`, because there is nothing to tune from.
- **Coordinate search:** it tunes one setting (knob) at a time and keeps a change only if it wins clearly. The order is:
  1. `gpu_layers`: only for split CPU/GPU placements. Candidates are +2, +4 and −2 layers.
  2. `n_cpu_moe`: MoE models with GPU layers only. Candidates are −2, −4 and +2.
  3. `threads`: physical cores, cores−1, cores/2 and logical threads.
  4. `flash_attn`: on and off. Skipped when the V cache is compressed, because a compressed V cache needs flash attention on.
  5. `batch`/`ubatch` pairs 512/512, 1024/256 and 2048/512: only for the `prompt` and `balanced` goals. Batch size only affects prompt reading, so the `generation` goal doesn't spend budget on it.
  6. KV cache f16 ↔ q8_0 (K and V set together): only when `allow_kv_compression=True`, or once any trial was skipped for memory ("memory is tight").
     - In the tight case, it also tries q8_0 together with +2 GPU layers, or with −2 `n_cpu_moe`, since the point of a smaller notepad is usually room for more of the model on the GPU.
- **Passes:** the search repeats in passes (at most 3) and stops as `converged` when a full pass finds no improvement.
- **One process per knob:** each knob's candidate values run in **one llama-bench process** with comma lists, and rows are matched back to their settings by the JSON fields.
  - Candidates that can't be expressed as a full comma-list product run as separate processes. This applies to the batch pairs, the paired KV types, and "llama.cpp default" values (unset `threads`, `flash_attn: auto`).
  - The current config is not re-measured on its own if it is a "default" value. The tuner reuses its earlier number instead of paying for an extra model load.
  - Inside one process, values are ordered from least to most memory-hungry. llama-bench exits at the first model-load failure, so this keeps the earlier rows. Combinations that never ran are recorded as `failed` with "Not tested: an earlier setting in the same run stopped llama-bench."
- **Noise rule:** a candidate is kept only if `score > ref × (1 + max(min_gain=3%, rel_sd_candidate + rel_sd_ref))`, where `rel_sd = stddev_ts / avg_ts` from llama-bench. `ref` is the current config's number from the same process when available (fresher and fairer), otherwise its last number.
- **Goals:**
  - `generation` maximises writing speed (tg tok/s).
  - `prompt` maximises reading speed (pp tok/s).
  - `balanced` maximises `tg^0.6 · pp^0.4`, a weighted geometric mean that leans towards writing because chat time is mostly spent writing.
- **Confirmation:** the winner is re-measured with `confirm_repetitions=5`. If the confirmed score does not beat the baseline, the tuner keeps the starting settings and says so. If there's no time left for the confirmation run, a note says the speed comes from a shorter test.
- **Budget:**
  - The cost of one combination comes from the slowest of the last 3 observed runs, per combination, scaled by repetitions. It's deliberately pessimistic because it includes model load.
  - Before each step, the tuner reserves time for the confirmation run and trims the sweep to what fits. It drops the re-measure of the current value first, then the lower-priority values. If nothing fits, it stops with `stopped="budget"`.
  - Every bench process gets `timeout = min(timeout, remaining + one combination estimate)`, so a slow run is killed instead of blowing the budget. Rows printed before the kill are still used.
- **Cancel:** `check_cancel` runs between processes. The default runner polls the cancel event every 0.25 s and kills llama-bench.
  - Cancel **before the baseline finishes** raises `domain.Cancelled`, so the job ends up `cancelled`.
  - Cancel **after the baseline** returns the best settings so far with `stopped="cancelled"`, as the contract's `stopped` enum says. See Requests.
- **Safety:** before any run, each candidate goes through `memory_check(config) -> bool`. Unsafe candidates are recorded as `skipped_memory` with `seconds: 0` and never run. If `memory_check` raises, the candidate counts as unsafe.
  - The default check (`default_memory_check`) uses `engine.allocations` with free memory from `hardware.scan(False)`, taken once per tune. It keeps 1 GiB of RAM and 0.5 GiB of VRAM spare, and uses the GPU matching `gpu_uuid`, or else the first GPU.
  - It passes the v0.4 keyword arguments (`kv_cache_type`, `n_cpu_moe`, `unified`) and falls back to the base-branch 4-argument signature on `TypeError`.
- **Failures:** a non-zero exit, OOM text or missing rows are recorded as `failed`, with a plain `error` ("Ran out of memory with these settings.", "This llama.cpp version does not support one of these settings.", "The test ran out of time.", …). The search then continues.
- **Notes:** plain sentences, for example:
  - "Using 7 CPU threads instead of the default made writing 56% faster."
  - "Putting 22 of 32 layers on the graphics card instead of 20 and storing the model's short-term notepad (KV cache) as q8_0 instead of f16 made writing 5% faster and reading 3% faster."
  - Summary lines: an overall change sentence, skipped and failed counts, and why it stopped.

## Public API as built

```python
bench_args(config, n_prompt=512, n_gen=128, depth=0, repetitions=2, sweep=None) -> list[str]
```
- Returns the argv **after** the llama-bench executable, including `-m <model_path>` and `-o json` (like `launch.server_args`, which includes `-m`).
- Flag order: `-m -p -n [-d] -ngl [-ncmoe] [-t] [-b] [-ub] [-fa] [-ctk] [-ctv] [-dev] [-mmp 0] -r -o json`.
- `sweep` maps launch names (`gpu_layers, n_cpu_moe, threads, batch, ubatch, flash_attn, cache_type_k, cache_type_v`) to lists.
- Every combination is validated with `launch.normalize`.
- `flash_attn` "auto" isn't allowed in a sweep, and isn't passed at all when it's the single value.
- `gpu_layers` goes through `launch.runtime_gpu_layers`, so a full offload becomes total + 1.
- `-dev none` is passed when there are 0 GPU layers with a GPU backend, mirroring `server_args`.

```python
tune(bench_command, variant, base_config, hardware, budget_seconds=300, goal="generation",
     memory_check=None, progress=None, cancel=None,
     # extras:
     allow_kv_compression=False, run_bench=None, clock=time.monotonic, n_prompt=512, n_gen=128,
     depth=None, repetitions=2, confirm_repetitions=5, min_gain=0.03, timeout=900) -> dict
```

The returned dict:

```python
{"best": launch config (normalized, total_layers filled from variant.layers if missing),
 "baseline": {"tps", "pp_tps"}, "best_result": {"tps", "pp_tps"},
 "improvement": float,            # goal score best / baseline (1.0 = no change)
 "trials": [{"changes": {key: value vs base_config}, "tps", "pp_tps", "seconds",
             "status": "ok"|"failed"|"skipped_memory", "step": "baseline"|"gpu_layers"|"n_cpu_moe"|"threads"|
             "flash_attn"|"batch"|"cache"|"confirm", "error"?: plain str}],
 "stopped": "budget"|"converged"|"cancelled", "notes": [plain str],
 # extras:
 "goal", "seconds", "confirmed": bool, "settings": {"n_prompt", "n_gen", "depth", "repetitions"}}
```

- `run_bench(argv, env, timeout, cancel) -> {"returncode", "stdout", "stderr", "seconds", "timed_out"?}` can be injected. The default is `run_process`.
- `trials[].seconds` is the process time divided evenly across the combinations in that process.
- `depth` defaults to `min(1024, context // 4)` rounded down to 256. This measures with some text already in context, like a real chat, without costing much time.
- The environment for each run is `launch.server_env(config)`, so CUDA runs are pinned by UUID.

Helpers you can use: `run_process(argv, env=None, timeout=900, cancel=None)`, `parse_rows(text)` (tolerates JSON, JSONL and output cut short by a crash), `score(tps, pp_tps, goal)`, `default_memory_check(variant, hardware=None)`.

## Deviations and decisions

- **`bench_args` includes `-m` and `-o json`.** The contract only says "flags". Including them makes the argv complete and testable, like `server_args`.
- **The `batch` step is skipped for `goal="generation"`.** Batch size doesn't change token-by-token writing, so it would waste budget.
- **`allow_kv_compression` is an extra kwarg.** The contract says "when the user allowed compressed notes". The api should pass `allow_kv_compression=(requirements.kv_cache_type != "f16")` or a UI toggle. If the base config already uses q8_0, the KV step also tries f16 (memory-checked).
- **Cancel:** see "Cancel" above. A cancelled tune *after* the baseline returns a result instead of raising, so the job ends `done` with `stopped="cancelled"`.
- **Baseline failures raise `ValueError`** instead of returning a result with no `best`.
- **`trials[].changes` are relative to `base_config`**, not to the previous best, so every trial is readable on its own.

## Known gaps

- `mlock`, `parallel` (users), `draft_*`: llama-bench has no equivalent, so they aren't tuned or passed. Speeds are single-user.
- Knob candidates are fixed heuristics, not a full grid. For example `gpu_layers` only tries ±2/+4 around the start. The engine's starting value is expected to be close already.
- No per-trial peak memory. The memory check is the only safety net, plus OOM detection after the fact.
- The time estimate assumes later runs cost about the same per combination as earlier ones. A full model reload (a `gpu_layers` or `n_cpu_moe` change) is covered because the baseline cost includes the load.
- Mixed K/V cache types (such as `q8_0` K with `f16` V) aren't tried.

## Requests

- **api:**
  - Call `tune(runtime_install.binary(store, "llama-bench"), variant, launch.from_candidate(...), hardware, budget_seconds, goal, progress=progress, cancel=cancel, allow_kv_compression=...)`.
  - Store the result with `store.append("tuned", {...result, "variant_id", "sha256", "fingerprint", "context", "gpu_layers", "timestamp"})`.
  - If you also want a §2.3 measurement record, build it from `best_result` with `kind="tune"`, `settings` from `best`, and `depth` from `result["settings"]["depth"]`.
  - When `result["stopped"] == "cancelled"`, show the partial result, or mark the job cancelled if you prefer.
- **engine:** `allocations(..., kv_cache_type=, n_cpu_moe=, unified=)` as in §4.12. The default memory check already calls it that way, with a fallback.
- **server (fake_llama bench mode):** please accept `-fa 0/1` (what the tuner sends) as well as on/off. Please exit with code 1 and an OOM message on stderr for `FAKE_LLAMA_FAIL=oom`, keeping rows already printed. That matches how the tuner parses a crash mid-sweep.
- **harness:** please confirm the llama.cpp facts listed below.

## How I tested

- `tests/test_tuner.py` has 35 tests with no network and no real llama.cpp. A `FakeBench` runner parses the real argv (comma lists) and returns llama-bench-style JSON rows from a deterministic speed surface:
  - writing speed peaks at 7 threads
  - `-fa 1` is +10%
  - more GPU layers help
  - fewer CPU MoE layers help
  - q8_0 KV is −2%
  - reading prefers 2048/512 batches
- A fake clock advances by simulated load and test time.
- What the tests cover:
  - exact `bench_args` argv (plain, every setting, sweeps, auto and CPU-only, bad sweeps)
  - finding the optimum (threads 7 plus flash attention on) and the confirmation run with `-r 5`
  - no `auto` in comma lists
  - noise rejection of a 1% gain
  - the prompt goal picking 2048/512, and the balanced goal
  - the budget respected across several budgets, never more than one trial over
  - failed trials recorded while the search continues
  - a crash mid-sweep keeping earlier rows
  - unsafe configs skipped and never run, and a `memory_check` that raises
  - tight memory leading to q8_0 plus more layers
  - KV step only when allowed or tight
  - the MoE `n_cpu_moe` knob, and dense models never getting `-ncmoe`
  - cancel between trials, during a trial, and before the baseline
  - baseline errors
  - a winner failing confirmation being dropped
  - progress events, argv prefix and CUDA env pinning
  - the default memory check (mocked `hardware.scan` / `engine.allocations`, including the v0.4 kwargs)
  - `run_process` output, cancel-kill, timeout and no-shell list args
- The full suite passes: `python3 -m unittest discover -s tests` (107 tests).

## llama.cpp facts assumed (harness: please confirm)

Checked against current `tools/llama-bench/README.md` and `llama-bench.cpp` on master via WebFetch (2026-09):

- **Comma lists** are accepted for `-p -n -d -ngl -ncmoe -t -b -ub -fa -ctk -ctv` (`-r` is a single value). llama-bench runs the **cartesian product**, with one JSON row per (combination × test). The pp test and the tg test are separate rows: `n_prompt>0, n_gen=0` and `n_prompt=0, n_gen>0`, both carrying `n_depth`.
- **`-fa`:** current master takes `on|off|auto` and parses it with `is_truthy`/`is_falsey`. **Assumed:** these accept `1`/`0`. Older builds took only `0|1`, so the tuner sends `1`/`0` for compatibility.
- **`flash_attn` in JSON** is an int in current builds (`-1` auto, `0`, `1`) and was a bool in older builds. Both are handled.
- **JSON fields used:** `n_prompt, n_gen, n_depth, avg_ts, stddev_ts, n_gpu_layers, n_threads, n_batch, n_ubatch, flash_attn, type_k, type_v, n_cpu_moe`. A missing field doesn't break matching (older builds lack `n_cpu_moe`).
- **`-ncmoe` / `--n-cpu-moe`** exists in current llama-bench (int ranges or lists). Older builds lack it. The tuner only passes it for MoE models or when it's non-zero.
- **`-ngl`:** the llama-bench default is −1 (all), so the tuner always passes `-ngl` explicitly.
- **`-dev none`** is accepted to force CPU.
- **`-mmp 0`** is marked "DEPRECATED IN FAVOUR OF --load-mode" but still listed. It's only sent when `mmap=False`. **Unverified** whether newer builds still accept it; if not, drop it or map it to `--load-mode`.
- **A model-load failure makes llama-bench exit with code 1 immediately**, without trying later combinations. Assumed: rows already finished were printed and flushed to stdout (the JSON printer flushes per test), so partial output is parseable.
- **OOM text** is matched with: "out of memory", "failed to allocate", "cudaMalloc failed", "ErrorOutOfDeviceMemory", "unable to allocate", "not enough memory", "insufficient memory".
- **Defaults when a flag is omitted:** `-t` = the llama.cpp default thread count, `-b` 2048, `-ub` 512, `-fa` auto (old builds: off).
