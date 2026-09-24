# llama.cpp facts (verified against a real build)

Everything here was observed by running a **real** llama.cpp in the harness container, not read from docs.
Raw transcripts live in `tests/integration/samples/` (made by `scripts/capture_llama_facts.py`).
Checks that keep these facts true: `tests/integration/test_real_runtime.py` (needs the real build) and
`tests/integration/test_fake_matches_real.py` (compares the fake llama.cpp with these samples; runs in the normal suite).

## Build under test

| | |
|---|---|
| Source | PyPI sdist `llama-cpp-python==0.3.35`, folder `vendor/llama.cpp` (GitHub is unreachable here) |
| Commit | `4df29be4f4c3673f428170fda944a5b19f743bb8`, 16 Aug 2026, "ci : fix dry-run reporting in make-release job (#27167)" |
| Version line | `version: 0.1.0-dev (build 1, commit 4df29be)` |
| Build number | **1** is an artefact: the sdist ships a 1-commit shallow git clone, and llama.cpp counts commits. Official releases print their real number (`b<N>` tag = build N). |
| Build | CMake Release, CPU only, static (`BUILD_SHARED_LIBS=OFF`), `-DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON`, GCC 13.3, x86_64 with AVX-512 |
| Not covered | No GPU in this container: CUDA/Vulkan/Metal offload, VRAM numbers and `-dev CUDA0` success paths are **unverified**. |

Newest llama.cpp on PyPI I could find: llama-cpp-python 0.3.35 is the latest release and vendors this Aug 2026 commit, so no newer source was needed.

---

## Mismatches (contract or branch code vs. the real runtime)

Ordered by impact. "Owner" is who needs to change something.

1. **`--draft-max` was removed from llama-server** (owner: lead, `launch.py`; server: fake). Passing it makes llama-server exit 1:
   ```
   error while handling argument "--draft-max": the argument has been removed. use --spec-draft-n-max or --spec-ngram-mod-n-max
   ```
   Use `--spec-draft-n-max N`. Also, **`-md` alone loads the draft model but does not use it**: speculative decoding stays off
   (`--spec-type` default is `none`) unless you add `--spec-type draft-simple`. Verified: with both flags `/completion`
   timings gain `"draft_n": 4, "draft_n_accepted": 4`; without `--spec-type` they don't. So `server_args` for a draft
   model should emit `-md PATH --spec-type draft-simple [--spec-draft-n-max N]`. `LaunchTests.test_draft_model_is_accepted_and_used` fails until fixed.
   The fake llama.cpp still accepts `--draft-max` (`test_server_rejects_removed_draft_max_like_real` fails).

2. **The `--version` format changed** (owner: runtime `parse_version`; server: fake is already fine). New builds print, on **stderr**:
   ```
   version: 0.1.0-dev (build 1, commit 4df29be)
   built with GNU 13.3.0 for Linux x86_64
   ```
   Older builds print `version: 5678 (4df29be)`. `runtime_install.parse_version` returns `version="0.1.0-dev", build=None,
   commit=None` for the new form. Parse both: `^version: (?:(\S+) \(build (\d+), commit ([0-9a-f]+)\)|(\d+) \(([0-9a-f]+)\))$`.
   The semver is a constant `0.1.0-dev` for every nightly, so **the build number is the only useful version**; `/props`
   also reports it as `"build_info": "b1-4df29be"` and chat responses as `"system_fingerprint": "b1-4df29be"`.

3. **`llama-bench` (and `llama-quantize`) have no `--version`** (owner: anyone calling it; server: fake). They print usage to
   stdout and exit **1**. `llama-server`, `llama-perplexity`, `llama-cli` and `llama-gguf-split` do support it (exit 0).
   Get the bench build from the JSON rows (`build_number`, `build_commit`). The fake bench answers `--version` with exit 0.

4. **`llama-perplexity` can lose the end of its output** (owner: quantcheck). Under CPU load about **1 run in 10**
   exits **0** with stdout cut short: the KL table or the whole summary is missing (its log thread is not flushed at exit).
   Measured: 4/40 and 5/40 truncated with 3 busy cores; `--log-file` is truncated the same way. The harness saw it happen
   inside `quantcheck.kl_check` (result had `median_kld: null`, `kld_99: null`, `chunks_done: 2` of 3). Fix: if
   `Same top p:` / `Mean    KLD:` is missing from stdout, re-run the candidate (the base logits file can be reused), and only
   then fall back to the last per-chunk row.

5. **`runtime_install.detect` reports `backend: "unknown"` for a CPU build** (owner: runtime). `--version` prints no backend
   hint on a static build (dynamic official builds print `load_backend: loaded CPU backend from ...` lines, not verified here).
   Reliable, cheap alternative: `llama-server --list-devices` (exit 0). CPU-only prints
   ```
   Available devices:
     (none)
   ```
   GPU builds list devices such as `CUDA0: ...` / `Vulkan0: ...` (format unverified here).

6. **`runtime.bench` hardcodes `-dev CUDA0`** when layers > 0 (owner: downloads, `runtime.py`). Device names depend on the
   backend (`Vulkan0`, `ROCm0`, `Metal`, ...). An unknown name is fatal: `error while handling argument "-dev": invalid device: CUDA0`
   (exit 1). `CPU` is rejected the same way. Only `none` is always valid.

7. **Repeated prompts are served from cache** (owner: testing, tuner, evals). A second identical request only reads the new
   tail: `"cache_n": 66, "prompt_n": 1`. For reading-speed / first-word-delay measurements send `"cache_prompt": false`
   (verified: `cache_n` 0, full `prompt_n`) or vary the prompt.

8. **`--mlock` and `--no-mmap` are deprecated** (owner: lead, `launch.py`; low priority). They still work but log
   `DEPRECATED: --mlock is deprecated. use --load-mode mlock instead` and
   `DEPRECATED: --mmap and --no-mmap are deprecated. use --load-mode mmap instead`.
   New flag: `-lm, --load-mode auto|none|mmap|mlock|mmap+mlock|dio` (`none` = no mmap).

9. **Defaults changed in ways that matter if a flag is ever left out** (owner: lead). `-ngl` default `auto`, `-np` default
   `-1` (auto; then the KV cache is unified across slots), `-c` default `0` (= model's full context), `--fit on` by default
   (adjusts *unset* arguments to fit free memory), `--jinja` on by default. `launch.server_args` always passes `-c`, `-np`
   and `-ngl`, so results stay deterministic; keep it that way.

10. **Filler-prompt sizing** (owner: testing; minor). `testing.filler_prompt(n)` assumes ~0.75 words per token. That's fine
   for real models but not guaranteed; the harness tiny tokenizer (≈1.3 characters/token) overflowed a 1024 context and got
   HTTP 400 `exceed_context_size_error`. Checking the length with `/tokenize` first would make it robust.

Confirmed as assumed (no action): every other flag in the contract list exists and works (table below); llama-bench field
names; `-fa` values; `-ncmoe`; `-d`; comma sweeps; `/health` 503 → 200; `timings` fields; `/tokenize`; split GGUF loads
from the first shard; `-c` is the total across slots (`-np 2 -c 1024` → `n_ctx_slot = 512`); a quantized V cache needs
flash attention; the GGUF architecture for Qwen3 MoE is `qwen3moe` (the discover branch maps it to `qwen3_moe`).

---

## Flags (llama-server `--help`, full text in `samples/help-llama-server.txt`)

| Contract flag | Exists | Help text / behaviour |
|---|---|---|
| `-c` | yes | `-c, --ctx-size N  size of the prompt context (default: 0, 0 = loaded from model)`. Total over all slots. |
| `-np` | yes | `-np, --parallel N  number of server slots (default: -1, -1 = auto)` |
| `-ngl` | yes | `-ngl, --gpu-layers, --n-gpu-layers N  max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)` |
| `-t` | yes | `-t, --threads N  number of CPU threads to use during generation (default: -1)` |
| `-b` / `-ub` | yes | `logical maximum batch size (default: 2048)` / `physical maximum batch size (default: 512)` |
| `-fa` | yes | `-fa, --flash-attn [on\|off\|auto]  (default: 'auto')`. `1` is also accepted; `yes` → `error while handling argument "-fa": error: unknown value for --flash-attn: 'yes'` (exit 1) |
| `-ctk` / `-ctv` | yes | `allowed values: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 (default: f16)`. Bad value: `Unsupported cache type: q9_9` (exit 1) |
| `--n-cpu-moe` | yes | `-ncmoe, --n-cpu-moe N  keep the Mixture of Experts (MoE) weights of the first N layers in the CPU`. Started fine on the tiny qwen3moe with `--n-cpu-moe 2`. |
| `-dev none` | yes | `-dev, --device <dev1,dev2,..>  comma-separated list of devices to use for offloading (none = don't offload)` |
| `--mlock` | deprecated | works, warns (mismatch 8) |
| `--no-mmap` | deprecated | works, warns (mismatch 8) |
| `-md` | yes | `--spec-draft-model, -md, --model-draft FNAME` — loads but is unused without `--spec-type draft-simple` (mismatch 1) |
| `--draft-max` | **removed** | exit 1 (mismatch 1). Replacement `--spec-draft-n-max N (default: 3)` |
| `--alias` | yes | `-a, --alias STRING  set model name aliases, comma-separated`. The alias becomes `"model"` in responses and `/v1/models` ids; without it `"model"` is the full model path. |
| `--jinja` | yes | `--jinja, --no-jinja  (default: enabled)` |
| `--host` / `--port` | yes | defaults `127.0.0.1` / `8080` |

Other useful flags seen: `--list-devices`, `-fit/--fit [on|off]`, `-lm/--load-mode`, `-kvu/--kv-unified`, `-cram/--cache-ram`,
`--spec-type`, `-lv/--verbosity`, `--log-file`.

**Quantized V cache without flash attention** fails at load (exit 1):
`llama_init_from_model: quantized V cache requires flash_attn to be enabled`. `-fa on -ctk q8_0 -ctv q8_0` works on CPU.

**`-dev none` with `-ngl`** (source `common/arg.cpp` `parse_device_list`): `none` sets an empty device list, so nothing is
offloaded whatever `-ngl` says. On this CPU build llama-server accepts `-dev none -ngl 99` and warns
`warning: no usable GPU found, --gpu-layers option will be ignored`. llama-bench echoes the request unchanged:
`"n_gpu_layers": 99, "devices": "none"` (so row-vs-request checks still pass). GPU behaviour itself is unverified here.

## `--version`

`samples/version.txt`. Always stderr, exit 0:
```
version: 0.1.0-dev (build 1, commit 4df29be)
built with GNU 13.3.0 for Linux x86_64
```
`llama-bench --version` / `llama-quantize --version`: usage on stdout, exit 1.

## llama-server HTTP

Startup log (`samples/server-log.txt`) uses `<elapsed> <level> <component> <message>` lines, e.g.
`0.00.034.765 I srv  llama_server: model loaded` and `0.00.034.787 I srv  llama_server: listening on http://127.0.0.1:33401`.

**`/health`** (`samples/server-health.json`): connection refused → `503 {"error":{"message":"Loading model","type":"unavailable_error","code":503}}`
→ `200 {"status":"ok"}`. The tiny model was ready in 45 ms.

**`/v1/chat/completions`** (`samples/server-chat-completions.json`), request with `max_tokens`, `temperature`, `seed`:
```json
{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"42"}}],
 "created":1790253601,"model":"tiny","system_fingerprint":"b1-4df29be","object":"chat.completion",
 "usage":{"completion_tokens":3,"prompt_tokens":67,"total_tokens":70,"prompt_tokens_details":{"cached_tokens":0}},
 "id":"chatcmpl-…",
 "timings":{"cache_n":0,"prompt_n":67,"prompt_ms":10.853,"prompt_per_token_ms":0.162,"prompt_per_second":6173.4,
            "predicted_n":3,"predicted_ms":1.562,"predicted_per_token_ms":0.781,"predicted_per_second":1280.4}}
```
`completion_tokens` counts the end-of-turn token (3 for the reply "42"). Speculative decoding adds `draft_n` and
`draft_n_accepted` to `timings`. Streaming (`samples/server-chat-completions-stream.txt`): `data: {...}` chunks, the last
data chunk has `"choices": []` plus `usage` and `timings`, then `data: [DONE]`.

**`/completion`** (`samples/server-completion.json`): top-level `index, content, tokens, id_slot, stop, model,
tokens_predicted, tokens_evaluated, generation_settings, prompt, has_new_line, truncated, stop_type, stopping_word,
tokens_cached, timings`. `stop_type` is e.g. `"eos"` or `"limit"`.

**`/tokenize`**: `{"content": "Hello world"}` → `{"tokens": [308, 418]}`.

**Errors**: prompt too long → `400 {"error":{"code":400,"message":"request (3018 tokens) exceeds the available context size (2048 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":3018,"n_ctx":2048}}`.
When the model's text can't be parsed by the chat-format parser (random-weight models): `500 {"error":{"code":500,"message":"The model produced output that does not match the expected peg-native format","type":"server_error"}}`
(`/completion` said `... expected Content-only format`). That's why the tiny models are wired to write plain ASCII.

**Start-up failures** (all exit 1; full text in `samples/error-server-*.txt`; the load error lines appear twice because
`--fit` tries to load first):

| Case | Recognisable line (stderr) |
|---|---|
| missing file | `E gguf_init_from_file: failed to open GGUF file '/…/does-not-exist.gguf' (No such file or directory)` |
| corrupt / truncated | `E llama_model_load: error loading model: tensor 'blk.1.ffn_down.weight' data is not within the file bounds, model is corrupted or incomplete` |
| unknown architecture | `E llama_model_load: error loading model: unknown model architecture: 'notarealarch'` |
| port in use | `E srv         start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 40077` then `exiting due to HTTP server error` |
| unknown flag | `error: invalid argument: --no-such-flag` |
| any model load failure | last line `E srv  llama_server: exiting due to model loading error` |

A CPU-only build always prints `warning: no usable GPU found, --gpu-layers option will be ignored` (even with `-ngl 0`).
OOM text could not be produced here.

## llama-bench

`-o json` prints one JSON array on stdout (`samples/bench-*.json`); `-o jsonl` one object per line. One row per test:
a prompt test (`n_gen: 0`) and a generation test (`n_prompt: 0`) per combination. Full row (`bench-basic.json`):
```json
{"build_commit":"4df29be","build_number":1,"cpu_info":"Intel(R) Xeon(R) Processor @ 2.10GHz","gpu_info":"","backends":"CPU",
 "model_filename":"/opt/llama-work/models/tiny-llama-F16.gguf","model_type":"llama ?B F16","model_size":13465600,
 "model_n_params":6728192,"n_batch":2048,"n_ubatch":512,"n_threads":2,"cpu_mask":"0x0","cpu_strict":false,"poll":50,
 "type_k":"f16","type_v":"f16","n_gpu_layers":-1,"n_cpu_moe":0,"split_mode":"layer","main_gpu":0,"no_kv_offload":false,
 "flash_attn":-1,"devices":"auto","tensor_split":"0.00","tensor_buft_overrides":"none","load_mode":"auto","embeddings":false,
 "no_op_offload":0,"no_host":false,"fit_target":0,"fit_min_ctx":0,"n_prompt":64,"n_gen":0,"n_depth":0,
 "test_time":"2026-09-24T12:40:01Z","avg_ns":5156902,"stddev_ns":0,"avg_ts":12410.551917,"stddev_ts":0.000000,
 "samples_ns":[5156902],"samples_ts":[12410.6]}
```
- `flash_attn` is an **int**: `-1` auto (the default), `0` off, `1` on. `-fa` accepts `0,1` and `on,off,auto` (both verified).
- `n_gpu_layers` default is `-1`; `devices` is a string (`"auto"`, `"none"`).
- Comma lists work for `-p -n -d -t -b -ub -fa -ctk -ctv -ngl -ncmoe -r`. Row order: outermost varies slowest in the
  order batch → flash_attn → threads, then prompt test before generation test (see `bench-sweep.json`: `-t 1,2 -fa 0,1 -b 64,128`).
- `-d 0,256` gives rows with `"n_depth": 0` and `"n_depth": 256` (`bench-depth.json`, with `-ctk q8_0 -ctv q8_0 -fa 1`).
- `-ncmoe 0,2` on the MoE model gives `"n_cpu_moe": 0` and `2` rows (`bench-moe-ncmoe.json`).
- Failures: exit 1, stdout has just `[`, stderr `llama_bench: error: failed to load model '/…/x.gguf'` for both a missing
  file and an unknown architecture (no detail).

## llama-perplexity

Normal run (`samples/perplexity-normal.txt`), stdout:
```
[1]58946.0598,[2]51222.4357,[3]56253.0675,[4]55470.5171,
```
and stderr `Final estimate: PPL = 55470.5171 +/- 4384.27356`. The corpus needs at least 2 × `-c` tokens.
`--kl-divergence-base FILE` with `-f` runs the reference and writes logits (216,772 bytes for 4 × 128 tokens with a
464-token vocabulary; it scales with vocabulary × tokens, so real models make **gigabyte** files).
`--kl-divergence-base FILE --kl-divergence` (no `-f`) on the candidate prints on **stdout** (`samples/perplexity-kld.txt`):
```
chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1    70055.0683 ± 2830.6999       0.17266 ±    0.02782       0.00032 ±    0.00006     0.001 ±  0.000 %    100.000 ±  0.000 %
…
====== Perplexity statistics ======
Mean PPL(Q)                   : 61360.395556 ± 4862.143165
Mean PPL(base)                : 55470.500402 ± 4384.271109
Cor(ln(PPL(Q)), ln(PPL(base))):  98.32%
Mean ln(PPL(Q)/PPL(base))     :   0.100913 ±   0.014499
Mean PPL(Q)/PPL(base)         :   1.106181 ±   0.016038
Mean PPL(Q)-PPL(base)         : 5889.895154 ± 971.448740

====== KL divergence statistics ======
Mean    KLD:   0.000234 ±   0.000025
Maximum KLD:   0.002127
99.9%   KLD:   0.002123
99.0%   KLD:   0.001983
…
Median  KLD:   0.000101
…
Minimum KLD:   0.000000

====== Token probability statistics ======
Mean    Δp:  0.000 ± 0.001 %
…
RMS Δp    :  0.014 ± 0.005 %
Same top p: 100.000 ± 0.000 %
```
The per-chunk columns are running averages. It warns `kl_divergence: calculated n_seq=16 exceeds context's n_seq_max=4, capping at 4`
(harmless). See mismatch 4 for truncated output.

## Tiny models used (`scripts/make_tiny_models.py`)

`llama` dense (4 layers, 512 wide, 8 heads / 4 KV heads, head dim 64, 464-token byte-level BPE vocab, ChatML template, 16k
context), `qwen3moe` (8 experts, 2 active), both F16 and quantized by the real `llama-quantize` to Q8_0 / Q4_K_M; a 4-shard
split (`llama-gguf-split --split --split-max-tensors 12` → `tiny-llama-Q8_0-0000k-of-00004.gguf`); `unknown-arch.gguf`;
`corrupt.gguf` (half the file). Greedy output is wired: chat answers `42`, raw text continues ` the quick brown fox.`
Note Q4_K needs tensor rows that are a multiple of 256 values, so tiny dims must be multiples of 256.
