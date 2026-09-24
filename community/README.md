# Community speed results

This folder holds real speed results that people shared from their own computers. LLM Configurator can show them before you download anything: "this model, on hardware like yours, wrote about 42 tokens (word pieces) per second for other people."

They are **other people's computers**, not yours. The app labels them that way and never mixes them up with your own measurements.

## What gets shared

Only when you choose to share, and only these things:

| Shared | Example |
|---|---|
| Model: repository, file name, file fingerprint (sha256), compression level (quant) | `Qwen/Qwen3-8B-GGUF`, `Qwen3-8B-Q4_K_M.gguf`, `Q4_K_M` |
| CPU and GPU model names, cleaned | `Intel Core i7-9700K`, `NVIDIA GeForce RTX 4090` |
| GPU type (backend) | `cuda`, `metal`, `rocm`, `vulkan`, `cpu` |
| Memory, rounded to 4 GB steps | `24` GB video memory, `32` GB RAM |
| Operating system family | `Linux`, `Windows 11`, `macOS 14` |
| Chip family | `x86_64`, `arm64` |
| Settings | context size, how many layers ran on the GPU, threads, flash attention, KV cache type |
| Speeds | writing speed, reading speed, time to first word |
| llama.cpp build tag | `b4500` |
| Month of the test | `2026-09` |
| A random id | `3f9c0a1b2d4e5f60` (new random value every time; not linked to your computer) |

## What never gets shared

- Your computer's name, your user name, or anything from file paths or folders.
- Serial numbers, GPU IDs (UUIDs), the app's hardware fingerprint, process lists or program names.
- The exact date or time.
- The name of a model you added yourself or found on disk. For those, only the file fingerprint (sha256) is shared, because a private file or repository name could identify you.

Hardware names are cleaned before sharing: brackets, serial-like codes, clock speeds and anything that matches your computer or user name are removed.

## How to contribute

1. Run a speed test in the app.
2. Choose **Share results**. The app shows you the exact JSON it would share.
3. Read it. If you are happy, open the prefilled GitHub link. It opens a new issue on this repository with the JSON already filled in.
4. Submit the issue yourself. **The app never posts anything for you.**

If you share many results at once, the link would be too long for GitHub. The app then opens an issue with a short note and asks you to paste the JSON it showed you.

The link asks for the `community-results` label. GitHub only applies it if you have permission, so maintainers may add it.

## File format (schema 1)

`results.json` is:

```json
{"schema": 1, "records": [ ... ]}
```

Each record:

```json
{
  "schema": 1, "id": "3f9c0a1b2d4e5f60", "month": "2026-09", "kind": "speed_test",
  "tps": 101.3, "pp_tps": 3000.0, "ttft_s": 0.21, "context": 8192, "depth": 0,
  "variant": {"repo": "Qwen/Qwen3-8B-GGUF", "filename": "Qwen3-8B-Q4_K_M.gguf",
              "sha256": "<64 hex characters>", "quant": "Q4_K_M", "layers": 36},
  "hardware": {"cpu": "Intel Core i7-9700K", "gpu": "NVIDIA GeForce RTX 4090", "backend": "cuda",
               "vram_gib": 24, "ram_gib": 32, "unified": false, "os": "Linux", "os_version": null, "arch": "x86_64"},
  "settings": {"placement": "gpu", "gpu_layers": 36, "users": 1, "threads": 8, "flash_attn": "on",
               "cache_type_k": "q8_0", "cache_type_v": "q8_0", "batch": 2048, "ubatch": 512, "n_cpu_moe": 0},
  "runtime": {"version": "b4500", "backend": "cuda"}
}
```

- `tps`: writing speed in tokens per second. `pp_tps`: reading speed. `ttft_s`: seconds until the first word.
- `placement`: `gpu` (the whole model on the graphics card), `split` (part on the card, part in RAM) or `cpu`.
- Memory values are whole multiples of 4. Unknown values are `null`, never guessed.

The app checks every row strictly when it imports this file: types, ranges, allowed values and finite numbers. Rows that fail are dropped and counted. Unknown keys are thrown away. The file may be at most 10 MB and 50,000 rows. It is only fetched over HTTPS.

## For maintainers: merging results

Nothing merges automatically. Shared results come from strangers, so a person checks them first.

1. Open issues labelled `community-results` (add the label if GitHub dropped it).
2. Read the JSON. Close the issue without merging if it looks wrong: impossible speeds for the hardware, many near-identical rows, or anything that looks personal.
3. Copy the JSON into a local file, then check it with the app's own validator:

   ```sh
   python -c "import json,sys; from llm_configurator import community as c; \
   rows, bad = c.parse(open(sys.argv[1]).read()); print(len(rows), 'valid,', bad, 'rejected'); \
   json.dump({'schema': 1, 'records': rows}, open(sys.argv[1] + '.clean.json', 'w'), indent=1)" issue.json
   ```

4. Add the cleaned rows to `records` in `results.json`. Skip ids that are already there.
5. Check the whole file still validates (`c.parse(open("community/results.json").read())` should report 0 rejected), then open a pull request that links the issue.
6. Close the issue once the pull request is merged.
