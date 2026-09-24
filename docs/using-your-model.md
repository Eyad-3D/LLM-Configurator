# Using your model

Once a model is downloaded (and ideally tested), you can use it in two ways:

1. **Start a local server** from LLM Configurator, and point your apps at it.
2. **Export settings** for another tool you already use (Ollama, LM Studio, Docker, and so on).

Both build the engine settings in one shared place, so the settings you tested are exactly the ones you run or export. If you have tuned this model at this context and placement, the app uses the tuned settings automatically. On the command line, add `--tuned` to also use a tune that moved layers, or to get an error if there is no matching tune (see [Testing and tuning](testing-and-tuning.md#the-result)).

## Start a local server

The server is llama.cpp's `llama-server`. It speaks the **OpenAI API**: the same "language" as ChatGPT's developer interface. Many apps and code libraries can plug straight into it, like a universal power socket.

**In the app:** open **Get it running → Use it** and click **Start server**. When it is ready, the app shows the address to use, with a **Copy** button, for example:

```text
http://127.0.0.1:PORT/v1
```

The app uses port 8080 when it is free, so the setups under **Copy a setup** work as shown. If another program already uses 8080, the app picks a different free port; then change the port in any copied setup to match. Click **Stop server** to stop it. One server runs at a time: starting another model replaces the running one. Quitting the app (Ctrl+C in its terminal) stops the server too.

**On the command line:**

```bash
llm-config run VARIANT_ID --context 8192 --port 8080 --tuned
```

This loads the model, prints `Running. OpenAI-compatible address: http://127.0.0.1:8080/v1` and keeps running until you press **Ctrl+C**. Without `--port` a free port is picked for you; if the port you name is already in use, it stops with a plain message. `--tuned` uses your saved tune for that model, context and placement (it stops with an error if there isn't one). `--gpu-layers` and `--kv` work as in `llm-config test`.

### Connecting an app

Use these settings in any OpenAI-compatible app or library:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:PORT/v1` (the address shown) |
| API key | anything, e.g. `local` (it is ignored, but some apps require a value) |
| Model name | any name usually works, since one model is loaded |

Python example (needs `pip install openai`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
reply = client.chat.completions.create(
    model="local",
    messages=[{"role": "user", "content": "Say hello in five words."}],
)
print(reply.choices[0].message.content)
```

The **openai-python** export (below) gives you this snippet ready to save.

### Only on this computer

The server listens on `127.0.0.1` only: other computers and phones on your network cannot reach it. This is deliberate. If you want to share a model across a network, you have to change that and set up access and security yourself.

Two more locks on the door:

- **Websites can't read it.** Normally any web page open in your browser could send the server a question and read the answer. The app starts the server with `LLAMA_ARG_CORS_ORIGINS=localhost`, so only pages served from this computer can. The exported llama-server scripts and the Docker setup set the same.
- **No request monitor.** llama-server's `/slots` page (which shows the settings and size of each request in progress) is switched off for the server the app starts.

The app also ignores any `LLAMA_ARG_…` settings in your environment when it starts the server, so it runs with exactly the settings you tested.

The Docker export keeps the same rule in its own way. Inside the container the server listens on all addresses (`0.0.0.0`), because Docker has to pass traffic in. But the port is published on `127.0.0.1` only, so from outside, only this computer can reach it.

## Export settings

**In the app:** **Use it** has a **Copy a setup** section with one tab per format, a **Mac or Linux / Windows** switch, and a **Copy** button.

**On the command line:**

```bash
llm-config export VARIANT_ID --format FORMAT [--platform posix|windows] [--context N] [--gpu-layers N] [--kv f16|q8_0|q4_0] [--port N] [--tuned] [--output FILE]
```

The file is printed (or saved with `--output FILE`); the instructions and notes follow on the terminal. `--platform windows` writes PowerShell and Windows paths; `posix` is for macOS and Linux. The default is the system you are on. A `.sh` file saved with `--output` is made runnable, and a `.ps1` file is saved so that Windows PowerShell reads any non-English characters correctly.

| Format (`--format`) | Suggested file name | What you get |
|---|---|---|
| `llama-server` | `start-llama-server.sh` / `.ps1` | A ready-to-run script that starts llama.cpp's server with your settings. It uses the llama.cpp the app found, or expects `llama-server` on your `PATH`. It sets `LLAMA_ARG_CORS_ORIGINS=localhost` (see above) and, on an NVIDIA card, `CUDA_VISIBLE_DEVICES` to pin the card you tested on. |
| `ollama` | `Modelfile` | An Ollama **Modelfile**; the instructions give the `ollama create` command to register it. |
| `docker-compose` | `docker-compose.yml` | Runs the official llama.cpp server image (`ghcr.io/ggml-org/llama.cpp`). When the model uses the graphics card, it picks the CUDA, Vulkan or ROCm image to match the llama.cpp build you tested with. The model folder is mounted read-only and the port is published on this computer only. |
| `openai-python` | `chat_local.py` | A Python snippet pointing the OpenAI library at the local server. |
| `continue` | `continue-config.yaml` | A models entry for the [Continue](https://continue.dev) coding assistant's `config.yaml`. |
| `open-webui` | `open-webui.txt` | Steps to connect [Open WebUI](https://openwebui.com) to the local server. |
| `lmstudio` | `lmstudio-settings.txt` | The settings to type into LM Studio's model load panel. |

**Port:** exports that talk to a server assume port **8080** (llama-server's usual port); on the command line, `--port N` picks another. If the server you started shows a different port, change the number in the copied setup to match.

**GPU layers:** when the graphics card is used, exported commands (`-ngl`, and Ollama's `num_gpu`) are one higher than the number of layers the app shows. That's because llama.cpp counts the output layer as one more layer (see [How it works](how-it-works.md#splitting-between-graphics-card-and-ram)).

Each export also lists plain **instructions** and **notes**. Read the notes: some tools can't express every setting. For example:

- **Ollama** can't set notepad (KV cache) compression or flash attention per model: they are settings for the whole Ollama server (`OLLAMA_KV_CACHE_TYPE`, `OLLAMA_FLASH_ATTENTION`), and so is serving several chats at once (`OLLAMA_NUM_PARALLEL`). The instructions show how to set them. Ollama also can't keep MoE experts in RAM the way llama.cpp's `--n-cpu-moe` does. For a model split into several files, the instructions show how to merge them into one first (recent Ollama versions can also load the parts directly). The notes tell you what was left out.
- **Docker** needs Docker installed. NVIDIA cards also need the NVIDIA Container Toolkit (Linux) or Docker Desktop with WSL 2 (Windows). Vulkan and ROCm images only get the graphics card on Linux. Docker on a Mac can't use the Apple graphics chip, so use the llama-server script there.
- **LM Studio** uses its own copy of llama.cpp, so its speed can differ a little from your test.

If the model isn't downloaded yet, the export uses the path where the file will be saved once you download it, and says so.

Exports contain no timestamps or secrets, so the same settings always produce the same file.
