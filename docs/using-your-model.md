# Using your model

Once a model is downloaded (and ideally tested), you can use it in two ways:

1. **Start a local server** from LLM Configurator, and point your apps at it.
2. **Export settings** for another tool you already use (Ollama, LM Studio, Docker, and so on).

Both build the engine settings in one shared place, so the settings you tested are exactly the ones you run or export. To use your saved tuned settings, add `--tuned` on the command line (the app's buttons use the recommended settings).

## Start a local server

The server is llama.cpp's `llama-server`. It speaks the **OpenAI API**: the same "language" as ChatGPT's developer interface. Many apps and code libraries can plug straight into it, like a universal power socket.

**In the app:** open **Get it running → Use it** and click **Start server**. When it is ready, the app shows the address to use, with a **Copy** button, for example:

```text
http://127.0.0.1:PORT/v1
```

The app picks a free port each time, so the number can change after a restart. Click **Stop server** to stop it. One server runs at a time: starting another model replaces the running one. Quitting the app (Ctrl+C in its terminal) stops the server too.

**On the command line:**

```bash
llm-config run VARIANT_ID --context 8192 --port 8080 --tuned
```

This loads the model, prints `Running. OpenAI-compatible address: http://127.0.0.1:8080/v1` and keeps running until you press **Ctrl+C**. Without `--port` a free port is picked for you. `--tuned` uses your saved tuned settings for that model (it stops with an error if you haven't run a tune yet). `--gpu-layers` and `--kv` work as in `llm-config test`.

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

The server listens on `127.0.0.1` only: other computers and phones on your network cannot reach it. This is deliberate. The Docker export publishes its port on `127.0.0.1` too. If you want to share a model across a network, you have to change that and set up access and security yourself.

## Export settings

**In the app:** **Use it** has a **Copy a setup** section with one tab per format, a **Mac or Linux / Windows** switch, and a **Copy** button.

**On the command line:**

```bash
llm-config export VARIANT_ID --format FORMAT [--platform posix|windows] [--context N] [--gpu-layers N] [--kv f16|q8_0|q4_0] [--tuned] [--output FILE]
```

The file is printed (or saved with `--output FILE`); the instructions and notes follow on the terminal. `--platform windows` writes PowerShell and Windows paths; `posix` (default) is for macOS and Linux.

| Format (`--format`) | Suggested file name | What you get |
|---|---|---|
| `llama-server` | `start-llama-server.sh` / `.ps1` | A ready-to-run script that starts llama.cpp's server with your settings. It uses the llama.cpp the app found, or expects `llama-server` on your `PATH`. |
| `ollama` | `Modelfile` | An Ollama **Modelfile**; the instructions give the `ollama create` command to register it. |
| `docker-compose` | `docker-compose.yml` | Runs the official llama.cpp server image (`ghcr.io/ggml-org/llama.cpp`, with the CUDA, Vulkan or ROCm version when your setup uses that graphics card), with the model folder mounted read-only and the port published on this computer only. |
| `openai-python` | `chat_local.py` | A Python snippet pointing the OpenAI library at the local server. |
| `continue` | `continue-config.yaml` | A models entry for the [Continue](https://continue.dev) coding assistant's `config.yaml`. |
| `open-webui` | `open-webui.txt` | Steps to connect [Open WebUI](https://openwebui.com) to the local server. |
| `lmstudio` | `lmstudio-settings.txt` | The settings to type into LM Studio's model load panel. |

**Port:** exports that talk to a server assume port **8080** (llama-server's usual port). If the server you started shows a different port, change the number in the copied setup to match.

Each export also lists plain **instructions** and **notes**. Read the notes: some tools can't express every setting. For example:

- **Ollama** can't set notepad (KV cache) compression per model: it is a setting for the whole Ollama server (`OLLAMA_KV_CACHE_TYPE`). It also can't keep MoE experts in RAM the way llama.cpp's `--n-cpu-moe` does, and it can't load a model split into several files, so the instructions show how to merge them first. The notes tell you what was left out.
- **Docker** needs Docker installed. NVIDIA cards also need the NVIDIA Container Toolkit (Linux) or Docker Desktop with WSL 2 (Windows). Docker on a Mac can't use the Apple graphics chip, so use the llama-server script there.

If the model isn't downloaded yet, the export uses the path where the file will be saved once you download it, and says so.

Exports contain no timestamps or secrets, so the same settings always produce the same file.
