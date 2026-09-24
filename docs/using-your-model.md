# Using your model

Once a model is downloaded (and ideally tested), you can use it in two ways:

1. **Start a local server** from LLM Configurator, and point your apps at it.
2. **Export settings** for another tool you already use (Ollama, LM Studio, Docker, and so on).

Both build the engine settings in one shared place, so the settings you tested are exactly the ones you run or export. Add your tuned settings with `--tuned` on the command line.

## Start a local server

The server is llama.cpp's `llama-server`. It speaks the **OpenAI API**: the same "language" as ChatGPT's developer interface. Many apps and code libraries can plug straight into it, like a universal power socket.

**In the app:** open **Get it running → Use it** and start the server. The app shows the address, for example:

```text
http://127.0.0.1:PORT/v1
```

Stop it from the same place. One server runs at a time.

**On the command line:**

```bash
llm-config run VARIANT_ID --context 8192 --port 8080 --tuned
```

This runs the server in the foreground and prints its address. Stop it with **Ctrl+C**. Without `--port` a free port is picked for you. `--tuned` uses your saved tuned settings for that model.

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

The **openai-python** export (below) gives you this snippet with your real address filled in.

### Only on this computer

The server listens on `127.0.0.1` only: other computers and phones on your network cannot reach it. This is deliberate. If you want to share a model across a network, use an export (for example Docker) and set up access and security yourself.

## Export settings

**In the app:** **Use it** shows one tab per format with a copy button.
**On the command line:**

```bash
llm-config export VARIANT_ID --format FORMAT [--platform posix|windows] [--context N] [--gpu-layers N] [--tuned]
```

`--platform windows` writes PowerShell; `posix` (default) writes bash for macOS and Linux.

| Format (`--format`) | What you get |
|---|---|
| `llama-server` | A ready-to-run script that starts llama.cpp's server with your settings. |
| `ollama` | An Ollama **Modelfile** plus the `ollama create` command to register it. |
| `docker-compose` | A `docker-compose.yml` using the official llama.cpp server image (CUDA version for NVIDIA), with the model folder mounted and the port published. |
| `openai-python` | A Python snippet pointing the OpenAI library at your local server. |
| `continue` | A config snippet for the [Continue](https://continue.dev) coding assistant. |
| `open-webui` | Steps to connect [Open WebUI](https://openwebui.com) to your local server. |
| `lmstudio` | The settings to type into LM Studio's model load panel. |

Each export also lists plain **instructions** and **notes**. Read the notes: some tools can't express every setting. For example:

- **Ollama** can't set notepad (KV cache) compression per model (it is a setting for the whole Ollama server), and can't keep MoE experts in RAM the way llama.cpp's `--n-cpu-moe` does. The note tells you what was left out.
- **Docker** needs Docker installed, and the NVIDIA Container Toolkit for graphics card use.

If the model isn't downloaded yet, exports use a placeholder path and say so. Replace it with the real file path.

Exports contain no timestamps or secrets, so the same settings always produce the same file.
