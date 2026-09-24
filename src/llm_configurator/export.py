"""Hand a tested launch configuration to the tools people already use.

Every format is built from `launch.server_args`, so the exported command is the one that was
tested. Paths are quoted for the target shell, never pasted raw. Output is deterministic
(no timestamps). Settings a target tool cannot express are listed in `notes`, never dropped
silently.
"""
import json
import ntpath
import posixpath
import re
import shlex

from . import launch

PLACEHOLDER = "<path-to-model.gguf>"
DRAFT_PLACEHOLDER = "<path-to-draft-model.gguf>"
DEFAULT_PORT = 8080  # llama-server's own default
API_KEY = "not-needed"
SPLIT = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)
PS_QUOTES = "'‘’‚‛"  # PowerShell treats all of these as single quotes
PS_BARE = re.compile(r"^(--?[A-Za-z][A-Za-z0-9-]*|[A-Za-z0-9_]+)$")
DOCKER_IMAGES = {"cuda": "server-cuda", "vulkan": "server-vulkan", "rocm": "server-rocm"}
RUNTIME_GPU_BACKENDS = {"cuda", "vulkan", "rocm", "metal"}

FORMATS = [
    {"id": "llama-server", "label": "llama-server script",
     "description": "A ready script (bash or PowerShell) that starts llama.cpp's server with the tested settings."},
    {"id": "ollama", "label": "Ollama Modelfile",
     "description": "A Modelfile so Ollama runs the same file with matching context, GPU layers and threads."},
    {"id": "docker-compose", "label": "docker-compose",
     "description": "A docker-compose.yml that runs the official llama.cpp server image, reachable from this computer only."},
    {"id": "openai-python", "label": "Python (OpenAI client)",
     "description": "A short Python snippet that chats with the local server through the OpenAI library."},
    {"id": "continue", "label": "Continue (VS Code / JetBrains)",
     "description": "A models entry for Continue's config.yaml pointing at the local server."},
    {"id": "open-webui", "label": "Open WebUI",
     "description": "Steps to add the local server as a connection in Open WebUI."},
    {"id": "lmstudio", "label": "LM Studio",
     "description": "The settings to enter in LM Studio's load panel to match the tested setup."},
]


def formats():
    return [dict(item) for item in FORMATS]


def _line(text):
    """One printable line for comments and prose: control characters become spaces."""
    return re.sub(r"[\x00-\x1f\x7f]+", " ", str(text)).strip()


def _slug(text):
    """Lowercase name safe for Ollama, compose services and model aliases."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", str(text).lower()).strip("-._")
    return re.sub(r"-{2,}", "-", slug)[:80].strip("-._") or "local-model"


def _title(variant):
    if variant is None:
        return "this model"
    quant = getattr(variant, "quant", None)
    return _line(f"{variant.name} {quant}" if quant else variant.name)


def _alias(config, variant):
    if config.get("alias"):
        return config["alias"]
    if variant is None:
        return "local-model"
    return _slug(f"{variant.name}-{getattr(variant, 'quant', '') or ''}")


def ps_quote(text):
    """PowerShell single-quoted string: nothing inside expands; each quote character is doubled."""
    return "'" + "".join(ch * 2 if ch in PS_QUOTES else ch for ch in str(text)) + "'"


def _ps_word(token):
    return token if PS_BARE.match(token) else ps_quote(token)


def _grouped(args):
    """Pair each flag with its value so scripts read one setting per line."""
    lines, index = [], 0
    while index < len(args):
        if args[index].startswith("-") and index + 1 < len(args) and not args[index + 1].startswith("-"):
            lines.append(args[index:index + 2])
            index += 2
        else:
            lines.append([args[index]])
            index += 1
    return lines


def _prepared(config, variant, uses_port=True, runtime_backend=None):
    """Normalised config with placeholders for missing files, a fixed port and an alias for clients.

    `runtime_backend` (runtime_install.detect()["backend"]) is the llama.cpp build the settings were
    tested with. When it is a GPU backend it wins over the card's own, so a Vulkan build on an NVIDIA
    card exports the Vulkan docker image and no CUDA_VISIBLE_DEVICES, like the tested run.
    """
    raw = dict(config)
    notes = []
    if runtime_backend in RUNTIME_GPU_BACKENDS and raw.get("gpu_backend") not in (None, "cpu", runtime_backend):
        raw["gpu_backend"] = runtime_backend
    if not raw.get("model_path"):
        raw["model_path"] = PLACEHOLDER
        name = f" ({variant.filename})" if variant is not None else ""
        notes.append(f"The model file{name} is not downloaded yet, so {PLACEHOLDER} stands in for its path. "
                     "Replace it with the real path after downloading.")
    for name in ["model_path", "draft_model_path", "alias", "device", "gpu_uuid"]:
        if raw.get(name) is None:
            continue
        if not isinstance(raw[name], str):
            raise ValueError(f"{name} must be text (a string), not {type(raw[name]).__name__}.")
        if re.search(r"[\x00-\x1f\x7f]", raw[name]):
            raise ValueError(f"The {name.replace('_', ' ')} contains line breaks or control characters. "
                             "Rename the file or folder and try again.")
    shard = SPLIT.search(raw["model_path"])
    if shard and int(shard.group(1)) != 1:
        # llama.cpp only loads a split model from its first part, and finds the others next to it.
        raw["model_path"] = raw["model_path"][:shard.start()] + f"-00001-of-{shard.group(2)}.gguf"
        notes.append("This model is split into parts; the command starts from the first part, "
                     "and llama.cpp finds the others in the same folder.")
    if not raw.get("port"):
        raw["port"] = DEFAULT_PORT
        if uses_port:
            notes.append(f"Uses port {DEFAULT_PORT} (llama-server's usual port). "
                         "Change it if another program already uses it.")
    raw["alias"] = _alias(raw, variant)
    return launch.normalize(raw), notes


def _base_url(config):
    return f"http://{config['host']}:{config['port']}/v1"


def _is_split(config, variant):
    if config["model_path"] != PLACEHOLDER:  # a real file name tells; it may already be merged
        return bool(SPLIT.search(config["model_path"]))
    return bool(variant is not None and len(getattr(variant, "files", None) or []) > 1)


def _tool(server_command, name):
    """Another llama.cpp tool next to the known llama-server, else the bare name on PATH."""
    command = [server_command] if isinstance(server_command, str) else list(server_command or [])
    if len(command) == 1:
        for pathlib in (posixpath, ntpath):
            folder, base = pathlib.split(command[0])
            if folder and base.lower() in {"llama-server", "llama-server.exe"}:
                return pathlib.join(folder, name + (".exe" if base.lower().endswith(".exe") else ""))
    return name


def _extra_env(config):
    return launch.server_env(config, base={})


# ---------------------------------------------------------------- llama-server script

def _llama_server(config, variant, platform, server_command, notes):
    command = server_command or ["llama-server"]
    command = [command] if isinstance(command, str) else list(command)
    args = launch.server_args(config)
    env = _extra_env(config)
    title, url = _title(variant), _base_url(config)
    if platform == "windows":
        if any('"' in token for token in command + args):
            raise ValueError("Windows PowerShell cannot pass a setting that contains a double quote (\") to a program "
                             "reliably. Rename the file or folder, or the model name, and try again.")
        lines = [f"# Start {title} with the settings LLM Configurator tested.",
                 f"# OpenAI-compatible address: {url}  (stop with Ctrl+C)",
                 "$ErrorActionPreference = 'Stop'"]
        lines += [f"$env:{key} = {ps_quote(value)}" for key, value in sorted(env.items())]
        body = [" ".join(_ps_word(t) for t in group) for group in _grouped(args)]
        lines.append("& " + " ".join(ps_quote(t) for t in command) + " `")
        lines += [f"    {part} `" for part in body[:-1]] + [f"    {body[-1]}"]
        filename = "start-llama-server.ps1"
        content = "\n".join(lines) + "\n"
        instructions = [
            f"Save this as {filename}.",
            f"Open PowerShell in that folder and run: powershell -ExecutionPolicy Bypass -File .\\{filename}",
            "Wait for the line saying the server is listening.",
            f"Point your apps at {url} (any API key works). Press Ctrl+C to stop the server.",
        ]
        if any(ord(ch) > 127 for ch in content):
            notes.append("The script contains non-English characters. Save it as \"UTF-8 with BOM\" so Windows "
                         "PowerShell 5.1 reads the path correctly (PowerShell 7 is fine either way).")
    else:
        lines = ["#!/usr/bin/env bash", f"# Start {title} with the settings LLM Configurator tested.",
                 f"# OpenAI-compatible address: {url}  (stop with Ctrl+C)", "set -euo pipefail"]
        lines += [f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())]
        body = [" ".join(shlex.quote(t) for t in group) for group in _grouped(args)]
        lines.append("exec " + " ".join(shlex.quote(t) for t in command) + " \\")
        lines += [f"    {part} \\" for part in body[:-1]] + [f"    {body[-1]}"]
        filename = "start-llama-server.sh"
        content = "\n".join(lines) + "\n"
        instructions = [
            f"Save this as {filename}.",
            f"Make it runnable once: chmod +x {filename}",
            f"Start it: ./{filename} and wait for the line saying the server is listening.",
            f"Point your apps at {url} (any API key works). Press Ctrl+C to stop the server.",
        ]
    if not server_command:
        notes.append("Expects llama-server on your PATH. Put the full path to llama-server in the script if it is elsewhere.")
    if "CUDA_VISIBLE_DEVICES" in env:
        notes.append("CUDA_VISIBLE_DEVICES pins the model to the graphics card it was tested on.")
    return filename, content, instructions


# ---------------------------------------------------------------- Ollama

def _merged_name(path, pathlib):
    folder, name = pathlib.split(path)
    return pathlib.join(folder, SPLIT.sub(".gguf", name)) if folder else SPLIT.sub(".gguf", name)


def _modelfile_value(text):
    """FROM takes the rest of the line; wrap paths with spaces or '#' in plain double quotes."""
    if '"' in text:
        raise ValueError("Ollama cannot read a model path containing a double quote. Rename the file or folder and try again.")
    return f'"{text}"' if re.search(r"[\s#]", text) else text


def _ollama(config, variant, platform, notes, server_command=None):
    pathlib = ntpath if platform == "windows" else posixpath
    name = _slug(_alias(config, variant))
    path = config["model_path"]
    instructions = []
    if _is_split(config, variant):
        merged = _merged_name(path, pathlib) if path != PLACEHOLDER else "<path-to-merged-model.gguf>"
        shell_quote = ps_quote if platform == "windows" else shlex.quote
        tool = _tool(server_command, "llama-gguf-split")
        merge = tool if tool == "llama-gguf-split" else f"& {ps_quote(tool)}" if platform == "windows" else shlex.quote(tool)
        merge += f" --merge {shell_quote(path)} {shell_quote(merged)}"
        notes.append("Ollama cannot load a model split into several files. Merge the parts into one file first "
                     f"(needs free disk space equal to the model size): {merge}")
        notes.append("Recent Ollama versions can also load the parts directly (one FROM line per part), "
                     "but merging works with every version.")
        instructions.append(f"Merge the split model into one file: {merge}")
        path = merged
    lines = [f"# Ollama Modelfile made by LLM Configurator for {_title(variant)}.",
             f"FROM {_modelfile_value(path)}",
             f"PARAMETER num_ctx {config['context']}",
             f"PARAMETER num_gpu {launch.runtime_gpu_layers(config['gpu_layers'], config['total_layers'])}"]
    if config["threads"]:
        lines.append(f"PARAMETER num_thread {config['threads']}")
    if config["batch"]:
        lines.append(f"PARAMETER num_batch {config['batch']}")
    content = "\n".join(lines) + "\n"
    env = {}
    compressed = config["cache_type_k"] != "f16" or config["cache_type_v"] != "f16"
    if config["flash_attn"] != "auto" or compressed:
        env["OLLAMA_FLASH_ATTENTION"] = "0" if config["flash_attn"] == "off" else "1"
    if compressed:
        kv = config["cache_type_k"] if config["cache_type_k"] == config["cache_type_v"] else "q8_0"
        env["OLLAMA_KV_CACHE_TYPE"] = kv
        notes.append("Ollama can only compress the KV cache (the model's short-term notepad) for the whole Ollama "
                     f"server, not per model: OLLAMA_KV_CACHE_TYPE={kv} applies to every model it runs, "
                     "and it needs flash attention on.")
        if config["cache_type_k"] != config["cache_type_v"]:
            notes.append(f"Ollama uses one type for both halves of the KV cache; the tested setup used "
                         f"{config['cache_type_k']} and {config['cache_type_v']}, so q8_0 is the closest safe choice. "
                         "Quality and memory use can differ a little from the tested run.")
    if config["parallel"] > 1:
        env["OLLAMA_NUM_PARALLEL"] = str(config["parallel"])
        notes.append(f"Serving {config['parallel']} users at once is an Ollama server setting "
                     f"(OLLAMA_NUM_PARALLEL={config['parallel']}); num_ctx is the space for each user.")
    if env:
        if platform == "windows":
            setting = "; ".join(f"setx {key} {value}" for key, value in env.items())
            instructions.append(f"In PowerShell set Ollama's server settings, then quit and reopen Ollama: {setting}")
        else:
            setting = " ".join(f"{key}={value}" for key, value in env.items())
            instructions.append(f"Restart the Ollama server with its settings: {setting} ollama serve "
                                "(for the background service, add these to its environment instead)")
    if config["n_cpu_moe"]:
        notes.append(f"Ollama cannot keep only the expert weights of {config['n_cpu_moe']} layers in main memory "
                     "(llama.cpp's --n-cpu-moe), so it may place this model differently and run slower.")
    if config["draft_model_path"]:
        notes.append("Ollama does not support a helper draft model (speculative decoding); it will run without one.")
    if config["batch"] and config["ubatch"] and config["ubatch"] != config["batch"]:
        notes.append("Ollama has no separate micro-batch (ubatch) setting; only num_batch is set.")
    if config["mlock"] or not config["mmap"]:
        notes.append("Memory locking and memory-mapping choices are not carried over; Ollama manages these itself.")
    if config["device"] or (config["gpu_backend"] == "cuda" and config["gpu_uuid"] and config["gpu_layers"]):
        notes.append("Ollama picks graphics cards itself. To pin one NVIDIA card, set CUDA_VISIBLE_DEVICES for the Ollama server.")
    notes.append("Ollama applies the model's own chat template from the file, like llama-server's --jinja.")
    instructions += [f"Save this as a file named Modelfile (no extension) in any folder.",
                     f"In that folder run: ollama create {name} -f Modelfile",
                     f"Chat with it: ollama run {name}  (apps can use http://127.0.0.1:11434/v1 with model {name})"]
    return "Modelfile", content, instructions


# ---------------------------------------------------------------- docker-compose

def _yaml(text):
    """Double-quoted YAML scalar; `$` is doubled because compose substitutes $VARIABLES."""
    return json.dumps(str(text).replace("$", "$$"), ensure_ascii=False)


def _docker(config, variant, platform, notes):
    pathlib = ntpath if platform == "windows" else posixpath
    backend = config["gpu_backend"] if config["gpu_layers"] else None
    image = "ghcr.io/ggml-org/llama.cpp:" + DOCKER_IMAGES.get(backend, "server")
    mounts, inside = [], {}
    for key, target in [("model_path", "/models"), ("draft_model_path", "/draft")]:
        path = config[key]
        if not path:
            continue
        if path in (PLACEHOLDER, DRAFT_PLACEHOLDER):
            filename = variant.filename if key == "model_path" and variant is not None else "model.gguf"
            folder = "<folder-containing-the-model>"
        else:
            if not (path.startswith("/") or re.match(r"^([A-Za-z]:[\\/]|\\\\)", path)):
                raise ValueError("Docker needs the full path to the model file (for example starting with / or C:\\). "
                                 "Export again with the full path.")
            folder, filename = pathlib.split(path)
            if platform == "windows" and path.startswith("\\\\"):
                notes.append("Docker Desktop cannot mount a network share (\\\\server\\share). Copy the model to a local drive "
                             "and export again.")
        mounts.append((folder, target))
        inside[key] = f"{target}/{filename}"
    args = launch.server_args({**config, **inside})
    # Inside the container the server must accept connections from Docker's network bridge;
    # the port mapping below still only opens it on this computer (127.0.0.1).
    args[args.index("--host") + 1] = "0.0.0.0"
    service = "llama-server"
    lines = [f"# docker-compose.yml made by LLM Configurator for {_title(variant)}.",
             f"# OpenAI-compatible address: {_base_url(config)}",
             "services:", f"  {service}:", f"    image: {image}",
             "    ports:", f"      - {_yaml('127.0.0.1:{0}:{0}'.format(config['port']))}",
             "    volumes:"]
    for folder, target in mounts:
        lines += ["      - type: bind", f"        source: {_yaml(folder)}", f"        target: {target}",
                  "        read_only: true"]
    lines.append("    command:")
    lines += [f"      - {_yaml(token)}" for token in args]
    if backend == "cuda":
        device = (f"          device_ids: [{_yaml(config['gpu_uuid'])}]" if config["gpu_uuid"]
                  else "          count: all")
        lines += ["    deploy:", "      resources:", "        reservations:", "          devices:",
                  "            - driver: nvidia", "              capabilities: [gpu]", device.replace("          ", "              ", 1)]
    elif backend == "vulkan":
        lines += ["    devices:", "      - /dev/dri:/dev/dri"]
    elif backend == "rocm":
        lines += ["    devices:", "      - /dev/kfd:/dev/kfd", "      - /dev/dri:/dev/dri",
                  "    group_add:", "      - video"]
    lines.append("    restart: unless-stopped")
    content = "\n".join(lines) + "\n"
    notes.append("Inside the container the server listens on 0.0.0.0 so Docker can forward traffic to it; "
                 f"the port is published on 127.0.0.1 only, so other computers still cannot reach it.")
    notes.append("The model folder is mounted read-only, so the container cannot change your files.")
    if backend == "cuda":
        notes.append("NVIDIA needs the NVIDIA Container Toolkit on Linux, or Docker Desktop with WSL 2 on Windows.")
    elif backend == "vulkan":
        notes.append("Vulkan in Docker works on Linux by passing /dev/dri; Docker Desktop on Windows and macOS "
                     "cannot pass the graphics card this way, so use the llama-server script there.")
        notes.append("This uses the Vulkan image because the settings were tested with a Vulkan build of llama.cpp. "
                     "On an NVIDIA card, Vulkan inside Docker also needs the NVIDIA Container Toolkit; the CUDA image "
                     "(server-cuda) is the usual choice there, and its speed may differ from the tested run.")
    elif backend == "rocm":
        notes.append("AMD ROCm in Docker works on Linux only and needs /dev/kfd and /dev/dri.")
    elif config["gpu_backend"] == "metal":
        notes.append("Docker on a Mac cannot use the Apple graphics chip (Metal), so this container runs on the "
                     "processor only and will be much slower. Use the llama-server script to run natively instead.")
    elif config["gpu_backend"] and config["gpu_backend"] != "cpu" and not config["gpu_layers"]:
        notes.append("The tested setup runs on the processor only, so the plain CPU image is used.")
    if config["n_cpu_moe"] or config["draft_model_path"]:
        notes.append("All other settings, including expert placement and the draft model, are passed unchanged.")
    desktop = "Docker Desktop" if platform == "windows" else "Docker with the compose plugin"
    instructions = [f"Install {desktop} if you have not already.",
                    "Save this as docker-compose.yml in an empty folder.",
                    "In that folder run: docker compose up -d  (the first run downloads the llama.cpp image)",
                    f"Point your apps at {_base_url(config)}. Stop it with: docker compose down"]
    return "docker-compose.yml", content, instructions


# ---------------------------------------------------------------- clients

def _openai_python(config, variant, platform, notes):
    alias, url = config["alias"], _base_url(config)
    content = "\n".join([
        f"# Chat with {_title(variant)} through the local server. Needs: pip install openai",
        "from openai import OpenAI",
        "",
        f"client = OpenAI(base_url={json.dumps(url)}, api_key={json.dumps(API_KEY)})",
        "reply = client.chat.completions.create(",
        f"    model={json.dumps(alias)},",
        "    messages=[{\"role\": \"user\", \"content\": \"Say hello in one short sentence.\"}],",
        ")",
        "print(reply.choices[0].message.content)",
    ]) + "\n"
    notes.append("The local server does not check the API key; \"not-needed\" just satisfies the library.")
    python = "py" if platform == "windows" else "python3"
    instructions = ["Start the local server first (llama-server script or docker-compose).",
                    f"Install the client once: {python} -m pip install openai",
                    f"Save this as chat_local.py and run: {python} chat_local.py"]
    return "chat_local.py", content, instructions


def _continue(config, variant, platform, notes):
    title = f"{_title(variant)} (local)"
    content = "\n".join([
        "name: Local models",
        "version: 1.0.0",
        "schema: v1",
        "models:",
        f"  - name: {_yaml(title)}",
        "    provider: openai",
        f"    model: {_yaml(config['alias'])}",
        f"    apiBase: {_yaml(_base_url(config))}",
        f"    apiKey: {_yaml(API_KEY)}",
        "    roles:",
        "      - chat",
        "      - edit",
        "      - apply",
        "    defaultCompletionOptions:",
        f"      contextLength: {config['context']}",
    ]) + "\n"
    notes.append("Uses Continue's generic OpenAI-compatible provider, which works with any llama-server version.")
    folder = "%USERPROFILE%\\.continue\\config.yaml" if platform == "windows" else "~/.continue/config.yaml"
    instructions = ["Start the local server first (llama-server script or docker-compose).",
                    f"Open Continue's config.yaml (gear icon in Continue, or {folder}).",
                    "If the file is empty, paste all of this. Otherwise copy only the entry under models: into your models list.",
                    f"Save, then pick \"{_line(title)}\" in Continue's model menu."]
    return "continue-config.yaml", content, instructions


def _open_webui(config, variant, platform, notes):
    url, docker_url = _base_url(config), f"http://host.docker.internal:{config['port']}/v1"
    content = "\n".join([
        f"Open WebUI connection for {_title(variant)}",
        "",
        "Where: Admin Panel > Settings > Connections > OpenAI API > + (Add Connection)",
        f"URL: {url}",
        f"URL if Open WebUI itself runs in Docker: {docker_url}",
        f"API key: {API_KEY}",
        f"Model: {config['alias']}",
        "Optional: under Advanced, set Provider to llama.cpp.",
    ]) + "\n"
    notes.append("Inside a Docker container 127.0.0.1 means the container itself, so Open WebUI in Docker "
                 "reaches the server through host.docker.internal.")
    if platform != "windows":
        notes.append("On Linux, host.docker.internal cannot reach a server that listens on 127.0.0.1 only. "
                     "Start the Open WebUI container with --network=host and use the first URL instead.")
    instructions = ["Start the local server first (llama-server script or docker-compose).",
                    "In Open WebUI open Admin Panel > Settings > Connections.",
                    f"Under OpenAI API add a connection with URL {url} (or the Docker URL above) and key {API_KEY}.",
                    f"Save, then choose {config['alias']} in the model menu of a new chat."]
    return "open-webui.txt", content, instructions


def _lmstudio(config, variant, platform, notes):
    layers = config["gpu_layers"]
    if config["total_layers"]:
        offload = f"{layers} of {config['total_layers']} layers" + (" (all the way right)" if layers == config["total_layers"] else "")
    else:
        offload = f"{layers} layers"
    rows = [("Context Length", str(config["context"])), ("GPU Offload", offload)]
    if config["threads"]:
        rows.append(("CPU Thread Pool Size", str(config["threads"])))
    if config["batch"]:
        rows.append(("Evaluation Batch Size", str(config["batch"])))
    rows.append(("Flash Attention", {"on": "On", "off": "Off", "auto": "leave as is"}[config["flash_attn"]]))
    if config["cache_type_k"] != "f16" or config["cache_type_v"] != "f16":
        rows += [("K Cache Quantization Type", config["cache_type_k"].upper()),
                 ("V Cache Quantization Type", config["cache_type_v"].upper())]
    if config["n_cpu_moe"]:
        rows.append(("Number of layers for which to force MoE weights onto CPU", str(config["n_cpu_moe"])))
    rows += [("Keep Model in Memory", "On" if config["mlock"] else "Off"), ("Try mmap()", "On" if config["mmap"] else "Off")]
    if config["draft_model_path"]:
        rows.append(("Speculative Decoding > Draft Model", _line(ntpath.basename(posixpath.basename(config["draft_model_path"])))))
    width = max(len(key) for key, _ in rows)
    model = variant.filename if variant is not None else posixpath.basename(ntpath.basename(config["model_path"]))
    content = "\n".join([f"LM Studio load settings for {_title(variant)}", f"Model file: {_line(model)}", ""]
                        + [f"{key.ljust(width)}  {value}" for key, value in rows]) + "\n"
    notes.append("LM Studio uses its own copy of llama.cpp, so speed can differ a little from the tested run.")
    if config["parallel"] > 1:
        notes.append(f"The tested setup served {config['parallel']} users at once; LM Studio's context length "
                     "above is for one conversation.")
    instructions = ["In LM Studio open My Models (or use lms import with the file) so it lists this model file.",
                    "Select the model to load, turn on \"Manually choose model load parameters\" and expand the settings.",
                    "Enter the values above, then click Load Model.",
                    "Optional: start LM Studio's local server from the Developer tab to use it from other apps."]
    return "lmstudio-settings.txt", content, instructions


BUILDERS = {"ollama": _ollama, "docker-compose": _docker, "openai-python": _openai_python,
            "continue": _continue, "open-webui": _open_webui, "lmstudio": _lmstudio}


def export(config, variant, fmt, platform="posix", server_command=None, runtime_backend=None):
    """Build one export. Returns {"format", "filename", "content", "instructions", "notes"}.

    Pass `runtime_backend` (runtime_install.detect(store)["backend"]) when the config did not come from
    launch.from_candidate(..., runtime_backend=...), so the GPU backend is the installed build's."""
    if fmt not in {item["id"] for item in FORMATS}:
        raise ValueError(f"Unknown export format {fmt!r}. Choose one of: {', '.join(item['id'] for item in FORMATS)}")
    if platform not in {"posix", "windows"}:
        raise ValueError("platform must be posix (macOS or Linux) or windows")
    config, notes = _prepared(config, variant, uses_port=fmt not in {"ollama", "lmstudio"}, runtime_backend=runtime_backend)
    if fmt == "llama-server":
        filename, content, instructions = _llama_server(config, variant, platform, server_command, notes)
    elif fmt == "ollama":
        filename, content, instructions = _ollama(config, variant, platform, notes, server_command)
    else:
        filename, content, instructions = BUILDERS[fmt](config, variant, platform, notes)
    return {"format": fmt, "filename": filename, "content": content, "instructions": instructions, "notes": notes}
