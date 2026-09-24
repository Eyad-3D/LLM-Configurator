"""Validated llama.cpp launch settings shared by testing, tuning, serving and export.

One place builds llama-server arguments so a tested configuration is exactly the one
that gets exported or served. Settings come from the recommendation engine or tuner,
never from free-form browser input.
"""
import os
from pathlib import Path

FLASH_ATTENTION = {"on", "off", "auto"}
CACHE_TYPES = {"f16", "q8_0", "q4_0"}
BACKENDS = {None, "cuda", "metal", "rocm", "vulkan", "cpu"}

DEFAULTS = {"model_path": None, "context": 8192, "parallel": 1, "gpu_layers": 0, "total_layers": None,
            "gpu_uuid": None, "gpu_backend": None, "device": None, "threads": None, "batch": None, "ubatch": None,
            "flash_attn": "auto", "cache_type_k": "f16", "cache_type_v": "f16", "n_cpu_moe": 0,
            "mlock": False, "mmap": True, "draft_model_path": None, "draft_max": None,
            "host": "127.0.0.1", "port": None, "alias": None}


def _integer(config, name, low, high, optional=False):
    value = config[name]
    if value is None and optional:
        return
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")


def normalize(config):
    """Fill defaults and reject unknown or out-of-range settings. Returns a new dict."""
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown launch settings: {', '.join(sorted(unknown))}")
    result = {**DEFAULTS, **config}
    if result["model_path"] is not None and not isinstance(result["model_path"], str):
        raise ValueError("model_path must be a string path")
    _integer(result, "context", 256, 1048576)
    _integer(result, "parallel", 1, 64)
    _integer(result, "total_layers", 1, 4096, optional=True)
    _integer(result, "gpu_layers", 0, result["total_layers"] or 4096)
    _integer(result, "threads", 1, 1024, optional=True)
    _integer(result, "batch", 1, 65536, optional=True)
    _integer(result, "ubatch", 1, 65536, optional=True)
    _integer(result, "n_cpu_moe", 0, result["total_layers"] or 4096)
    _integer(result, "draft_max", 1, 64, optional=True)
    _integer(result, "port", 1, 65535, optional=True)
    if result["batch"] and result["ubatch"] and result["ubatch"] > result["batch"]:
        raise ValueError("ubatch cannot exceed batch")
    if result["flash_attn"] not in FLASH_ATTENTION:
        raise ValueError("flash_attn must be on, off or auto")
    for name in ["cache_type_k", "cache_type_v"]:
        if result[name] not in CACHE_TYPES:
            raise ValueError(f"{name} must be f16, q8_0 or q4_0")
    # llama.cpp needs flash attention to store a compressed V cache.
    if result["cache_type_v"] != "f16":
        if result["flash_attn"] == "off":
            raise ValueError("A compressed V cache requires flash attention")
        result["flash_attn"] = "on"
    if result["gpu_backend"] not in BACKENDS:
        raise ValueError("Unsupported GPU backend")
    for name in ["mlock", "mmap"]:
        if type(result[name]) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if result["host"] not in {"127.0.0.1", "localhost"}:
        raise ValueError("Servers started by this app listen on this computer only")
    for name in ["alias", "device", "gpu_uuid", "draft_model_path"]:
        value = result[name]
        if value is not None and (not isinstance(value, str) or not value or any(ord(ch) < 32 or ord(ch) == 127 for ch in value)):
            raise ValueError(f"{name} must be a non-empty single-line string")
    if result["draft_max"] and not result["draft_model_path"]:
        raise ValueError("draft_max requires a draft model")
    return result


def runtime_gpu_layers(gpu_layers, total_layers):
    """The -ngl value that puts `gpu_layers` transformer blocks on the GPU.

    llama.cpp offloads from the top: -ngl N places blocks n_layer+1-N .. n_layer-1 plus the output
    layer, i.e. N-1 blocks (llama-model.cpp, i_gpu_start). So g blocks need g+1, at any split.
    """
    return gpu_layers + 1 if gpu_layers > 0 else 0


def server_args(config):
    """Arguments after the llama-server executable; list form, never a shell string."""
    c = normalize(config)
    if not c["model_path"]:
        raise ValueError("A downloaded model file is required")
    args = ["-m", c["model_path"], "-c", str(c["context"] * c["parallel"]), "-np", str(c["parallel"]),
            "-ngl", str(runtime_gpu_layers(c["gpu_layers"], c["total_layers"])), "--host", c["host"], "--jinja"]
    if c["port"]:
        args += ["--port", str(c["port"])]
    device = c["device"] or ("none" if c["gpu_layers"] == 0 and c["gpu_backend"] not in {None, "cpu"} else None)
    if device:
        args += ["-dev", device]
    for flag, name in [("-t", "threads"), ("-b", "batch"), ("-ub", "ubatch")]:
        if c[name]:
            args += [flag, str(c[name])]
    if c["flash_attn"] != "auto":
        args += ["-fa", c["flash_attn"]]
    if c["cache_type_k"] != "f16":
        args += ["-ctk", c["cache_type_k"]]
    if c["cache_type_v"] != "f16":
        args += ["-ctv", c["cache_type_v"]]
    if c["n_cpu_moe"]:
        args += ["--n-cpu-moe", str(c["n_cpu_moe"])]
    if c["mlock"]:
        args.append("--mlock")
    if not c["mmap"]:
        args.append("--no-mmap")
    if c["draft_model_path"]:
        # -md alone only loads the draft model; --spec-type turns speculative decoding on.
        args += ["-md", c["draft_model_path"], "--spec-type", "draft-simple"]
        if c["draft_max"]:
            # Current llama.cpp removed --draft-max; --spec-draft-n-max replaced it.
            args += ["--spec-draft-n-max", str(c["draft_max"])]
    if c["alias"]:
        args += ["--alias", c["alias"]]
    return args


def server_env(config, base=None):
    """Process environment: pin NVIDIA work to one GPU by UUID, matching the benchmark runner."""
    c = normalize(config)
    # LLAMA_ARG_* variables silently fill llama-server settings; drop them so a run matches what was tested.
    env = {k: v for k, v in (os.environ if base is None else base).items() if not k.upper().startswith("LLAMA_ARG_")}
    # llama-server lets any website read its answers by default (CORS "*"); allow only pages on this computer.
    # An environment variable, not a flag, so builds without the option ignore it instead of refusing to start.
    env["LLAMA_ARG_CORS_ORIGINS"] = "localhost"
    if c["gpu_backend"] == "cuda" and c["gpu_uuid"] and c["gpu_layers"]:
        env["CUDA_VISIBLE_DEVICES"] = c["gpu_uuid"]
    return env


def from_candidate(candidate, model_path=None, hardware=None, runtime_backend=None, **overrides):
    """Launch settings for one engine candidate; overrides come from the tuner or CLI flags.

    `runtime_backend` is the installed llama.cpp build's backend (runtime_install.detect). It wins over the
    GPU's own backend: a Vulkan build on an NVIDIA card ignores CUDA_VISIBLE_DEVICES.
    """
    gpu = None
    if hardware and candidate.get("gpu_index") is not None:
        gpu = next((g for g in hardware.get("gpus", []) if g.get("index") == candidate["gpu_index"]), None)
    launch = candidate.get("launch") or {}
    config = {"model_path": str(Path(model_path)) if model_path else None, "context": candidate["context"],
              "parallel": candidate.get("users", 1), "gpu_layers": candidate["gpu_layers"],
              "total_layers": candidate.get("total_layers"), "threads": candidate.get("threads"),
              "gpu_uuid": gpu.get("uuid") if gpu and candidate["gpu_layers"] else None,
              "gpu_backend": ((runtime_backend if runtime_backend in BACKENDS - {None, "cpu"} else None)
                              or gpu.get("backend") or "cuda") if gpu else None,
              "cache_type_k": candidate.get("kv_cache_type", "f16"), "cache_type_v": candidate.get("kv_cache_type", "f16"),
              "n_cpu_moe": candidate.get("n_cpu_moe", 0) or 0}
    config.update({k: v for k, v in launch.items() if k in DEFAULTS and k != "model_path"})
    config.update(overrides)
    return normalize(config)
