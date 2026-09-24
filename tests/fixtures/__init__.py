"""Test helpers: run the fake llama.cpp (fake_llama.py) exactly like a real binary."""
from pathlib import Path
import importlib.util
import sys

FAKE_LLAMA = Path(__file__).resolve().with_name("fake_llama.py")
MODES = ("server", "bench", "perplexity", "cli")


def fake_command(mode):
    """argv prefix for one fake llama.cpp tool, for example fake_command("server") + ["-m", path]."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {', '.join(MODES)}")
    return [sys.executable, str(FAKE_LLAMA), "--as", mode]


def _module():
    spec = importlib.util.spec_from_file_location("fake_llama", FAKE_LLAMA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_gguf(path, architecture="llama", layers=32, experts=0, name=None, context_length=32768):
    """Write a tiny but valid GGUF v3 header (metadata only, no tensors) that the fake loads like a model."""
    return _module().write_gguf(path, architecture=architecture, layers=layers, experts=experts, name=name,
                                context_length=context_length)
