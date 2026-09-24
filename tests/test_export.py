import ast
import json
import re
import shlex
import unittest

from llm_configurator import export as ex
from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import Variant
from llm_configurator.launch import server_args

try:
    import yaml
except ImportError:  # PyYAML is optional; parsing checks are skipped without it
    yaml = None

NASTY = "/home/me/My Models/it's $HOME `whoami` \"q\" é 模型.gguf"
WIN_NASTY = "C:\\Users\\Zoë O'Neil\\My Models\\$env:PATH `n ‘x’.gguf"
ALL_FORMATS = ["llama-server", "ollama", "docker-compose", "openai-python", "continue", "open-webui", "lmstudio"]


def variant(**changes):
    return Variant(**{**demo_variants()[0].to_dict(), **changes})


def base(**changes):
    return {"model_path": "/models/demo.gguf", "context": 8192, "gpu_layers": 32, "total_layers": 32,
            "threads": 8, "port": 8080, "alias": "demo", **changes}


def bash_argv(content):
    """Undo line continuations and let a POSIX shell lexer split the exec line."""
    text = content.replace(" \\\n", " ")
    line = next(l for l in text.splitlines() if l.startswith("exec "))
    return shlex.split(line)[1:]


def ps_argv(content):
    """Tiny PowerShell lexer for what the exporter emits: bare words and single-quoted strings."""
    text = content.replace(" `\n", " ")
    line = next(l for l in text.splitlines() if l.startswith("& "))[2:]
    quotes = ex.PS_QUOTES
    tokens, i = [], 0
    while i < len(line):
        if line[i] == " ":
            i += 1
        elif line[i] in quotes:
            value, i = [], i + 1
            while True:
                if line[i] in quotes:
                    if i + 1 < len(line) and line[i + 1] in quotes:
                        value.append(line[i])
                        i += 2
                        continue
                    i += 1
                    break
                value.append(line[i])
                i += 1
            tokens.append("".join(value))
        else:
            end = line.find(" ", i)
            end = len(line) if end == -1 else end
            word = line[i:end]
            if not ex.PS_BARE.match(word):
                raise AssertionError(f"unsafe bare word {word!r}")
            tokens.append(word)
            i = end
    return tokens


class FormatListTests(unittest.TestCase):
    def test_formats_listed_with_labels(self):
        items = ex.formats()
        self.assertEqual([item["id"] for item in items], ALL_FORMATS)
        for item in items:
            self.assertTrue(item["label"] and item["description"])
        items[0]["label"] = "changed"
        self.assertNotEqual(ex.formats()[0]["label"], "changed")

    def test_bad_format_or_platform(self):
        with self.assertRaisesRegex(ValueError, "Unknown export format"):
            ex.export(base(), variant(), "exe")
        with self.assertRaisesRegex(ValueError, "platform"):
            ex.export(base(), variant(), "ollama", platform="amiga")

    def test_every_format_both_platforms_deterministic_and_complete(self):
        for fmt in ALL_FORMATS:
            for platform in ["posix", "windows"]:
                with self.subTest(fmt=fmt, platform=platform):
                    first = ex.export(base(), variant(), fmt, platform)
                    self.assertEqual(first, ex.export(base(), variant(), fmt, platform))
                    self.assertEqual(set(first), {"format", "filename", "content", "instructions", "notes"})
                    self.assertEqual(first["format"], fmt)
                    self.assertTrue(first["content"].endswith("\n"))
                    self.assertTrue(2 <= len(first["instructions"]) <= 5)
                    self.assertNotRegex(first["content"], r"20\d\d-\d\d-\d\d")

    def test_model_names_cannot_inject_lines(self):
        record = variant(name="Evil\nrm -rf ~\r\n#")
        for fmt in ALL_FORMATS:
            for platform in ["posix", "windows"]:
                content = ex.export(base(), record, fmt, platform)["content"]
                self.assertFalse(any(line.startswith("rm -rf") for line in content.splitlines()), (fmt, platform))

    def test_line_breaks_in_paths_rejected(self):
        with self.assertRaisesRegex(ValueError, "line breaks"):
            ex.export(base(model_path="/a\nb.gguf"), variant(), "llama-server")


class LlamaServerScriptTests(unittest.TestCase):
    def test_bash_snapshot(self):
        result = ex.export(base(), variant(), "llama-server")
        self.assertEqual(result["filename"], "start-llama-server.sh")
        self.assertEqual(result["content"], """#!/usr/bin/env bash
# Start Demo Small Q4_K_M with the settings LLM Configurator tested.
# OpenAI-compatible address: http://127.0.0.1:8080/v1  (stop with Ctrl+C)
set -euo pipefail
exec llama-server \\
    -m /models/demo.gguf \\
    -c 8192 \\
    -np 1 \\
    -ngl 33 \\
    --host 127.0.0.1 \\
    --jinja \\
    --port 8080 \\
    -t 8 \\
    --alias demo
""")
        self.assertIn("chmod +x start-llama-server.sh", result["instructions"][1])

    def test_powershell_snapshot(self):
        result = ex.export(base(model_path="C:\\AI Models\\demo.gguf"), variant(), "llama-server", "windows")
        self.assertEqual(result["filename"], "start-llama-server.ps1")
        self.assertEqual(result["content"], """# Start Demo Small Q4_K_M with the settings LLM Configurator tested.
# OpenAI-compatible address: http://127.0.0.1:8080/v1  (stop with Ctrl+C)
$ErrorActionPreference = 'Stop'
& 'llama-server' `
    -m 'C:\\AI Models\\demo.gguf' `
    -c 8192 `
    -np 1 `
    -ngl 33 `
    --host '127.0.0.1' `
    --jinja `
    --port 8080 `
    -t 8 `
    --alias demo
""")
        self.assertIn("-ExecutionPolicy Bypass -File .\\start-llama-server.ps1", result["instructions"][1])

    def test_bash_round_trip_with_hostile_paths(self):
        config = base(model_path=NASTY, draft_model_path="/d/$(rm -rf ~)/draft;x.gguf", draft_max=8,
                      alias="my `model` $x")
        result = ex.export(config, variant(), "llama-server", server_command=["/opt/llama cpp/llama-server", "--log-disable"])
        argv = bash_argv(result["content"])
        self.assertEqual(argv[:2], ["/opt/llama cpp/llama-server", "--log-disable"])
        self.assertEqual(argv[2:], server_args(config))
        self.assertNotIn("PATH", " ".join(result["notes"]))

    def test_exported_args_equal_tested_args(self):
        config = base(cache_type_k="q8_0", cache_type_v="q8_0", batch=1024, ubatch=256, n_cpu_moe=6,
                      mlock=True, mmap=False, parallel=2, device="CUDA0")
        self.assertEqual(bash_argv(ex.export(config, variant(), "llama-server")["content"])[1:], server_args(config))
        self.assertEqual(ps_argv(ex.export(config, variant(), "llama-server", "windows")["content"])[1:],
                         server_args(config))

    def test_powershell_round_trip_with_hostile_paths(self):
        config = base(model_path=WIN_NASTY, draft_model_path="D:\\it’s; & calc.exe\\d.gguf", draft_max=4)
        result = ex.export(config, variant(), "llama-server", "windows",
                           server_command="C:\\Program Files\\llama.cpp\\llama-server.exe")
        argv = ps_argv(result["content"])
        self.assertEqual(argv[0], "C:\\Program Files\\llama.cpp\\llama-server.exe")
        self.assertEqual(argv[1:], server_args(config))
        self.assertTrue(any("UTF-8 with BOM" in note for note in result["notes"]))

    def test_cuda_pin_exported_as_environment(self):
        config = base(gpu_backend="cuda", gpu_uuid="GPU-1234 'x'")
        bash = ex.export(config, variant(), "llama-server")["content"]
        self.assertIn("export CUDA_VISIBLE_DEVICES='GPU-1234 '\"'\"'x'\"'\"''\n", bash)
        ps = ex.export(config, variant(), "llama-server", "windows")["content"]
        self.assertIn("$env:CUDA_VISIBLE_DEVICES = 'GPU-1234 ''x'''\n", ps)

    def test_defaults_filled_for_port_and_alias(self):
        result = ex.export(base(port=None, alias=None), variant(), "llama-server")
        argv = bash_argv(result["content"])
        self.assertEqual(argv[argv.index("--port") + 1], "8080")
        self.assertEqual(argv[argv.index("--alias") + 1], "demo-small-q4_k_m")
        self.assertTrue(any("port 8080" in note for note in result["notes"]))


class PowerShellQuotingTests(unittest.TestCase):
    def test_single_quotes_doubled_and_nothing_expands(self):
        self.assertEqual(ex.ps_quote("it's"), "'it''s'")
        self.assertEqual(ex.ps_quote("$env:HOME `n $(x)"), "'$env:HOME `n $(x)'")
        self.assertEqual(ex.ps_quote(""), "''")
        self.assertEqual(ex.ps_quote("é 模型"), "'é 模型'")

    def test_typographic_quotes_are_also_doubled(self):
        # PowerShell ends a single-quoted string at ‘ ’ ‚ ‛ too.
        self.assertEqual(ex.ps_quote("a‘b’c‚d‛e"), "'a‘‘b’’c‚‚d‛‛e'")

    def test_only_plain_flags_and_words_stay_bare(self):
        self.assertEqual(ex._ps_word("--n-cpu-moe"), "--n-cpu-moe")
        self.assertEqual(ex._ps_word("q8_0"), "q8_0")
        self.assertEqual(ex._ps_word("127.0.0.1"), "'127.0.0.1'")  # PowerShell splits -x.y style words
        self.assertEqual(ex._ps_word("a,b"), "'a,b'")
        self.assertEqual(ex._ps_word("-x.y"), "'-x.y'")


class PlaceholderTests(unittest.TestCase):
    def test_every_format_works_before_download(self):
        for fmt in ALL_FORMATS:
            for platform in ["posix", "windows"]:
                with self.subTest(fmt=fmt, platform=platform):
                    result = ex.export(base(model_path=None), variant(), fmt, platform)
                    self.assertTrue(any("not downloaded yet" in note for note in result["notes"]))
        script = ex.export(base(model_path=None), variant(), "llama-server")["content"]
        self.assertIn("-m '<path-to-model.gguf>'", script)
        self.assertIn(ex.PLACEHOLDER, ex.export(base(model_path=None), variant(), "ollama")["content"])
        compose = ex.export(base(model_path=None), variant(), "docker-compose")["content"]
        self.assertIn('source: "<folder-containing-the-model>"', compose)
        self.assertIn('"/models/Demo Small-Q4_K_M.gguf"', compose)

    def test_variant_optional(self):
        result = ex.export(base(model_path=None, alias=None), None, "openai-python")
        self.assertIn('model="local-model"', result["content"])


class OllamaTests(unittest.TestCase):
    def test_snapshot(self):
        result = ex.export(base(batch=512, alias="Demo Small: Q4/K_M!"), variant(), "ollama")
        self.assertEqual(result["filename"], "Modelfile")
        self.assertEqual(result["content"], """# Ollama Modelfile made by LLM Configurator for Demo Small Q4_K_M.
FROM /models/demo.gguf
PARAMETER num_ctx 8192
PARAMETER num_gpu 33
PARAMETER num_thread 8
PARAMETER num_batch 512
""")
        self.assertIn("ollama create demo-small-q4-k_m -f Modelfile", " ".join(result["instructions"]))
        self.assertIn("ollama run demo-small-q4-k_m", " ".join(result["instructions"]))
        self.assertFalse(any("port" in note for note in result["notes"]))

    def test_partial_offload_and_paths_with_spaces(self):
        result = ex.export(base(gpu_layers=20, model_path="C:\\My Models\\x #1.gguf"), variant(), "ollama", "windows")
        self.assertIn('FROM "C:\\My Models\\x #1.gguf"\n', result["content"])
        self.assertIn("PARAMETER num_gpu 20\n", result["content"])
        with self.assertRaisesRegex(ValueError, "double quote"):
            ex.export(base(model_path='/a/"b".gguf'), variant(), "ollama")

    def test_split_model_needs_merge(self):
        path = "/m/My Model-00001-of-00003.gguf"
        result = ex.export(base(model_path=path), variant(), "ollama")
        self.assertIn('FROM "/m/My Model.gguf"', result["content"])
        merge = "llama-gguf-split --merge '/m/My Model-00001-of-00003.gguf' '/m/My Model.gguf'"
        self.assertIn(merge, result["instructions"][0])
        self.assertTrue(any(merge in note for note in result["notes"]))
        windows = ex.export(base(model_path="C:\\m\\x-00001-of-00002.gguf"), variant(), "ollama", "windows")
        self.assertIn("llama-gguf-split --merge 'C:\\m\\x-00001-of-00002.gguf' 'C:\\m\\x.gguf'",
                      windows["instructions"][0])

    def test_split_detected_from_variant_before_download(self):
        record = variant(size_bytes=2, files=[{"filename": "a-00001-of-00002.gguf", "size_bytes": 1, "sha256": "x"},
                                              {"filename": "a-00002-of-00002.gguf", "size_bytes": 1, "sha256": "y"}])
        result = ex.export(base(model_path=None), record, "ollama")
        self.assertIn("FROM <path-to-merged-model.gguf>", result["content"])

    def test_what_ollama_cannot_express_is_listed(self):
        config = base(cache_type_k="q8_0", cache_type_v="q4_0", n_cpu_moe=12, parallel=4,
                      draft_model_path="/m/draft.gguf", batch=2048, ubatch=512)
        result = ex.export(config, variant(), "ollama")
        notes = " ".join(result["notes"])
        for text in ["OLLAMA_KV_CACHE_TYPE=q8_0", "whole Ollama server", "--n-cpu-moe", "OLLAMA_NUM_PARALLEL=4",
                     "draft model", "micro-batch"]:
            self.assertIn(text, notes)
        steps = " ".join(result["instructions"])
        self.assertIn("OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=4 ollama serve", steps)
        windows = " ".join(ex.export(config, variant(), "ollama", "windows")["instructions"])
        self.assertIn("setx OLLAMA_KV_CACHE_TYPE q8_0", windows)

    def test_names_sanitised(self):
        self.assertEqual(ex._slug("Qwen3 30B-A3B / Q4_K_M"), "qwen3-30b-a3b-q4_k_m")
        self.assertEqual(ex._slug("../../etc"), "etc")
        self.assertEqual(ex._slug("模型"), "local-model")
        self.assertLessEqual(len(ex._slug("x" * 300)), 80)


class DockerTests(unittest.TestCase):
    def compose(self, config, platform="posix", record=None):
        result = ex.export(config, record or variant(), "docker-compose", platform)
        self.assertEqual(result["filename"], "docker-compose.yml")
        return result

    def test_cpu_snapshot(self):
        result = self.compose(base(gpu_layers=0))
        self.assertEqual(result["content"], """# docker-compose.yml made by LLM Configurator for Demo Small Q4_K_M.
# OpenAI-compatible address: http://127.0.0.1:8080/v1
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - type: bind
        source: "/models"
        target: /models
        read_only: true
    command:
      - "-m"
      - "/models/demo.gguf"
      - "-c"
      - "8192"
      - "-np"
      - "1"
      - "-ngl"
      - "0"
      - "--host"
      - "0.0.0.0"
      - "--jinja"
      - "--port"
      - "8080"
      - "-t"
      - "8"
      - "--alias"
      - "demo"
    restart: unless-stopped
""")
        self.assertTrue(any("0.0.0.0" in note and "127.0.0.1 only" in note for note in result["notes"]))

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_cuda_compose_parses_and_matches_tested_args(self):
        config = base(model_path=NASTY, gpu_backend="cuda", gpu_uuid="GPU-abc", cache_type_k="q8_0",
                      cache_type_v="q8_0", n_cpu_moe=4, draft_model_path="/d/draft $1.gguf", draft_max=8)
        doc = yaml.safe_load(self.compose(config)["content"])
        service = doc["services"]["llama-server"]
        self.assertEqual(service["image"], "ghcr.io/ggml-org/llama.cpp:server-cuda")
        self.assertEqual(service["ports"], ["127.0.0.1:8080:8080"])
        self.assertEqual(service["volumes"], [
            {"type": "bind", "source": "/home/me/My Models", "target": "/models", "read_only": True},
            {"type": "bind", "source": "/d", "target": "/draft", "read_only": True}])
        expected = server_args({**config, "model_path": "/models/" + NASTY.rsplit("/", 1)[1],
                                "draft_model_path": "/draft/draft $1.gguf"})
        expected[expected.index("--host") + 1] = "0.0.0.0"
        # Compose turns $$ back into $.
        self.assertEqual([arg.replace("$$", "$") for arg in service["command"]], expected)
        device = service["deploy"]["resources"]["reservations"]["devices"][0]
        self.assertEqual(device, {"driver": "nvidia", "capabilities": ["gpu"], "device_ids": ["GPU-abc"]})

    def test_dollar_signs_escaped_for_compose(self):
        content = self.compose(base(model_path="/m/$HOME/x.gguf"))["content"]
        self.assertIn('source: "/m/$$HOME"', content)
        self.assertNotRegex(content, r"(?<!\$)\$HOME")

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_windows_paths(self):
        config = base(model_path="C:\\Users\\Zoë\\AI Models\\demo.gguf", gpu_backend="cuda")
        result = self.compose(config, "windows")
        service = yaml.safe_load(result["content"])["services"]["llama-server"]
        self.assertEqual(service["volumes"][0]["source"], "C:\\Users\\Zoë\\AI Models")
        self.assertEqual(service["command"][1], "/models/demo.gguf")
        self.assertEqual(service["deploy"]["resources"]["reservations"]["devices"][0]["count"], "all")
        self.assertIn("Docker Desktop", result["instructions"][0])

    def test_vulkan_rocm_and_metal(self):
        vulkan = self.compose(base(gpu_backend="vulkan"))
        self.assertIn("image: ghcr.io/ggml-org/llama.cpp:server-vulkan", vulkan["content"])
        self.assertIn("- /dev/dri:/dev/dri", vulkan["content"])
        rocm = self.compose(base(gpu_backend="rocm"))
        self.assertIn(":server-rocm", rocm["content"])
        self.assertIn("- /dev/kfd:/dev/kfd", rocm["content"])
        metal = self.compose(base(gpu_backend="metal"))
        self.assertIn("image: ghcr.io/ggml-org/llama.cpp:server\n", metal["content"])
        self.assertTrue(any("Metal" in note and "llama-server script" in note for note in metal["notes"]))

    def test_placeholder_split_and_host_only_changed(self):
        config = base(model_path="/m/a-00001-of-00002.gguf")
        content = self.compose(config)["content"]
        self.assertIn('"/models/a-00001-of-00002.gguf"', content)
        self.assertNotIn("127.0.0.1\"\n      - \"--jinja", content)


class ClientTests(unittest.TestCase):
    def test_openai_python_is_valid_python(self):
        for platform in ["posix", "windows"]:
            result = ex.export(base(port=9000, alias='we"ird\\alias'), variant(), "openai-python", platform)
            tree = ast.parse(result["content"])
            strings = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
            self.assertIn("http://127.0.0.1:9000/v1", strings)
            self.assertIn("not-needed", strings)
            self.assertIn('we"ird\\alias', strings)
        self.assertIn("py -m pip install openai", result["instructions"][1])

    @unittest.skipIf(yaml is None, "PyYAML not installed")
    def test_continue_config(self):
        result = ex.export(base(alias="a: b # c"), variant(), "continue", "windows")
        doc = yaml.safe_load(result["content"])
        self.assertEqual(doc["schema"], "v1")
        model = doc["models"][0]
        self.assertEqual(model["provider"], "openai")
        self.assertEqual(model["model"], "a: b # c")
        self.assertEqual(model["apiBase"], "http://127.0.0.1:8080/v1")
        self.assertEqual(model["apiKey"], "not-needed")
        self.assertEqual(model["defaultCompletionOptions"]["contextLength"], 8192)
        self.assertIn("%USERPROFILE%", result["instructions"][1])

    def test_open_webui(self):
        posix = ex.export(base(), variant(), "open-webui")
        self.assertIn("URL: http://127.0.0.1:8080/v1\n", posix["content"])
        self.assertIn("http://host.docker.internal:8080/v1", posix["content"])
        self.assertIn("API key: not-needed", posix["content"])
        self.assertIn("Model: demo\n", posix["content"])
        self.assertTrue(any("--network=host" in note for note in posix["notes"]))
        windows = ex.export(base(), variant(), "open-webui", "windows")
        self.assertFalse(any("--network=host" in note for note in windows["notes"]))

    def test_lmstudio_settings(self):
        config = base(gpu_layers=20, cache_type_k="q8_0", cache_type_v="q8_0", n_cpu_moe=6, batch=512,
                      draft_model_path="C:\\m\\draft.gguf", mlock=True)
        result = ex.export(config, variant(), "lmstudio", "windows")
        rows = dict(re.split(r"\s{2,}", line, maxsplit=1) for line in result["content"].splitlines()[3:])
        self.assertEqual(rows["Context Length"], "8192")
        self.assertEqual(rows["GPU Offload"], "20 of 32 layers")
        self.assertEqual(rows["CPU Thread Pool Size"], "8")
        self.assertEqual(rows["Evaluation Batch Size"], "512")
        self.assertEqual(rows["Flash Attention"], "On")
        self.assertEqual(rows["K Cache Quantization Type"], "Q8_0")
        self.assertEqual(rows["Number of layers for which to force MoE weights onto CPU"], "6")
        self.assertEqual(rows["Keep Model in Memory"], "On")
        self.assertEqual(rows["Speculative Decoding > Draft Model"], "draft.gguf")
        self.assertIn("Model file: Demo Small-Q4_K_M.gguf", result["content"])
        full = ex.export(base(), variant(), "lmstudio")["content"]
        self.assertIn("32 of 32 layers (all the way right)", full)
        self.assertNotIn("Cache Quantization", full)


if __name__ == "__main__":
    unittest.main()
