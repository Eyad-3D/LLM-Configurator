"""v0.4 CLI commands with fake workstream modules: output, progress, exit codes and Ctrl+C."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import llm_configurator
from llm_configurator import cli
from llm_configurator.hardware import scan
from llm_configurator.storage import Store

from test_api_v04 import SECRET, Fakes, real_variant

HARDWARE = {**scan(False), "fingerprint": "fp", "ram_available": 64 * 1024**3, "ram_total": 96 * 1024**3, "gpus": []}


class CliCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.variant, self.other = real_variant(), real_variant(0, "Q8_0")
        self.store.put("variants", [self.variant.to_dict(), self.other.to_dict()])
        self.fakes = Fakes(self.store)
        self.stack = self.fakes.installed()
        for target in ["llm_configurator.cli.scan", "llm_configurator.app.scan"]:
            self.stack.enter_context(patch(target, return_value=HARDWARE))

    def tearDown(self):
        self.fakes.release.set()
        self.stack.close()
        self.temp.cleanup()

    def run_cli(self, *argv, stdin=None):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch("builtins.input", side_effect=stdin or ["n"]):
            code = cli.main(["--data-dir", self.temp.name, *argv])
        return code, out.getvalue(), err.getvalue()

    def downloaded(self, variant=None):
        variant = variant or self.variant
        path = Path(self.temp.name) / "models" / Path(variant.filename).name
        self.fakes.local[variant.id] = path
        return path


class HelpTests(unittest.TestCase):
    def test_every_command_has_help(self):
        commands = ["runtime", "runtime status", "runtime install", "runtime install-archive", "runtime use", "download", "local",
                    "test", "tune", "quiz", "quant-check", "export", "run", "community", "community import",
                    "community share", "community status", "settings", "models", "models add", "models remove",
                    "recommend", "serve", "bench"]
        for command in commands:
            out = io.StringIO()
            with redirect_stdout(out), self.assertRaises(SystemExit) as exit_code:
                cli.main([*command.split(), "--help"])
            self.assertEqual(exit_code.exception.code, 0, command)
            self.assertIn("usage:", out.getvalue())

    def test_existing_ci_commands(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(["--data-dir", directory, "recommend", "--demo", "--context", "2048"]), 0)
        self.assertIn("DEMO", out.getvalue())


class ProgressTests(unittest.TestCase):
    def test_plain_lines_when_piped(self):
        stream = io.StringIO()
        bar = cli.ProgressBar(stream)
        for done in [0, 5, 10, 55, 60, 100]:
            bar.update({"stage": "download", "done": done * 2**30, "total": 100 * 2**30, "bytes_per_second": 2**20, "eta_seconds": 90})
        bar.update({"stage": "verify", "done": 1, "total": None, "message": "Checking the file"})
        lines = stream.getvalue().splitlines()
        self.assertEqual([l.split(" · ")[1].strip() for l in lines[:-1]], ["0%", "10%", "55%", "60%", "100%"])
        self.assertIn("1.0 MB/s", lines[0])
        self.assertIn("about 1m 30s left", lines[0])
        self.assertEqual(lines[-1], "verify · Checking the file")
        self.assertNotIn("\r", stream.getvalue())

    def test_single_line_bar_on_terminals(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        bar = cli.ProgressBar(stream, width=10)
        bar.update({"stage": "tune", "done": 1, "total": 2, "message": "threads"})
        bar.update({"stage": "tune", "done": 2, "total": 2, "message": "done"})
        bar.close()
        self.assertIn("\r[#####-----] tune ·  50%", stream.getvalue())
        self.assertIn("[##########]", stream.getvalue())
        self.assertNotIn("\n", stream.getvalue())


class RuntimeAndSettingsTests(CliCase):
    def test_runtime_commands(self):
        code, out, _ = self.run_cli("runtime", "status")
        self.assertEqual(code, 0)
        self.assertIn("llama.cpp b1 (cpu backend)", out)
        code, out, _ = self.run_cli("runtime", "status", "--json")
        self.assertTrue(json.loads(out)["installed"])
        code, out, err = self.run_cli("runtime", "install", "--allow-unverified")
        self.assertEqual(code, 0)
        self.assertIn(("install", True), self.fakes.calls)
        self.assertIn("download ·  50%", err)
        self.assertEqual(self.run_cli("runtime", "use", self.temp.name)[0], 0)
        self.assertEqual(self.run_cli("runtime", "install-archive", "x.zip")[0], 0)

    def test_settings_and_models(self):
        target = Path(self.temp.name) / "my-models"
        code, out, _ = self.run_cli("settings", "--models-dir", str(target))
        self.assertEqual((code, json.loads(out)["models_dir"]), (0, str(target.resolve())))
        self.assertEqual(self.store.get("settings")["models_dir"], str(target.resolve()))
        code, out, _ = self.run_cli("models")
        self.assertEqual([v["id"] for v in json.loads(out)], [self.variant.id, self.other.id])
        with patch.object(llm_configurator.catalogue, "add_entry", create=True, return_value={"added": "a/b"}) as add, \
             patch.object(llm_configurator.catalogue, "remove_entry", create=True, return_value={"removed": "a/b"}) as remove:
            self.assertEqual(self.run_cli("models", "add", "a/b", "a/b-GGUF")[0], 0)
            self.assertEqual(self.run_cli("models", "remove", "a/b")[0], 0)
        add.assert_called_once()
        self.assertEqual(add.call_args.args[1:], ("a/b", "a/b-GGUF"))
        remove.assert_called_once()


class DownloadTests(CliCase):
    def test_download_with_progress(self):
        code, out, err = self.run_cli("download", self.variant.id, "--yes")
        self.assertEqual(code, 0, err)
        self.assertIn("Download Demo Small Q4_K_M", err)
        self.assertIn("download · 100%", err)
        self.assertTrue(out.strip().endswith(Path(self.variant.filename).name))
        self.assertEqual(self.fakes.calls[0][:2], ("download", self.variant.id))
        # Second time: the local copy is reused, nothing downloaded.
        code, out, _ = self.run_cli("download", self.variant.id)
        self.assertIn("Already on this computer", out)
        self.assertEqual(len([c for c in self.fakes.calls if c[0] == "download"]), 1)

    def test_prompt_decline_unknown_model_and_disk_space(self):
        self.assertEqual(self.run_cli("download", self.variant.id, stdin=["n"])[0], 0)
        self.assertEqual(self.fakes.calls, [])
        code, _, err = self.run_cli("download", "nope", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("Model ID not found", err)
        plan = self.fakes.modules["downloads"].plan
        self.fakes.modules["downloads"].plan = lambda v, d: {**plan(v, d), "enough_space": False, "disk_free": 0}
        code, _, err = self.run_cli("download", self.variant.id, "--yes")
        self.assertEqual(code, 1)
        self.assertIn("Not enough disk space", err)

    def test_ctrl_c_cancels_the_job(self):
        self.fakes.block_download = True
        original = cli.ProgressBar.update
        calls = []
        def interrupt(bar, value):
            calls.append(value)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return original(bar, value)
        with patch.object(cli.ProgressBar, "update", interrupt):
            code, _, err = self.run_cli("download", self.variant.id, "--yes")
        self.assertEqual(code, 130)
        self.assertIn("Stopping", err)
        self.assertIn("Cancelled.", err)
        self.assertNotIn(self.variant.id, self.fakes.local)


class ModelCommandTests(CliCase):
    def test_test_command_outputs_and_exit_codes(self):
        code, _, err = self.run_cli("test", self.variant.id, "--kind", "smoke")
        self.assertEqual(code, 1)
        self.assertIn("Download it first", err)
        self.downloaded()
        code, out, err = self.run_cli("test", self.variant.id, "--kind", "smoke", "--context", "4096", "--gpu-layers", "0")
        self.assertEqual(code, 0, err)
        self.assertIn("It works.", out)
        self.assertIn("smoke ·  50%", err)
        config = self.fakes.calls[-1][2]
        self.assertEqual((config["context"], config["gpu_layers"]), (4096, 0))
        code, out, _ = self.run_cli("test", self.variant.id, "--json", "--kv", "q8_0")
        self.assertEqual(json.loads(out)["verdict"], "works")
        self.assertEqual(self.fakes.calls[-1][2]["cache_type_k"], "q8_0")
        self.fakes.modules["testing"].run_tests = lambda *a, **k: {"verdict": "failed", "verdict_text": "It did not start."}
        code, out, _ = self.run_cli("test", self.variant.id)
        self.assertEqual((code, out.strip()), (1, "It did not start."))

    def test_invalid_model_settings(self):
        for args in [["--context", "100"], ["--context", "999999"], ["--gpu-layers", "999"], ["--gpu-layers", "3"]]:
            code, _, err = self.run_cli("test", self.variant.id, *args)
            self.assertEqual(code, 1, args)
            self.assertTrue(err.startswith("Error: "))

    def test_tune_then_use_tuned_settings(self):
        self.downloaded()
        code, _, err = self.run_cli("tune", self.variant.id, "--budget", "30")
        self.assertEqual(code, 1)
        self.assertIn("between 60 and 1800", err)
        code, _, err = self.run_cli("export", self.variant.id, "--format", "llama-server", "--tuned")
        self.assertIn("No saved tune", err)
        code, out, err = self.run_cli("tune", self.variant.id, "--budget", "60", "--goal", "prompt")
        self.assertEqual(code, 0, err)
        self.assertIn("10.0 → 12.0 tokens/s", out)
        self.assertEqual(self.store.get("tuned")[-1]["goal"], "prompt")
        code, out, _ = self.run_cli("export", self.variant.id, "--format", "llama-server", "--tuned")
        self.assertEqual(code, 0)
        self.assertIn(str(self.downloaded()), out)
        self.run_cli("test", self.variant.id, "--tuned")
        self.assertEqual(self.fakes.calls[-1][2]["threads"], 8)

    def test_quiz_and_quant_check(self):
        self.downloaded()
        code, out, err = self.run_cli("quiz", self.variant.id, "--workload", "coding", "--needle")
        self.assertEqual(code, 0, err)
        self.assertIn("3 of 4 correct (75%", out)
        self.assertEqual([r["kind"] for r in self.store.get("quality_results")], ["quiz", "needle"])
        code, _, err = self.run_cli("quant-check", self.other.id, self.variant.id)
        self.assertEqual(code, 1)
        self.assertIn("Download it first", err)
        self.downloaded(self.other)
        code, out, err = self.run_cli("quant-check", self.other.id, self.variant.id)
        self.assertEqual(code, 0, err)
        self.assertIn("Q4_K_M: Nearly identical.", out)
        self.assertEqual(self.run_cli("quant-check", self.other.id, self.other.id)[0], 1)

    def test_export_writes_content_and_rejects_unknown_formats(self):
        code, out, err = self.run_cli("export", self.variant.id, "--format", "ollama")
        self.assertEqual(code, 0, err)
        self.assertIn("not downloaded yet", err)
        target = Path(self.temp.name) / "Modelfile"
        self.assertEqual(self.run_cli("export", self.variant.id, "--format", "ollama", "--output", str(target))[0], 0)
        self.assertTrue(target.read_text().startswith("['"))
        code, _, err = self.run_cli("export", self.variant.id, "--format", "made-up")
        self.assertEqual(code, 1)
        self.assertIn("Unknown export format", err)

    def test_run_reports_address_and_failure(self):
        self.downloaded()
        with patch.object(cli, "RUN_POLL_SECONDS", 0):
            self.fakes.release.set()  # the fake server reports not-ready, as if it crashed
            code, out, err = self.run_cli("run", self.variant.id, "--port", "8090")
        self.assertEqual(code, 1)
        self.assertIn("OpenAI-compatible address: http://127.0.0.1:9999/v1", out)
        self.assertIn("ran out of memory", err)
        self.assertEqual(self.fakes.calls[-1][2]["port"], 8090)
        self.assertEqual(self.fakes.modules["llama_server"].LlamaServer.live, 0)

    def test_run_stops_on_ctrl_c(self):
        self.downloaded()
        with patch.object(cli.time, "sleep", side_effect=KeyboardInterrupt):
            code, _, err = self.run_cli("run", self.variant.id)
        self.assertEqual(code, 0)
        self.assertIn("Stopping the server", err)
        self.assertEqual(self.fakes.modules["llama_server"].LlamaServer.live, 0)


class LocalAndCommunityTests(CliCase):
    def test_local_scan_list_and_add(self):
        code, out, _ = self.run_cli("local")
        self.assertIn("No model files found yet", out)
        code, out, _ = self.run_cli("local", "--scan", "--json")
        self.assertEqual(json.loads(out)["files"][0]["path"], f"{SECRET}/models/found.gguf")
        code, _, err = self.run_cli("local", "--add", str(Path(self.temp.name) / "missing.gguf"))
        self.assertEqual(code, 1)
        self.assertIn("existing .gguf", err)
        model = Path(self.temp.name) / "mine.gguf"
        model.write_bytes(b"GGUF")
        code, out, err = self.run_cli("local", "--add", str(model))
        self.assertEqual(code, 0, err)
        self.assertIn("Model ID: local:cdcdcdcdcdcd", out)
        self.assertIn("local:cdcdcdcdcdcd", [v["id"] for v in json.loads(self.run_cli("models")[1])])
        # A registered file is used directly, without discover.
        self.run_cli("test", "local:cdcdcdcdcdcd", "--kind", "smoke")
        self.assertEqual(self.fakes.calls[-1][2]["model_path"], str(model.resolve()))

    def test_community_commands(self):
        code, out, _ = self.run_cli("community", "status")
        self.assertIn("0 community results", out)
        code, out, _ = self.run_cli("community", "import", "--source", "https://example.org/r.json")
        self.assertEqual(code, 0)
        self.assertIn("Imported 1 results; skipped 2", out)
        self.assertEqual(self.store.get("community")["source"], "https://example.org/r.json")
        code, out, err = self.run_cli("community", "share", "a1b2c3")
        self.assertEqual(code, 0)
        self.assertIn("issues/new", out)
        self.assertIn("Nothing has been sent", err)


if __name__ == "__main__":
    unittest.main()
