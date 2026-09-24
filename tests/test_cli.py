"""v0.4 CLI commands with fake workstream modules: output, progress, exit codes and Ctrl+C."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import llm_configurator
from llm_configurator import app, cli
from llm_configurator.domain import Variant
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
        self.match_real_signatures()
        self.stack = self.fakes.installed()
        for target in ["llm_configurator.cli.scan", "llm_configurator.app.scan"]:
            self.stack.enter_context(patch(target, return_value=HARDWARE))

    def tearDown(self):
        self.fakes.release.set()
        self.stack.close()
        self.temp.cleanup()

    def match_real_signatures(self):
        """The shared fakes predate some v0.4 functions; add them with the real modules' signatures."""
        discover, fakes = self.fakes.modules["discover"], self.fakes
        self.located = []

        def find_for_variant(store, variant, verify=True, progress=None, cancel=None):
            return fakes.local.get(variant.id)

        def locations(store=None, extra_dirs=()):
            self.located.append(list(extra_dirs))
            return [{"source": "models_dir", "path": f"{SECRET}/models", "exists": True}] + [
                {"source": "custom", "path": str(d), "exists": True} for d in extra_dirs]

        def add_file(store, path):
            path = Path(path).absolute()
            if not path.is_file():
                raise ValueError(f"{path.name} was not found. Check the path and try again.")
            variant = Variant(**{**self.variant.to_dict(), "id": "local:cdcdcdcdcdcd", "source": "local", "sha256": "cd" * 32})
            record = {"path": str(path), "filename": path.name, "size_bytes": path.stat().st_size, "variant_id": None,
                      "local_variant_id": variant.id, "local_variant": variant.to_dict(), "added": True}
            store.update("local_files", lambda items: [r for r in items or [] if r["path"] != record["path"]] + [record], [])
            fakes.local[variant.id] = path
            fakes.calls.append(("add_file", str(path)))
            return record

        discover.find_for_variant, discover.locations, discover.add_file = find_for_variant, locations, add_file
        discover.local_variants = lambda store: [Variant(**r["local_variant"]) for r in store.get("local_files") or [] if r.get("local_variant")]
        self.fakes.modules["gguf"].shard_paths = lambda path: [Path(path)]

    def run_cli(self, *argv, stdin=None, tty=True):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), patch("builtins.input", side_effect=stdin or ["n"]), \
             patch.object(cli.sys, "stdin", Keyboard(tty)):
            code = cli.main(["--data-dir", self.temp.name, *argv])
        return code, out.getvalue(), err.getvalue()

    def downloaded(self, variant=None):
        variant = variant or self.variant
        path = Path(self.temp.name) / "models" / Path(variant.filename).name
        self.fakes.local[variant.id] = path
        return path


class Keyboard(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self.tty = tty

    def isatty(self):
        return self.tty


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
        self.assertTrue(lines[0].startswith("Download · "))
        self.assertIn("1.0 MB/s", lines[0])
        self.assertIn("about 1m 30s left", lines[0])
        self.assertEqual(lines[-1], "Checking the file")
        self.assertNotIn("\r", stream.getvalue())

    def test_model_loading_is_open_ended_and_uses_plain_words(self):
        stream = io.StringIO()
        bar = cli.ProgressBar(stream)
        bar.update({"stage": "loading", "done": 10.5, "total": 300, "message": "Loading the model into memory…"})
        bar.update({"stage": "first_word", "done": 1, "total": 3})
        lines = stream.getvalue().splitlines()
        self.assertEqual(lines[0], "Loading the model into memory… · 10s so far")
        self.assertNotIn("%", lines[0])
        self.assertTrue(lines[1].startswith("First word · "))

    def test_seconds_and_small_files_read_as_units(self):
        stream = io.StringIO()
        bar = cli.ProgressBar(stream)
        bar.update({"stage": "tune", "done": 7.6, "total": 60, "message": "Trying threads"})
        bar.update({"stage": "verifying", "done": 559392, "total": 559392, "message": "Checking x.gguf"})
        lines = stream.getvalue().splitlines()
        self.assertIn("7s of 1m 00s", lines[0])
        self.assertIn("0.5 MB of 0.5 MB", lines[1])

    def test_ctrl_c_waits_until_the_job_has_stopped(self):
        import threading, time
        stopped = threading.Event()
        def job(progress, cancel):
            progress({"stage": "tune", "done": 1, "total": 60})
            cancel.wait(5)
            time.sleep(0.3)  # like killing llama-bench
            stopped.set()
        def interrupt(bar, value):
            if value:
                raise KeyboardInterrupt
        with patch.object(cli.ProgressBar, "update", interrupt), redirect_stderr(io.StringIO()), \
             self.assertRaises(KeyboardInterrupt):
            cli.run_job(job, "Tuning")
        self.assertTrue(stopped.is_set())

    def test_terminal_line_fits_the_terminal_width(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        with patch.object(cli.shutil, "get_terminal_size", return_value=__import__("os").terminal_size((60, 20))):
            cli.ProgressBar(stream).update({"stage": "download", "done": 45, "total": 100, "message": "x" * 200})
        self.assertEqual(len(stream.getvalue().lstrip("\r").rstrip()), 59)

    def test_single_line_bar_on_terminals(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        stream = Terminal()
        bar = cli.ProgressBar(stream, width=10)
        bar.update({"stage": "tune", "done": 1, "total": 2, "message": "Trying threads"})
        bar.update({"stage": "tune", "done": 2, "total": 2, "message": "done"})
        bar.close()
        self.assertIn("\r[#####-----] Trying threads ·  50%", stream.getvalue())
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
        self.assertIn(f"Saving to {SECRET}/runtime ·  50%", err)
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
             patch.object(llm_configurator.catalogue, "refresh_entry", create=True, return_value={"base_repo": "a/b", "variants": 3}) as fetch, \
             patch.object(llm_configurator.catalogue, "remove_entry", create=True, return_value={"removed": "a/b"}) as remove:
            code, out, _ = self.run_cli("models", "add", "a/b", "a/b-GGUF")
            self.assertEqual(code, 0)
            self.assertIn("Added a/b: 3 downloadable versions", out)
            fetch.side_effect = ValueError("Could not reach Hugging Face")
            code, out, _ = self.run_cli("models", "add", "a/b", "a/b-GGUF")
            self.assertEqual(code, 0)
            self.assertIn("llm-config refresh", out)
            self.assertEqual(self.run_cli("models", "remove", "a/b")[0], 0)
        self.assertEqual(add.call_args.args[1:], ("a/b", "a/b-GGUF"))
        self.assertEqual(fetch.call_args.args[1:], ("a/b",))
        remove.assert_called_once()

    def test_refresh_runs_as_a_cancellable_job(self):
        seen = {}
        def refresh(store, progress=None, cancel=None):
            seen["cancel"] = cancel
            return {"variants": 0}
        with patch.object(cli, "refresh", refresh):
            code, out, _ = self.run_cli("refresh")
        self.assertEqual((code, json.loads(out)), (0, {"variants": 0}))
        self.assertIsNotNone(seen["cancel"])

    def test_runtime_install_prints_the_reason(self):
        detect = self.fakes.modules["runtime_install"].detect
        self.fakes.modules["runtime_install"].install = lambda store, hardware, progress=None, cancel=None, allow_unverified=False: {
            **detect(store), "reason": "No supported GPU found, so the CPU build was chosen.", "warnings": ["Old driver."]}
        code, out, _ = self.run_cli("runtime", "install")
        self.assertEqual(code, 0)
        self.assertIn("Why this build: No supported GPU found", out)
        self.assertIn("Note: Old driver.", out)

    def test_bench_uses_the_installed_llama_bench(self):
        with patch.object(cli, "bench", return_value={"id": "m1", "tps": 5.0}) as bench:
            code, _, err = self.run_cli("bench", self.variant.id, "--model", "x.gguf")
        self.assertEqual(code, 0, err)
        self.assertEqual(bench.call_args.args[2], f"{SECRET}/runtime/llama-bench")


class DownloadTests(CliCase):
    def test_download_with_progress(self):
        code, out, err = self.run_cli("download", self.variant.id, "--yes")
        self.assertEqual(code, 0, err)
        self.assertIn("Download Demo Small Q4_K_M", err)
        self.assertIn("Download · 100%", err)
        self.assertTrue(out.strip().endswith(Path(self.variant.filename).name))
        self.assertEqual(self.fakes.calls[0][:2], ("download", self.variant.id))
        # Second time: the local copy is reused, nothing downloaded.
        code, out, _ = self.run_cli("download", self.variant.id)
        self.assertIn("Already on this computer", out)
        self.assertEqual(len([c for c in self.fakes.calls if c[0] == "download"]), 1)

    def test_prompt_decline_unknown_model_and_disk_space(self):
        self.assertEqual(self.run_cli("download", self.variant.id, stdin=["n"])[0], 0)
        self.assertEqual(self.fakes.calls, [])
        code, _, err = self.run_cli("download", self.variant.id, tty=False)
        self.assertEqual(code, 1)
        self.assertIn("Add --yes", err)
        self.assertEqual(self.run_cli("download", self.variant.id, stdin=EOFError())[0], 0)
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
            if value:
                calls.append(value)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return original(bar, value)
        with patch.object(cli.ProgressBar, "update", interrupt):
            code, _, err = self.run_cli("download", self.variant.id, "--yes")
        self.assertEqual(code, 130)
        self.assertIn("Stopping", err)
        self.assertIn("Paused. Run the same command again to resume.", err)
        self.assertNotIn(self.variant.id, self.fakes.local)

    def test_download_elsewhere_is_registered_so_other_commands_find_it(self):
        elsewhere = Path(self.temp.name) / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / Path(self.variant.filename).name).write_bytes(b"GGUF")  # what the real download leaves behind
        code, out, err = self.run_cli("download", self.variant.id, "--yes", "--directory", str(elsewhere))
        self.assertEqual(code, 0, err)
        self.assertIn(("add_file", str(elsewhere.resolve() / Path(self.variant.filename).name)), self.fakes.calls)

    def test_own_models_are_never_downloaded(self):
        local = Variant(**{**self.variant.to_dict(), "id": "local:abc", "source": "local"})
        self.store.put("local_files", [{"path": "x", "local_variant": local.to_dict()}])
        code, _, err = self.run_cli("download", "local:abc", "--yes")
        self.assertEqual(code, 1)
        self.assertIn("nothing to download", err)
        self.assertEqual(self.fakes.calls, [])


class ModelCommandTests(CliCase):
    def test_test_command_outputs_and_exit_codes(self):
        code, _, err = self.run_cli("test", self.variant.id, "--kind", "smoke")
        self.assertEqual(code, 1)
        self.assertIn("Download it first", err)
        self.downloaded()
        code, out, err = self.run_cli("test", self.variant.id, "--kind", "smoke", "--context", "4096", "--gpu-layers", "0")
        self.assertEqual(code, 0, err)
        self.assertIn("It works.", out)
        self.assertIn(" ·  50%", err)
        config = self.fakes.calls[-1][2]
        self.assertEqual((config["context"], config["gpu_layers"]), (4096, 0))
        seen = {}
        run_tests = self.fakes.run_tests
        def with_target(*args, min_tps=None, **kwargs):
            seen["min_tps"] = min_tps
            return run_tests(*args, **kwargs)
        self.fakes.modules["testing"].run_tests = with_target
        self.run_cli("test", self.variant.id, "--kind", "smoke", "--min-tps", "20")
        self.assertEqual(seen["min_tps"], 20)
        self.run_cli("test", self.variant.id, "--kind", "smoke", "--min-tps", "0")
        self.assertIsNone(seen["min_tps"])
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
        self.assertIn("Writing speed: 10.0 -> 12.0 tokens/s (1.20x)", out)
        self.assertIn("Reading speed: 100.0 -> 110.0 tokens/s (1.10x)", out)
        self.assertIn("Changed settings: CPU threads = 8, flash attention = on", out)
        self.assertNotIn("None", out)
        record = self.store.get("tuned")[-1]
        self.assertEqual(record["goal"], "prompt")
        for key in ["best", "baseline", "best_result", "improvement", "variant_id", "sha256", "fingerprint", "context",
                    "gpu_layers", "n_cpu_moe", "kv_cache_type", "timestamp"]:
            self.assertIn(key, record)
        self.assertNotIn("model_path", record["best"])
        self.assertEqual((record["kv_cache_type"], record["users"]), ("f16", 1))
        self.assertFalse(self.store.get("measurements"))  # shallow tune speeds are not "tested" speeds
        # A tune made for the f16 notepad does not apply to a q8_0 run.
        code, _, err = self.run_cli("export", self.variant.id, "--format", "llama-server", "--tuned", "--kv", "q8_0")
        self.assertEqual(code, 1)
        self.assertIn("No saved tune", err)
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
        self.assertIn("Long-document recall: 3 of 3 hidden facts found", out)
        quiz = self.store.get("quality_results")[0]
        self.assertEqual([r["kind"] for r in self.store.get("quality_results")], ["quiz", "needle"])
        self.assertEqual((quiz["workload"], quiz["score"], quiz["needle"]["found"], quiz["result"]["correct"]), ("coding", 0.75, 3, 3))
        self.assertIn("candidate_id", quiz)
        code, _, err = self.run_cli("quant-check", self.other.id, self.variant.id)
        self.assertEqual(code, 1)
        self.assertIn("Download it first", err)
        self.downloaded(self.other)
        code, out, err = self.run_cli("quant-check", self.other.id, self.variant.id)
        self.assertEqual(code, 0, err)
        self.assertIn("Q4_K_M: Nearly identical.", out)
        check = self.store.get("quality_results")[-1]
        self.assertEqual((check["kind"], check["variant_id"]), ("quant_check", self.other.id))
        self.assertIn("Q4_K_M", check["result"]["results"])
        self.assertEqual(self.run_cli("quant-check", self.other.id, self.other.id)[0], 1)

    def test_export_writes_content_and_rejects_unknown_formats(self):
        code, out, err = self.run_cli("export", self.variant.id, "--format", "ollama")
        self.assertEqual(code, 0, err)
        self.assertIn("not downloaded yet", err)
        target = Path(self.temp.name) / "Modelfile"
        self.assertEqual(self.run_cli("export", self.variant.id, "--format", "ollama", "--output", str(target))[0], 0)
        self.assertTrue(target.read_text().startswith("['"))
        script = Path(self.temp.name) / "start.ps1"
        self.assertEqual(self.run_cli("export", self.variant.id, "--format", "llama-server", "--output", str(script))[0], 0)
        self.assertTrue(script.read_bytes().startswith(b"\xef\xbb\xbf"))
        shell = Path(self.temp.name) / "start.sh"
        self.fakes.modules["export"].export = lambda config, variant, fmt, platform="posix", server_command=None: {
            "format": fmt, "filename": "start.sh", "content": "#!/bin/sh\nexec llama-server\n", "instructions": [],
            "notes": ['The script contains non-English characters. Save it as "UTF-8 with BOM" so Windows reads it.']}
        code, _, err = self.run_cli("export", self.variant.id, "--format", "llama-server", "--output", str(shell))
        self.assertEqual(code, 0)
        self.assertEqual(shell.read_bytes(), b"#!/bin/sh\nexec llama-server\n")
        self.assertNotIn("BOM", err)
        if __import__("os").name != "nt":
            self.assertTrue(shell.stat().st_mode & 0o111)
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
        self.assertIn("was not found", err)
        model = Path(self.temp.name) / "mine.gguf"
        model.write_bytes(b"GGUF")
        code, out, err = self.run_cli("local", "--add", str(model))
        self.assertEqual(code, 0, err)
        self.assertIn("Model ID: local:cdcdcdcdcdcd", out)
        self.assertIn("local:cdcdcdcdcdcd", [v["id"] for v in json.loads(self.run_cli("models")[1])])
        code, out, _ = self.run_cli("local")
        self.assertIn("your own Model ID: local:cdcdcdcdcdcd", out)
        self.run_cli("test", "local:cdcdcdcdcdcd", "--kind", "smoke")
        self.assertEqual(self.fakes.calls[-1][2]["model_path"], str(model.absolute()))

    def test_scan_folders_are_saved_as_full_paths(self):
        seen = {}
        scan = self.fakes.modules["discover"].scan
        def remember(store, extra_dirs=(), progress=None, cancel=None):
            seen["dirs"] = extra_dirs
            return scan(store, extra_dirs, progress, cancel)
        self.fakes.modules["discover"].scan = remember
        code, out, _ = self.run_cli("local", "--dir", "relative/models", "--json")
        self.assertEqual(code, 0)
        full = str(Path("relative/models").resolve())
        self.assertEqual([str(d) for d in seen["dirs"]], [full])
        self.assertIn(full, [l["path"] for l in json.loads(out)["locations"]])

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


class AppTests(unittest.TestCase):
    """app.py against the real discover module: files are trusted only after the hash matches."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        (Path(self.temp.name) / "models").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def variant(self, content, filename="Q4_K_M/model-Q4_K_M.gguf"):
        import hashlib
        base = real_variant()
        return Variant(**{**base.to_dict(), "filename": filename, "size_bytes": len(content),
                          "sha256": hashlib.sha256(content).hexdigest(), "files": []})

    def test_same_size_file_with_other_contents_is_not_the_model(self):
        wanted = self.variant(b"A" * 4096)
        path = Path(self.temp.name) / "models" / "model-Q4_K_M.gguf"
        path.write_bytes(b"B" * 4096)
        self.assertIsNone(app.local_model(self.store, wanted))
        self.assertEqual(app.local_model(self.store, wanted, verify=False), path)  # "probably here" for plans only
        path.write_bytes(b"A" * 4096)
        self.assertEqual(app.local_model(self.store, wanted), path)

    def test_best_tune_matches_users_and_notepad_format(self):
        record = {"variant_id": "v", "context": 4096, "placement": "cpu", "fingerprint": "fp", "users": 1,
                  "kv_cache_type": "f16", "best": {"threads": 8, "cache_type_k": "f16"}, "best_result": {"tps": 9}}
        self.store.put("tuned", [record])
        self.assertIsNotNone(app.best_tune(self.store, "v", 4096, "cpu", "fp", users=1, kv_cache_type="f16"))
        self.assertIsNone(app.best_tune(self.store, "v", 4096, "cpu", "fp", users=4, kv_cache_type="f16"))
        self.assertIsNone(app.best_tune(self.store, "v", 4096, "cpu", "fp", users=1, kv_cache_type="q4_0"))
        self.assertIsNotNone(app.best_tune(self.store, "v", 4096, "cpu", "fp"))  # old callers keep working

    def test_scanned_own_models_become_variants(self):
        local = Variant(**{**real_variant().to_dict(), "id": "local:abc", "source": "local"})
        self.store.put("local_files", [{"path": "x.gguf", "local_variant": local.to_dict()}, {"path": "y.gguf", "local_variant": None}])
        self.assertIn("local:abc", [v.id for v in app.variants(self.store)])
        with self.assertRaises(ValueError):
            app.download_job(self.store, local)

    def test_adding_a_later_part_registers_the_whole_split_model(self):
        from llm_configurator import discover
        folder = Path(self.temp.name) / "mine"
        folder.mkdir()
        for index in (1, 2):
            (folder / f"split-Q4_K_M-0000{index}-of-00002.gguf").write_bytes(b"GGUF" + bytes([index]))
        with patch.object(discover, "add_file", return_value={"path": "p", "variant_id": None, "local_variant_id": "local:x"}) as add:
            record = app.add_local_job(self.store, folder / "split-Q4_K_M-00002-of-00002.gguf")(None, None)
        self.assertEqual(add.call_args.args[1].name, "split-Q4_K_M-00001-of-00002.gguf")
        self.assertEqual(record["model_id"], "local:x")
        self.assertEqual(len(self.store.get("hash_cache")), 2)  # both parts hashed first, so the ID is stable


class OutputTests(unittest.TestCase):
    def test_redirected_windows_output_never_crashes_on_symbols(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(raw, encoding="cp1252")
        with patch.object(cli.sys, "stdout", stream), patch.object(cli.sys, "stderr", stream):
            cli._safe_streams()
            print("✓ → ×", file=stream)
            stream.flush()
        self.assertEqual(raw.getvalue(), "? ? ×\n".encode("cp1252").replace(b"\n", __import__("os").linesep.encode()))

    def test_tune_summary_for_a_cancelled_run(self):
        out = io.StringIO()
        with redirect_stdout(out):
            cli.show_tune({"baseline": {"tps": 10, "pp_tps": None}, "best_result": {"tps": 10}, "changes": {},
                           "stopped": "cancelled", "notes": ["Stopped because you cancelled."]})
        self.assertNotIn("--tuned", out.getvalue())


if __name__ == "__main__":
    unittest.main()
