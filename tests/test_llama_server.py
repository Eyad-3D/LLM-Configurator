import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import psutil

from fixtures import FAKE_LLAMA, fake_command, write_fake_gguf
from llm_configurator import llama_server
from llm_configurator.domain import Cancelled
from llm_configurator.llama_server import LlamaServer, PeakMemory, ServerRegistry, failure_from_log, free_port

FIELDS = ["build_commit", "build_number", "cpu_info", "gpu_info", "backends", "model_filename", "model_type", "model_size",
          "model_n_params", "n_batch", "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll", "type_k", "type_v",
          "n_gpu_layers", "n_cpu_moe", "split_mode", "main_gpu", "no_kv_offload", "flash_attn", "devices",
          "tensor_split", "tensor_buft_overrides", "load_mode", "embeddings", "no_op_offload", "no_host",
          "fit_target", "fit_min_ctx", "n_prompt", "n_gen", "n_depth", "test_time", "avg_ns", "stddev_ns", "avg_ts",
          "stddev_ts", "samples_ns", "samples_ts"]


def env(**values):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_LLAMA_")}
    return mock.patch.dict(os.environ, {**clean, **{f"FAKE_LLAMA_{k}": str(v) for k, v in values.items()}}, clear=True)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.model = str(write_fake_gguf(self.dir / "Qwen3-8B-Q4_K_M.gguf", "qwen3", 36))
        self.servers = []
        patcher = env(LOAD_SECONDS="0.2")
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        for server in self.servers:
            server.stop()
        self.tmp.cleanup()

    def server(self, **config):
        server = LlamaServer(fake_command("server"), {"model_path": self.model, "context": 2048, **config})
        self.servers.append(server)
        return server

    def run_fake(self, mode, *args, extra_env=None):
        environment = {**os.environ, **(extra_env or {})}
        return subprocess.run(fake_command(mode) + list(args), capture_output=True, text=True, timeout=60, env=environment)


class FailureParsingTests(unittest.TestCase):
    def test_real_llama_cpp_messages_become_plain_reasons(self):
        cases = {
            "ggml_backend_cuda_buffer_type_alloc_buffer: allocating 9216.00 MiB on device 0: cudaMalloc failed: out of memory": "Not enough memory",
            "llama_model_load: error loading model: unable to allocate CPU buffer": "Not enough memory",
            "ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 123": "Not enough memory",
            "llama_model_load: error loading model: error loading model hyperparameters: unknown model architecture: 'foo'": "(foo)",
            "gguf_init_from_file: failed to open GGUF file '/x.gguf' (No such file or directory)": "missing",
            "gguf_init_from_reader: invalid magic characters: 'abcd', expected 'GGUF'": "damaged",
            "gguf_init_from_reader: failed to read magic": "damaged",
            "srv          start: couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080": "port",
            "error while handling argument \"-dev\": invalid device: CUDA3": "graphics card",
            "ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected": "graphics card",
            "llama_init_from_model: V cache quantization requires flash_attn": "flash attention",
            "0.00.018.104 E llama_init_from_model: quantized V cache requires flash_attn to be enabled": "flash attention",
            "error while handling argument \"--draft-max\": the argument has been removed. use --spec-draft-n-max":
                "no longer accepts",
            "error: invalid argument: --made-up": "does not understand",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertIn(expected, failure_from_log("some noise\n" + text + "\nmore"))

    def test_unknown_logs_stay_unknown_but_oom_kills_are_named(self):
        self.assertIsNone(failure_from_log("all good\nmain: model loaded"))
        self.assertIsNone(failure_from_log(""))
        self.assertIn("memory ran out", failure_from_log("", returncode=-9))

    def test_command_is_an_argv_prefix(self):
        self.assertEqual(llama_server.argv("/opt/llama-server"), ["/opt/llama-server"])
        self.assertEqual(llama_server.argv(["python", Path("x.py")]), ["python", "x.py"])
        for bad in [[], None, ["ok", 3]]:
            with self.assertRaises(ValueError):
                llama_server.argv(bad)

    def test_timings_are_normalised_without_inventing_numbers(self):
        timings = llama_server._normalize_timings({"prompt_n": 5, "predicted_per_second": 12.5, "junk": "x", "flag": True})
        self.assertEqual(timings["prompt_n"], 5)
        self.assertIsNone(timings["prompt_ms"])
        self.assertNotIn("junk", timings)
        self.assertNotIn("flag", timings)
        self.assertEqual(set(llama_server.TIMING_KEYS) - set(llama_server._normalize_timings(None)), set())

    def test_free_port_is_bindable(self):
        port = free_port()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    def test_config_is_validated_and_needs_a_model(self):
        with self.assertRaises(ValueError):
            LlamaServer("llama-server", {"context": 2048})
        with self.assertRaises(ValueError):
            LlamaServer("llama-server", {"model_path": "/m.gguf", "host": "0.0.0.0"})


class PeakMemoryTests(unittest.TestCase):
    def test_ram_is_sampled_for_the_process_tree(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(30_000_000); time.sleep(5)"])
        try:
            time.sleep(0.3)
            monitor = PeakMemory(os.getpid(), interval=0.05).start()
            time.sleep(0.3)
            result = monitor.stop()
        finally:
            child.kill()
            child.wait()
        self.assertGreater(result["peak_ram_bytes"], psutil.Process().memory_info().rss)
        self.assertIsNone(result["peak_vram_bytes"])
        self.assertGreater(result["samples"], 1)

    def test_cuda_vram_sums_the_tree_from_nvidia_smi(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        try:
            output = f"{os.getpid()}, 1000\n{child.pid}, 24\n99999999, 5000\n"
            done = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            with mock.patch.object(llama_server.subprocess, "run", return_value=done) as run:
                result = PeakMemory(os.getpid(), gpu_backend="cuda", interval=0.05).start().stop()
            self.assertEqual(run.call_args[0][0], ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                                   "--format=csv,noheader,nounits"])
            self.assertEqual(result["peak_vram_bytes"], 1024 * 1024**2)
        finally:
            child.kill()
            child.wait()

    def test_vram_is_unknown_without_nvidia_smi_or_on_other_backends(self):
        with mock.patch.object(llama_server.subprocess, "run", side_effect=FileNotFoundError):
            self.assertIsNone(PeakMemory(os.getpid(), gpu_backend="cuda").start().stop()["peak_vram_bytes"])
        with mock.patch.object(llama_server.subprocess, "run") as run:
            self.assertIsNone(PeakMemory(os.getpid(), gpu_backend="metal").start().stop()["peak_vram_bytes"])
            run.assert_not_called()
        # WDDM drivers print [N/A] instead of numbers.
        na = subprocess.CompletedProcess([], 0, stdout=f"{os.getpid()}, [N/A]\n", stderr="")
        with mock.patch.object(llama_server.subprocess, "run", return_value=na):
            self.assertIsNone(PeakMemory(os.getpid(), gpu_backend="cuda").start().stop()["peak_vram_bytes"])
        # Our process not listed (e.g. inside a container) or no GPU work at all: unknown, not 0 bytes.
        for output in ["", "99999999, 5000\n"]:
            done = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            with mock.patch.object(llama_server.subprocess, "run", return_value=done):
                self.assertIsNone(PeakMemory(os.getpid(), gpu_backend="cuda").start().stop()["peak_vram_bytes"])

    def test_vram_rows_without_numbers_only_affect_their_process(self):
        other = f"{os.getpid()}, 700\n1, [N/A]\n2, [Insufficient Permissions]\n"
        ours_na = f"{os.getpid()}, [N/A]\n1, 50\n"
        for output, expected in [(other, 700 * 1024**2), (ours_na, None)]:
            done = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
            with mock.patch.object(llama_server.subprocess, "run", return_value=done):
                self.assertEqual(PeakMemory(os.getpid(), gpu_backend="cuda").start().stop()["peak_vram_bytes"], expected)
        timeout = subprocess.TimeoutExpired("nvidia-smi", 5)
        with mock.patch.object(llama_server.subprocess, "run", side_effect=timeout):
            monitor = PeakMemory(os.getpid(), gpu_backend="cuda")
            self.assertIsNone(monitor.start().stop()["peak_vram_bytes"])
            self.assertTrue(monitor._vram_ok, "a slow nvidia-smi is retried, not given up on")

    @unittest.skipUnless(sys.platform.startswith("linux") or os.name == "nt", "OS keeps a peak RSS")
    def test_peak_before_sampling_started_is_counted(self):
        code = ("import sys, time\nx = bytearray(80_000_000)\nx[::4096] = b'1' * len(x[::4096])\ndel x\n"
                "print('ready', flush=True)\ntime.sleep(10)\n")
        child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
        try:
            child.stdout.readline()
            result = PeakMemory(child.pid, interval=0.05).start().stop()
            now_rss = psutil.Process(child.pid).memory_info().rss
        finally:
            child.kill()
            child.wait()
            child.stdout.close()
        self.assertGreater(result["peak_ram_bytes"], now_rss + 60_000_000)

    def test_monitor_can_be_restarted(self):
        monitor = PeakMemory(os.getpid(), interval=0.02)
        monitor.start().stop()
        first = monitor.samples
        monitor.start()
        time.sleep(0.2)
        self.assertGreater(monitor.stop()["samples"], first + 2)

    def test_gone_process_reports_no_ram(self):
        self.assertIsNone(PeakMemory(99999999).start().stop()["peak_ram_bytes"])


class ServerTests(Base):
    def test_start_waits_for_health_then_chats_with_real_timings(self):
        events = []
        with env(LOAD_SECONDS="0.8"):
            server = self.server(threads=8, gpu_layers=36, total_layers=36, flash_attn="on").start(progress=events.append)
        self.assertTrue(server.ready())
        self.assertTrue(server.base_url.startswith("http://127.0.0.1:"))
        self.assertGreaterEqual(server.load_seconds, 0.7)
        self.assertIn("Loading the model into memory…", [e["message"] for e in events])
        self.assertEqual(events[-1]["stage"], "ready")
        for event in events:
            self.assertTrue({"stage", "done", "total", "message"} <= set(event))
        reply = server.chat([{"role": "user", "content": "What is 2+2? Answer with just the number."}])
        self.assertEqual(reply["text"], "4")
        self.assertEqual(reply["finish_reason"], "stop")
        timings = reply["timings"]
        for key in llama_server.TIMING_KEYS:
            self.assertIsInstance(timings[key], (int, float), key)
        self.assertGreater(timings["prompt_n"], 10)
        self.assertEqual(timings["predicted_n"], 1)
        self.assertAlmostEqual(timings["predicted_per_second"], 40.0, delta=2)
        self.assertEqual(reply["usage"]["prompt_tokens"], timings["prompt_n"])
        self.assertEqual(reply["usage"]["completion_tokens"], 1)

    def test_prompt_cache_is_off_by_default_so_timings_stay_honest(self):
        server = self.server().start()
        messages = [{"role": "user", "content": "Tell me something about memory."}]
        first, second = server.chat(messages), server.chat(messages)
        self.assertEqual(first["timings"]["prompt_n"], second["timings"]["prompt_n"])
        self.assertEqual(second["timings"]["cache_n"], 0)
        cached = server.chat(messages, cache_prompt=True)
        self.assertGreater(cached["timings"]["cache_n"], 0)
        self.assertLess(cached["timings"]["prompt_n"], first["timings"]["prompt_n"])

    def test_settings_change_speed(self):
        slow = self.server(threads=2).start().chat([{"role": "user", "content": "hi"}], max_tokens=16)
        fast = self.server(threads=8, gpu_layers=36, total_layers=36).start().chat([{"role": "user", "content": "hi"}],
                                                                                    max_tokens=16)
        self.assertGreater(fast["timings"]["predicted_per_second"], 2 * slow["timings"]["predicted_per_second"])

    def test_thinking_can_be_switched_off(self):
        with env(REASONING="1", LOAD_SECONDS="0.1"):
            server = self.server().start()
        messages = [{"role": "user", "content": "What is 6*7?"}]
        thinking = server.chat(messages)
        self.assertEqual(thinking["text"], "42")
        self.assertIn("think", thinking["reasoning"])
        plain = server.chat(messages, enable_thinking=False)
        self.assertEqual(plain["text"], "42")
        self.assertIsNone(plain["reasoning"])
        self.assertLess(plain["timings"]["predicted_n"], thinking["timings"]["predicted_n"])

    def test_completion_needle_limits_stop_words_and_tokenize(self):
        server = self.server().start()
        prompt = "Some filler. The secret code is ALPHA-7. More filler.\nQuestion: what is the secret code?\nAnswer:"
        full = server.complete(prompt)
        self.assertEqual(full["text"], "The secret code is ALPHA-7.")
        self.assertEqual(full["finish_reason"], "stop")
        self.assertEqual(full["usage"]["completion_tokens"], full["timings"]["predicted_n"])
        cut = server.complete(prompt, max_tokens=3)
        self.assertEqual(cut["finish_reason"], "length")
        self.assertEqual(cut["timings"]["predicted_n"], 3)
        stopped = server.chat([{"role": "user", "content": "The secret code is ZULU. Repeat it."}], stop=["ZULU"])
        self.assertEqual(stopped["text"], "The secret code is ")
        tokens = server.tokenize("hello world")
        self.assertEqual(len(tokens), 3)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        self.assertEqual(server.tokenize("hello world", add_special=True)[1:], tokens)

    def test_too_long_prompt_is_a_plain_error(self):
        server = self.server(context=256).start()
        with self.assertRaisesRegex(ValueError, "longer than the model's context window"):
            server.chat([{"role": "user", "content": "word " * 400}])
        self.assertTrue(server.ready())

    def test_raw_http_matches_llama_cpp_shapes(self):
        server = self.server(alias="my-model").start()
        body = server.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Say 'hello'"}]})
        self.assertEqual(list(body)[:7], ["choices", "created", "model", "system_fingerprint", "object", "usage", "id"])
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["model"], "my-model")
        self.assertEqual(body["choices"][0]["message"], {"role": "assistant", "content": "hello"})
        self.assertEqual(list(body["timings"]), ["cache_n", "prompt_n", "prompt_ms", "prompt_per_token_ms",
                                                 "prompt_per_second", "predicted_n", "predicted_ms",
                                                 "predicted_per_token_ms", "predicted_per_second"])
        completion = server.request("/completion", {"prompt": "1+1=", "n_predict": 4})
        for key in ["content", "stop", "tokens_predicted", "tokens_evaluated", "stop_type", "stopping_word", "timings"]:
            self.assertIn(key, completion)
        self.assertEqual(server.request("/health"), {"status": "ok"})
        self.assertEqual(server.request("/v1/models")["data"][0]["id"], "my-model")
        with self.assertRaisesRegex(ValueError, "refused the request \\(404\\)"):
            server.request("/nope", {})

    def test_health_is_503_with_llama_cpp_body_while_loading(self):
        import urllib.error
        with env(LOAD_SECONDS="3"):
            server = self.server(port=free_port())
            server._spawn(server.config["port"])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = server._health()
            if state == "loading":
                break
            time.sleep(0.05)
        self.assertEqual(state, "loading")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            llama_server._OPENER.open(f"{server.base_url}/health", timeout=2)
        self.assertEqual(caught.exception.code, 503)
        self.assertEqual(json.loads(caught.exception.read()),
                         {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}})
        caught.exception.close()
        self.assertFalse(server.ready())

    def test_stop_kills_the_whole_process_tree(self):
        wrapper = self.dir / "wrapper.py"
        wrapper.write_text("import subprocess, sys\n"
                           f"sys.exit(subprocess.call([sys.executable, {str(FAKE_LLAMA)!r}, '--as', 'server'] + sys.argv[1:]))\n")
        server = LlamaServer([sys.executable, str(wrapper)], {"model_path": self.model, "context": 2048})
        self.servers.append(server)
        server.start()
        children = psutil.Process(server.pid).children(recursive=True)
        self.assertTrue(children)
        self.assertTrue(server.ready())
        server.stop(timeout=5)
        for process in children:
            self.assertFalse(process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
        self.assertFalse(server.running())
        server.stop()
        self.assertFalse(server.running())

    def test_context_manager_starts_and_stops(self):
        with self.server() as server:
            pid = server.pid
            self.assertTrue(server.ready())
        self.assertFalse(psutil.pid_exists(pid) and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE)
        self.assertFalse(server.running())
        with self.assertRaisesRegex(ValueError, "stopped|not been started"):
            server.chat([{"role": "user", "content": "hi"}])

    def test_log_file_is_kept_when_given_and_temp_logs_are_removed(self):
        log = self.dir / "logs" / "server.log"
        server = LlamaServer(fake_command("server"), {"model_path": self.model}, log_path=log)
        self.servers.append(server)
        server.start()
        server.stop()
        self.assertRegex(log.read_text(), r"(?m)^\d+\.\d\d\.\d{3}\.\d{3} I srv  llama_server: model loaded$")
        temp = self.server().start()
        path = temp.log_path
        temp.stop()
        self.assertFalse(path.exists())
        self.assertIn("listening on http://127.0.0.1", temp.log_tail())

    def test_explicit_port_is_used(self):
        port = free_port()
        self.assertEqual(self.server(port=port).start().base_url, f"http://127.0.0.1:{port}")

    def test_busy_auto_port_is_retried_once(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            taken = busy.getsockname()[1]
            fresh = free_port()
            with mock.patch.object(llama_server, "free_port", side_effect=[taken, fresh]):
                server = self.server().start()
            self.assertEqual(server.port, fresh)

    def test_busy_explicit_port_fails_plainly(self):
        with socket.socket() as busy:
            busy.bind(("127.0.0.1", 0))
            busy.listen()
            with self.assertRaisesRegex(ValueError, "port is already used"):
                self.server(port=busy.getsockname()[1]).start()

    def test_failures_become_plain_reasons_and_leave_nothing_running(self):
        cases = [({"FAIL": "oom"}, {}, "Not enough memory"),
                 ({"FAIL": "arch"}, {}, "does not know this model's design \\(made-up-arch\\)"),
                 ({"FAIL": "port"}, {}, "port is already used"),
                 ({}, {"model_path": str(self.dir / "missing.gguf")}, "model file is missing"),
                 ({"MAX_GPU_LAYERS": "10"}, {"gpu_layers": 20}, "Not enough memory"),
                 ({}, {"flash_attn": "off", "cache_type_k": "q8_0"}, None)]
        corrupt = self.dir / "broken.gguf"
        corrupt.write_bytes(b"NOPE" + b"\0" * 64)
        cases.append(({}, {"model_path": str(corrupt)}, "damaged or incomplete"))
        other = write_fake_gguf(self.dir / "odd.gguf", "strangeformer", 12)
        cases.append(({}, {"model_path": str(other)}, "\\(strangeformer\\)"))
        for knobs, config, expected in cases:
            with self.subTest(knobs=knobs, config=config), env(LOAD_SECONDS="0.05", **knobs):
                server = self.server(**config)
                if expected is None:
                    server.start()
                    self.assertTrue(server.ready())
                    continue
                with self.assertRaisesRegex(ValueError, expected):
                    server.start(timeout=20)
                self.assertFalse(server.running())
                self.assertIn(".gguf" if "port" not in expected else "port", server.log_tail())

    def test_backend_warning_is_reported_while_the_server_still_runs(self):
        with env(FAIL="backend", LOAD_SECONDS="0.05"):
            server = self.server(gpu_layers=10, total_layers=36).start()
        self.assertTrue(server.ready())
        self.assertIn("processor only", server.warnings()[0])

    def test_timeout_stops_the_server(self):
        with env(LOAD_SECONDS="30"):
            server = self.server()
            with self.assertRaisesRegex(ValueError, "did not finish loading within 1 seconds"):
                server.start(timeout=1)
        self.assertFalse(server.running())

    def test_cancel_during_load_stops_the_server(self):
        cancel = threading.Event()
        seen = []

        def progress(event):
            seen.append(event)
            if event["message"].startswith("Loading"):
                cancel.set()
        with env(LOAD_SECONDS="30"):
            server = self.server()
            with self.assertRaises(Cancelled):
                server.start(progress=progress, cancel=cancel)
        self.assertFalse(server.running())
        self.assertTrue(seen)

    def test_missing_executable_is_plain(self):
        server = LlamaServer([str(self.dir / "no-such-llama-server")], {"model_path": self.model})
        with self.assertRaisesRegex(ValueError, "Could not run llama-server"):
            server.start()

    def test_proxy_settings_are_ignored_for_loopback(self):
        with mock.patch.dict(os.environ, {"http_proxy": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9",
                                          "no_proxy": "", "NO_PROXY": ""}):
            server = self.server().start()
            self.assertEqual(server.chat([{"role": "user", "content": "What is 3*3?"}])["text"], "9")


class RegistryTests(Base):
    def test_start_status_restart_and_stop(self):
        registry = ServerRegistry(log_dir=self.dir / "logs")
        idle = registry.status()
        self.assertEqual(idle["running"], False)
        self.assertIsNone(idle["openai_base_url"])
        status = registry.start(fake_command("server"), {"model_path": self.model, "alias": "qwen"})
        self.assertTrue(status["running"])
        self.assertEqual(status["openai_base_url"], status["base_url"] + "/v1")
        self.assertEqual(status["model"], "qwen")
        self.assertEqual(status["config"]["model_path"], self.model)
        self.assertIn("listening on", status["log_tail"])
        self.assertIsNotNone(status["started_at"])
        first_pid = status["pid"]
        with self.assertRaises(ValueError):
            registry.start(fake_command("server"), {"model_path": self.model, "context": 1})
        self.assertEqual(registry.status()["pid"], first_pid)
        again = registry.start(fake_command("server"), {"model_path": self.model})
        self.assertNotEqual(again["pid"], first_pid)
        self.assertEqual(again["model"], "Qwen3-8B-Q4_K_M.gguf")
        self.assertFalse(psutil.pid_exists(first_pid) and psutil.Process(first_pid).status() != psutil.STATUS_ZOMBIE)
        stopped = registry.stop()
        self.assertFalse(stopped["running"])
        self.assertIsNone(stopped["base_url"])
        self.assertFalse(psutil.pid_exists(again["pid"]) and psutil.Process(again["pid"]).status() != psutil.STATUS_ZOMBIE)
        self.assertEqual(registry.stop()["running"], False)

    def test_crash_and_failed_start_show_errors(self):
        registry = ServerRegistry()
        status = registry.start(fake_command("server"), {"model_path": self.model})
        psutil.Process(status["pid"]).kill()
        time.sleep(0.3)
        crashed = registry.status()
        self.assertFalse(crashed["running"])
        self.assertTrue(crashed["error"])
        with env(FAIL="oom", LOAD_SECONDS="0.05"):
            with self.assertRaisesRegex(ValueError, "Not enough memory"):
                registry.start(fake_command("server"), {"model_path": self.model})
        failed = registry.status()
        self.assertFalse(failed["running"])
        self.assertIn("Not enough memory", failed["error"])
        registry.stop()

    def test_status_is_readable_while_starting_and_cancel_works(self):
        registry = ServerRegistry()
        cancel = threading.Event()
        errors = []

        def run():
            try:
                registry.start(fake_command("server"), {"model_path": self.model}, cancel=cancel)
            except Cancelled as error:
                errors.append(error)
        with env(LOAD_SECONDS="30"):
            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 10
            while not registry.status()["starting"] and time.monotonic() < deadline:
                time.sleep(0.02)
            status = registry.status()
            self.assertTrue(status["starting"])
            self.assertFalse(status["running"])
            cancel.set()
            thread.join(10)
        self.assertTrue(errors)
        self.assertFalse(registry.status()["starting"])


SAMPLES = Path(__file__).resolve().parent / "integration" / "samples"


def sample_stderr(name):
    text = (SAMPLES / name).read_text()
    return text.split("# ---- stderr ----\n", 1)[1] if "# ---- stderr ----" in text else text


class RealLogTests(unittest.TestCase):
    """failure_from_log against logs captured from a REAL CPU-only llama.cpp (every one starts with the
    'no usable GPU found' warning, even with -ngl 0)."""

    def test_real_error_samples(self):
        expected = {"error-server-bad-flag.txt": "does not understand one of the settings",
                    "error-server-bad-cache-type.txt": "does not understand one of the settings",
                    "error-server-bad-fa-value.txt": "does not understand one of the settings",
                    "error-server-missing-file.txt": "model file is missing",
                    "error-server-corrupt.txt": "damaged or incomplete",
                    "error-server-unknown-arch.txt": "(notarealarch)",
                    "error-server-port-in-use.txt": "port is already used"}
        for name, reason in expected.items():
            with self.subTest(name=name):
                self.assertIn(reason, failure_from_log(sample_stderr(name), 1))

    def test_more_real_failures(self):
        # Lines captured from the real llama-server 4df29be (review of the fix-up round).
        cases = {
            "E gguf_init_from_file: failed to open GGUF file '/m/noperm.gguf' (Permission denied)": "not allowed to read the model file (noperm.gguf)",
            "E gguf_init_from_file: failed to open GGUF file '/m/nope-draft.gguf' (No such file or directory)": "missing (nope-draft.gguf)",
            "E llama_model_load: error loading model: illegal split file idx: 1 (file: /m/x-00002-of-00004.gguf), "
            "model must be loaded with the first split": "first part of a split model",
            "/src/llama-context.cpp:123: GGML_ASSERT(n_outputs >= 1) failed\n"
            "warning: 30 ../sysdeps/unix/sysv/linux/wait4.c: No such file or directory": "crashed",
        }
        for text, reason in cases.items():
            with self.subTest(text=text[:60]):
                found = failure_from_log(text, 1)
                self.assertIn(reason, found)
                self.assertNotIn("/m/", found)
        self.assertIn("crashed", failure_from_log("nothing useful", -11))
        self.assertIn("crashed", failure_from_log("nothing useful", 3221225477))

    def test_cpu_only_warning_is_not_a_failure(self):
        log = sample_stderr("server-log.txt")
        self.assertIsNone(failure_from_log(log))
        self.assertIsNone(failure_from_log(log, -15))
        self.assertIn("memory ran out", failure_from_log(log, -9))
        self.assertIn("Not enough memory", failure_from_log(log + "\nllama_model_load: error: failed to allocate buffer", 1))
        self.assertIn("graphics card", failure_from_log(log + "\nggml_metal_init: error: no device", 1))


class SeamTests(Base):
    """What other modules call, run against the fake end to end."""

    def test_pid_after_start_and_stop_is_idempotent_even_after_a_failed_start(self):
        server = self.server().start()
        self.assertTrue(psutil.pid_exists(server.pid))
        server.stop()
        server.stop()
        with env(FAIL="oom", LOAD_SECONDS="0.05"):
            failed = self.server()
            with self.assertRaises(ValueError):
                failed.start()
        failed.stop()
        failed.stop()
        self.assertIsNone(LlamaServer(fake_command("server"), {"model_path": self.model}).stop())

    def test_chat_template_kwargs_reach_the_server(self):
        with env(REASONING="1", LOAD_SECONDS="0.05"):
            server = self.server().start()
            thinking = server.chat([{"role": "user", "content": "What is 6*7?"}])
            plain = server.chat([{"role": "user", "content": "What is 6*7?"}],
                                chat_template_kwargs={"enable_thinking": False})
        self.assertTrue(thinking["reasoning"])
        self.assertIsNone(plain["reasoning"])
        self.assertEqual(plain["text"], "42")
        body = {}
        with mock.patch.object(server, "request", side_effect=lambda path, b, timeout: body.update(b) or
                               {"choices": [{"message": {"content": "x"}}]}):
            server.chat([{"role": "user", "content": "hi"}], chat_template_kwargs={"a": 1, "enable_thinking": True},
                        enable_thinking=False)
        self.assertEqual(body["chat_template_kwargs"], {"a": 1, "enable_thinking": False})

    def test_testing_smoke_test_sends_no_thinking(self):
        from llm_configurator import testing
        with env(REASONING="1", LOAD_SECONDS="0.05"):
            result = testing.smoke_test(fake_command("server"), {"model_path": self.model, "context": 2048})
        self.assertTrue(result["ok"], result)

    def test_evals_needle_test_passes_against_the_fake(self):
        from llm_configurator import evals
        server = self.server(context=8192).start()
        result = evals.needle_test(server.chat, server.tokenize, 2000)
        self.assertEqual(result["found"], len(result["results"]), result)

    def test_user_llama_env_does_not_change_the_server(self):
        with mock.patch.dict(os.environ, {"LLAMA_API_KEY": "secret", "LLAMA_ARG_CTX_SIZE": "8"}):
            server = self.server().start()
            self.assertEqual(server.chat([{"role": "user", "content": "What is 5+5?"}])["text"], "10")
        env_used = llama_server.server_env(server.config)
        self.assertNotIn("LLAMA_API_KEY", env_used)
        self.assertEqual(env_used["LLAMA_ARG_CORS_ORIGINS"], "localhost")

    def test_slots_monitoring_page_is_off(self):
        import urllib.error
        import urllib.request
        server = self.server().start()
        self.assertEqual(llama_server.server_env(server.config)["LLAMA_ARG_ENDPOINT_SLOTS"], "0")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.assertRaises(urllib.error.HTTPError) as caught:
            opener.open(f"{server.base_url}/slots", timeout=10)
        self.assertEqual(caught.exception.code, 501)  # what real llama-server answers with the variable set
        caught.exception.close()
        self.assertEqual(server.chat([{"role": "user", "content": "What is 5+5?"}])["text"], "10")

    def test_other_web_pages_cannot_read_the_server(self):
        import urllib.request
        server = self.server().start()
        for origin, allowed in [("https://evil.example", False), ("http://localhost:3000", True),
                                ("http://127.0.0.1:8000", True)]:
            request = urllib.request.Request(server.base_url + "/props", headers={"Origin": origin})
            with llama_server._OPENER.open(request, timeout=10) as response:
                self.assertEqual(response.headers.get("Access-Control-Allow-Origin") == origin, allowed, origin)

    def test_fixed_port_held_by_another_server_is_not_mistaken_for_ours(self):
        port = free_port()
        other = self.server(port=port, alias="OLD-MODEL").start()
        with mock.patch.object(llama_server.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "port is already used"):
                self.server(port=port).start()
            popen.assert_not_called()
        self.assertTrue(other.ready())

    def test_load_progress_has_no_invented_total(self):
        events = []
        with env(LOAD_SECONDS="0.5"):
            self.server().start(progress=events.append)
        loading = [e for e in events if e["stage"] == "loading"]
        self.assertTrue(loading)
        self.assertTrue(all(e["total"] is None and isinstance(e["done"], float) for e in loading))

    def test_registry_never_stays_starting_after_an_os_error(self):
        blocker = self.dir / "notadir"
        blocker.write_text("")
        registry = ServerRegistry(log_dir=blocker / "logs")
        with self.assertRaisesRegex(ValueError, "log file"):
            registry.start(fake_command("server"), {"model_path": self.model})
        status = registry.status()
        self.assertFalse(status["starting"])
        self.assertIn("log file", status["error"])
        with mock.patch.object(LlamaServer, "start", side_effect=RuntimeError("boom /secret/path")):
            with self.assertRaises(RuntimeError):
                ServerRegistry().start(fake_command("server"), {"model_path": self.model})
        registry = ServerRegistry()
        with mock.patch.object(LlamaServer, "start", side_effect=RuntimeError("boom /secret/path")):
            with self.assertRaises(RuntimeError):
                registry.start(fake_command("server"), {"model_path": self.model})
        self.assertFalse(registry.status()["starting"])
        self.assertNotIn("/secret", registry.status()["error"])

    def test_cpu_only_build_bad_setting_gets_the_settings_reason(self):
        with env(GPU="none", LOAD_SECONDS="0.05"):
            server = self.server(gpu_layers=0, cache_type_k="q8_0")
            with mock.patch.object(llama_server.launch, "server_args",
                                   side_effect=lambda c: ["-m", c["model_path"], "-ngl", "0", "--port", str(c["port"]),
                                                          "--no-such-flag"]):
                with self.assertRaisesRegex(ValueError, "does not understand one of the settings"):
                    server.start()
            self.assertIn("no usable GPU found", server.log_tail())
            ok = self.server(gpu_layers=0).start()
        self.assertIn("processor only", ok.warnings()[0])
        self.assertIsNone(ok.failure_reason())

    def test_draft_model_turns_on_speculative_decoding(self):
        draft = str(write_fake_gguf(self.dir / "Qwen3-0.6B-Q8_0.gguf", "qwen3", 28))
        server = self.server(draft_model_path=draft, draft_max=4).start()
        timings = server.complete("The quick brown fox", max_tokens=16)["timings"]
        self.assertGreater(timings["draft_n"], 0)
        self.assertLessEqual(timings["draft_n_accepted"], timings["draft_n"])
        plain = self.server().start().complete("The quick brown fox", max_tokens=16)["timings"]
        self.assertNotIn("draft_n", plain)

    @unittest.skipIf(os.name == "nt", "POSIX exit codes")
    def test_stop_keeps_the_real_exit_code_and_dropped_servers_are_still_tracked(self):
        server = self.server().start()
        self.assertEqual(server.stop(), -15)
        dropped = self.server().start()
        pid = dropped.pid
        self.servers.remove(dropped)
        del dropped
        import gc
        gc.collect()
        live = [s for s in llama_server._LIVE if s.pid == pid]
        self.assertEqual(len(live), 1, "atexit must still know about a server nobody holds any more")
        live[0].stop()

    def test_odd_answers_become_plain_errors(self):
        server = self.server(host="localhost").start()
        self.assertTrue(server.base_url.startswith("http://127.0.0.1:"))
        for reply in [[1, 2], {"choices": [None]}, {"choices": [{"message": "text"}]}, {"choices": "x"}]:
            with self.subTest(reply=reply), mock.patch.object(server, "request", return_value=reply
                                                              if isinstance(reply, dict) else {"tokens": None}):
                with self.assertRaises(ValueError):
                    server.chat([{"role": "user", "content": "hi"}]) if isinstance(reply, dict) else server.tokenize("hi")
        garbage = socket.socket()
        garbage.bind(("127.0.0.1", 0))
        garbage.listen()

        def answer():
            connection, _ = garbage.accept()
            connection.recv(65536)
            connection.sendall(b"this is not HTTP\r\n\r\n")
            connection.close()
        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        real_port, server.port = server.port, garbage.getsockname()[1]
        try:
            with self.assertRaisesRegex(ValueError, "not responding"):
                server.request("/v1/models", timeout=5)
        finally:
            server.port = real_port
            garbage.close()

    @unittest.skipIf(os.name == "nt", "POSIX signals")
    def test_exit_handlers_stop_servers_when_the_app_is_terminated(self):
        import signal
        script = ("import sys, time\n"
                  "sys.path[:0] = [%r, %r]\n"
                  "from fixtures import fake_command\n"
                  "from llm_configurator import llama_server\n"
                  "llama_server.install_exit_handlers()\n"
                  "s = llama_server.LlamaServer(fake_command('server'), {'model_path': %r}).start()\n"
                  "print(s.pid, flush=True)\n"
                  "time.sleep(60)\n") % (str(Path(__file__).parent), str(Path(llama_server.__file__).parents[1]), self.model)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum):
                parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
                child = int(parent.stdout.readline())
                parent.send_signal(signum)
                parent.wait(20)
                parent.stdout.close()
                deadline = time.monotonic() + 10
                while psutil.pid_exists(child) and time.monotonic() < deadline:
                    try:
                        if psutil.Process(child).status() == psutil.STATUS_ZOMBIE:
                            break
                    except psutil.NoSuchProcess:
                        break
                    time.sleep(0.05)
                alive = psutil.pid_exists(child) and psutil.Process(child).status() != psutil.STATUS_ZOMBIE
                self.assertFalse(alive, "llama-server outlived the app")


class FakeBenchTests(Base):
    def bench(self, *args, extra_env=None):
        result = self.run_fake("bench", "-m", self.model, "-o", "json", *args, extra_env=extra_env)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_comma_lists_expand_like_llama_bench(self):
        rows = self.bench("-p", "512", "-n", "128", "-t", "4,8", "-fa", "0,1", "-ngl", "99", "-r", "2")
        self.assertEqual(len(rows), 8)
        self.assertEqual(list(rows[0]), FIELDS)
        # Outer to inner: ngl, ..., fa, threads, then the tests (pp rows first, then tg rows).
        self.assertEqual([(r["flash_attn"], r["n_threads"], r["n_prompt"], r["n_gen"]) for r in rows],
                         [(0, 4, 512, 0), (0, 4, 0, 128), (0, 8, 512, 0), (0, 8, 0, 128),
                          (1, 4, 512, 0), (1, 4, 0, 128), (1, 8, 512, 0), (1, 8, 0, 128)])
        self.assertEqual(len(rows[0]["samples_ts"]), 2)
        self.assertIsInstance(rows[0]["flash_attn"], int)
        self.assertIsInstance(rows[0]["no_kv_offload"], bool)
        self.assertEqual(rows[0]["build_number"], 6512)
        self.assertEqual(rows[0]["model_filename"], self.model)
        best = max((r for r in rows if r["n_gen"]), key=lambda r: r["avg_ts"])
        self.assertEqual((best["n_threads"], best["flash_attn"]), (8, 1))

    def test_tuner_has_a_real_optimum_and_knobs_matter(self):
        rows = self.bench("-p", "0", "-n", "32", "-t", "2-16*2", "-r", "1", "-ngl", "99")
        speeds = {r["n_threads"]: r["avg_ts"] for r in rows}
        self.assertEqual(max(speeds, key=speeds.get), 8)
        ub = {r["n_ubatch"]: r["avg_ts"] for r in self.bench("-p", "512", "-n", "0", "-ub", "128,512,2048", "-r", "1")}
        self.assertEqual(max(ub, key=ub.get), 512)
        ngl = {r["n_gpu_layers"]: r["avg_ts"] for r in self.bench("-n", "16", "-p", "0", "-ngl", "0,18,99", "-r", "1")}
        self.assertLess(ngl[0], ngl[18])
        self.assertLess(ngl[18], ngl[99])
        depth = {r["n_depth"]: r["avg_ts"] for r in self.bench("-n", "16", "-p", "0", "-d", "0,8192", "-r", "1")}
        self.assertLess(depth[8192], depth[0])
        kv = {r["type_k"]: r["avg_ts"] for r in self.bench("-n", "16", "-p", "0", "-fa", "1", "-ctk", "f16,q4_0",
                                                           "-ctv", "q4_0", "-r", "1")}
        self.assertLess(kv["q4_0"], kv["f16"])
        moe = str(write_fake_gguf(self.dir / "moe.gguf", "qwen3moe", 48, experts=128))
        result = self.run_fake("bench", "-m", moe, "-o", "json", "-n", "16", "-p", "0", "-ncmoe", "0,24", "-r", "1")
        rows = json.loads(result.stdout)
        self.assertGreater(rows[0]["avg_ts"], rows[1]["avg_ts"])
        tps = self.bench("-n", "16", "-p", "0", "-t", "8", "-fa", "1", "-r", "1", extra_env={"FAKE_LLAMA_TPS": "100"})
        self.assertAlmostEqual(tps[0]["avg_ts"], 100, delta=1)

    def test_jsonl_and_markdown_outputs(self):
        result = self.run_fake("bench", "-m", self.model, "-o", "jsonl", "-p", "64", "-n", "8", "-r", "1")
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([(r["n_prompt"], r["n_gen"]) for r in lines], [(64, 0), (0, 8)])
        md = self.run_fake("bench", "-m", self.model, "-p", "64", "-n", "8", "-r", "1").stdout
        self.assertIn("| pp64 |", md)
        self.assertIn("| tg8 |", md)

    def test_out_of_memory_stops_mid_array_like_the_real_tool(self):
        result = self.run_fake("bench", "-m", self.model, "-o", "json", "-ngl", "10,40", "-p", "0", "-n", "8", "-r", "1",
                               extra_env={"FAKE_LLAMA_MAX_GPU_LAYERS": "20"})
        self.assertEqual(result.returncode, 1)
        # Like real llama-bench without -v: the cause is not on stderr, just one line with the path as given.
        self.assertEqual(result.stderr, f"llama_bench: error: failed to load model '{self.model}'\n")
        self.assertTrue(result.stdout.startswith("[\n"))
        verbose = self.run_fake("bench", "-m", self.model, "-o", "json", "-ngl", "40", "-p", "0", "-n", "8", "-r", "1",
                                "-v", extra_env={"FAKE_LLAMA_MAX_GPU_LAYERS": "20"})
        self.assertIn("cudaMalloc failed: out of memory", verbose.stderr)
        self.assertNotIn("]", result.stdout.rstrip()[-1:])
        self.assertIn('"n_gpu_layers": 10', result.stdout)

    def test_bad_flags_and_missing_models_fail(self):
        self.assertEqual(self.run_fake("bench", "--made-up", "1").returncode, 1)
        missing = self.run_fake("bench", "-m", str(self.dir / "nope.gguf"), "-o", "json")
        self.assertEqual(missing.returncode, 1)
        self.assertEqual(missing.stderr, f"llama_bench: error: failed to load model '{self.dir / 'nope.gguf'}'\n")
        self.assertEqual(missing.stdout, "[\n")
        self.assertEqual(self.run_fake("server", "-m", self.model, "--made-up").returncode, 1)
        self.assertIn("error: invalid argument: --made-up",
                      self.run_fake("server", "-m", self.model, "--made-up").stderr)


class FakeToolTests(Base):
    def test_version_output_is_realistic_and_on_stderr(self):
        for mode in ["server", "perplexity", "cli"]:
            result = self.run_fake(mode, "--version")
            self.assertEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertRegex(result.stderr, r"^version: 0\.1\.0-dev \(build 6512, commit fa4ec0d\)\n"
                                            r"built with GNU 13\.3\.0 for (Linux x86_64|Darwin arm64|Windows AMD64)\n$")
        classic = self.run_fake("server", "--version", extra_env={"FAKE_LLAMA_VERSION_STYLE": "classic"})
        self.assertRegex(classic.stderr, r"^version: 6512 \(fa4ec0d\)\nbuilt with .+ for .+\n$")
        bench = self.run_fake("bench", "--version")  # real llama-bench has no --version
        self.assertEqual(bench.returncode, 1)
        self.assertTrue(bench.stdout.startswith("usage: "))
        self.assertEqual(bench.stderr, "error: invalid parameter for argument: --version\n")

    def test_mode_can_come_from_the_program_name(self):
        link = self.dir / "llama-bench.py"
        link.write_text(FAKE_LLAMA.read_text())
        result = subprocess.run([sys.executable, str(link), "-m", self.model, "-o", "json", "-p", "8", "-n", "0", "-r", "1"],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(json.loads(result.stdout)[0]["n_prompt"], 8)
        self.assertEqual(subprocess.run([sys.executable, str(FAKE_LLAMA)], capture_output=True, timeout=60).returncode, 2)

    def test_kl_divergence_round_trip(self):
        corpus = self.dir / "corpus.txt"
        corpus.write_text("The quick brown fox jumps over the lazy dog. " * 400)
        base = self.dir / "base.kld"
        reference = str(write_fake_gguf(self.dir / "m-F16.gguf", "qwen3", 36))
        first = self.run_fake("perplexity", "-m", reference, "-f", str(corpus), "-c", "512", "--kl-divergence-base", str(base))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertRegex(first.stdout, r"^0\.00 minutes\n\[1\]\d+\.\d{4},\[2\]")
        self.assertRegex(first.stderr, r"Final estimate: PPL = \d+\.\d{4} \+/- \d+\.\d{5}")
        self.assertEqual(base.read_bytes()[:8], b"_logits_")
        results = {}
        for quant in ["Q8_0", "Q4_K_M", "Q2_K"]:
            candidate = str(write_fake_gguf(self.dir / f"m-{quant}.gguf", "qwen3", 36))
            out = self.run_fake("perplexity", "-m", candidate, "--kl-divergence-base", str(base), "--kl-divergence")
            self.assertEqual(out.returncode, 0, out.stderr)
            for header in ["====== Perplexity statistics ======", "====== KL divergence statistics ======",
                           "====== Token probability statistics ======", "Mean PPL(base)                :",
                           "Cor(ln(PPL(Q)), ln(PPL(base))):", "99.0%   KLD:", "Median  KLD:", " 5.0%   KLD:",
                           "RMS Δp    :", "Same top p:"]:
                self.assertIn(header, out.stdout)
            mean = next(line for line in out.stdout.splitlines() if line.startswith("Mean    KLD:"))
            self.assertRegex(mean, r"^Mean    KLD: +\d+\.\d{6} ± +\d+\.\d{6}$")
            results[quant] = float(mean.split(":")[1].split("±")[0])
        self.assertLess(results["Q8_0"], results["Q4_K_M"])
        self.assertLess(results["Q4_K_M"], results["Q2_K"])
        same = self.run_fake("perplexity", "-m", reference, "--kl-divergence-base", str(base), "--kl-divergence")
        self.assertIn("Mean    KLD:   0.000000", same.stdout)

    def test_perplexity_needs_enough_text_and_a_valid_base_file(self):
        short = self.dir / "short.txt"
        short.write_text("too short")
        result = self.run_fake("perplexity", "-m", self.model, "-f", str(short))
        self.assertEqual(result.returncode, 1)
        self.assertIn("you need at least 1024 tokens", result.stderr)
        bad = self.dir / "bad.kld"
        bad.write_bytes(b"garbage")
        result = self.run_fake("perplexity", "-m", self.model, "--kl-divergence-base", str(bad), "--kl-divergence")
        self.assertEqual(result.returncode, 1)

    def test_cli_mode_answers(self):
        result = self.run_fake("cli", "-m", self.model, "-p", "What is 12*3?", "-n", "8", "-no-cnv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "What is 12*3?36\n")
        self.assertIn("tokens per second", result.stderr)

    def test_empty_file_is_accepted_as_a_model(self):
        empty = self.dir / "empty.gguf"
        empty.touch()
        server = self.server(model_path=str(empty)).start()
        self.assertTrue(server.ready())

    def test_answers_are_deterministic(self):
        fake = _load_fake()
        self.assertEqual(fake.answer("", "What is (3 + 4) * 2?"), "14")
        self.assertEqual(fake.answer("", "Compute 7 / 2"), "3.5")
        self.assertEqual(fake.answer("", "What is the capital of France?"), "Paris")
        self.assertEqual(fake.answer("", "Reply with only the word ready."), "ready")
        self.assertEqual(fake.answer("x The magic number is 4312. y", "What was the magic number?"), "The magic number is 4312.")
        self.assertEqual(fake.answer("", "Tell me a story"), fake.answer("", "Tell me a story"))
        self.assertTrue(fake.answer("", "Tell me a story").endswith("."))
        with env(REPLY="<|im_start|>"):
            self.assertEqual(fake.answer("", "anything"), "<|im_start|>")


def _load_fake():
    import importlib.util
    spec = importlib.util.spec_from_file_location("fake_llama_module", FAKE_LLAMA)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
