import tempfile
import threading
import time
import unittest

from llm_configurator.catalogue import demo_variants
from llm_configurator.domain import Cancelled, Requirements, Variant, check_cancel
from llm_configurator.jobs import JobManager
from llm_configurator.launch import from_candidate, normalize, runtime_gpu_layers, server_args, server_env
from llm_configurator.storage import Store, models_dir, runtime_dir


class DomainTests(unittest.TestCase):
    def test_old_records_still_load_and_new_fields_default(self):
        record = demo_variants()[0].to_dict()
        for key in ["files", "experts", "active_experts", "expert_fraction", "sliding_window", "sliding_layers",
                    "parameters", "active_parameters", "family", "license", "source"]:
            record.pop(key)
        variant = Variant(**record)
        self.assertFalse(variant.moe)
        self.assertEqual(variant.all_files(), [{"filename": variant.filename, "size_bytes": variant.size_bytes, "sha256": None}])

    def test_shards_must_add_up(self):
        record = demo_variants()[0].to_dict()
        record["files"] = [{"filename": "a-00001-of-00002.gguf", "size_bytes": 1, "sha256": "x"},
                           {"filename": "a-00002-of-00002.gguf", "size_bytes": 1, "sha256": "y"}]
        with self.assertRaises(ValueError):
            Variant(**record)
        record["size_bytes"] = 2
        self.assertEqual(len(Variant(**record).all_files()), 2)

    def test_moe_fields_validated(self):
        record = {**demo_variants()[0].to_dict(), "architecture": "qwen3_moe", "experts": 128, "active_experts": 8,
                  "expert_fraction": 0.9}
        self.assertTrue(Variant(**record).moe)
        with self.assertRaises(ValueError):
            Variant(**{**record, "active_experts": 200})
        with self.assertRaises(ValueError):
            Variant(**{**record, "architecture": "made_up"})

    def test_kv_cache_type(self):
        self.assertEqual(Requirements(kv_cache_type="q8_0").kv_cache_type, "q8_0")
        with self.assertRaises(ValueError):
            Requirements(kv_cache_type="q2")

    def test_check_cancel(self):
        event = threading.Event()
        check_cancel(event)
        check_cancel(None)
        event.set()
        with self.assertRaises(Cancelled):
            check_cancel(event)


class StoreTests(unittest.TestCase):
    def test_append_is_atomic_across_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            threads = [threading.Thread(target=lambda i=i: store.append("items", i)) for i in range(20)]
            [t.start() for t in threads]
            [t.join() for t in threads]
            self.assertEqual(sorted(store.get("items")), list(range(20)))
            store.append("capped", 1, limit=2), store.append("capped", 2, limit=2), store.append("capped", 3, limit=2)
            self.assertEqual(store.get("capped"), [2, 3])

    def test_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            self.assertTrue(models_dir(store).is_dir())
            self.assertTrue(runtime_dir(store).is_dir())
            store.put("settings", {"models_dir": f"{directory}/elsewhere"})
            self.assertTrue(str(models_dir(store)).endswith("elsewhere"))


class JobTests(unittest.TestCase):
    def test_progress_result_and_failure(self):
        jobs = JobManager()
        def work(progress, cancel):
            progress({"stage": "work", "done": 1, "total": 2, "message": "half"})
            return {"ok": True}
        done = jobs.wait(jobs.submit("test", "Test", work)["id"], 5)
        self.assertEqual(done["state"], "done")
        self.assertEqual(done["result"], {"ok": True})
        self.assertEqual(done["progress"]["done"], 1)
        failed = jobs.wait(jobs.submit("test", "Fail", lambda p, c: (_ for _ in ()).throw(ValueError("plain message")))["id"], 5)
        self.assertEqual((failed["state"], failed["error"]), ("failed", "plain message"))

    def test_cancel_and_exclusive_queue(self):
        jobs = JobManager()
        started = threading.Event()
        def slow(progress, cancel):
            started.set()
            while True:
                check_cancel(cancel)
                time.sleep(0.01)
        first = jobs.submit("tune", "Slow", slow, exclusive="compute")
        started.wait(5)
        second = jobs.submit("test", "Queued", lambda p, c: "ran", exclusive="compute")
        time.sleep(0.1)
        self.assertEqual(jobs.get(second["id"])["state"], "queued")
        self.assertEqual(len(jobs.active()), 2)
        jobs.cancel(first["id"])
        self.assertEqual(jobs.wait(first["id"], 5)["state"], "cancelled")
        self.assertEqual(jobs.wait(second["id"], 5)["result"], "ran")
        with self.assertRaises(ValueError):
            jobs.cancel("job-missing")


class LaunchTests(unittest.TestCase):
    def test_server_args_are_explicit_and_local(self):
        args = server_args({"model_path": "/m/a.gguf", "context": 4096, "parallel": 2, "gpu_layers": 32, "total_layers": 32,
                            "threads": 8, "cache_type_k": "q8_0", "cache_type_v": "q8_0", "n_cpu_moe": 4, "port": 9000,
                            "gpu_backend": "cuda"})
        joined = " ".join(args)
        self.assertIn("-c 8192 -np 2 -ngl 33 --host 127.0.0.1", joined)
        self.assertIn("-fa on", joined)
        self.assertIn("-ctk q8_0 -ctv q8_0", joined)
        self.assertIn("--n-cpu-moe 4", joined)
        self.assertNotIn("-dev", args)
        self.assertIn("-dev none", " ".join(server_args({"model_path": "a", "gpu_backend": "cuda"})))
        draft = server_args({"model_path": "a", "draft_model_path": "d.gguf", "draft_max": 8})
        self.assertIn("-md d.gguf --spec-type draft-simple --spec-draft-n-max 8", " ".join(draft))
        self.assertNotIn("--draft-max", draft)

    def test_rejects_unsafe_or_invalid(self):
        for bad in [{"host": "0.0.0.0"}, {"flash_attn": "maybe"}, {"cache_type_v": "q8_0", "flash_attn": "off"},
                    {"batch": 256, "ubatch": 512}, {"surprise": 1}, {"context": 10}, {"alias": "a\nb"}]:
            with self.assertRaises(ValueError, msg=bad):
                normalize(bad)
        with self.assertRaises(ValueError):
            server_args({})

    def test_runtime_layers_and_env(self):
        self.assertEqual(runtime_gpu_layers(32, 32), 33)
        self.assertEqual(runtime_gpu_layers(10, 32), 11)  # 10 blocks + the output layer
        self.assertEqual(runtime_gpu_layers(0, 32), 0)
        self.assertEqual(server_env({}, base={"LLAMA_ARG_N_PARALLEL": "4", "PATH": "/bin"}),
                         {"PATH": "/bin", "LLAMA_ARG_CORS_ORIGINS": "localhost"})
        with self.assertRaises(ValueError):
            normalize({"alias": "a\tb"})
        vulkan = from_candidate({"context": 4096, "gpu_layers": 2, "total_layers": 8, "gpu_index": 0}, "m.gguf",
                                {"gpus": [{"index": 0, "uuid": "GPU-x", "backend": "cuda"}]}, runtime_backend="vulkan")
        self.assertEqual(vulkan["gpu_backend"], "vulkan")
        self.assertNotIn("CUDA_VISIBLE_DEVICES", server_env(vulkan, base={}))
        # The installed build wins over the candidate's own launch fragment too.
        fragment = {"context": 4096, "gpu_layers": 2, "total_layers": 8, "gpu_index": 0, "launch": {"gpu_backend": "cuda"}}
        self.assertEqual(from_candidate(fragment, "m.gguf", {"gpus": [{"index": 0, "uuid": "GPU-x"}]},
                                        runtime_backend="vulkan")["gpu_backend"], "vulkan")
        # CPU-only on a GPU build: -dev none keeps llama.cpp off cards the app is not tracking.
        cpu = from_candidate({"context": 4096, "gpu_layers": 0, "total_layers": 8, "gpu_index": None}, "m.gguf", {"gpus": []},
                             runtime_backend="cuda")
        self.assertIn("-dev none", " ".join(server_args(cpu)))
        self.assertEqual(from_candidate({"context": 4096, "gpu_layers": 0, "total_layers": 8}, "m.gguf", {},
                                        runtime_backend="cpu")["gpu_backend"], None)
        # A placeholder UUID must not hide every CUDA device.
        self.assertNotIn("CUDA_VISIBLE_DEVICES", server_env({"gpu_backend": "cuda", "gpu_uuid": "[N/A]", "gpu_layers": 3}, base={}))
        env = server_env({"gpu_backend": "cuda", "gpu_uuid": "GPU-1", "gpu_layers": 3}, base={})
        self.assertEqual(env, {"CUDA_VISIBLE_DEVICES": "GPU-1", "LLAMA_ARG_CORS_ORIGINS": "localhost"})
        self.assertEqual(server_env({"gpu_layers": 0}, base={}), {"LLAMA_ARG_CORS_ORIGINS": "localhost"})

    def test_from_candidate(self):
        candidate = {"context": 8192, "users": 1, "gpu_layers": 20, "total_layers": 36, "threads": 8, "gpu_index": 0,
                     "kv_cache_type": "q8_0", "n_cpu_moe": 0}
        hardware = {"gpus": [{"index": 0, "uuid": "GPU-x", "backend": "cuda"}]}
        config = from_candidate(candidate, "/m/x.gguf", hardware, batch=1024)
        self.assertEqual((config["gpu_uuid"], config["cache_type_v"], config["flash_attn"], config["batch"]),
                         ("GPU-x", "q8_0", "on", 1024))


if __name__ == "__main__":
    unittest.main()
