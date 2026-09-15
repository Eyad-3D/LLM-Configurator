from dataclasses import replace
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from llm_configurator.catalogue import apply_scores, demo_variants, fetch_scores, fetch_variants, refresh
from llm_configurator.runtime import bench
from llm_configurator.storage import Store
from llm_configurator.domain import GIB


class AdapterTests(unittest.TestCase):
    def test_refresh_reports_completed_requests_and_failures(self):
        updates = []
        entries = [{"base_repo": "test/one", "gguf_repo": "test/one"},
                   {"base_repo": "test/two", "gguf_repo": "test/two"}]
        with tempfile.TemporaryDirectory() as directory:
            with patch("llm_configurator.catalogue.definitions", return_value=entries), \
                 patch("llm_configurator.catalogue.fetch_variants", side_effect=[[], ValueError("offline")]), \
                 patch("llm_configurator.catalogue.fetch_scores", side_effect=ValueError("key unavailable")):
                refresh(Store(directory), progress=updates.append)
        self.assertEqual(updates[0]["models_done"], 0)
        self.assertEqual(updates[-1], {"models_done": 2, "models_total": 2,
                                      "models_failed": 1, "scores": "failed"})
        self.assertEqual(len(updates), 4)

    def test_model_and_score_requests_overlap(self):
        scores_started = threading.Event()
        models_started = threading.Event()
        def scores():
            scores_started.set()
            if not models_started.wait(2):
                raise AssertionError("Models waited for scores")
            return {"version": "test", "data": [], "fetched_at": "today"}
        def models(entry):
            models_started.set()
            if not scores_started.wait(2):
                raise AssertionError("Scores waited for models")
            return []
        with tempfile.TemporaryDirectory() as directory:
            with patch("llm_configurator.catalogue.fetch_scores", side_effect=scores), \
                 patch("llm_configurator.catalogue.fetch_variants", side_effect=models):
                result = refresh(Store(directory))
                self.assertEqual(result["warnings"], [])

    def test_scores_only_refresh_preserves_models_without_hf_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            variant = demo_variants()[0]
            store.put("variants", [variant.to_dict()])
            entries = [{"base_repo": variant.base_repo, "gguf_repo": variant.repo, "aa_slug": "rated"}]
            scores = {"version": "test", "fetched_at": "today", "data": [
                {"slug": "rated", "evaluations": {"artificial_analysis_intelligence_index": 42}}]}
            with patch("llm_configurator.catalogue.definitions", return_value=entries), \
                 patch("llm_configurator.catalogue.fetch_variants") as hf, \
                 patch("llm_configurator.catalogue.fetch_scores", return_value=scores):
                refresh(store, include_models=False)
                hf.assert_not_called()
            cached = store.get("variants")[0]
            self.assertEqual(cached["id"], variant.id)
            self.assertEqual(cached["size_bytes"], variant.size_bytes)
            self.assertEqual(cached["scores"]["general"], 42)

    def test_model_only_refresh_does_not_fetch_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory)
            with patch("llm_configurator.catalogue.fetch_variants", return_value=[]), \
                 patch("llm_configurator.catalogue.fetch_scores") as scores:
                refresh(store, include_scores=False)
                scores.assert_not_called()

    def test_hf_uses_real_file_size_and_skips_shards(self):
        responses = [{"sha": "base123"}, {"model_type": "qwen3", "num_hidden_layers": 32, "num_key_value_heads": 8,
                     "head_dim": 128, "max_position_embeddings": 32768}, {"sha": "gguf123", "siblings": [
                         {"rfilename": "Model-Q4_K_M.gguf", "size": 12345, "lfs": {"sha256": "hash"}},
                         {"rfilename": "Model-Q8_0-00001-of-00002.gguf", "size": 12345}]}]
        with patch("llm_configurator.catalogue.get_json", side_effect=responses):
            variants = fetch_variants({"base_repo": "test/base", "gguf_repo": "test/gguf"})
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0].size_bytes, 12345)
        self.assertEqual(variants[0].base_revision, "base123")
        self.assertEqual(variants[0].revision, "gguf123")

    def test_unsupported_architecture_is_rejected(self):
        with patch("llm_configurator.catalogue.get_json", side_effect=[{"sha": "x"}, {"model_type": "qwen3_moe"}, {}]):
            with self.assertRaises(ValueError):
                fetch_variants({"base_repo": "test/base", "gguf_repo": "test/gguf"})

    def test_aa_pagination_and_null_scores(self):
        pages = [{"intelligence_index_version": 4.3, "pagination": {"has_more": True}, "data": [{"slug": "a"}]},
                 {"intelligence_index_version": 4.3, "pagination": {"has_more": False}, "data": [{"slug": "b"}]}]
        with patch.dict("os.environ", {"AA_API_KEY": "test"}), patch("llm_configurator.catalogue.get_json", side_effect=pages):
            cache = fetch_scores()
        self.assertEqual(len(cache["data"]), 2)
        model = demo_variants()[0]
        apply_scores(model, {"aa_slug": "a"}, {"version": "4.3", "data": [{"slug": "a", "evaluations": {
            "artificial_analysis_intelligence_index": None, "artificial_analysis_coding_index": 0}}]})
        self.assertNotIn("general", model.scores)
        self.assertEqual(model.scores["coding"], 0)

    def test_no_fuzzy_score_matching(self):
        model = demo_variants()[0]
        apply_scores(model, {"aa_slug": None}, {"version": "4.3", "data": [{"slug": "looks-similar", "name": model.name, "evaluations": {}}]})
        self.assertEqual(model.scores, {})

    def test_bench_rejects_wrong_local_artifact_before_execution(self):
        model = replace(demo_variants()[0], demo=False, sha256="correct")
        with patch("llm_configurator.runtime.digest", return_value="wrong"), patch("llm_configurator.runtime.subprocess.run") as run:
            with self.assertRaisesRegex(ValueError, "SHA256"):
                bench(model, "file.gguf", "llama-bench", 8192, 0)
            run.assert_not_called()

    def test_bench_full_gpu_includes_output_layer_and_keeps_identity(self):
        model = replace(demo_variants()[0], demo=False, sha256="correct")
        hw = {"fingerprint": "machine", "ram_available": 32 * GIB, "cores": 8, "threads": 16,
              "gpus": [{"index": 0, "uuid": "GPU-abc", "available": 24 * GIB}]}
        row = {"n_prompt": 0, "n_gen": 128, "n_depth": 8192-128, "avg_ts": 30,
               "n_gpu_layers": model.layers + 1, "n_threads": 8, "type_k": "f16", "type_v": "f16", "build_commit": "test"}
        with patch("llm_configurator.runtime.digest", return_value="correct"), \
                patch("llm_configurator.runtime.shutil.which", return_value="llama-bench"), \
                patch("llm_configurator.runtime.scan", return_value=hw), \
                patch("llm_configurator.runtime.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=json.dumps([row]))) as run:
            result = bench(model, "file.gguf", "llama-bench", 8192, model.layers)
        self.assertEqual(result["tps"], 30)
        self.assertEqual(result["gpu_layers"], model.layers)
        self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-abc")

    def test_bench_does_not_accept_empty_cache_test_as_long_context(self):
        model = replace(demo_variants()[0], demo=False, sha256="correct")
        hw = {"fingerprint": "machine", "ram_available": 32 * GIB, "cores": 8, "threads": 16, "gpus": []}
        with patch("llm_configurator.runtime.digest", return_value="correct"), \
                patch("llm_configurator.runtime.shutil.which", return_value="llama-bench"), \
                patch("llm_configurator.runtime.scan", return_value=hw), \
                patch("llm_configurator.runtime.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout='[{"n_prompt":0,"n_gen":128,"n_depth":0,"avg_ts":100}]')):
            with self.assertRaisesRegex(ValueError, "Unrecognised"):
                bench(model, "file.gguf", "llama-bench", 8192, 0)

    def test_cache_roundtrip_and_missing_value(self):
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            self.assertIsNone(store.get("missing"))
            store.put("variants", [{"name": "model"}])
            self.assertEqual(Store(folder).get("variants"), [{"name": "model"}])


if __name__ == "__main__":
    unittest.main()
