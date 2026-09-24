import hashlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from llm_configurator import catalogue
from llm_configurator.catalogue import (add_entry, definitions, fetch_variants, gguf_artifacts, quant_of, refresh,
                                        refresh_entry, remove_entry)
from llm_configurator.domain import ARCHITECTURES, Cancelled
from llm_configurator.storage import Store

GB = 10**9


def digest(label):
    return hashlib.sha256(label.encode()).hexdigest()


def sibling(name, size, sha=None):
    return {"rfilename": name, "size": size, "lfs": {"sha256": digest(sha or name), "size": size}}


class FakeHub:
    """Answers get_json by URL so tests do not depend on request order."""

    def __init__(self, models, configs):
        self.models, self.configs, self.urls = models, configs, []

    def __call__(self, url, headers=None):
        self.urls.append(url)
        if "/resolve/" in url:
            repo = url.split("huggingface.co/")[1].split("/resolve/")[0]
            if repo not in self.configs:
                raise ValueError("Metadata request returned HTTP 401; check access, API key or rate limit")
            return self.configs[repo]
        repo = url.split("/api/models/")[1].split("?")[0]
        if repo not in self.models:
            raise ValueError("Metadata request returned HTTP 404; check access, API key or rate limit")
        return self.models[repo]


def fetch(entry, models, configs, env=None):
    hub = FakeHub(models, configs)
    with patch("llm_configurator.catalogue.get_json", side_effect=hub), \
         patch.dict("os.environ", {"HF_TOKEN": "", **(env or {})}):
        return fetch_variants(entry), hub


QWEN3_8B = {"model_type": "qwen3", "hidden_size": 4096, "intermediate_size": 12288, "num_hidden_layers": 36,
            "num_attention_heads": 32, "num_key_value_heads": 8, "head_dim": 128, "max_position_embeddings": 40960,
            "vocab_size": 151936, "tie_word_embeddings": False, "sliding_window": None, "use_sliding_window": False,
            "max_window_layers": 36}
QWEN3_30B_A3B = {"model_type": "qwen3_moe", "hidden_size": 2048, "intermediate_size": 6144, "moe_intermediate_size": 768,
                 "num_hidden_layers": 48, "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
                 "num_experts": 128, "num_experts_per_tok": 8, "decoder_sparse_step": 1, "mlp_only_layers": [],
                 "max_position_embeddings": 40960, "vocab_size": 151936, "tie_word_embeddings": False,
                 "use_sliding_window": False, "sliding_window": None, "max_window_layers": 48}
GPT_OSS_20B = {"model_type": "gpt_oss", "hidden_size": 2880, "intermediate_size": 2880, "num_hidden_layers": 24,
               "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64, "num_local_experts": 32,
               "num_experts_per_tok": 4, "experts_per_token": 4, "sliding_window": 128,
               "layer_types": ["sliding_attention", "full_attention"] * 12, "max_position_embeddings": 131072,
               "vocab_size": 201088, "tie_word_embeddings": False}
MIXTRAL = {"model_type": "mixtral", "hidden_size": 4096, "intermediate_size": 14336, "num_hidden_layers": 32,
           "num_attention_heads": 32, "num_key_value_heads": 8, "num_local_experts": 8, "num_experts_per_tok": 2,
           "max_position_embeddings": 32768, "vocab_size": 32000, "tie_word_embeddings": False, "sliding_window": None}
# Gemma 3 multimodal config: the language model is nested and leaves many fields to transformers defaults.
GEMMA3_4B = {"model_type": "gemma3", "architectures": ["Gemma3ForConditionalGeneration"],
             "text_config": {"model_type": "gemma3_text", "hidden_size": 2560, "intermediate_size": 10240,
                             "num_hidden_layers": 34, "sliding_window": 1024},
             "vision_config": {"model_type": "siglip_vision_model", "num_hidden_layers": 27}}


class FetchVariantsTests(unittest.TestCase):
    def test_dense_quant_shortlist_and_metadata(self):
        files = [sibling("Qwen3-8B-Q4_K_M.gguf", 5 * GB), sibling("Qwen3-8B-UD-Q4_K_M.gguf", 5 * GB),
                 sibling("Qwen3-8B-Q6_K.gguf", 6 * GB), sibling("Qwen3-8B-Q6_K_L.gguf", 7 * GB),
                 sibling("Qwen3-8B-Q2_K.gguf", 3 * GB), sibling("Qwen3-8B-Q4_0_4_4.gguf", 4 * GB),
                 sibling("Qwen3-8B-F16.gguf", 16 * GB), sibling("mmproj-F16.gguf", GB), sibling("README.md", 10)]
        models = {"test/base": {"sha": "b1", "safetensors": {"total": 8_190_735_360}, "cardData": {"license": "apache-2.0"}},
                  "test/gguf": {"sha": "g1", "siblings": files}}
        variants, _ = fetch({"base_repo": "test/base", "gguf_repo": "test/gguf", "family": "qwen3"}, models,
                            {"test/base": QWEN3_8B})
        self.assertEqual([v.filename for v in variants], ["Qwen3-8B-Q4_K_M.gguf", "Qwen3-8B-Q6_K.gguf"])
        model = variants[0]
        self.assertEqual((model.parameters, model.active_parameters), (8_190_735_360, 8_190_735_360))
        self.assertEqual((model.experts, model.expert_fraction, model.sliding_layers), (0, 0.0, 0))
        self.assertEqual((model.family, model.license, model.source, model.files), ("qwen3", "apache-2.0", "catalogue", []))
        self.assertEqual(model.sha256, digest("Qwen3-8B-Q4_K_M.gguf"))

    def test_qwen3_moe_expert_fields(self):
        models = {"Qwen/Qwen3-30B-A3B": {"sha": "b", "safetensors": {"total": 30_532_122_624}},
                  "Qwen/Qwen3-30B-A3B-GGUF": {"sha": "g", "siblings": [sibling("Qwen3-30B-A3B-Q4_K_M.gguf", 18_556_689_568)]}}
        [model], _ = fetch({"base_repo": "Qwen/Qwen3-30B-A3B", "gguf_repo": "Qwen/Qwen3-30B-A3B-GGUF"}, models,
                           {"Qwen/Qwen3-30B-A3B": QWEN3_30B_A3B})
        self.assertTrue(model.moe)
        self.assertEqual((model.experts, model.active_experts, model.architecture), (128, 8, "qwen3_moe"))
        self.assertAlmostEqual(model.expert_fraction, 0.9496, places=3)  # 48*128*3*2048*768 / 30.53B
        self.assertAlmostEqual(model.active_parameters / 1e9, 3.35, delta=0.1)  # advertised "A3B" = 3.3B active

    def test_decoder_sparse_step_and_mlp_only_layers_limit_expert_layers(self):
        config = dict(QWEN3_30B_A3B, decoder_sparse_step=2, mlp_only_layers=[1])
        self.assertEqual(len(catalogue.moe_layers(config, 8)), 3)  # layers 3, 5, 7
        self.assertEqual(catalogue.moe_shape(config, 8)["routed"], 3 * 128 * 3 * 2048 * 768)

    def test_gpt_oss_mxfp4_sharded_and_sliding(self):
        shards = [sibling(f"gpt-oss-120b-mxfp4-0000{i}-of-00003.gguf", size) for i, size in
                  [(3, 20 * GB), (1, 13 * 10**6), (2, 40 * GB)]]
        config = dict(GPT_OSS_20B, num_hidden_layers=36, num_local_experts=128,
                      layer_types=["sliding_attention", "full_attention"] * 18)
        # Packed MXFP4 checkpoints under-report parameters; the config shapes win.
        models = {"openai/gpt-oss-120b": {"sha": "b", "safetensors": {"total": 60 * 10**9}},
                  "ggml-org/gpt-oss-120b-GGUF": {"sha": "g", "siblings": shards}}
        [model], _ = fetch({"base_repo": "openai/gpt-oss-120b", "gguf_repo": "ggml-org/gpt-oss-120b-GGUF"}, models,
                           {"openai/gpt-oss-120b": config})
        self.assertEqual(model.quant, "MXFP4")
        self.assertEqual(model.filename, "gpt-oss-120b-mxfp4-00001-of-00003.gguf")
        self.assertEqual([f["filename"][-19:] for f in model.files],
                         ["00001-of-00003.gguf", "00002-of-00003.gguf", "00003-of-00003.gguf"])
        self.assertEqual(model.size_bytes, 60 * GB + 13 * 10**6)
        self.assertEqual(model.sha256, model.files[0]["sha256"])
        self.assertEqual(model.all_files(), model.files)
        self.assertEqual((model.experts, model.active_experts), (128, 4))
        self.assertAlmostEqual(model.parameters / 1e9, 116.8, delta=1.5)
        self.assertAlmostEqual(model.active_parameters / 1e9, 5.9, delta=0.6)
        self.assertEqual((model.sliding_window, model.sliding_layers), (128, 18))
        self.assertEqual((model.kv_heads, model.head_dim), (8, 64))

    def test_gpt_oss_20b_single_file(self):
        models = {"openai/gpt-oss-20b": {"sha": "b"},
                  "ggml-org/gpt-oss-20b-GGUF": {"sha": "g", "siblings": [sibling("gpt-oss-20b-mxfp4.gguf", 12_109_566_560)]}}
        [model], _ = fetch({"base_repo": "openai/gpt-oss-20b", "gguf_repo": "ggml-org/gpt-oss-20b-GGUF"}, models,
                           {"openai/gpt-oss-20b": GPT_OSS_20B})
        self.assertAlmostEqual(model.parameters / 1e9, 20.9, delta=0.3)
        # Expert share of the file's BYTES (contract §2.1): 19.11e9 expert weights x 17/32 bytes / 12.11e9 bytes.
        # By parameters it would be 0.914, which would leave ~0.9 GB of non-expert weights uncounted on the GPU.
        self.assertAlmostEqual(model.expert_fraction, 19_110_297_600 * 17 / 32 / 12_109_566_560, places=3)
        self.assertAlmostEqual(model.expert_fraction, 0.838, places=2)
        self.assertEqual(model.sliding_layers, 12)

    def test_mixtral_like(self):
        models = {"test/mixtral": {"sha": "b", "safetensors": {"total": 46_702_792_704}},
                  "test/mixtral-gguf": {"sha": "g", "siblings": [sibling("mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf", 26 * GB)]}}
        [model], _ = fetch({"base_repo": "test/mixtral", "gguf_repo": "test/mixtral-gguf"}, models,
                           {"test/mixtral": MIXTRAL})
        self.assertEqual((model.experts, model.active_experts, model.head_dim), (8, 2, 128))
        self.assertAlmostEqual(model.active_parameters / 1e9, 12.9, delta=0.2)
        self.assertEqual(model.sliding_layers, 0)

    def test_incomplete_moe_config_is_rejected(self):
        config = dict(MIXTRAL)
        del config["num_experts_per_tok"]
        models = {"t/b": {"sha": "b"}, "t/g": {"sha": "g", "siblings": [sibling("m-Q4_K_M.gguf", GB)]}}
        with self.assertRaisesRegex(ValueError, "mixture-of-experts"):
            fetch({"base_repo": "t/b", "gguf_repo": "t/g"}, models, {"t/b": config})

    def test_subfolder_shards_incomplete_sets_and_single_preference(self):
        files = [sibling("Q4_K_M/Big-Q4_K_M-00002-of-00003.gguf", 50 * GB, "s2"),
                 sibling("Q4_K_M/Big-Q4_K_M-00001-of-00003.gguf", 49 * GB, "s1"),
                 sibling("Q4_K_M/Big-Q4_K_M-00003-of-00003.gguf", 43 * GB, "s3"),
                 sibling("Q8_0/Big-Q8_0-00001-of-00004.gguf", 50 * GB),  # shards 2-4 missing
                 sibling("Q8_0/Big-Q8_0-00002-of-00004.gguf", 50 * GB),
                 {"rfilename": "Q6_K/Big-Q6_K-00001-of-00002.gguf", "size": 50 * GB},
                 {"rfilename": "Q6_K/Big-Q6_K-00002-of-00002.gguf"},  # no size: cannot verify
                 sibling("big-q5_k_m-00001-of-00002.gguf", 40 * GB), sibling("big-q5_k_m-00002-of-00002.gguf", 40 * GB),
                 sibling("big-q5_k_m.gguf", 80 * GB), sibling("UD-Q4_K_XL/Big-UD-Q4_K_XL-00001-of-00001.gguf", GB)]
        artifacts = {a["quant"]: a for a in catalogue.pick_artifacts(gguf_artifacts(files), set(catalogue.QUANTS))}
        self.assertEqual(sorted(artifacts), ["Q4_K_M", "Q5_K_M"])
        big = artifacts["Q4_K_M"]
        self.assertEqual(big["filename"], "Q4_K_M/Big-Q4_K_M-00001-of-00003.gguf")
        self.assertEqual([f["sha256"] for f in big["files"]], [digest("s1"), digest("s2"), digest("s3")])
        self.assertEqual(big["size_bytes"], 142 * GB)
        self.assertEqual((artifacts["Q5_K_M"]["filename"], artifacts["Q5_K_M"]["files"]), ("big-q5_k_m.gguf", []))

    def test_files_without_a_checksum_are_not_offered(self):
        # Downloads refuse a file without sha256, and discover cannot verify one, so it must never become a variant.
        files = [{"rfilename": "m-Q4_K_M.gguf", "size": GB},  # not stored in LFS: no sha256
                 {"rfilename": "m-Q8_0.gguf", "size": GB, "lfs": {"oid": "not-a-sha", "size": GB}},
                 sibling("m-Q6_K-00001-of-00002.gguf", GB),
                 {"rfilename": "m-Q6_K-00002-of-00002.gguf", "size": GB},  # one shard lacks its checksum
                 {"rfilename": "m-Q5_K_M.gguf", "size": GB, "lfs": {"oid": digest("x").upper(), "size": GB}}]
        artifacts = gguf_artifacts(files)
        self.assertEqual([a["quant"] for a in artifacts], ["Q5_K_M"])
        self.assertEqual(artifacts[0]["sha256"], digest("x"))

    def test_quant_names(self):
        cases = {"Phi-4-mini-instruct.Q8_0.gguf": "Q8_0", "qwen2.5-coder-7b-instruct-q4_k_m.gguf": "Q4_K_M",
                 "granite-3.3-8b-instruct-f16.gguf": "F16", "model-BF16.gguf": "BF16", "gpt-oss-20b-MXFP4.gguf": "MXFP4",
                 "m-IQ4_XS.gguf": "IQ4_XS", "m-Q6_K_L.gguf": None, "m-Q4_0_4_8.gguf": None, "m-UD-Q4_K_XL.gguf": None,
                 "Q3_K_M/m-00001-of-00002": "Q3_K_M"}
        for name, quant in cases.items():
            self.assertEqual(quant_of(name), quant, name)

    def test_full_precision_only_for_small_models_and_quants_override(self):
        files = [sibling("m-Q4_K_M.gguf", GB), sibling("m-F16.gguf", 3 * GB), sibling("m-Q2_K.gguf", GB // 2)]
        small = dict(QWEN3_8B, hidden_size=1024, intermediate_size=3072, num_hidden_layers=28, num_attention_heads=16,
                     tie_word_embeddings=True)
        models = {"t/b": {"sha": "b"}, "t/g": {"sha": "g", "siblings": files}}
        variants, _ = fetch({"base_repo": "t/b", "gguf_repo": "t/g"}, models, {"t/b": small})
        self.assertEqual([v.quant for v in variants], ["Q4_K_M", "F16"])
        variants, _ = fetch({"base_repo": "t/b", "gguf_repo": "t/g"}, models, {"t/b": QWEN3_8B})
        self.assertEqual([v.quant for v in variants], ["Q4_K_M"])
        variants, _ = fetch({"base_repo": "t/b", "gguf_repo": "t/g", "quants": ["Q2_K"]}, models, {"t/b": QWEN3_8B})
        self.assertEqual([v.quant for v in variants], ["Q2_K"])
        # F16 and BF16 are the same weights; only one 16-bit file is offered.
        both = {"t/b": {"sha": "b"}, "t/g": {"sha": "g", "siblings": files + [sibling("m-BF16.gguf", 3 * GB)]}}
        variants, _ = fetch({"base_repo": "t/b", "gguf_repo": "t/g"}, both, {"t/b": small})
        self.assertEqual([v.quant for v in variants], ["Q4_K_M", "F16"])

    def test_config_that_is_not_an_object_is_a_plain_error(self):
        models = {"t/b": {"sha": "b"}, "t/g": {"sha": "g", "siblings": [sibling("m-Q4_K_M.gguf", GB)]}}
        with self.assertRaisesRegex(ValueError, "config.json"):
            fetch({"base_repo": "t/b", "gguf_repo": "t/g"}, models, {"t/b": [1, 2]})

    def test_gemma3_text_config_defaults_and_sliding_pattern(self):
        models = {"google/gemma-3-4b-it": {"sha": "b", "gated": "manual"}, "mirror/gemma-3-4b-it": {"sha": "m"},
                  "gguf/gemma-3-4b-it-GGUF": {"sha": "g", "siblings": [sibling("gemma-3-4b-it-Q4_K_M.gguf", 2_489_757_856),
                                                                       sibling("mmproj-F16.gguf", 851_251_104)]}}
        entry = {"base_repo": "google/gemma-3-4b-it", "gguf_repo": "gguf/gemma-3-4b-it-GGUF",
                 "config_repo": "mirror/gemma-3-4b-it"}
        [model], hub = fetch(entry, models, {"mirror/gemma-3-4b-it": GEMMA3_4B})
        self.assertEqual(model.architecture, "gemma3")
        self.assertEqual((model.layers, model.kv_heads, model.head_dim, model.max_context), (34, 4, 256, 131072))
        self.assertEqual((model.sliding_window, model.sliding_layers), (1024, 29))  # every 6th layer is global
        self.assertEqual(model.base_revision, "b")
        self.assertFalse(any("google/gemma-3-4b-it/resolve" in url for url in hub.urls), "gated base skipped without token")

    def test_gemma3_layer_types_win_over_pattern(self):
        config = {"model_type": "gemma3_text", "num_hidden_layers": 26, "max_position_embeddings": 32768,
                  "sliding_window": 512, "layer_types": ["sliding_attention"] * 20 + ["full_attention"] * 6}
        flat, architecture = catalogue.flatten_config(config)
        self.assertEqual(architecture, "gemma3_text")
        self.assertEqual(catalogue.sliding(flat, 26, 32768), (512, 20))

    def test_qwen2_max_window_layers_quirk(self):
        base = {"model_type": "qwen2", "sliding_window": 4096, "max_window_layers": 28, "use_sliding_window": False}
        self.assertEqual(catalogue.sliding(base, 28, 131072), (None, 0))
        # When enabled, layers at or above max_window_layers use the window (not below, as the name suggests).
        self.assertEqual(catalogue.sliding(dict(base, use_sliding_window=True, max_window_layers=20), 28, 131072), (4096, 8))
        # A window as long as the context never caps anything (Qwen2.5 ships sliding_window=131072).
        self.assertEqual(catalogue.sliding(dict(base, use_sliding_window=True, sliding_window=131072, max_window_layers=0),
                                           28, 32768), (None, 0))
        self.assertEqual(catalogue.sliding({"model_type": "mistral", "sliding_window": 4096}, 32, 32768), (None, 0))
        self.assertEqual(catalogue.sliding({"model_type": "gemma2", "sliding_window": 4096}, 42, 8192), (4096, 21))

    def test_gated_with_token_tries_base_then_mirror(self):
        models = {"meta/llama": {"sha": "b", "gated": "manual"}, "mirror/llama": {"sha": "m"},
                  "gguf/llama": {"sha": "g", "siblings": [sibling("Llama-Q4_K_M.gguf", 5 * GB)]}}
        config = dict(QWEN3_8B, model_type="llama", head_dim=None)
        entry = {"base_repo": "meta/llama", "gguf_repo": "gguf/llama", "config_repo": "mirror/llama"}
        [model], hub = fetch(entry, models, {"mirror/llama": config}, env={"HF_TOKEN": "secret"})
        self.assertEqual(model.head_dim, 128)
        self.assertTrue(any("meta/llama/resolve/b/config.json" in url for url in hub.urls))
        self.assertTrue(any("mirror/llama/resolve/m/config.json" in url for url in hub.urls))

    def test_gated_without_mirror_explains_next_step(self):
        models = {"meta/llama": {"sha": "b", "gated": "manual"}, "gguf/llama": {"sha": "g", "siblings": []}}
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            fetch({"base_repo": "meta/llama", "gguf_repo": "gguf/llama"}, models, {})

    def test_multimodal_mistral3_uses_text_model(self):
        config = {"model_type": "mistral3", "text_config": dict(QWEN3_8B, model_type="mistral")}
        flat, architecture = catalogue.flatten_config(config)
        self.assertEqual((architecture, flat["num_hidden_layers"]), ("mistral", 36))

    def test_user_entries_are_marked_custom(self):
        models = {"t/b": {"sha": "b"}, "t/g": {"sha": "g", "siblings": [sibling("m-Q4_K_M.gguf", GB)]}}
        [model], _ = fetch({"base_repo": "t/b", "gguf_repo": "t/g", "user": True}, models, {"t/b": QWEN3_8B})
        self.assertEqual(model.source, "custom")


class CatalogueFileTests(unittest.TestCase):
    ALLOWED_TAGS = {"general", "coding", "reasoning", "small", "moe", "large"}

    def setUp(self):
        self.entries = json.loads((Path(catalogue.__file__).with_name("catalogue.json")).read_text(encoding="utf-8"))

    def test_structure(self):
        self.assertTrue(30 <= len(self.entries) <= 50, len(self.entries))
        for key in ["base_repo", "gguf_repo"]:
            values = [e[key] for e in self.entries]
            self.assertEqual(len(values), len(set(values)), f"duplicate {key}")
        for entry in self.entries:
            catalogue.validate_entry(entry)
            self.assertIn("aa_slug", entry)
            self.assertIsInstance(entry.get("family"), str, entry["base_repo"])
            self.assertIsInstance(entry.get("license"), str, entry["base_repo"])
            self.assertTrue(entry.get("tags"), entry["base_repo"])
            self.assertLessEqual(set(entry["tags"]), self.ALLOWED_TAGS, entry["base_repo"])
            self.assertNotIn("user", entry)
            self.assertTrue(set(entry) <= {"base_repo", "gguf_repo", "config_repo", "aa_slug", "family", "tags",
                                           "license", "architecture", "quants", "note"}, entry)
            # Recorded for reviewers; fetch_variants reads the real model_type from config.json.
            self.assertIn(entry.get("architecture"), ARCHITECTURES, entry["base_repo"])

    def test_gated_families_have_ungated_config_source(self):
        for entry in self.entries:
            if entry["base_repo"].split("/")[0] in {"meta-llama", "google"}:
                self.assertTrue(entry.get("config_repo"), entry["base_repo"])

    def test_moe_entries_are_tagged(self):
        from llm_configurator.domain import MOE_ARCHITECTURES
        for entry in self.entries:
            self.assertEqual("moe" in entry["tags"], entry["architecture"] in MOE_ARCHITECTURES, entry["base_repo"])


class UserEntryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.store = Store(self.folder.name)
        self.addCleanup(self.folder.cleanup)
        self.shipped = definitions(self.store)

    def test_add_validates_rejects_duplicates_and_marks_user(self):
        entry = add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")
        self.assertTrue(entry["user"])
        self.assertIn("custom", entry["tags"])
        saved = definitions(self.store)
        self.assertEqual(len(saved), len(self.shipped) + 1)
        self.assertEqual(saved[-1]["base_repo"], "me/Model-7B")
        with self.assertRaisesRegex(ValueError, "already"):
            add_entry(self.store, "me/Model-7B", "other/gguf")
        with self.assertRaisesRegex(ValueError, "already"):
            add_entry(self.store, self.shipped[0]["base_repo"], "other/gguf")
        with self.assertRaisesRegex(ValueError, "already used"):
            add_entry(self.store, "me/Other", "me/Model-7B-GGUF")
        for bad in ["noslash", "a/b/c", "../etc", "a/..", "a/.hidden", "a b/c", ""]:
            with self.assertRaisesRegex(ValueError, "Invalid Hugging Face repository"):
                add_entry(self.store, bad, "me/gguf")
        with self.assertRaisesRegex(ValueError, "config_repo"):
            add_entry(self.store, "me/x", "me/x-gguf", config_repo="bad repo")
        self.assertEqual([p.name for p in Path(self.folder.name).glob("*.tmp")], [])

    def test_remove_user_and_shipped_entries(self):
        add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")
        shipped = self.shipped[0]["base_repo"]
        self.store.put("variants", [{"base_repo": shipped, "repo": "x"}, {"base_repo": "keep/me", "repo": "y"}])
        result = remove_entry(self.store, shipped)
        self.assertEqual((result["removed"], result["user"], result["variants"]), (True, False, 1))
        self.assertEqual(self.store.get("variants"), [{"base_repo": "keep/me", "repo": "y"}])
        self.assertNotIn(shipped, [e["base_repo"] for e in definitions(self.store)])
        self.assertTrue(remove_entry(self.store, "me/Model-7B")["user"])
        with self.assertRaisesRegex(ValueError, "not in the catalogue"):
            remove_entry(self.store, "me/Model-7B")
        # Adding a removed shipped model back brings it back.
        add_entry(self.store, shipped, self.shipped[0]["gguf_repo"])
        self.assertEqual(sum(e["base_repo"] == shipped for e in definitions(self.store)), 1)

    def test_old_user_copy_keeps_slugs_and_gains_new_models(self):
        old = [{"base_repo": "Qwen/Qwen3-8B", "gguf_repo": "Qwen/Qwen3-8B-GGUF", "aa_slug": "qwen3-8b"},
               {"base_repo": "hand/edited", "gguf_repo": "hand/edited-GGUF", "aa_slug": None}]
        (Path(self.folder.name) / "catalogue.json").write_text(json.dumps(old), encoding="utf-8")
        entries = {e["base_repo"]: e for e in definitions(self.store)}
        self.assertEqual(len(entries), len(self.shipped) + 1)
        self.assertEqual(entries["Qwen/Qwen3-8B"]["aa_slug"], "qwen3-8b")
        self.assertTrue(entries["hand/edited"]["user"])

    def test_damaged_user_copy_does_not_break_the_app(self):
        path = Path(self.folder.name) / "catalogue.json"
        path.write_text("{not json", encoding="utf-8")
        self.assertEqual(len(definitions(self.store)), len(self.shipped))
        self.assertFalse(path.exists())
        self.assertEqual(len(list(Path(self.folder.name).glob("catalogue.json.damaged-*"))), 1)  # kept, not deleted
        path.write_text(json.dumps([{"base_repo": "bad name", "gguf_repo": "x/y"},
                                    {"base_repo": "me/ok", "gguf_repo": "me/ok-GGUF"}]), encoding="utf-8")
        names = [e["base_repo"] for e in definitions(self.store)]
        self.assertIn("me/ok", names)
        self.assertNotIn("bad name", names)

    def test_dropped_shipped_model_does_not_return_as_custom(self):
        add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")  # writes the merged list, shipped entries included
        fewer = [e for e in catalogue.packaged_entries() if e["base_repo"] != "Qwen/Qwen3-8B"]
        with patch("llm_configurator.catalogue.packaged_entries", return_value=fewer):
            names = {e["base_repo"]: e for e in definitions(self.store)}
        self.assertNotIn("Qwen/Qwen3-8B", names)
        self.assertTrue(names["me/Model-7B"]["user"])

    def test_names_are_ascii_and_duplicates_ignore_case(self):
        with self.assertRaisesRegex(ValueError, "Invalid Hugging Face repository"):
            add_entry(self.store, "m\u00e9/model", "me/gguf")
        with self.assertRaisesRegex(ValueError, "already"):
            add_entry(self.store, self.shipped[0]["base_repo"].lower(), "other/gguf")

    def test_set_slug_keeps_user_entries(self):
        add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")
        catalogue.set_slug(self.store, "Qwen/Qwen3-8B", "qwen3-8b")
        entries = {e["base_repo"]: e for e in definitions(self.store)}
        self.assertEqual(entries["Qwen/Qwen3-8B"]["aa_slug"], "qwen3-8b")
        self.assertIn("me/Model-7B", entries)
        with self.assertRaisesRegex(ValueError, "not in the configured catalogue"):
            catalogue.set_slug(self.store, "no/such", "x")

    def test_benchmark_mapping_keeps_user_entries(self):
        from llm_configurator.app import map_benchmark
        add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")
        map_benchmark(self.store, "me/Model-7B", None)
        remove_entry(self.store, "Qwen/Qwen3-8B")
        map_benchmark(self.store, "me/Model-7B", None)
        names = [e["base_repo"] for e in definitions(self.store)]
        self.assertIn("me/Model-7B", names)
        self.assertNotIn("Qwen/Qwen3-8B", names)

    def test_concurrent_adds_are_not_lost(self):
        threads = [threading.Thread(target=add_entry, args=(self.store, f"me/m{i}", f"me/m{i}-GGUF")) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(bool(e.get("user")) for e in definitions(self.store)), 8)

    def test_refresh_entry_replaces_only_that_model(self):
        add_entry(self.store, "me/Model-7B", "me/Model-7B-GGUF")
        self.store.put("variants", [{"base_repo": "me/Model-7B", "id": "old"}, {"base_repo": "other", "id": "keep"}])
        models = {"me/Model-7B": {"sha": "b"}, "me/Model-7B-GGUF": {"sha": "g", "siblings": [sibling("m-Q4_K_M.gguf", GB)]}}
        hub = FakeHub(models, {"me/Model-7B": QWEN3_8B})
        with patch("llm_configurator.catalogue.get_json", side_effect=hub):
            self.assertEqual(refresh_entry(self.store, "me/Model-7B")["variants"], 1)
        stored = self.store.get("variants")
        self.assertEqual([v["id"] for v in stored][0], "keep")
        self.assertEqual(stored[1]["source"], "custom")


class RefreshTests(unittest.TestCase):
    def test_cancel_stops_refresh(self):
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as folder:
            with patch("llm_configurator.catalogue.fetch_variants", return_value=[]), \
                 patch("llm_configurator.catalogue.fetch_scores", side_effect=ValueError("no key")):
                with self.assertRaises(Cancelled):
                    refresh(Store(folder), cancel=cancel)

    def test_cancel_does_not_wait_for_slow_requests(self):
        import time
        cancel, release = threading.Event(), threading.Event()
        started = threading.Event()

        def slow(entry, cancel=None):
            started.set()
            release.wait(10)
            return []
        entries = [{"base_repo": f"t/{i}", "gguf_repo": f"t/{i}-g"} for i in range(3)]
        threading.Thread(target=lambda: (started.wait(5), cancel.set()), daemon=True).start()
        began = time.monotonic()
        try:
            with tempfile.TemporaryDirectory() as folder:
                with patch("llm_configurator.catalogue.definitions", return_value=entries), \
                     patch("llm_configurator.catalogue.fetch_variants", side_effect=slow):
                    with self.assertRaises(Cancelled):
                        refresh(Store(folder), include_scores=False, cancel=cancel)
            self.assertLess(time.monotonic() - began, 3)
        finally:
            release.set()

    @staticmethod
    def odd_reply_for_b(entry):
        if entry["base_repo"] == "t/b":
            raise AttributeError("odd reply")
        return []

    def test_unexpected_error_keeps_other_models(self):
        entries = [{"base_repo": "t/a", "gguf_repo": "t/a-g"}, {"base_repo": "t/b", "gguf_repo": "t/b-g", "user": True}]
        cached = {"id": "t/b-g@old:b.gguf", "name": "b", "base_repo": "t/b", "repo": "t/b-g", "revision": "old",
                  "base_revision": "old", "filename": "b.gguf", "sha256": None, "quant": "Q4_K_M", "size_bytes": GB,
                  "layers": 1, "kv_heads": 1, "head_dim": 1, "max_context": 1, "architecture": "qwen3",
                  "source": "catalogue"}  # a v0.3-era cache of a user's own repo
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            store.put("variants", [cached])
            with patch("llm_configurator.catalogue.definitions", return_value=entries), \
                 patch("llm_configurator.catalogue.fetch_variants", side_effect=self.odd_reply_for_b):
                status = refresh(store, include_scores=False)
            self.assertIn("t/b: unexpected reply", status["warnings"][0])
            [kept] = store.get("variants")
            self.assertEqual(kept["source"], "custom")  # privacy rule for sharing relies on this

    def test_one_failure_keeps_other_models_and_cached_copy(self):
        entries = [{"base_repo": "t/a", "gguf_repo": "t/a-g"}, {"base_repo": "t/b", "gguf_repo": "t/b-g"}]
        models = {"t/a": {"sha": "a"}, "t/a-g": {"sha": "g", "siblings": [sibling("a-Q4_K_M.gguf", GB)]}}
        hub = FakeHub(models, {"t/a": QWEN3_8B})
        with tempfile.TemporaryDirectory() as folder:
            store = Store(folder)
            cached = {"id": "t/b-g@old:b.gguf", "name": "b", "base_repo": "t/b", "repo": "t/b-g", "revision": "old",
                      "base_revision": "old", "filename": "b.gguf", "sha256": None, "quant": "Q4_K_M", "size_bytes": GB,
                      "layers": 1, "kv_heads": 1, "head_dim": 1, "max_context": 1, "architecture": "qwen3"}
            store.put("variants", [cached])
            with patch("llm_configurator.catalogue.definitions", return_value=entries), \
                 patch("llm_configurator.catalogue.get_json", side_effect=hub):
                status = refresh(store, include_scores=False)
            self.assertEqual(len(status["warnings"]), 1)
            self.assertEqual(sorted(v["base_repo"] for v in store.get("variants")), ["t/a", "t/b"])


class RedirectHeaderTests(unittest.TestCase):
    """get_json must not carry HF_TOKEN or the rankings key to another host on a redirect (like downloads.py)."""

    def serve(self, handler):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        server = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (BaseHTTPRequestHandler,), {
            "do_GET": handler, "log_message": lambda *a: None}))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def redirect_through(self, target_host):
        """A server that redirects /api to `target_host` (same port) and records the headers /final receives."""
        seen, port = [], []

        def handle(handler):
            if handler.path == "/api":
                handler.send_response(302)
                handler.send_header("Location", f"http://{target_host}:{port[0]}/final")
                handler.send_header("Content-Length", "0")
                handler.end_headers()
                return
            seen.append({k.lower(): v for k, v in handler.headers.items()})
            body = b'{"ok": true}'
            handler.send_response(200)
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        port.append(self.serve(handle))
        # Local servers only: keep the machine's proxy settings out of the way.
        with patch.dict("os.environ", {"no_proxy": "*", "NO_PROXY": "*"}):
            result = catalogue.get_json(f"http://127.0.0.1:{port[0]}/api",
                                        {"Authorization": "Bearer hf_secret", "x-api-key": "aa_secret"})
        self.assertEqual(result, {"ok": True})
        return seen[0]

    def test_cross_host_redirect_drops_credentials(self):
        headers = self.redirect_through("localhost")
        self.assertNotIn("authorization", headers)
        self.assertNotIn("x-api-key", headers)
        self.assertIn("user-agent", headers)

    def test_same_host_redirect_keeps_credentials(self):
        headers = self.redirect_through("127.0.0.1")
        self.assertEqual(headers.get("authorization"), "Bearer hf_secret")
        self.assertEqual(headers.get("x-api-key"), "aa_secret")

    def test_https_to_http_redirect_is_refused(self):
        from urllib.request import Request
        request = Request("https://huggingface.co/api/models/x", headers={"Authorization": "Bearer hf_secret"})
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            catalogue._MetadataRedirect().redirect_request(request, None, 302, "Found", {}, "http://cdn.example/x")


if __name__ == "__main__":
    unittest.main()
