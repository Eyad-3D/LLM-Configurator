import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from llm_configurator import discover
from llm_configurator.domain import Cancelled, Variant
from llm_configurator.storage import Store

try:
    from test_gguf import ARR, STR, U16, U32, build_gguf, llama_kvs
except ImportError:  # run as tests.test_discover
    from tests.test_gguf import ARR, STR, U16, U32, build_gguf, llama_kvs

ENV_KEYS = ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME", "OLLAMA_MODELS", "LLM_CONFIG_MODELS"]


def model_bytes(tag, arch="llama", extra=()):
    return build_gguf(llama_kvs(arch, extra=[("general.basename", STR, tag), *extra]), [("blk.0.attn_q.weight", [64, 64], 0)])


def sha(data):
    return hashlib.sha256(data).hexdigest()


def catalogue_variant(vid, filename, data, with_sha=True, files=None):
    return Variant(id=vid, name=vid, base_repo="org/base", repo="org/repo-GGUF", revision="r", base_revision="r",
                   filename=filename, sha256=sha(data) if with_sha and data else None, quant="Q4_K_M",
                   size_bytes=sum(f["size_bytes"] for f in files) if files else len(data), layers=4, kv_heads=2,
                   head_dim=32, max_context=4096, architecture="llama", files=files or []).to_dict()


class DiscoverTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        env = {k: v for k, v in os.environ.items() if k not in ENV_KEYS}
        env["LLM_CONFIG_MODELS"] = str(self.root / "appmodels")
        patches = [mock.patch.dict(os.environ, env, clear=True), mock.patch.object(discover, "_home", return_value=self.home),
                   mock.patch.object(discover, "_system", return_value="Linux")]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.store = Store(self.root / "data")

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def hf_file(self, repo, filename, data, snapshot="abc123"):
        repo_dir = self.home / ".cache/huggingface/hub" / ("models--" + repo.replace("/", "--"))
        blob = self.write(repo_dir / "blobs" / sha(data), data)
        link = repo_dir / "snapshots" / snapshot / filename
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
        return link

    def ollama_model(self, name, tag, data, root=None):
        root = root or self.home / ".ollama/models"
        digest = sha(data)
        self.write(root / "blobs" / f"sha256-{digest}", data)
        manifest = {"schemaVersion": 2, "layers": [
            {"mediaType": "application/vnd.ollama.image.template", "digest": "sha256:" + "0" * 64, "size": 10},
            {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}", "size": len(data)}]}
        self.write(root / "manifests/registry.ollama.ai/library" / name / tag, json.dumps(manifest).encode())
        return digest

    # locations -------------------------------------------------------------------------------------------------

    def test_default_locations_per_system(self):
        paths = {(loc["source"], loc["path"]) for loc in discover.locations()}
        self.assertIn(("hf_cache", str(self.home / ".cache/huggingface/hub")), paths)
        self.assertIn(("ollama", str(self.home / ".ollama/models")), paths)
        self.assertIn(("ollama", "/usr/share/ollama/.ollama/models"), paths)
        self.assertIn(("lmstudio", str(self.home / ".lmstudio/models")), paths)
        self.assertIn(("lmstudio", str(self.home / ".cache/lm-studio/models")), paths)
        self.assertTrue(all(loc["exists"] is False for loc in discover.locations()))
        for system in ["Windows", "Darwin"]:
            with mock.patch.object(discover, "_system", return_value=system):
                sources = [loc["path"] for loc in discover.locations()]
                self.assertNotIn("/usr/share/ollama/.ollama/models", sources)
                self.assertIn(str(self.home / ".ollama/models"), sources)

    def test_windows_deduplicates_case_insensitively(self):
        with mock.patch.object(discover, "_system", return_value="Windows"):
            found = discover.locations(extra_dirs=[str(self.root / "Models"), str(self.root / "models")])
        self.assertEqual(sum(1 for loc in found if loc["source"] == "custom"), 1)

    def test_environment_overrides(self):
        os.environ["HF_HUB_CACHE"] = str(self.root / "hub")
        os.environ["OLLAMA_MODELS"] = str(self.root / "ollama")
        found = [(loc["source"], loc["path"]) for loc in discover.locations(self.store, ["~/extra"])]
        self.assertEqual(found[0], ("hf_cache", str(self.root / "hub")))
        self.assertIn(("hf_cache", str(self.home / ".cache/huggingface/hub")), found)  # older downloads stay reusable
        self.assertIn(("ollama", str(self.root / "ollama")), found)
        self.assertIn(("models_dir", str(self.root / "appmodels")), found)
        self.assertEqual(found[-1][0], "custom")
        del os.environ["HF_HUB_CACHE"]
        os.environ["HF_HOME"] = str(self.root / "hfhome")
        self.assertEqual(discover.locations()[0]["path"], str(self.root / "hfhome/hub"))
        del os.environ["HF_HOME"]
        os.environ["XDG_CACHE_HOME"] = str(self.root / "xdg")
        self.assertEqual(discover.locations()[0]["path"], str(self.root / "xdg/huggingface/hub"))

    def test_lmstudio_custom_folder_and_home_pointer(self):
        self.write(self.home / ".lmstudio/settings.json", json.dumps({"downloadsFolder": "~/MyModels"}).encode())
        paths = [loc["path"] for loc in discover.locations() if loc["source"] == "lmstudio"]
        self.assertEqual(paths[0], str(self.home / "MyModels"))
        self.write(self.home / ".lmstudio-home-pointer", str(self.root / "lmhome").encode())
        self.write(self.root / "lmhome/settings.json", b"not json")
        paths = [loc["path"] for loc in discover.locations() if loc["source"] == "lmstudio"]
        self.assertEqual(paths[0], str(self.root / "lmhome/models"))

    # scan ------------------------------------------------------------------------------------------------------

    def test_scan_finds_and_matches_every_source(self):
        hf_data, lm_data, ol_data, own_data = model_bytes("hf"), model_bytes("lm"), model_bytes("ol"), model_bytes("own")
        self.hf_file("org/repo-GGUF", "Q4_K_M/model-Q4_K_M.gguf", hf_data)
        self.hf_file("org/repo-GGUF", "Q4_K_M/model-Q4_K_M.gguf", hf_data, snapshot="older")  # same blob, two snapshots
        self.write(self.home / ".lmstudio/models/pub/repo/lm-Q4_K_M.gguf", lm_data)
        self.write(self.home / ".lmstudio/models/.hidden/secret.gguf", lm_data)
        digest = self.ollama_model("tiny", "latest", ol_data)
        self.ollama_model("tiny", "q4", ol_data)
        self.write(self.root / "appmodels/own.gguf", own_data)
        self.write(self.root / "appmodels/weird-mamba.gguf", model_bytes("m", arch="mamba"))
        self.write(self.root / "appmodels/broken.gguf", b"GGUF\x03\0\0\0")
        self.write(self.root / "appmodels/partial.gguf.part", own_data)
        self.store.put("variants", [
            catalogue_variant("hf-v", "Q4_K_M/model-Q4_K_M.gguf", hf_data),
            catalogue_variant("lm-v", "lm-Q4_K_M.gguf", lm_data, with_sha=False),
            catalogue_variant("ol-v", "ollama-name.gguf", ol_data),
            {"id": "broken-record"}])
        events = []
        with mock.patch.object(discover.hashlib, "sha256", side_effect=AssertionError("a scan must not hash")):
            records = discover.scan(self.store, progress=events.append)
        by_name = {r["filename"]: r for r in records}
        self.assertEqual(len(records), 6)
        hf = by_name["Q4_K_M/model-Q4_K_M.gguf"]
        self.assertEqual((hf["source"], hf["repo"], hf["sha256"], hf["hash_source"]), ("hf_cache", "org/repo-GGUF", sha(hf_data), "blob_name"))
        self.assertEqual((hf["variant_id"], hf["match"], hf["verified"]), ("hf-v", "sha256", True))
        lm = by_name["lm-Q4_K_M.gguf"]
        self.assertEqual((lm["variant_id"], lm["match"], lm["verified"], lm["sha256"]), ("lm-v", "name_size", False, None))
        self.assertIn("not yet verified", lm["match_note"])
        ol = by_name[f"sha256-{digest}"]
        self.assertEqual((ol["variant_id"], ol["verified"], sorted(ol["names"])), ("ol-v", True, ["tiny:latest", "tiny:q4"]))
        own = by_name["own.gguf"]
        self.assertIsNone(own["variant_id"])
        self.assertEqual(own["local_variant"]["source"], "local")
        self.assertEqual(own["gguf"]["layers"], 4)
        self.assertIn("'mamba' design", by_name["weird-mamba.gguf"]["local_error"])
        self.assertIn("not a readable GGUF", by_name["broken.gguf"]["gguf_error"])
        self.assertEqual(self.store.get("local_files"), records)
        self.assertEqual(events[-1]["message"], "Found 6 model files on this computer.")
        self.assertTrue(all(set(e) >= {"stage", "done", "total", "message"} and "/" not in e["message"] for e in events))
        self.assertEqual([v.id for v in discover.local_variants(self.store)], [own["local_variant_id"]])

    def test_scan_never_follows_links_out_of_the_folder(self):
        outside = self.write(self.root / "outside/secret.gguf", model_bytes("x"))
        lm = self.home / ".lmstudio/models"
        lm.mkdir(parents=True)
        (lm / "escape.gguf").symlink_to(outside)
        (lm / "linked-dir").symlink_to(outside.parent, target_is_directory=True)
        self.write(lm / "real/inside.gguf", model_bytes("in"))
        (lm / "alias.gguf").symlink_to(lm / "real/inside.gguf")
        snap = self.home / ".cache/huggingface/hub/models--a--b/snapshots/s1"
        snap.mkdir(parents=True)
        (snap / "evil.gguf").symlink_to(outside)  # not the snapshot -> own blobs/ pattern
        records = discover.scan(self.store)
        self.assertEqual(len(records), 1)  # the in-folder alias and the file itself count once
        self.assertIn(records[0]["filename"], {"alias.gguf", "inside.gguf"})

    def test_hf_copies_without_symlinks_match_on_name_and_size(self):
        data = model_bytes("win")
        self.write(self.home / ".cache/huggingface/hub/models--org--repo/snapshots/s/m-Q4_K_M.gguf", data)
        self.store.put("variants", [catalogue_variant("v", "m-Q4_K_M.gguf", data)])
        record = discover.scan(self.store)[0]
        self.assertEqual((record["sha256"], record["match"], record["verified"]), (None, "name_size", False))

    def test_same_name_different_fingerprint_is_not_a_match(self):
        data = model_bytes("a")
        self.hf_file("org/repo", "m.gguf", data)
        other = catalogue_variant("v", "m.gguf", data)
        other["sha256"] = "f" * 64
        self.store.put("variants", [other])
        self.assertIsNone(discover.scan(self.store)[0]["variant_id"])

    def test_split_models_are_grouped(self):
        first = build_gguf(llama_kvs(extra=[("split.count", U16, 3)]), [("a", [64], 0)])
        rest = build_gguf([("split.count", U16, 3)], [("b", [64], 0)])
        base = self.root / "appmodels"
        parts = [self.write(base / f"big-00001-of-00003.gguf", first), self.write(base / f"big-00002-of-00003.gguf", rest),
                 self.write(base / f"big-00003-of-00003.gguf", rest)]
        self.write(base / "half-00001-of-00002.gguf", first)
        self.write(base / "orphan-00002-of-00002.gguf", rest)
        files = [{"filename": p.name, "size_bytes": p.stat().st_size, "sha256": sha(p.read_bytes())} for p in parts]
        self.store.put("variants", [catalogue_variant("big", parts[0].name, None, files=files)])
        records = {r["filename"]: r for r in discover.scan(self.store)}
        self.assertEqual(set(records), {"big-00001-of-00003.gguf", "half-00001-of-00002.gguf"})
        big = records["big-00001-of-00003.gguf"]
        self.assertEqual((big["complete"], big["size_bytes"], len(big["files"])), (True, sum(f["size_bytes"] for f in files), 3))
        self.assertEqual((big["variant_id"], big["match"]), ("big", "name_size"))
        self.assertEqual(big["local_variant"], None)
        self.assertEqual((records["half-00001-of-00002.gguf"]["complete"], records["half-00001-of-00002.gguf"]["variant_id"]), (False, None))

    def test_bad_ollama_manifests_are_ignored(self):
        root = self.home / ".ollama/models"
        data = model_bytes("o")
        self.write(root / "blobs" / f"sha256-{sha(data)}", data)
        self.write(root / "manifests/r/l/bad/json", b"{not json")
        self.write(root / "manifests/r/l/bad/traversal", json.dumps({"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:../../../../etc/passwd"}]}).encode())
        self.write(root / "manifests/r/l/bad/huge", b" " * (discover.MAX_SMALL_FILE + 1))
        self.write(root / "manifests/r/l/bad/list", b"[1, 2]")
        self.write(root / "manifests/r/l/bad/missingblob", json.dumps({"layers": [
            {"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:" + "a" * 64}]}).encode())
        self.assertEqual(discover.scan(self.store), [])

    def test_scan_cancel_and_depth_limit(self):
        self.write(self.root / "appmodels/a.gguf", model_bytes("a"))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(Cancelled):
            discover.scan(self.store, cancel=cancel)
        deep = self.root / "custom" / "/".join(f"d{i}" for i in range(discover.MAX_DEPTH + 1)) / "deep.gguf"
        self.write(deep, model_bytes("d"))
        self.write(self.root / "custom/shallow.gguf", model_bytes("s"))
        names = {r["filename"] for r in discover.scan(self.store, extra_dirs=[self.root / "custom"])}
        self.assertEqual(names, {"a.gguf", "shallow.gguf"})
        self.assertEqual({r["source"] for r in self.store.get("local_files")}, {"models_dir", "custom"})

    def test_rescan_reuses_earlier_hashes(self):
        data = model_bytes("r")
        path = self.write(self.home / ".lmstudio/models/p/r/m.gguf", data)
        self.store.put("variants", [catalogue_variant("v", "m.gguf", data)])
        self.assertEqual(discover.scan(self.store)[0]["match"], "name_size")
        discover.hash_cached(self.store, path)
        record = discover.scan(self.store)[0]
        self.assertEqual((record["match"], record["verified"], record["hash_source"]), ("sha256", True, "hashed"))

    # hashing and lookup ----------------------------------------------------------------------------------------

    def test_hash_cached(self):
        data = os.urandom(3000)
        path = self.write(self.root / "f.bin", data)
        events = []
        self.assertEqual(discover.hash_cached(self.store, path, progress=events.append), sha(data))
        self.assertEqual((events[-1]["stage"], events[-1]["done"], events[-1]["total"]), ("verifying", 3000, 3000))
        expected = sha(data)
        with mock.patch.object(discover.hashlib, "sha256", side_effect=AssertionError("should be cached")):
            self.assertEqual(discover.hash_cached(self.store, path), expected)
        os.utime(path, (1, 1))
        self.assertEqual(discover.hash_cached(self.store, path), sha(data))
        self.assertEqual(self.store.get("hash_cache")[str(path.resolve())]["mtime"], 1)
        cancel = threading.Event()
        cancel.set()
        os.utime(path, (2, 2))
        with self.assertRaises(Cancelled):
            discover.hash_cached(self.store, path, cancel=cancel)
        with self.assertRaisesRegex(ValueError, "changed while it was being checked"):
            discover.hash_cached(self.store, path, progress=lambda e: path.write_bytes(data + b"x"))
        with self.assertRaisesRegex(ValueError, "Could not read"):
            discover.hash_cached(self.store, self.root / "missing.bin")

    def test_hash_cache_is_bounded(self):
        with mock.patch.object(discover, "HASH_CACHE_LIMIT", 3):
            for i in range(5):
                discover.hash_cached(self.store, self.write(self.root / f"{i}.bin", bytes([i])))
        self.assertEqual(len(self.store.get("hash_cache")), 3)

    def test_find_for_variant_in_models_dir_verifies_shards(self):
        base = self.root / "appmodels"
        datas = [b"part-one", b"part-two!"]
        paths = [self.write(base / f"m-0000{i + 1}-of-00002.gguf", d) for i, d in enumerate(datas)]
        files = [{"filename": p.name, "size_bytes": len(d), "sha256": sha(d)} for p, d in zip(paths, datas)]
        variant = Variant(**catalogue_variant("v", paths[0].name, None, files=files))
        self.assertEqual(discover.find_for_variant(self.store, variant), paths[0])
        self.assertEqual(len(self.store.get("hash_cache")), 2)
        self.assertEqual(discover.find_for_variant(self.store, variant.to_dict(), verify=False), paths[0])
        paths[1].write_bytes(b"part-TWO!")  # same size, different content
        self.assertIsNone(discover.find_for_variant(self.store, variant))
        self.assertEqual(discover.find_for_variant(self.store, variant, verify=False), paths[0])
        paths[1].unlink()
        self.assertIsNone(discover.find_for_variant(self.store, variant, verify=False))

    def test_find_for_variant_uses_blob_names_without_rehashing(self):
        data = model_bytes("hf")
        link = self.hf_file("org/repo-GGUF", "model.gguf", data)
        variant = catalogue_variant("hf-v", "model.gguf", data)
        self.store.put("variants", [variant])
        discover.scan(self.store)
        with mock.patch.object(discover, "hash_cached", side_effect=AssertionError("must not rehash")):
            self.assertEqual(discover.find_for_variant(self.store, variant), link)

    def test_find_for_variant_hashes_likely_matches_and_marks_them(self):
        data = model_bytes("lm")
        path = self.write(self.home / ".lmstudio/models/p/r/lm.gguf", data)
        variant = catalogue_variant("v", "lm.gguf", data)
        self.store.put("variants", [variant])
        self.assertFalse(discover.scan(self.store)[0]["verified"])
        self.assertEqual(discover.find_for_variant(self.store, variant), path)
        record = self.store.get("local_files")[0]
        self.assertEqual((record["verified"], record["match"]), (True, "sha256"))
        bad = dict(variant, sha256="0" * 64, id="other")
        self.assertIsNone(discover.find_for_variant(self.store, bad))

    def test_find_for_variant_never_leaves_models_dir(self):
        self.write(self.root / "secret.gguf", b"data")
        variant = catalogue_variant("v", "../secret.gguf", b"data", with_sha=False)
        self.assertIsNone(discover.find_for_variant(self.store, variant, verify=False))

    def test_local_variant_round_trip_and_add_file(self):
        path = self.write(self.root / "elsewhere/my-Q4_K_M.gguf", model_bytes("mine"))
        record = discover.add_file(self.store, path)
        self.assertTrue(record["added"])
        variant = discover.local_variants(self.store)[0]
        self.assertEqual(discover.find_for_variant(self.store, variant), path)
        discover.scan(self.store)  # the added file is outside every scanned folder but is kept
        self.assertEqual([r["path"] for r in self.store.get("local_files")], [str(path)])
        with self.assertRaisesRegex(ValueError, "not found"):
            discover.add_file(self.store, self.root / "nope.gguf")
        with self.assertRaisesRegex(ValueError, "not a readable GGUF"):
            discover.add_file(self.store, self.write(self.root / "bad.gguf", b"nope"))
        with self.assertRaisesRegex(ValueError, "'mamba' design"):
            discover.add_file(self.store, self.write(self.root / "m.gguf", model_bytes("m", arch="mamba")))
        path.unlink()
        self.assertEqual(discover.scan(self.store), [])


if __name__ == "__main__":
    unittest.main()
