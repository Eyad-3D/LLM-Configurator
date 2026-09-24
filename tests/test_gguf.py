import struct
import tempfile
import time
import unittest
from pathlib import Path

from llm_configurator import gguf

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FMT = {U8: "B", I8: "b", U16: "H", I16: "h", U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q", I64: "q", F64: "d"}


def _string(text):
    data = text.encode() if isinstance(text, str) else text
    return struct.pack("<Q", len(data)) + data


def _value(kind, value):
    if kind in _FMT:
        return struct.pack("<" + _FMT[kind], value)
    if kind == STR:
        return _string(value)
    item_kind, items = value
    body = b"".join(_value(item_kind, item) for item in items)
    return struct.pack("<IQ", item_kind, len(items)) + body


def build_gguf(kvs, tensors=(), version=3, magic=b"GGUF"):
    """Tiny valid GGUF: kvs = [(key, type, value)], tensors = [(name, dims, ggml_type)]; arrays are (item_type, items)."""
    out = bytearray(magic + struct.pack("<I", version) + struct.pack("<QQ", len(tensors), len(kvs)))
    for key, kind, value in kvs:
        out += _string(key) + struct.pack("<I", kind) + _value(kind, value)
    offset = 0
    for name, dims, kind in tensors:
        out += _string(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        out += struct.pack("<IQ", kind, offset)
        count = 1
        for d in dims:
            count *= d
        block, width = gguf.TENSOR_TYPES[kind]
        size = -(-count // block) * width
        offset += size + (-size % 32)
    out += b"\0" * (-len(out) % 32) + b"\0" * offset  # aligned (zero) tensor data
    return bytes(out)


def llama_kvs(arch="llama", layers=4, heads=8, kv_heads=2, embed=256, context=4096, file_type=15, extra=()):
    return [("general.architecture", STR, arch), ("general.name", STR, "Tiny Test"), ("general.file_type", U32, file_type),
            (f"{arch}.block_count", U32, layers), (f"{arch}.context_length", U32, context),
            (f"{arch}.embedding_length", U32, embed), (f"{arch}.attention.head_count", U32, heads),
            (f"{arch}.attention.head_count_kv", U32, kv_heads),
            ("tokenizer.ggml.tokens", ARR, (STR, [f"tok{i}" for i in range(5000)])),
            ("tokenizer.ggml.scores", ARR, (F32, [0.0] * 5000)), *extra]


class GGUFTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def test_reads_dense_summary_and_skips_token_list(self):
        path = self.write("tiny-Q4_K_M.gguf", build_gguf(llama_kvs()))
        result = gguf.read_metadata(path)
        self.assertEqual(result["version"], 3)
        self.assertEqual(result["architecture"], "llama")
        self.assertEqual(result["metadata"]["general.name"], "Tiny Test")
        self.assertNotIn("tokenizer.ggml.tokens", result["metadata"])
        self.assertIn("tokenizer.ggml.tokens", result["skipped_keys"])
        self.assertEqual(result["summary"], {
            "name": "Tiny Test", "layers": 4, "kv_heads": 2, "head_dim": 32, "context_length": 4096, "experts": 0,
            "active_experts": 0, "sliding_window": None, "sliding_layers": 0, "file_type": 15, "quant": "Q4_K_M",
            "split_count": 1})

    def test_all_value_types_and_nested_arrays_version_2(self):
        kvs = [("general.architecture", STR, "qwen3"), ("a.u8", U8, 255), ("a.i8", I8, -3), ("a.u16", U16, 65535),
               ("a.i16", I16, -300), ("a.u32", U32, 7), ("a.i32", I32, -7), ("a.f32", F32, 1.5), ("a.bool", BOOL, True),
               ("a.u64", U64, 2**40), ("a.i64", I64, -(2**40)), ("a.f64", F64, 0.25), ("a.str", STR, "héllo"),
               ("a.nums", ARR, (I32, [1, -2, 3])), ("a.nested", ARR, (ARR, [(U8, [1, 2]), (STR, ["x"])])),
               ("a.bools", ARR, (BOOL, [True, False]))]
        result = gguf.read_metadata(self.write("m.gguf", build_gguf(kvs, version=2)))
        meta = result["metadata"]
        self.assertEqual(result["version"], 2)
        self.assertEqual((meta["a.u8"], meta["a.i8"], meta["a.u16"], meta["a.i16"]), (255, -3, 65535, -300))
        self.assertEqual((meta["a.u32"], meta["a.i32"], meta["a.f32"], meta["a.bool"]), (7, -7, 1.5, True))
        self.assertEqual((meta["a.u64"], meta["a.i64"], meta["a.f64"], meta["a.str"]), (2**40, -(2**40), 0.25, "héllo"))
        self.assertNotIn("a.nums", meta)  # arrays never appear in the scalar metadata
        self.assertIn("a.nested", result["skipped_keys"])

    def test_per_layer_kv_heads_key_length_and_sliding_window(self):
        extra = [("gemma3.attention.key_length", U32, 64), ("gemma3.attention.sliding_window", U32, 1024)]
        kvs = [kv for kv in llama_kvs("gemma3", layers=12, extra=extra) if kv[0] != "gemma3.attention.head_count_kv"]
        kvs.append(("gemma3.attention.head_count_kv", ARR, (U32, [4, 2] * 6)))
        summary = gguf.read_metadata(self.write("g.gguf", build_gguf(kvs)))["summary"]
        self.assertEqual((summary["kv_heads"], summary["head_dim"]), (4, 64))
        self.assertEqual((summary["sliding_window"], summary["sliding_layers"]), (1024, 10))  # every 6th layer is full

    def test_sliding_pattern_from_header_and_unknown_pattern(self):
        extra = [("qwen3.attention.sliding_window", U32, 512),
                 ("qwen3.attention.sliding_window_pattern", ARR, (BOOL, [True, True, False, False]))]
        summary = gguf.read_metadata(self.write("a.gguf", build_gguf(llama_kvs("qwen3", extra=extra))))["summary"]
        self.assertEqual(summary["sliding_layers"], 2)
        extra = [("qwen3.attention.sliding_window", U32, 512)]
        summary = gguf.read_metadata(self.write("b.gguf", build_gguf(llama_kvs("qwen3", extra=extra))))["summary"]
        self.assertEqual(summary["sliding_layers"], 0)  # never guess a pattern: full attention is the safe side

    def test_missing_kv_heads_falls_back_to_head_count_and_quant_from_name(self):
        kvs = [kv for kv in llama_kvs() if kv[0] not in {"llama.attention.head_count_kv", "general.file_type"}]
        summary = gguf.read_metadata(self.write("Model-IQ4_XS.gguf", build_gguf(kvs)))["summary"]
        self.assertEqual((summary["kv_heads"], summary["file_type"], summary["quant"]), (8, None, "IQ4_XS"))
        summary = gguf.read_metadata(self.write("plain.gguf", build_gguf(kvs)))["summary"]
        self.assertIsNone(summary["quant"])

    def test_file_type_names_follow_llama_h(self):
        for value, name in [(0, "F32"), (1, "F16"), (7, "Q8_0"), (15, "Q4_K_M"), (17, "Q5_K_M"), (18, "Q6_K"),
                            (30, "IQ4_XS"), (32, "BF16"), (38, "MXFP4")]:
            self.assertEqual(gguf.quant_name(value), name)
        self.assertIsNone(gguf.quant_name(4))  # removed enum value
        self.assertEqual(gguf.quant_name(None, "gpt-oss-20b-mxfp4.gguf"), "MXFP4")
        self.assertEqual(gguf.quant_name(None, "x-q4_k_m.gguf"), "Q4_K_M")

    def test_variant_from_dense_file(self):
        tensors = [("token_embd.weight", [256, 1000], 12), ("blk.0.attn_q.weight", [256, 256], 12)]
        path = self.write("tiny-Q4_K_M.gguf", build_gguf(llama_kvs(extra=[("general.license", STR, "mit")]), tensors))
        variant = gguf.variant_from_file(path)
        self.assertEqual(variant.source, "local")
        self.assertTrue(variant.id.startswith("local:") and variant.id.endswith(":tiny-Q4_K_M.gguf"))
        self.assertEqual((variant.architecture, variant.layers, variant.kv_heads, variant.head_dim), ("llama", 4, 2, 32))
        self.assertEqual((variant.max_context, variant.quant, variant.size_bytes), (4096, "Q4_K_M", path.stat().st_size))
        self.assertEqual((variant.parameters, variant.active_parameters, variant.experts), (256 * 1256, 256 * 1256, 0))
        self.assertEqual((variant.license, variant.files, variant.sha256), ("mit", [], None))
        hashed = gguf.variant_from_file(path, sha256="AB" * 32)
        self.assertEqual((hashed.id, hashed.sha256), (f"local:{'ab' * 8}:tiny-Q4_K_M.gguf", "ab" * 32))
        self.assertEqual(gguf.variant_from_file(path).id, variant.id)  # stable id per path

    def test_variant_from_moe_file_uses_expert_bytes(self):
        extra = [("qwen3moe.expert_count", U32, 8), ("qwen3moe.expert_used_count", U32, 2),
                 ("qwen3moe.attention.key_length", U32, 128)]
        tensors = [("blk.0.ffn_up_exps.weight", [256, 64, 8], 8), ("blk.0.ffn_up_shexp.weight", [256, 64], 8),
                   ("blk.0.attn_q.weight", [256, 256], 0)]
        variant = gguf.variant_from_file(self.write("moe.gguf", build_gguf(llama_kvs("qwen3moe", extra=extra), tensors)))
        self.assertEqual((variant.architecture, variant.experts, variant.active_experts, variant.head_dim), ("qwen3_moe", 8, 2, 128))
        expert_bytes = 256 * 64 * 8 // 32 * 34
        total = expert_bytes + 256 * 64 // 32 * 34 + 256 * 256 * 4
        self.assertAlmostEqual(variant.expert_fraction, expert_bytes / total)
        experts = 256 * 64 * 8
        self.assertEqual(variant.parameters, experts + 256 * 64 + 256 * 256)
        self.assertEqual(variant.active_parameters, variant.parameters - experts * 3 // 4)

    def test_arch_mapping(self):
        for gguf_arch, expected in [("gpt-oss", "gpt_oss"), ("qwen2moe", "qwen2_moe"), ("gemma3", "gemma3"), ("phi3", "phi3")]:
            path = self.write(f"{gguf_arch}.gguf", build_gguf(llama_kvs(gguf_arch)))
            self.assertEqual(gguf.variant_from_file(path).architecture, expected)
        mixtral = llama_kvs(extra=[("llama.expert_count", U32, 8), ("llama.expert_used_count", U32, 2)])
        self.assertEqual(gguf.variant_from_file(self.write("mix.gguf", build_gguf(mixtral))).architecture, "mixtral")
        gpt_oss = gguf.variant_from_file(self.write("oss.gguf", build_gguf(llama_kvs("gpt-oss", layers=24, extra=[
            ("gpt-oss.attention.sliding_window", U32, 128)]))))
        self.assertEqual((gpt_oss.sliding_window, gpt_oss.sliding_layers), (128, 12))

    def test_unsupported_architecture_is_a_plain_error(self):
        path = self.write("mamba.gguf", build_gguf(llama_kvs("mamba")))
        with self.assertRaisesRegex(ValueError, "'mamba' design"):
            gguf.variant_from_file(path)
        missing = [kv for kv in llama_kvs() if kv[0] != "llama.context_length"]
        with self.assertRaisesRegex(ValueError, "context length"):
            gguf.variant_from_file(self.write("nocontext.gguf", build_gguf(missing)))

    def test_split_files(self):
        first = build_gguf(llama_kvs(extra=[("split.count", U16, 3), ("split.no", U16, 0)]),
                           [("blk.0.attn_q.weight", [256, 256], 1)])
        rest = build_gguf([("split.count", U16, 3)], [("blk.1.attn_q.weight", [256, 256], 1)])
        paths = [self.write("big-Q8_0-00001-of-00003.gguf", first), self.write("big-Q8_0-00002-of-00003.gguf", rest)]
        with self.assertRaisesRegex(ValueError, "1 are missing .*00003-of-00003"):
            gguf.variant_from_file(paths[1])
        paths.append(self.write("big-Q8_0-00003-of-00003.gguf", rest))
        variant = gguf.variant_from_file(paths[2])  # any part works; the first part is what llama.cpp loads
        self.assertEqual(variant.filename, "big-Q8_0-00001-of-00003.gguf")
        self.assertEqual([f["filename"] for f in variant.files], [p.name for p in paths])
        self.assertEqual(variant.size_bytes, sum(p.stat().st_size for p in paths))
        self.assertEqual(variant.parameters, 3 * 256 * 256)
        self.assertEqual(gguf.read_metadata(paths[0])["summary"]["split_count"], 3)
        self.assertEqual(gguf.shard_paths(self.write("single.gguf", first)), [self.dir / "single.gguf"])

    def test_corrupt_and_truncated_files_give_plain_errors(self):
        good = build_gguf(llama_kvs())
        cases = {
            "empty.gguf": b"",
            "magic.gguf": b"GGML" + good[4:],
            "v1.gguf": build_gguf(llama_kvs(), version=1),
            "v9.gguf": build_gguf(llama_kvs(), version=9),
            "bigendian.gguf": b"GGUF" + struct.pack(">I", 3) + good[8:],
            "truncated.gguf": good[:200],
            "badtype.gguf": b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + _string("x") + struct.pack("<I", 42),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(ValueError) as caught:
                gguf.read_metadata(self.write(name, data))
            self.assertIn("not a readable GGUF", str(caught.exception))
        with self.assertRaisesRegex(ValueError, "Could not open"):
            gguf.read_metadata(self.dir / "missing.gguf")

    def test_malicious_counts_are_rejected_quickly(self):
        header = b"GGUF" + struct.pack("<I", 3)
        huge_kv = header + struct.pack("<QQ", 0, 2**40)
        huge_string = header + struct.pack("<QQ", 0, 1) + struct.pack("<Q", 2**62)
        huge_array = header + struct.pack("<QQ", 0, 1) + _string("k") + struct.pack("<IIQ", ARR, STR, 2**60)
        huge_numeric = header + struct.pack("<QQ", 0, 1) + _string("k") + struct.pack("<IIQ", ARR, U64, 2**58)
        deep = _string("k") + struct.pack("<I", ARR) + struct.pack("<IQ", ARR, 1) * 10 + struct.pack("<IQ", U8, 0)
        too_deep = header + struct.pack("<QQ", 0, 1) + deep + b"\0" * 64
        huge_tensors = header + struct.pack("<QQ", 2**50, 0)
        many_dims = header + struct.pack("<QQ", 1, 0) + _string("t") + struct.pack("<I", 1000) + b"\0" * 64
        started = time.monotonic()
        for name, data in [("kv", huge_kv), ("s", huge_string), ("a", huge_array), ("n", huge_numeric),
                           ("deep", too_deep), ("t", huge_tensors)]:
            with self.subTest(name), self.assertRaisesRegex(ValueError, "not a readable GGUF"):
                gguf.read_metadata(self.write(name + ".gguf", data))
        with self.assertRaisesRegex(ValueError, "too many dimensions"):
            gguf.read_metadata(self.write("dims.gguf", many_dims), tensors=True)
        self.assertLess(time.monotonic() - started, 2)

    def test_header_byte_cap(self):
        data = build_gguf(llama_kvs())
        path = self.write("capped.gguf", data)
        original = gguf.MAX_HEADER_BYTES
        gguf.MAX_HEADER_BYTES = 1024
        try:
            with self.assertRaisesRegex(ValueError, "larger than"):
                gguf.read_metadata(path)
        finally:
            gguf.MAX_HEADER_BYTES = original

    def test_string_lists_crossing_read_chunks(self):
        items = ["a"] * 1000 + ["b" * (1536 * 1024)] + [f"t{i}" for i in range(200_000)]
        kvs = llama_kvs(extra=[("tokenizer.ggml.merges", ARR, (STR, items)), ("after.value", U32, 42)])
        data = build_gguf(kvs)
        result = gguf.read_metadata(self.write("chunks.gguf", data))
        self.assertEqual(result["metadata"]["after.value"], 42)
        cut = data.index(b"after.value") - 100  # inside the long list
        with self.assertRaisesRegex(ValueError, "cut off"):
            gguf.read_metadata(self.write("cut.gguf", data[:cut]))

    def test_long_strings_are_skipped_not_stored(self):
        kvs = llama_kvs(extra=[("tokenizer.chat_template", STR, "x" * (gguf.MAX_KEPT_STRING + 1))])
        result = gguf.read_metadata(self.write("t.gguf", build_gguf(kvs)))
        self.assertNotIn("tokenizer.chat_template", result["metadata"])
        self.assertEqual(result["summary"]["layers"], 4)


try:  # optional cross-check against the reference implementation, only when it is already installed
    import gguf as reference_gguf  # noqa: F401
    from gguf import GGUFReader
except Exception:  # pragma: no cover - reference package absent
    GGUFReader = None


@unittest.skipUnless(GGUFReader, "reference gguf package not installed")
class ReferenceReaderTest(unittest.TestCase):
    def test_writer_output_is_valid_for_reference_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ref.gguf"
            path.write_bytes(build_gguf(llama_kvs(), [("blk.0.attn_q.weight", [32, 32], 0)]))
            reader = GGUFReader(path)
            self.assertEqual(reader.fields["llama.block_count"].parts[-1][0], 4)
            self.assertEqual(len(reader.tensors), 1)


if __name__ == "__main__":
    unittest.main()
