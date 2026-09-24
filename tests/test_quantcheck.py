import os
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from llm_configurator import quantcheck
from llm_configurator.domain import Cancelled
from llm_configurator.quantcheck import (CORPUS_PATH, estimate_logits_bytes, interpret, kl_check, parse_kld_output,
                                         quant_label)

# Output layouts follow llama.cpp's perplexity.cpp printf/LOG format strings; numbers are from the
# LLaMA 3 8B q4_K_M row of tools/perplexity/README.md.

# Current builds (LOG macros, late 2024 onwards): 95%/90% lines, Δp block, RMS and Same top p.
KLD_CURRENT = """\
build: 6500 (1a2b3c4d) with cc (GCC) 13.2.0 for x86_64-linux-gnu
llama_model_loader: loaded meta data with 30 key-value pairs and 291 tensors from model-Q4_K_M.gguf (version GGUF V3 (latest))
kl_divergence: computing over 12 chunks, n_ctx=512, batch_size=512, n_seq=1
kl_divergence: 0.61 seconds per pass - ETA 0.12 minutes

chunk             PPL               ln(PPL(Q)/PPL(base))          KL Divergence              Δp RMS            Same top p
   1       7.0123 ±    1.4211       0.02101 ±    0.01213       0.02911 ±    0.00302     5.012 ±  0.588 %    92.157 ±  1.687 %
   2       6.8841 ±    0.9876       0.02543 ±    0.00911       0.03002 ±    0.00221     5.301 ±  0.402 %    91.765 ±  1.219 %
  12       6.4071 ±    0.0391       0.02778 ±    0.00072       0.03127 ±    0.00024     5.519 ±  0.050 %    91.901 ±  0.072 %

====== Perplexity statistics ======
Mean PPL(Q)                   :   6.407115 ±   0.039119
Mean PPL(base)                :   6.231633 ±   0.037716
Cor(ln(PPL(Q)), ln(PPL(base))):  99.34%
Mean ln(PPL(Q)/PPL(base))     :   0.027772 ±   0.000723
Mean PPL(Q)/PPL(base)         :   1.028160 ±   0.000723
Mean PPL(Q)-PPL(base)         :   0.175482 ±   0.004620

====== KL divergence statistics ======
Mean    KLD:   0.031273 ±   0.000238
Maximum KLD:   5.911021
99.9%   KLD:   1.834118
99.0%   KLD:   0.412901
95.0%   KLD:   0.118233
90.0%   KLD:   0.066712
Median  KLD:   0.011044
10.0%   KLD:   0.000312
 5.0%   KLD:   0.000105
 1.0%   KLD:   0.000011
 0.1%   KLD:  -0.000002
Minimum KLD:  -0.000011

====== Token probability statistics ======
Mean    Δp: -0.596 ± 0.014 %
Maximum Δp: 95.054%
99.9%   Δp: 27.084%
99.0%   Δp: 12.084%
95.0%   Δp:  4.911%
90.0%   Δp:  2.534%
75.0%   Δp:  0.402%
Median  Δp: -0.024%
25.0%   Δp: -0.611%
10.0%   Δp: -3.101%
 5.0%   Δp: -6.201%
 1.0%   Δp: -19.567%
 0.1%   Δp: -56.054%
Minimum Δp: -98.699%
RMS Δp    :  5.519 ± 0.050 %
Same top p: 91.901 ± 0.072 %

llama_perf_context_print:        load time =    1021.31 ms
"""

# Mid-2024 builds (printf, roughly b2800-b3900): the "95%" line was printed as a second "99.0%".
KLD_MID_2024 = """\
kl_divergence: computing over 12 chunks with n_ctx=512
====== KL divergence statistics ======
Mean    KLD:   0.005452 ±   0.000035
Maximum KLD:   1.204551
99.9%   KLD:   0.301002
99.0%   KLD:   0.061203
99.0%   KLD:   0.061203
Median  KLD:   0.001950
10.0%   KLD:   0.000031
 5.0%   KLD:   0.000009
 1.0%   KLD:  -0.000001
Minimum KLD:  -0.000009

====== Token probability statistics ======
Mean    Δp: -0.007 ± 0.006 %
Maximum Δp: 53.601%
RMS Δp    :  2.295 ± 0.019 %
Same top p: 96.031 ± 0.051 %
"""

# Early 2024 builds (b1960-b2700): "Average:"/"KLD_99 :" and no final same-top or PPL lines.
KLD_EARLY_2024 = """\
kl_divergence : computing over 12 chunks with n_ctx=512

chunk        PPL          ln(PPL(Q)/PPL(base))          KL-Divergence           Same top
   1        9.1201       0.30112 ±    0.05511       0.40125 ±    0.03311    0.73725 ± 0.02762
  12        9.7516       0.44722 ±    0.00362       0.44513 ±    0.00184    0.71138 ± 0.00119

===== KL-divergence statistics
Average:   0.445132 ±  0.001835
Median :   0.201003
Maximum:  14.512002
KLD_99 :   3.901223
KLD_95 :   1.804419
KLD_90 :   1.100187
Minimum:  -0.000031
KLD_01 :   0.000301
KLD_05 :   0.002119
KLD_10 :   0.006013
"""

REFERENCE_RUN = """\
perplexity: saving all logits to /tmp/x/reference.kld
perplexity: tokenizing the input ..
perplexity: tokenization took 12.5 ms
perplexity: calculating perplexity over 12 chunks, n_ctx=512, batch_size=512, n_seq=1
perplexity: 0.52 seconds per pass - ETA 0.10 minutes
[1]6.1201,[2]6.3012,[3]6.2210,[4]6.1804,[5]6.2002,[6]6.2511,[7]6.2233,[8]6.2401,[9]6.2352,[10]6.2290,[11]6.2301,[12]6.2316,
Final estimate: PPL = 6.2316 +/- 0.03772
"""

FAKE_PERPLEXITY = r'''
import os, sys, time
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1] if flag in args else None
log = os.environ.get("FAKE_ARGV_LOG")
if log:
    with open(log, "a", encoding="utf-8") as handle:
        handle.write(" ".join(args) + "\n")
model = value("-m")
base = value("--kl-divergence-base")
if os.environ.get("FAKE_SLEEP") and "--kl-divergence" in args:
    print("kl_divergence: computing over 12 chunks", flush=True)
    time.sleep(float(os.environ["FAKE_SLEEP"]))
if "--kl-divergence" not in args:
    if os.environ.get("FAKE_SHORT"):
        print("perplexity: you need at least 1024 tokens to evaluate perplexity with a context of 512")
        sys.exit(0)  # the real tool exits 0 here too
    with open(base, "wb") as handle:
        handle.write(b"_logits_" + b"\0" * 4096)
    sys.stdout.write(open(os.environ["FAKE_REFERENCE"], encoding="utf-8").read())
    sys.exit(0)
if not os.path.exists(base):
    print("kl_divergence: failed to open " + base)
    sys.exit(1)
if "BROKEN" in model:
    print("llama_model_load: error loading model: tensor data is not within the file bounds")
    print("main: unable to load model")
    sys.exit(1)
if os.environ.get("FAKE_KLD_ERROR"):
    print(os.environ["FAKE_KLD_ERROR"])
    sys.exit(0)  # the real tool logs these errors and still exits 0
name = "FAKE_OUTPUT_" + ("Q6" if "Q6_K" in model else "Q4")
text = open(os.environ[name], encoding="utf-8").read()
counter = os.environ.get("FAKE_COUNTER")
if counter:
    runs = int(open(counter).read() or 0) + 1 if os.path.exists(counter) else 1
    open(counter, "w").write(str(runs))
    if runs <= int(os.environ.get("FAKE_TRUNCATE_RUNS", "0")):
        text = text[:text.index(os.environ.get("FAKE_CUT_AT", "====== KL divergence"))]
sys.stdout.write(text)
'''

SAMPLES = Path(__file__).resolve().parent / "integration" / "samples"


def real_sample(name):
    """stdout+stderr of a real llama-perplexity run recorded by the harness."""
    return (SAMPLES / name).read_text(encoding="utf-8")


class ParserTests(unittest.TestCase):
    def test_current_layout(self):
        parsed = parse_kld_output(KLD_CURRENT)
        self.assertEqual(parsed["mean_kld"], 0.031273)
        self.assertEqual(parsed["mean_kld_uncertainty"], 0.000238)
        self.assertEqual(parsed["median_kld"], 0.011044)
        self.assertEqual(parsed["kld_99"], 0.412901)
        self.assertEqual(parsed["kld_999"], 1.834118)
        self.assertEqual(parsed["max_kld"], 5.911021)
        self.assertEqual(parsed["same_top_p"], 91.901)
        self.assertEqual(parsed["same_top_p_uncertainty"], 0.072)
        self.assertEqual(parsed["ppl"], 6.407115)
        self.assertEqual(parsed["ppl_base"], 6.231633)
        self.assertEqual(parsed["ppl_ratio"], 1.028160)
        self.assertEqual(parsed["mean_delta_p"], -0.596)
        self.assertEqual(parsed["rms_delta_p"], 5.519)
        self.assertEqual(parsed["chunks_done"], 12)

    def test_mid_2024_duplicate_99_line_and_missing_ppl_block(self):
        parsed = parse_kld_output(KLD_MID_2024)
        self.assertEqual(parsed["mean_kld"], 0.005452)
        self.assertEqual(parsed["kld_99"], 0.061203)
        self.assertEqual(parsed["same_top_p"], 96.031)
        self.assertIsNone(parsed["ppl"])
        self.assertIsNone(parsed["ppl_base"])
        self.assertIsNone(parsed["ppl_ratio"])

    def test_early_2024_layout_uses_last_chunk_row_for_gaps(self):
        parsed = parse_kld_output(KLD_EARLY_2024)
        self.assertEqual(parsed["mean_kld"], 0.445132)
        self.assertEqual(parsed["median_kld"], 0.201003)
        self.assertEqual(parsed["kld_99"], 3.901223)
        self.assertEqual(parsed["max_kld"], 14.512002)
        self.assertAlmostEqual(parsed["same_top_p"], 71.138)
        self.assertEqual(parsed["ppl"], 9.7516)
        self.assertIsNone(parsed["mean_delta_p"])
        self.assertEqual(parsed["chunks_done"], 12)

    def test_reference_run_final_estimate(self):
        parsed = parse_kld_output(REFERENCE_RUN)
        self.assertEqual(parsed["final_ppl"], 6.2316)
        self.assertIsNone(parsed["mean_kld"])

    def test_missing_lines_stay_none(self):
        parsed = parse_kld_output("Mean    KLD:   0.1\n")
        self.assertEqual(parsed["mean_kld"], 0.1)
        for key in ["median_kld", "kld_99", "same_top_p", "ppl_base", "ppl", "mean_delta_p"]:
            self.assertIsNone(parsed[key])
        for empty in ["", None]:
            parsed = parse_kld_output(empty)
            self.assertFalse(parsed.pop("complete"))
            self.assertEqual(parsed.pop("from_rows"), [])
            self.assertTrue(all(value is None for value in parsed.values()))

    def test_spacing_colour_codes_crlf_and_mangled_delta(self):
        text = ("\x1b[32mMean KLD :0.0200 +/- 0.0010\x1b[0m\r\n"
                "Same top p:97.5±0.1%\r\n"
                "Mean    Î”p: -0.250 ± 0.010 %\r\n"
                "mean ppl(q) : 6.5\r\n")
        parsed = parse_kld_output(text)
        self.assertEqual(parsed["mean_kld"], 0.02)
        self.assertEqual(parsed["mean_kld_uncertainty"], 0.001)
        self.assertEqual(parsed["same_top_p"], 97.5)
        self.assertEqual(parsed["mean_delta_p"], -0.25)
        self.assertEqual(parsed["ppl"], 6.5)

    def test_nan_and_scientific_values(self):
        parsed = parse_kld_output("Mean    KLD:        nan ±        nan\nMedian  KLD: 1.5e-05\n")
        self.assertIsNone(parsed["mean_kld"])
        self.assertEqual(parsed["median_kld"], 1.5e-05)

    def test_complete_flag(self):
        self.assertTrue(parse_kld_output(KLD_CURRENT)["complete"])
        self.assertTrue(parse_kld_output(KLD_MID_2024)["complete"])
        self.assertTrue(parse_kld_output(KLD_EARLY_2024)["complete"])   # never prints a Same top summary
        self.assertEqual(parse_kld_output(KLD_CURRENT)["from_rows"], [])
        self.assertFalse(parse_kld_output(REFERENCE_RUN)["complete"])

    def test_output_cut_before_the_last_line(self):
        parsed = parse_kld_output(KLD_CURRENT[:KLD_CURRENT.index("Same top p:")])
        self.assertFalse(parsed["complete"])
        self.assertEqual(parsed["mean_kld"], 0.031273)           # the summary line survived
        self.assertEqual(parsed["same_top_p"], 91.901)           # from the last row (running average)
        self.assertEqual(parsed["from_rows"], ["same_top_p"])

    def test_output_cut_before_the_kl_block(self):
        parsed = parse_kld_output(KLD_CURRENT[:KLD_CURRENT.index("====== KL divergence")])
        self.assertFalse(parsed["complete"])
        self.assertEqual(parsed["mean_kld"], 0.03127)            # last row's running average
        self.assertIsNone(parsed["median_kld"])
        self.assertEqual(parsed["chunks_done"], 12)
        self.assertEqual(sorted(parsed["from_rows"]), ["mean_kld", "same_top_p"])

    def test_real_samples(self):
        parsed = parse_kld_output(real_sample("perplexity-kld.txt"))
        self.assertTrue(parsed["complete"])
        self.assertEqual(parsed["from_rows"], [])
        expected = {"mean_kld": 0.000234, "mean_kld_uncertainty": 0.000025, "median_kld": 0.000101,
                    "kld_99": 0.001983, "kld_999": 0.002123, "max_kld": 0.002127, "same_top_p": 100.0,
                    "ppl": 61360.395556, "ppl_base": 55470.500402, "ppl_ratio": 1.106181, "mean_delta_p": 0.0,
                    "rms_delta_p": 0.014, "chunks_done": 4}
        for key, value in expected.items():
            self.assertEqual(parsed[key], value, key)
        for name in ["perplexity-kld-base.txt", "perplexity-normal.txt"]:
            parsed = parse_kld_output(real_sample(name))
            self.assertIsNotNone(parsed["final_ppl"], name)
            self.assertIsNone(parsed["mean_kld"], name)
            self.assertIsNone(parsed["chunks_done"], name)     # "[1]58946.06," progress is not a row

    def test_real_sample_cut_short(self):
        text = real_sample("perplexity-kld.txt")
        parsed = parse_kld_output(text[:text.index("====== Perplexity statistics")])
        self.assertFalse(parsed["complete"])
        self.assertEqual(parsed["mean_kld"], 0.00023)
        self.assertEqual(parsed["same_top_p"], 100.0)
        self.assertEqual(parsed["ppl"], 61360.3956)
        self.assertIsNone(parsed["ppl_base"])

    def test_percentile_lines_are_not_mistaken_for_chunk_rows(self):
        parsed = parse_kld_output(" 5.0%   KLD:   0.000105\n 1.0%   KLD:   0.000011\n")
        self.assertIsNone(parsed["chunks_done"])
        self.assertIsNone(parsed["kld_99"])


class InterpretTests(unittest.TestCase):
    def test_bands_follow_scoreboard_anchors(self):
        cases = [(0.001355, 97.674, "negligible"), (0.031273, 91.901, "small"), (0.101913, 85.0, "moderate"),
                 (0.445132, 71.138, "large"), (1.39, 55.0, "severe")]
        for kld, same_top, verdict in cases:
            self.assertEqual(interpret({"mean_kld": kld, "same_top_p": same_top})["verdict"], verdict)

    def test_sentence(self):
        result = interpret({"mean_kld": 0.031, "same_top_p": 96.0}, "Q8_0")
        self.assertEqual(result["plain"], "Picks a different top word about 4% of the time compared with Q8_0"
                                          " — usually hard to notice in chat.")
        tiny = interpret({"mean_kld": 0.0002, "same_top_p": 99.7}, "BF16")["plain"]
        self.assertIn("less than 1%", tiny)
        self.assertIn("very unlikely to be noticed", tiny)

    def test_partial_and_missing(self):
        self.assertEqual(interpret({"mean_kld": None, "same_top_p": 80.0})["verdict"], "large")
        only_kld = interpret({"mean_kld": 0.2, "same_top_p": None}, "Q8_0")
        self.assertEqual(only_kld["verdict"], "large")
        self.assertIn("0.200", only_kld["plain"])
        self.assertIsNone(interpret({})["verdict"])

    def test_quant_label(self):
        self.assertEqual(quant_label("/m/Qwen3-8B-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(quant_label("model.q8_0.gguf"), "Q8_0")
        self.assertEqual(quant_label("gemma-3-IQ4_XS.gguf"), "IQ4_XS")
        self.assertEqual(quant_label("x-BF16-00001-of-00002.gguf"), "BF16")
        self.assertEqual(quant_label("mystery.gguf"), "mystery.gguf")


class EstimateTests(unittest.TestCase):
    def test_matches_llama_cpp_writer_layout(self):
        # Header, tokens at 4 bytes each, then per chunk (n_ctx - 1 - n_ctx/2) rows of (2*((V+1)/2)+4) uint16.
        self.assertEqual(estimate_logits_bytes(32000, 512, 1), 20 + 512 * 4 + 255 * 32004 * 2)
        big = estimate_logits_bytes(151936, 512, 12)
        self.assertGreater(big, 0.9e9)
        self.assertLess(big, 1.0e9)

    def test_default_chunks_target_about_six_thousand_tokens(self):
        self.assertEqual(quantcheck._default_chunks(512, 45000), 12)
        self.assertEqual(quantcheck._default_chunks(2048, 45000), 3)
        self.assertEqual(quantcheck._default_chunks(512, 8000), 3)   # capped by the text length


class CorpusTests(unittest.TestCase):
    def test_corpus_size_and_content(self):
        data = CORPUS_PATH.read_bytes()
        self.assertTrue(30_000 <= len(data) <= 60_000, len(data))
        text = data.decode("utf-8")
        self.assertIn("def ", text)            # a little code
        self.assertIn('"', text)               # some dialogue


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


class KlCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.work = self.dir / "work"
        self.work.mkdir()
        self.reference = _write(self.dir / "model-Q8_0.gguf", "ref")
        self.q4 = _write(self.dir / "model-Q4_K_M.gguf", "q4")
        self.q6 = _write(self.dir / "model-Q6_K.gguf", "q6")
        script = _write(self.dir / "fake_perplexity.py", FAKE_PERPLEXITY)
        self.command = [sys.executable, str(script)]
        self.argv_log = self.dir / "argv.log"
        self.env = {"FAKE_REFERENCE": str(_write(self.dir / "ref.txt", REFERENCE_RUN)),
                    "FAKE_OUTPUT_Q4": str(_write(self.dir / "q4.txt", KLD_CURRENT)),
                    "FAKE_OUTPUT_Q6": str(_write(self.dir / "q6.txt", KLD_MID_2024)),
                    "FAKE_ARGV_LOG": str(self.argv_log)}
        patcher = mock.patch.dict(os.environ, self.env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp.cleanup)

    def check(self, **kwargs):
        options = {"work_dir": self.work, "vocab_size": 32000, "timeout": 60}
        options.update(kwargs)
        return kl_check(self.command, self.reference, options.pop("candidates", {"Q4_K_M": self.q4, "Q6_K": self.q6}),
                        **options)

    def test_two_step_flow_with_fake_llama_perplexity(self):
        events = []
        result = self.check(progress=events.append)
        self.assertEqual(result["reference"]["label"], "Q8_0")
        self.assertEqual(result["reference"]["ppl"], 6.2316)
        self.assertEqual(result["reference"]["logits_bytes"], 8 + 4096)
        q4, q6 = result["results"]["Q4_K_M"], result["results"]["Q6_K"]
        self.assertEqual(q4["mean_kld"], 0.031273)
        self.assertEqual(q4["verdict"], "small")
        self.assertIn("about 8% of the time compared with Q8_0", q4["plain"])
        self.assertIsNone(q4["error"])
        self.assertEqual(q6["verdict"], "negligible")
        self.assertEqual(q6["ppl_base"], 6.2316)   # old layout: filled from the reference run
        self.assertEqual(result["corpus"]["chunks"], quantcheck._default_chunks(512, CORPUS_PATH.stat().st_size))
        self.assertTrue(any("deleted afterwards" in note for note in result["notes"]))
        self.assertEqual(os.listdir(self.work), [])
        stages = [event["stage"] for event in events]
        self.assertEqual(stages[0], "reference")
        self.assertIn("candidate", stages)
        self.assertEqual(events[-1]["stage"], "done")
        self.assertEqual(events[-1]["done"], events[-1]["total"])
        self.assertTrue(all(set(event) >= {"stage", "done", "total", "message"} for event in events))
        self.assertTrue(any("window 12 of 12" in event["message"] for event in events))
        calls = self.argv_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 3)
        self.assertIn("--kl-divergence-base", calls[0])
        self.assertNotIn("--kl-divergence ", calls[0] + " ")
        self.assertIn("--chunks", calls[0])
        self.assertIn("--kl-divergence", calls[1].split())
        self.assertIn("-c 512 -b 512", calls[1])
        self.assertIn("model-Q4_K_M.gguf", calls[1])
        self.assertIn("model-Q6_K.gguf", calls[2])

    def test_launch_config_mapping_skips_server_only_flags(self):
        seen = []

        def run(argv, env=None, timeout=None, cancel=None, on_line=None):
            seen.append((argv, env))
            base = Path(argv[argv.index("--kl-divergence-base") + 1])
            if "--kl-divergence" not in argv:
                base.write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            return 0, KLD_CURRENT
        config = {"model_path": "/elsewhere.gguf", "gpu_layers": 32, "total_layers": 32, "threads": 8,
                  "device": "CUDA0", "gpu_backend": "cuda", "gpu_uuid": "GPU-1234", "n_cpu_moe": 4, "port": 8080,
                  "alias": "x", "cache_type_k": "q8_0", "parallel": 2, "batch": 2048, "context": 8192}
        self.check(candidates={"Q4_K_M": self.q4}, config=config, run=run, context=1024, chunks=4)
        argv, env = seen[1]
        joined = " ".join(argv)
        self.assertIn("-ngl 33", joined)          # full offload counts the output layer
        self.assertIn("-dev CUDA0", joined)
        self.assertIn("-t 8", joined)
        self.assertIn("--n-cpu-moe 4", joined)
        self.assertIn("-c 1024 -b 1024", joined)
        for flag in ["--port", "--host", "--jinja", "--alias", "-np", "-ctk", "-ctv", "/elsewhere.gguf"]:
            self.assertNotIn(flag, argv)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "GPU-1234")
        self.assertNotIn("--chunks", seen[1][0])
        self.assertIn("--chunks", seen[0][0])

    def test_cpu_only_config_hides_gpus(self):
        seen = []

        def run(argv, env=None, **_):
            seen.append(argv)
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
            return 0, KLD_CURRENT
        self.check(candidates={"Q4_K_M": self.q4}, config={"gpu_layers": 0, "gpu_backend": "metal"}, run=run)
        self.assertIn("-dev none", " ".join(seen[0]))
        self.assertIn("-ngl 0", " ".join(seen[0]))

    def test_disk_space_refusal_runs_nothing(self):
        run = mock.Mock()
        with mock.patch.object(quantcheck.shutil, "disk_usage", return_value=mock.Mock(free=100 * 1024**2)):
            with self.assertRaises(ValueError) as caught:
                self.check(run=run, vocab_size=151936)
        self.assertIn("GB of temporary space", str(caught.exception))
        self.assertIn("Free up some space", str(caught.exception))
        run.assert_not_called()
        self.assertEqual(os.listdir(self.work), [])

    def test_unknown_vocab_assumes_largest(self):
        with mock.patch.object(quantcheck, "_vocab_size", return_value=None):
            result = self.check(vocab_size=None, candidates={"Q4_K_M": self.q4})
        self.assertEqual(result["estimated_temp_bytes"],
                         estimate_logits_bytes(quantcheck.ASSUMED_VOCAB, 512, result["corpus"]["chunks"]))
        self.assertIn("assumes a very large one", result["notes"][0])

    def test_reference_failure_is_plain_and_cleans_up(self):
        with mock.patch.dict(os.environ, {"FAKE_SHORT": "1"}):
            with self.assertRaises(ValueError) as caught:
                self.check()
        self.assertIn("too short", str(caught.exception))
        self.assertEqual(os.listdir(self.work), [])

    def test_candidate_failure_is_reported_per_model(self):
        broken = _write(self.dir / "BROKEN-Q3_K_M.gguf", "x")
        result = self.check(candidates={"Q3_K_M": broken, "Q4_K_M": self.q4})
        self.assertIsNone(result["results"]["Q3_K_M"]["mean_kld"])
        self.assertIsNone(result["results"]["Q3_K_M"]["verdict"])
        self.assertIn("could not load", result["results"]["Q3_K_M"]["error"])
        self.assertIn("Could not compare Q3_K_M", result["results"]["Q3_K_M"]["plain"])
        self.assertEqual(result["results"]["Q4_K_M"]["verdict"], "small")
        self.assertEqual(os.listdir(self.work), [])

    def test_cancel_between_models_cleans_up(self):
        cancel = threading.Event()
        calls = []

        def run(argv, env=None, timeout=None, cancel=None, on_line=None):
            calls.append(argv)
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            cancel.set()
            return 0, KLD_CURRENT
        with self.assertRaises(Cancelled):
            self.check(run=run, cancel=cancel)
        self.assertEqual(len(calls), 2)            # stopped before the second candidate
        self.assertEqual(os.listdir(self.work), [])

    def test_cancel_kills_running_process(self):
        cancel = threading.Event()
        threading.Timer(1.0, cancel.set).start()
        started = time.monotonic()
        with mock.patch.dict(os.environ, {"FAKE_SLEEP": "30"}):
            with self.assertRaises(Cancelled):
                self.check(cancel=cancel)
        self.assertLess(time.monotonic() - started, 15)
        self.assertEqual(os.listdir(self.work), [])

    def test_candidate_timeout_is_reported_per_model(self):
        with mock.patch.dict(os.environ, {"FAKE_SLEEP": "30"}):
            result = self.check(timeout=1)
        for label in ["Q4_K_M", "Q6_K"]:
            entry = result["results"][label]
            self.assertIn("took longer than", entry["error"])
            self.assertIsNone(entry["verdict"])
            self.assertEqual(entry["attempts"], 1)
        self.assertEqual(result["reference"]["ppl"], 6.2316)
        self.assertEqual(os.listdir(self.work), [])

    def test_reference_timeout_is_plain_and_cleans_up(self):
        def run(argv, **_):
            raise quantcheck._TimedOut("llama-perplexity took longer than 1 minutes and was stopped.")
        with self.assertRaises(ValueError) as caught:
            self.check(run=run)
        self.assertIn("took longer than", str(caught.exception))
        self.assertEqual(os.listdir(self.work), [])

    def test_timeout_on_a_retry_keeps_the_partial_result(self):
        cut = KLD_CURRENT[:KLD_CURRENT.index("Same top p:")]
        outputs = [(0, cut)]

        def run(argv, **_):
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            if outputs:
                return outputs.pop()
            raise quantcheck._TimedOut("llama-perplexity took longer than 1 minutes and was stopped.")
        q4 = self.check(run=run, candidates={"Q4_K_M": self.q4})["results"]["Q4_K_M"]
        self.assertIsNone(q4["error"])
        self.assertEqual(q4["mean_kld"], 0.031273)
        self.assertIn("Partial result", q4["plain"])
        self.assertEqual(q4["attempts"], 2)

    def test_output_lost_mid_line_is_not_trusted(self):
        text = KLD_CURRENT[:KLD_CURRENT.index("Same top p:") + len("Same top p: 9")]
        outputs = [(0, KLD_CURRENT), (0, text)]
        events = []

        def run(argv, on_line=None, **_):
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            code, output = outputs.pop()
            for line in output.splitlines(keepends=True):
                on_line(line)
            return code, output
        q4 = self.check(run=run, candidates={"Q4_K_M": self.q4}, progress=events.append)["results"]["Q4_K_M"]
        self.assertEqual(q4["attempts"], 2)
        self.assertEqual(q4["same_top_p"], 91.901)
        dones = [event["done"] for event in events]
        self.assertEqual(dones, sorted(dones))                  # the re-run does not move the bar back

    def test_unrelated_server_settings_do_not_block_the_check(self):
        seen = []

        def run(argv, env=None, **_):
            seen.append(argv)
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
            return 0, KLD_CURRENT
        config = {"gpu_layers": 0, "cache_type_v": "q8_0", "flash_attn": "off", "draft_max": 8, "host": "0.0.0.0",
                  "threads": 4, "surprise": True}
        self.check(candidates={"Q4_K_M": self.q4}, config=config, run=run)
        self.assertIn("-t 4", " ".join(seen[1]))

    def test_missing_work_dir_is_created_and_chunk_count_follows_llama_cpp(self):
        work = self.work / "new" / "folder"
        reference = REFERENCE_RUN.replace("over 12 chunks", "over 9 chunks")
        with mock.patch.dict(os.environ, {"FAKE_REFERENCE": str(_write(self.dir / "r9.txt", reference))}):
            result = self.check(work_dir=work, candidates={"Q4_K_M": self.q4})
        self.assertEqual(os.listdir(work), [])
        self.assertEqual(result["corpus"]["chunks"], 9)
        self.assertEqual(result["corpus"]["tokens"], 9 * 512)

    def test_unexpected_error_cleans_up(self):
        def run(argv, **_):
            Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.check(run=run)
        self.assertEqual(os.listdir(self.work), [])

    def test_cut_short_output_is_run_again_reusing_the_reference(self):
        counter = self.dir / "runs.txt"
        events = []
        with mock.patch.dict(os.environ, {"FAKE_COUNTER": str(counter), "FAKE_TRUNCATE_RUNS": "2"}):
            result = self.check(candidates={"Q4_K_M": self.q4}, progress=events.append)
        q4 = result["results"]["Q4_K_M"]
        self.assertEqual(q4["attempts"], 3)
        self.assertTrue(q4["complete"])
        self.assertIsNone(q4["partial"])
        self.assertEqual(q4["median_kld"], 0.011044)
        self.assertNotIn("Partial", q4["plain"])
        calls = self.argv_log.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(calls), 4)                              # reference once, candidate three times
        self.assertEqual(sum("--kl-divergence " in call + " " for call in calls), 3)
        self.assertTrue(any("running it again with fewer CPU threads (try 2 of 3)" in event["message"]
                            for event in events))
        self.assertEqual(events[-1]["done"], events[-1]["total"])
        self.assertEqual(os.listdir(self.work), [])

    def test_always_cut_short_falls_back_to_last_row_and_says_so(self):
        with mock.patch.dict(os.environ, {"FAKE_COUNTER": str(self.dir / "runs.txt"), "FAKE_TRUNCATE_RUNS": "99"}):
            result = self.check(candidates={"Q4_K_M": self.q4})
        q4 = result["results"]["Q4_K_M"]
        self.assertEqual(q4["attempts"], quantcheck.MAX_ATTEMPTS)
        self.assertFalse(q4["complete"])
        self.assertIsNone(q4["error"])
        self.assertEqual(q4["mean_kld"], 0.03127)                 # last row's running average
        self.assertEqual(q4["verdict"], "small")
        self.assertIsNone(q4["median_kld"])
        self.assertIn("cut short 3 times in a row", q4["partial"])
        self.assertIn("last progress line, which covers all 12 windows", q4["partial"])
        self.assertIn("missing: median, worst 1%", q4["partial"])
        self.assertIn("(Partial result:", q4["plain"])
        self.assertTrue(any(note.startswith("Q4_K_M: llama-perplexity's final report was cut short")
                            for note in result["notes"]))
        self.assertEqual(len(self.argv_log.read_text(encoding="utf-8").splitlines()), 1 + quantcheck.MAX_ATTEMPTS)

    def test_re_runs_use_fewer_threads(self):
        self.assertEqual(quantcheck._fewer_threads(["x", "-t", "8", "-c", "512"]), ["x", "-t", "4", "-c", "512"])
        self.assertEqual(quantcheck._fewer_threads(["x", "-t", "1"]), ["x", "-t", "1"])
        with mock.patch.object(quantcheck.os, "cpu_count", return_value=6):
            self.assertEqual(quantcheck._fewer_threads(["x"]), ["x", "-t", "3"])
        with mock.patch.object(quantcheck.os, "cpu_count", return_value=None):
            self.assertEqual(quantcheck._fewer_threads(["x"]), ["x", "-t", "1"])
        seen = []
        cut = KLD_CURRENT[:KLD_CURRENT.index("Same top p:")]

        def run(argv, **_):
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            seen.append(argv)
            return 0, cut
        self.check(run=run, candidates={"Q4_K_M": self.q4}, config={"gpu_layers": 0, "threads": 8})
        self.assertEqual([argv[argv.index("-t") + 1] for argv in seen], ["8", "4", "2"])
        self.assertEqual([argv.count("-t") for argv in seen], [1, 1, 1])

    def test_partial_note_wording(self):
        rows = {"from_rows": ["mean_kld"], "chunks_done": 2, "median_kld": 0.1, "kld_99": 0.2, "same_top_p": 90.0,
                "mean_delta_p": 0.1}
        self.assertEqual(quantcheck._partial_note(rows, 3, 12),
                         "llama-perplexity's final report was cut short 3 times in a row, so some figures are a "
                         "running average over the first 2 of 12 windows.")
        only_end = quantcheck._partial_note({"from_rows": [], "median_kld": 0.1, "kld_99": 0.2, "same_top_p": None}, 1, 4)
        self.assertEqual(only_end, "llama-perplexity's final report was cut short; "
                                   "missing: same top word, change in confidence.")

    def test_missing_last_line_only_is_retried(self):
        with mock.patch.dict(os.environ, {"FAKE_COUNTER": str(self.dir / "runs.txt"), "FAKE_TRUNCATE_RUNS": "1",
                                          "FAKE_CUT_AT": "Same top p:"}):
            result = self.check(candidates={"Q4_K_M": self.q4})
        self.assertEqual(result["results"]["Q4_K_M"]["attempts"], 2)
        self.assertTrue(result["results"]["Q4_K_M"]["complete"])

    def test_real_errors_are_not_retried(self):
        broken = _write(self.dir / "BROKEN-Q3_K_M.gguf", "x")      # exit 1
        result = self.check(candidates={"Q3_K_M": broken})
        self.assertEqual(result["results"]["Q3_K_M"]["attempts"], 1)
        for message, words in [("kl_divergence: inconsistent vocabulary (151936 vs 32000)", "different vocabularies"),
                               ("kl_divergence: /x/reference.kld has been computed with 1024, while the current "
                                "context is 512. Increase it with -c and retry", "different window size")]:
            self.argv_log.write_text("")
            with mock.patch.dict(os.environ, {"FAKE_KLD_ERROR": message}):
                result = self.check(candidates={"Q4_K_M": self.q4})
            q4 = result["results"]["Q4_K_M"]
            self.assertEqual(q4["attempts"], 1, message)          # exit 0, but a known error: no retry
            self.assertIn(words, q4["error"])
            self.assertIsNone(q4["verdict"])
            self.assertEqual(len(self.argv_log.read_text(encoding="utf-8").splitlines()), 2)

    def test_unknown_silent_failure_is_retried_then_reported(self):
        with mock.patch.dict(os.environ, {"FAKE_KLD_ERROR": "llama_perf_context_print: load time = 1 ms"}):
            result = self.check(candidates={"Q4_K_M": self.q4})
        q4 = result["results"]["Q4_K_M"]
        self.assertEqual(q4["attempts"], quantcheck.MAX_ATTEMPTS)
        self.assertIn("gave no comparison", q4["error"])
        self.assertIsNone(q4["partial"])
        self.assertIn("Could not compare Q4_K_M", q4["plain"])

    def test_cancel_during_retries_cleans_up(self):
        cancel = threading.Event()
        calls = []

        def run(argv, env=None, timeout=None, cancel=None, on_line=None):
            calls.append(argv)
            if "--kl-divergence" not in argv:
                Path(argv[argv.index("--kl-divergence-base") + 1]).write_bytes(b"_logits_" + b"\0" * 64)
                return 0, REFERENCE_RUN
            cancel.set()
            return 0, KLD_CURRENT[:KLD_CURRENT.index("Same top p:")]
        with self.assertRaises(Cancelled):
            self.check(run=run, cancel=cancel)
        self.assertEqual(len(calls), 2)
        self.assertEqual(os.listdir(self.work), [])

    def test_reference_ppl_from_progress_when_final_line_is_lost(self):
        cut = REFERENCE_RUN[:REFERENCE_RUN.index("Final estimate")]
        with mock.patch.dict(os.environ, {"FAKE_REFERENCE": str(_write(self.dir / "cut.txt", cut))}):
            result = self.check(candidates={"Q6_K": self.q6})
        self.assertEqual(result["reference"]["ppl"], 6.2316)
        self.assertEqual(result["results"]["Q6_K"]["ppl_base"], 6.2316)

    def test_results_and_errors_hide_folder_names(self):
        import json
        result = self.check()
        dumped = json.dumps(result)
        self.assertNotIn(str(self.dir), dumped)
        self.assertEqual(result["reference"]["file"], "model-Q8_0.gguf")
        self.assertEqual(result["results"]["Q4_K_M"]["file"], "model-Q4_K_M.gguf")
        self.assertEqual(result["corpus"]["file"], CORPUS_PATH.name)

        def run(argv, **_):
            return 0, f"llama_model_load: loading model from {argv[argv.index('-m') + 1]}\nsomething odd\n"
        with self.assertRaises(ValueError) as caught:
            self.check(run=run)
        self.assertIn("model-Q8_0.gguf", str(caught.exception))
        self.assertNotIn(str(self.dir), str(caught.exception))
        with self.assertRaises(ValueError) as caught:
            self.check(candidates={"Q4": self.dir / "missing.gguf"})
        self.assertNotIn(str(self.dir), str(caught.exception))

    def test_vocab_size_from_gguf_summary(self):
        from llm_configurator import gguf
        info = {"metadata": {}, "architecture": "llama", "summary": {"vocab_size": 151936}}
        with mock.patch.object(gguf, "read_metadata", return_value=info):
            self.assertEqual(quantcheck._vocab_size(self.reference), 151936)
            result = self.check(vocab_size=None, candidates={"Q4_K_M": self.q4})
        self.assertEqual(result["estimated_temp_bytes"], estimate_logits_bytes(151936, 512, result["corpus"]["chunks"]))
        self.assertNotIn("assumes", result["notes"][0])
        without = {"metadata": {"llama.vocab_size": 32000}, "architecture": "llama", "summary": {"layers": 4}}
        with mock.patch.object(gguf, "read_metadata", return_value=without):
            self.assertEqual(quantcheck._vocab_size(self.reference), 32000)
        with mock.patch.object(gguf, "read_metadata", return_value={"metadata": {}, "summary": {}}):
            self.assertIsNone(quantcheck._vocab_size(self.reference))
        with mock.patch.object(gguf, "read_metadata", side_effect=ValueError("not a GGUF")):
            self.assertIsNone(quantcheck._vocab_size(self.reference))

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            self.check(candidates={})
        with self.assertRaises(ValueError):
            self.check(candidates={"Q4": self.dir / "missing.gguf"})
        with self.assertRaises(ValueError):
            self.check(context=64)
        with self.assertRaises(ValueError):
            self.check(chunks=0)
        with self.assertRaises(ValueError):
            self.check(corpus_path=_write(self.dir / "tiny.txt", "short text"))
        with self.assertRaises(ValueError):
            self.check(config={"gpu_layers": -1})


if __name__ == "__main__":
    unittest.main()
