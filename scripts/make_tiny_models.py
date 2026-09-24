#!/usr/bin/env python3
"""Write tiny, deterministic GGUF models for the real-runtime integration tests.

Most weights are seeded random numbers, but the embeddings and output layer are
wired as a "next word" table so greedy decoding always writes plain ASCII:
a chat answer is "42" (the smoke-test answer), and any other text continues with
" the quick brown fox." and then stops. Random bytes would otherwise make
llama-server's reply parser fail with HTTP 500. The random attention/FFN weights
still give quantization something to lose, so KL divergence is non-zero.

Usage: python3 scripts/make_tiny_models.py OUT_DIR [--bin LLAMA_BIN_DIR]

Writes into OUT_DIR (never into the repository):
  tiny-llama-F16.gguf         dense `llama` architecture, byte-level BPE vocab, ChatML template
  tiny-qwen3moe-F16.gguf      mixture-of-experts `qwen3moe` (8 experts, 2 active)
  unknown-arch.gguf           valid GGUF whose architecture llama.cpp does not know
  corrupt.gguf                the dense model truncated half way (a broken download)
With --bin (a directory holding llama-quantize and llama-gguf-split) it also writes
  tiny-llama-Q8_0.gguf, tiny-llama-Q4_K_M.gguf, tiny-qwen3moe-Q4_K_M.gguf and
  split/tiny-llama-Q8_0-0000k-of-0000n.gguf (a sharded copy).
Needs `pip install gguf` (it brings numpy).
"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

try:
    import gguf
except ImportError:
    sys.exit("The gguf package is required: python3 -m pip install gguf")

SEED = 1234
CHATML = ("{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
          "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}")
TRAINING_TEXT = """The quick brown fox jumps over the lazy dog. What is two plus two? Two plus two is four.
The capital of France is Paris. A model reads tokens and writes tokens. Hello world, hello there.
The needle in the haystack is a small fact hidden inside a long document. Answer with one word.
def add(a, b):\n    return a + b\nprint(add(2, 3))\n""" * 3
SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]


def byte_unicode():
    """GPT-2's reversible byte -> printable character table."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars, extra = keep[:], 0
    for b in range(256):
        if b not in keep:
            keep.append(b)
            chars.append(256 + extra)
            extra += 1
    return {b: chr(c) for b, c in zip(keep, chars)}


def bpe_vocab(merges_wanted=200):
    """Train a small deterministic byte-level BPE; returns (tokens, merges)."""
    table = byte_unicode()
    tokens = [table[b] for b in range(256)]
    words = {}
    for word in re.findall(r" ?\w+| ?[^\w\s]+|\s+", TRAINING_TEXT):
        key = tuple(table[b] for b in word.encode())
        words[key] = words.get(key, 0) + 1
    merges = []
    for _ in range(merges_wanted):
        pairs = {}
        for word, count in words.items():
            for pair in zip(word, word[1:]):
                pairs[pair] = pairs.get(pair, 0) + count
        if not pairs:
            break
        best = max(sorted(pairs), key=lambda p: pairs[p])
        merges.append(f"{best[0]} {best[1]}")
        tokens.append(best[0] + best[1])
        merged = {}
        for word, count in words.items():
            out, i = [], 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    out.append(word[i] + word[i + 1])
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            merged[tuple(out)] = merged.get(tuple(out), 0) + count
        words = merged
    return tokens, merges


CHAIN_WORDS = [" the", " quick", " brown", " fox", "."]


def next_token_table(tokens):
    """Greedy continuation: end of prompt -> "4" -> "2" -> end; anything else -> " the quick brown fox." -> end."""
    table = byte_unicode()
    ids = {t: i for i, t in enumerate(tokens)}
    enc = lambda text: ids["".join(table[b] for b in text.encode())]
    end = ids["<|im_end|>"]
    chain = [enc(w) for w in CHAIN_WORDS]
    nxt = {i: chain[0] for i in range(len(tokens))}
    for a, b in zip(chain, chain[1:] + [end]):
        nxt[a] = b
    nxt[enc("\n")], nxt[enc("4")], nxt[enc("2")] = enc("4"), enc("2"), end
    return nxt


def write_model(path, arch, *, n_embd=512, n_layer=4, n_head=8, n_head_kv=4, n_ff=512, n_expert=0, n_expert_used=0,
                context=16384, arch_name=None):
    rng = np.random.default_rng(SEED)
    tokens, merges = bpe_vocab()
    table = byte_unicode()
    tokens += ["".join(table[b] for b in w.encode()) for w in CHAIN_WORDS
               if "".join(table[b] for b in w.encode()) not in tokens]
    tokens = tokens + SPECIALS
    types = [gguf.TokenType.NORMAL] * (len(tokens) - len(SPECIALS)) + [gguf.TokenType.CONTROL] * len(SPECIALS)
    n_vocab = len(tokens)
    name = arch_name or arch
    w = gguf.GGUFWriter(str(path), name)
    w.add_name(f"tiny-{arch}")
    w.add_context_length(context)
    w.add_embedding_length(n_embd)
    w.add_block_count(n_layer)
    w.add_feed_forward_length(n_ff)
    w.add_head_count(n_head)
    w.add_head_count_kv(n_head_kv)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_rope_freq_base(10000.0)
    w.add_rope_dimension_count(n_embd // n_head)
    w.add_vocab_size(n_vocab)
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    if n_expert:
        w.add_expert_count(n_expert)
        w.add_expert_used_count(n_expert_used)
        w.add_expert_feed_forward_length(n_ff)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("default")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    w.add_bos_token_id(tokens.index("<|endoftext|>"))
    w.add_eos_token_id(tokens.index("<|im_end|>"))
    w.add_add_bos_token(False)
    w.add_chat_template(CHATML)

    def weight(*shape, scale=0.02):  # numpy shape is ggml's dimensions reversed
        return (rng.standard_normal(shape) * scale).astype(np.float16)

    def norm(n):
        return np.ones(n, dtype=np.float32)

    head = n_embd // n_head
    assert n_vocab <= n_embd, "one-hot embeddings need n_embd >= vocabulary size"
    embd = rng.standard_normal((n_vocab, n_embd)) * 0.02
    embd[np.arange(n_vocab), np.arange(n_vocab)] = 100.0  # the residual stream stays dominated by the current token
    out = rng.standard_normal((n_vocab, n_embd)) * 0.01
    for current, following in next_token_table(tokens).items():
        out[following, current] += 0.5
    w.add_tensor("token_embd.weight", embd.astype(np.float16))
    w.add_tensor("output_norm.weight", norm(n_embd))
    w.add_tensor("output.weight", out.astype(np.float16))
    for i in range(n_layer):
        p = f"blk.{i}."
        w.add_tensor(p + "attn_norm.weight", norm(n_embd))
        w.add_tensor(p + "attn_q.weight", weight(n_head * head, n_embd))
        w.add_tensor(p + "attn_k.weight", weight(n_head_kv * head, n_embd))
        w.add_tensor(p + "attn_v.weight", weight(n_head_kv * head, n_embd))
        w.add_tensor(p + "attn_output.weight", weight(n_embd, n_head * head))
        w.add_tensor(p + "ffn_norm.weight", norm(n_embd))
        if n_expert:
            w.add_tensor(p + "attn_q_norm.weight", norm(head))
            w.add_tensor(p + "attn_k_norm.weight", norm(head))
            w.add_tensor(p + "ffn_gate_inp.weight", weight(n_expert, n_embd).astype(np.float32))
            w.add_tensor(p + "ffn_gate_exps.weight", weight(n_expert, n_ff, n_embd))
            w.add_tensor(p + "ffn_up_exps.weight", weight(n_expert, n_ff, n_embd))
            w.add_tensor(p + "ffn_down_exps.weight", weight(n_expert, n_embd, n_ff))
        else:
            w.add_tensor(p + "ffn_gate.weight", weight(n_ff, n_embd))
            w.add_tensor(p + "ffn_up.weight", weight(n_ff, n_embd))
            w.add_tensor(p + "ffn_down.weight", weight(n_embd, n_ff))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def run(argv):
    print("$", " ".join(map(str, argv)), flush=True)
    subprocess.run([str(a) for a in argv], check=True, timeout=600, stdout=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir")
    parser.add_argument("--bin", help="llama.cpp bin directory (for llama-quantize and llama-gguf-split)")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dense = write_model(out / "tiny-llama-F16.gguf", "llama")
    moe = write_model(out / "tiny-qwen3moe-F16.gguf", "qwen3moe", n_ff=256, n_expert=8, n_expert_used=2)
    write_model(out / "unknown-arch.gguf", "llama", n_layer=1, arch_name="notarealarch")
    data = dense.read_bytes()
    (out / "corrupt.gguf").write_bytes(data[: len(data) // 2])
    files = {"tiny-llama-F16.gguf": dense.stat().st_size, "tiny-qwen3moe-F16.gguf": moe.stat().st_size}
    if args.bin:
        bin_dir = Path(args.bin)
        for source, target, kind in [(dense, "tiny-llama-Q8_0.gguf", "Q8_0"), (dense, "tiny-llama-Q4_K_M.gguf", "Q4_K_M"),
                                     (moe, "tiny-qwen3moe-Q4_K_M.gguf", "Q4_K_M")]:
            run([bin_dir / "llama-quantize", source, out / target, kind])
            files[target] = (out / target).stat().st_size
        split = out / "split"
        split.mkdir(exist_ok=True)
        for old in split.glob("*.gguf"):
            old.unlink()
        run([bin_dir / "llama-gguf-split", "--split", "--split-max-tensors", "12", out / "tiny-llama-Q8_0.gguf",
             split / "tiny-llama-Q8_0"])
        for shard in sorted(split.glob("*.gguf")):
            files["split/" + shard.name] = shard.stat().st_size
    (out / "manifest.json").write_text(json.dumps({"seed": SEED, "files": files}, indent=2) + "\n")
    print(out.resolve())


if __name__ == "__main__":
    main()
