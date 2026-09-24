#!/usr/bin/env python3
"""Run the real llama.cpp tools against the tiny models and save their raw output.

Usage: python3 scripts/capture_llama_facts.py BIN_DIR MODELS_DIR OUT_DIR

OUT_DIR (normally tests/integration/samples) receives small text/JSON files that
docs/v0.4/llama-cpp-facts.md quotes and that the fake llama.cpp can be checked
against. Nothing here is interpreted: it records what the binaries really print.
"""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

BIN, MODELS, OUT = (Path(a) for a in sys.argv[1:4])
OUT.mkdir(parents=True, exist_ok=True)
F16 = MODELS / "tiny-llama-F16.gguf"
Q8 = MODELS / "tiny-llama-Q8_0.gguf"
Q4 = MODELS / "tiny-llama-Q4_K_M.gguf"
MOE = MODELS / "tiny-qwen3moe-Q4_K_M.gguf"
CORPUS = Path(__file__).with_name("tiny_corpus.txt")


def tool(name):
    return str(BIN / name)


def save(name, text):
    (OUT / name).write_text(text if text.endswith("\n") else text + "\n")
    print("wrote", OUT / name)


def run(argv, timeout=300):
    done = subprocess.run([str(a) for a in argv], capture_output=True, text=True, timeout=timeout, errors="replace")
    return done.returncode, done.stdout, done.stderr


def record(argv, timeout=300):
    """Full transcript of one command: argv, exit code, stdout, stderr."""
    code, out, err = run(argv, timeout)
    shown = [Path(str(a)).name if str(a).startswith(("/", ".")) and "/" in str(a) else str(a) for a in argv]
    return f"$ {' '.join(shown)}\n# exit code: {code}\n# ---- stdout ----\n{out}# ---- stderr ----\n{err}"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except OSError as error:
        return None, f"{type(error).__name__}: {error}"


def versions():
    text = []
    for name in ["llama-server", "llama-bench", "llama-perplexity", "llama-cli", "llama-quantize", "llama-gguf-split"]:
        text.append(record([tool(name), "--version"], 30))
    save("version.txt", "\n".join(text))
    for name in ["llama-server", "llama-bench", "llama-perplexity"]:
        code, out, err = run([tool(name), "--help"], 30)
        save(f"help-{name}.txt", out + err)


def server():
    port = free_port()
    log = OUT / "server-log.txt"
    argv = [tool("llama-server"), "-m", F16, "-c", "2048", "-np", "1", "-ngl", "0", "--host", "127.0.0.1", "--jinja",
            "--port", str(port), "-t", "2", "--alias", "tiny"]
    base = f"http://127.0.0.1:{port}"
    health = []
    with log.open("w") as stream:
        process = subprocess.Popen([str(a) for a in argv], stdout=stream, stderr=subprocess.STDOUT)
        try:
            start = time.monotonic()
            while time.monotonic() - start < 60:
                status, body = http("GET", base + "/health", timeout=5)
                if not health or health[-1][1:] != (status, body):
                    health.append((round(time.monotonic() - start, 3), status, body))
                if status == 200:
                    break
                time.sleep(0.02)
            save("server-health.json", json.dumps([{"t": t, "status": s, "body": b} for t, s, b in health], indent=2))
            chat = {"messages": [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
                    "max_tokens": 16, "temperature": 0.0, "seed": 1}
            status, body = http("POST", base + "/v1/chat/completions", chat)
            save("server-chat-completions.json", json.dumps({"request": chat, "status": status, "response": json.loads(body)}, indent=2))
            stream_chat = {**chat, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 4}
            status, body = http("POST", base + "/v1/chat/completions", stream_chat)
            save("server-chat-completions-stream.txt", f"# status {status}\n{body}")
            completion = {"prompt": "The capital of France is", "n_predict": 8, "temperature": 0.0, "seed": 1}
            status, body = http("POST", base + "/completion", completion)
            save("server-completion.json", json.dumps({"request": completion, "status": status, "response": json.loads(body)}, indent=2))
            status, body = http("POST", base + "/tokenize", {"content": "Hello world"})
            save("server-tokenize.json", json.dumps({"request": {"content": "Hello world"}, "status": status, "response": json.loads(body)}, indent=2))
            nocache = {**chat, "cache_prompt": False}
            status, body = http("POST", base + "/v1/chat/completions", nocache)
            save("server-chat-no-cache.json", json.dumps({"request": nocache, "status": status,
                                                          "timings": json.loads(body).get("timings")}, indent=2))
            long = {"messages": [{"role": "user", "content": "hello " * 3000}], "max_tokens": 4}
            status, body = http("POST", base + "/v1/chat/completions", long)
            save("server-error-context-exceeded.json", json.dumps({"request": "3000 x 'hello ' with -c 2048",
                                                                   "status": status, "response": json.loads(body)}, indent=2))
            status, body = http("GET", base + "/v1/models")
            save("server-models.json", json.dumps({"status": status, "response": json.loads(body)}, indent=2))
            status, body = http("GET", base + "/props")
            props = json.loads(body)
            props.pop("chat_template", None)
            save("server-props.json", json.dumps({"status": status, "response": props}, indent=2))
            # A second server on the same port shows the "port in use" failure.
            save("error-server-port-in-use.txt", record(argv, 60))
        finally:
            process.terminate()
            process.wait(20)


def server_errors():
    base = [tool("llama-server"), "-c", "512", "-ngl", "0", "--host", "127.0.0.1", "--port", str(free_port())]
    save("error-server-missing-file.txt", record(base + ["-m", MODELS / "does-not-exist.gguf"], 60))
    save("error-server-corrupt.txt", record(base + ["-m", MODELS / "corrupt.gguf"], 60))
    save("error-server-unknown-arch.txt", record(base + ["-m", MODELS / "unknown-arch.gguf"], 60))
    save("error-server-bad-flag.txt", record(base + ["-m", F16, "--no-such-flag"], 60))
    save("error-server-bad-fa-value.txt", record(base + ["-m", F16, "-fa", "yes"], 60))
    save("error-server-bad-cache-type.txt", record(base + ["-m", F16, "-ctk", "q9_9"], 60))


def bench():
    common = [tool("llama-bench"), "-m", F16, "-r", "1", "-o", "json"]
    cases = {
        "bench-basic.json": ["-p", "64", "-n", "16", "-t", "2"],
        "bench-sweep.json": ["-p", "32", "-n", "8", "-t", "1,2", "-fa", "0,1", "-b", "64,128", "-ub", "32"],
        "bench-depth.json": ["-p", "32", "-n", "8", "-d", "0,256", "-t", "2", "-ctk", "q8_0", "-ctv", "q8_0", "-fa", "1"],
        "bench-dev-none.json": ["-p", "32", "-n", "8", "-t", "2", "-ngl", "99", "-dev", "none"],
    }
    for name, extra in cases.items():
        save(name, record(common + extra))
    save("bench-moe-ncmoe.json", record([tool("llama-bench"), "-m", MOE, "-r", "1", "-o", "json", "-p", "32", "-n", "8",
                                         "-t", "2", "-ncmoe", "0,2"]))
    save("bench-jsonl.txt", record(common + ["-p", "32", "-n", "8", "-t", "2", "-o", "jsonl"]))
    save("bench-fa-words.txt", record(common + ["-p", "32", "-n", "8", "-t", "2", "-fa", "on,off,auto"]))
    save("error-bench-missing-file.txt", record([tool("llama-bench"), "-m", MODELS / "does-not-exist.gguf", "-o", "json"]))
    save("error-bench-unknown-arch.txt", record([tool("llama-bench"), "-m", MODELS / "unknown-arch.gguf", "-o", "json", "-p", "8", "-n", "4"]))


def perplexity():
    base = [tool("llama-perplexity"), "-f", CORPUS, "-c", "128", "-b", "128", "-t", "2", "--chunks", "4"]
    save("perplexity-normal.txt", record(base + ["-m", Q8]))
    logits = OUT.parent / "kld-base.bin"
    save("perplexity-kld-base.txt", record(base + ["-m", Q8, "--kl-divergence-base", logits]))
    save("perplexity-kld.txt", record([tool("llama-perplexity"), "-m", Q4, "-t", "2", "--kl-divergence-base", logits,
                                       "--kl-divergence"]))
    save("perplexity-kld-bytes.txt", f"kl-divergence-base file: {logits.stat().st_size} bytes for 4 chunks of 128 tokens\n")
    logits.unlink()


if __name__ == "__main__":
    wanted = sys.argv[4:] or ["versions", "server", "server_errors", "bench", "perplexity"]
    for step in wanted:
        globals()[step]()
