#!/usr/bin/env bash
# Build a real CPU-only llama.cpp from the llama.cpp sources vendored inside the
# llama-cpp-python sdist on PyPI (works where GitHub is unreachable).
#
# Usage: scripts/build_llama_cpp.sh [WORK_DIR]
#   WORK_DIR defaults to $LLAMA_WORK_DIR or /opt/llama-work (outside the repo).
#   LLAMA_CPP_PYTHON_VERSION pins the sdist version (default 0.3.35).
# Idempotent: re-running skips the download/extract and only rebuilds what changed.
# The last line printed is the directory holding the binaries; point
# LLM_CONFIG_REAL_RUNTIME at it.
set -euo pipefail

WORK="${1:-${LLAMA_WORK_DIR:-/opt/llama-work}}"
VERSION="${LLAMA_CPP_PYTHON_VERSION:-0.3.35}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
mkdir -p "$WORK"
cd "$WORK"

SDIST="llama_cpp_python-${VERSION}.tar.gz"
if [ ! -f "$SDIST" ]; then
  python3 -m pip download --no-binary :all: --no-deps "llama-cpp-python==${VERSION}" -d "$WORK" >&2
fi
SRC="$WORK/llama_cpp_python-${VERSION}/vendor/llama.cpp"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  tar xzf "$SDIST" -C "$WORK"
fi

command -v cmake >/dev/null || python3 -m pip install cmake >&2
GEN=()
command -v ninja >/dev/null && GEN=(-G Ninja)

BUILD="$SRC/build"
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  cmake -S "$SRC" -B "$BUILD" "${GEN[@]}" -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF >&2
fi
cmake --build "$BUILD" --config Release -j "$JOBS" --target \
  llama-server llama-bench llama-perplexity llama-cli llama-gguf-split llama-quantize >&2

BIN="$BUILD/bin"
for tool in llama-server llama-bench llama-perplexity llama-cli llama-gguf-split llama-quantize; do
  [ -x "$BIN/$tool" ] || { echo "missing $BIN/$tool" >&2; exit 1; }
done
( cd "$SRC" && echo "llama.cpp commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)" ) >&2
"$BIN/llama-server" --version >&2 || true
echo "$BIN"
