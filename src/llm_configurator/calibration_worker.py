"""Isolated synthetic probes. No model downloads, shell commands or persisted tensors."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import json
import os
import statistics
import time

MIB = 1024**2


def rate(operation, amount, duration=0.25):
    operation()  # first-touch / warm-up excluded
    samples = []
    deadline = time.perf_counter() + duration
    while len(samples) < 3 or time.perf_counter() < deadline:
        start = time.perf_counter()
        operation()
        elapsed = time.perf_counter() - start
        if elapsed > 0:
            samples.append(amount / elapsed)
        if len(samples) >= 100:
            break
    return statistics.median(samples)


def cpu_probe():
    import numpy as np
    import psutil
    available = psutil.virtual_memory().available
    size = min(128 * MIB, int(max(0, available - 512 * MIB) / 8))
    size = size // MIB * MIB
    if size < 32 * MIB:
        raise ValueError("Not enough free RAM for calibration")
    workers = min(8, psutil.cpu_count(logical=False) or 1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        source = np.full(size, 73, dtype=np.uint8)
        target = np.empty_like(source)
        chunks = [(lo, min(size, lo + (size + workers - 1) // workers))
                  for lo in range(0, size, (size + workers - 1) // workers)]
        def copy_chunk(bounds):
            lo, hi = bounds
            np.copyto(target[lo:hi], source[lo:hi])
        bandwidth = rate(lambda: list(pool.map(copy_chunk, chunks)), 2 * size)
        del source, target
        rows, cols = min(4096, size // (4096 * 8)), 4096
        packed = np.full((rows, cols // 2), 0x79, dtype=np.uint8)
        matrix = np.empty((rows, cols), dtype=np.float32)
        scratch = np.empty_like(packed)
        vector = np.full(cols, 0.125, dtype=np.float32)
        output = np.empty(rows, dtype=np.float32)
        chunks = [(lo, min(rows, lo + (rows + workers - 1) // workers))
                  for lo in range(0, rows, (rows + workers - 1) // workers)]
        def q4_chunk(bounds):
            lo, hi = bounds
            np.bitwise_and(packed[lo:hi], 15, out=scratch[lo:hi])
            matrix[lo:hi, 0::2] = scratch[lo:hi]
            np.right_shift(packed[lo:hi], 4, out=scratch[lo:hi])
            matrix[lo:hi, 1::2] = scratch[lo:hi]
            matrix[lo:hi] -= 8
            matrix[lo:hi] *= 0.125
            np.matmul(matrix[lo:hi], vector, out=output[lo:hi])
        parameters_s = rate(lambda: list(pool.map(q4_chunk, chunks)), rows * cols)
        if not np.isfinite(output).all():
            raise ValueError("Synthetic matrix test failed")
    return {"ram_bytes_s": bandwidth, "q4_parameters_s": parameters_s,
            "buffer_bytes": size, "matrix_shape": [rows, cols], "threads": workers,
            "method": "NumPy copy and unpacked Q4 matrix-vector proxy; not llama.cpp"}


def gpu_probes():
    """CUDA driver API only: no CUDA toolkit or PyTorch dependency."""
    library = C.WinDLL("nvcuda.dll") if os.name == "nt" else C.CDLL("libcuda.so.1")
    def bind(name, args):
        fn = getattr(library, name)
        fn.argtypes, fn.restype = args, C.c_int
        def call(*values):
            code = fn(*values)
            if code:
                raise ValueError(f"CUDA probe {name} returned {code}")
        return call
    pointer, size_t, deviceptr = C.c_void_p, C.c_size_t, C.c_uint64
    init = bind("cuInit", [C.c_uint])
    count = bind("cuDeviceGetCount", [C.POINTER(C.c_int)])
    uuid = bind("cuDeviceGetUuid", [pointer, C.c_int])
    retain = bind("cuDevicePrimaryCtxRetain", [C.POINTER(pointer), C.c_int])
    release = bind("cuDevicePrimaryCtxRelease_v2", [C.c_int])
    current = bind("cuCtxSetCurrent", [pointer])
    memory = bind("cuMemGetInfo_v2", [C.POINTER(size_t), C.POINTER(size_t)])
    alloc = bind("cuMemAlloc_v2", [C.POINTER(deviceptr), size_t])
    free = bind("cuMemFree_v2", [deviceptr])
    host_alloc = bind("cuMemAllocHost_v2", [C.POINTER(pointer), size_t])
    host_free = bind("cuMemFreeHost", [pointer])
    h2d = bind("cuMemcpyHtoD_v2", [deviceptr, pointer, size_t])
    d2h = bind("cuMemcpyDtoH_v2", [pointer, deviceptr, size_t])
    d2d = bind("cuMemcpyDtoD_v2", [deviceptr, deviceptr, size_t])
    sync = bind("cuCtxSynchronize", [])
    init(0)
    n = C.c_int()
    count(C.byref(n))
    results, warnings = {}, []
    for index in range(min(n.value, 4)):
        ctx, host, a, b = pointer(), pointer(), deviceptr(), deviceptr()
        try:
            raw = (C.c_ubyte * 16)()
            uuid(raw, index)
            value = bytes(raw).hex()
            key = 'GPU-' + '-'.join([value[:8], value[8:12], value[12:16], value[16:20], value[20:]])
            retain(C.byref(ctx), index)
            current(ctx)
            available, total = size_t(), size_t()
            memory(C.byref(available), C.byref(total))
            size = min(128 * MIB, max(0, available.value - 256 * MIB) // 8)
            if size < 32 * MIB:
                raise ValueError("Not enough free VRAM for calibration")
            alloc(C.byref(a), size)
            alloc(C.byref(b), size)
            host_alloc(C.byref(host), size)
            C.memset(host, 73, size)
            h2d(a, host, size)
            def copied():
                d2d(b, a, size)
                sync()
            ram_to_gpu = rate(lambda: h2d(a, host, size), size, 0.15)
            gpu_to_ram = rate(lambda: d2h(host, a, size), size, 0.15)
            results[key] = {"vram_bytes_s": rate(copied, 2 * size, 0.15),
                            "h2d_bytes_s": ram_to_gpu, "d2h_bytes_s": gpu_to_ram,
                            "transfer_latency_s": 1 / rate(lambda: h2d(a, host, 4096), 1, 0.1),
                            "buffer_bytes": size,
                            "method": "CUDA driver copy proxy; inference compute throughput unmeasured"}
        except (ValueError, OSError) as error:
            warnings.append(f"GPU {index}: {error}")
        finally:
            for fn, argument in [(free, a), (free, b), (host_free, host), (release, index if ctx.value else None)]:
                if argument is not None and (not hasattr(argument, 'value') or argument.value):
                    try:
                        fn(argument)
                    except (ValueError, OSError):
                        pass
    return results, warnings


if __name__ == "__main__":
    result = {"cpu": None, "gpus": {}, "warnings": []}
    try:
        result["cpu"] = cpu_probe()
    except Exception as error:
        result["warnings"].append(f"CPU calibration unavailable: {type(error).__name__}: {error}")
    # Emit each completed phase so a slow driver cannot discard a valid CPU probe.
    print(json.dumps(result, allow_nan=False), flush=True)
    try:
        result["gpus"], warnings = gpu_probes()
        result["warnings"].extend(warnings)
    except (OSError, ValueError, AttributeError) as error:
        result["warnings"].append(f"GPU calibration unavailable: {type(error).__name__}")
    print(json.dumps(result, allow_nan=False), flush=True)
