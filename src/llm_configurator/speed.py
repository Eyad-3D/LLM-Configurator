"""Unvalidated hardware-calibrated generation ranges, never verified speed gates.

Also holds the shared memory geometry (conversation-memory size, expert share) so the
memory model and the speed model can never disagree about how many bytes exist.
"""
import math

from .calibration import positive, valid
from .domain import GIB, KV_BYTES_PER_ELEMENT, QUANT_BYTES_PER_PARAMETER
from .launch import runtime_gpu_layers

# Kept for callers of v0.3; the domain table is the single source of truth.
BYTES_PER_PARAMETER = QUANT_BYTES_PER_PARAMETER
# llama-server's default micro-batch; a sliding-window cache holds the window plus one micro-batch per user.
DEFAULT_UBATCH = 512
SLIDING_MARGIN_TOKENS = DEFAULT_UBATCH  # kept for v0.3 callers
CELL_PAD = 256  # llama.cpp rounds every per-user cache up to a multiple of 256 cells


def _pad(tokens):
    return -(-tokens // CELL_PAD) * CELL_PAD


def kv_bytes(variant, context, users, kv_cache_type="f16", ubatch=None):
    """K + V bytes for all users, sized the way llama-server allocates them.

    `launch` passes `-c context*users -np users`, so every user gets a separate cache of
    pad256(context) cells; a sliding-window layer holds pad256(min(context, window + ubatch))
    per user (checked against llama.cpp's own `llama_kv_cache: size = …` log lines).
    """
    per_token_layer = 2 * variant.kv_heads * variant.head_dim * KV_BYTES_PER_ELEMENT[kv_cache_type]
    sliding = variant.sliding_layers if variant.sliding_window else 0
    cells = _pad(context)
    full = (variant.layers - sliding) * cells * users
    window = _pad(min(cells, variant.sliding_window + (ubatch or DEFAULT_UBATCH))) if sliding else 0
    return math.ceil(per_token_layer * (full + sliding * window * users))


def gpu_blocks(variant, gpu_layers):
    """(transformer blocks on the GPU, output layer on the GPU) for our gpu_layers setting.

    llama.cpp counts the output layer as one more layer after the last block: `-ngl n`
    offloads the output layer plus the last n-1 blocks. `launch.runtime_gpu_layers` decides n.
    """
    n = runtime_gpu_layers(gpu_layers, variant.layers)
    return min(max(n - 1, 0), variant.layers), n >= 1


def output_bytes(variant):
    """Allowance for the output layer (vocabulary × width), which is usually bigger than one block.

    The catalogue has no vocabulary size, so this is a bounded share of the file: large enough
    for the tied 262k-token output of Gemma-3-27B (≈1.2 GB) and never below one average block.
    """
    return max(variant.size_bytes / variant.layers, min(0.15 * variant.size_bytes, 2 * GIB))


def moved_expert_layers(variant, gpu_layers, n_cpu_moe):
    """GPU blocks whose expert tensors `--n-cpu-moe` keeps in RAM.

    llama.cpp pins the experts of blocks 0..n-1 to the CPU and offloads the *last*
    blocks, so only GPU blocks in [layers - on_gpu, n_cpu_moe) actually move.
    """
    if not variant.moe or not n_cpu_moe:
        return 0
    on_gpu = gpu_blocks(variant, gpu_layers)[0]
    return max(0, min(on_gpu, n_cpu_moe - (variant.layers - on_gpu)))


def active_fraction(variant):
    """Share of weight bytes read per generated token: dense part plus the routed experts in use."""
    if not variant.moe or not variant.experts:
        return 1.0
    return (1 - variant.expert_fraction) + variant.expert_fraction * variant.active_experts / variant.experts


def estimate(variant, hardware, calibration, context, layers, users, gpu, target,
             kv_cache_type="f16", n_cpu_moe=0, unified=False, efficiency=None):
    missing = lambda reason: {'available': False, 'reason': reason}
    if variant.demo:
        return missing('Demo models have no speed estimate.')
    if users != 1:
        return missing('Concurrent throughput requires a batching-aware benchmark; no per-session estimate yet.')
    if not valid(calibration, hardware):
        return missing('Hardware calibration is missing, expired or belongs to different hardware.')
    bp = QUANT_BYTES_PER_PARAMETER.get(variant.quant)
    if bp is None and not variant.parameters:
        return missing('Quantization is not supported by the estimator.')
    total = variant.layers
    moved = moved_expert_layers(variant, layers, n_cpu_moe)
    # One decode step reads the active weights and the whole conversation memory once.
    share = active_fraction(variant)
    layer_bytes = variant.size_bytes / total
    expert_active = layer_bytes * variant.expert_fraction * (variant.active_experts / variant.experts if variant.moe else 0)
    kv_layer = kv_bytes(variant, context, 1, kv_cache_type) / total
    on_gpu = gpu_blocks(variant, layers)[0]
    gpu_traffic = on_gpu * (layer_bytes * share + kv_layer) - moved * expert_active
    cpu_traffic = (total - on_gpu) * (layer_bytes * share + kv_layer) + moved * expert_active
    parameters = variant.active_parameters or (variant.parameters or variant.size_bytes / bp) * share
    cpu_share = cpu_traffic / (cpu_traffic + gpu_traffic)
    cpu = calibration.get('cpu') or {}
    slow, fast = 0.0, 0.0
    methods = []
    if cpu_traffic > 0:
        bandwidth, throughput = cpu.get('ram_bytes_s'), cpu.get('q4_parameters_s')
        if not positive(bandwidth) or not positive(throughput):
            return missing('CPU memory and matrix calibration are unavailable.')
        # Proxy sensitivity envelope, not a statistical confidence interval.
        memory_time = cpu_traffic / bandwidth
        compute_time = parameters * cpu_share / throughput
        slow += max(memory_time / 0.35, compute_time / 1.0)
        fast += memory_time / 0.8  # optimized fused kernels may greatly outperform our unpacking proxy
        methods.append('CPU memory + synthetic Q4 matrix proxy')
    if layers:
        measured = calibration.get('gpus', {}).get(gpu['uuid'] if gpu else '', {})
        bandwidth = measured.get('vram_bytes_s')
        if not positive(bandwidth):
            return missing('The graphics chip has no memory-speed calibration yet.' if unified
                           else 'Selected GPU has no VRAM bandwidth calibration.')
        gpu_time = gpu_traffic / bandwidth
        # Broad bandwidth-only envelope; no measured GPU inference kernel efficiency.
        slow += gpu_time / 0.10
        fast += gpu_time / 0.55
        methods.append('GPU bandwidth proxy; compute unmeasured')
        crossings = 2 * moved + (2 if layers < total else 0)
        if crossings and not unified:
            link = min(measured.get('h2d_bytes_s', 0), measured.get('d2h_bytes_s', 0))
            latency = measured.get('transfer_latency_s')
            if not positive(link) or not positive(latency):
                return missing('CPU–GPU transfer calibration is unavailable.')
            # Activations, not weights, cross the link: once per split point and twice per CPU-expert layer.
            overhead = crossings * (max(4096, variant.kv_heads * variant.head_dim * 4) / link + latency)
            slow += overhead * 4
            fast += overhead
    if not positive(slow) or not positive(fast):
        return missing('Calibration cannot produce a finite estimate.')
    low, high = 1 / slow, 1 / fast
    result = {'available': True, 'low_tps': round(low, 2), 'high_tps': round(high, 2), 'mid_tps': math.sqrt(low * high),
              'confidence': 'low', 'method': '; '.join(methods), 'calibrated_at': calibration['timestamp'],
              'scope': 'Single-session token generation at the selected context; not prompt processing or TTFT.',
              'caveat': 'Unvalidated heuristic range. Synthetic probes are not llama.cpp kernels; actual speed can fall outside this range.'}
    if efficiency:
        mode = 'cpu' if layers == 0 else 'gpu' if layers == total and not moved else 'split'
        fit = (efficiency.get('by_mode') or {}).get(mode)
        spread = fit['spread'] if fit else None
        if not fit:
            # No tests in this placement: shift by the machine-wide factor but keep the full raw width.
            fit, spread = efficiency, max(efficiency.get('spread') or 0, math.log(high / low) / 2)
        if fit.get('factor'):
            middle = math.sqrt(low * high) * fit['factor']
            low, high = middle * math.exp(-spread), middle * math.exp(spread)
            result.update(low_tps=round(low, 2), high_tps=round(high, 2), mid_tps=middle, raw_low_tps=round(1 / slow, 2), raw_high_tps=round(1 / fast, 2),
                          adjusted={'n': fit['n'], 'factor': round(fit['factor'], 3)},
                          method=result['method'] + f"; adjusted from {fit['n']} local measurements",
                          caveat='Range corrected using speed tests on this computer; still an estimate for this exact setup.')
    result['target_status'] = 'likely_meets' if low >= target else 'likely_below' if high < target else 'borderline'
    return result
