"""Unvalidated hardware-calibrated generation ranges, never verified speed gates."""
from .calibration import positive, valid

# File bytes / parameter approximations include typical GGUF overhead/mixed precision.
BYTES_PER_PARAMETER = {'Q4_K_M': 0.62, 'Q5_K_M': 0.73, 'Q6_K': 0.84, 'Q8_0': 1.1}


def estimate(variant, hardware, calibration, context, layers, users, gpu, target):
    missing = lambda reason: {'available': False, 'reason': reason}
    if variant.demo:
        return missing('Demo models have no speed estimate.')
    if users != 1:
        return missing('Concurrent throughput requires a batching-aware benchmark; no per-session estimate yet.')
    if not valid(calibration, hardware):
        return missing('Hardware calibration is missing, expired or belongs to different hardware.')
    bp = BYTES_PER_PARAMETER.get(variant.quant)
    if bp is None:
        return missing('Quantization is not supported by the estimator.')
    fraction = layers / variant.layers
    # One decode step reads dense weights and the existing FP16 key/value cache.
    kv = 2 * variant.layers * variant.kv_heads * variant.head_dim * 2 * context
    traffic = variant.size_bytes + kv
    parameters = variant.size_bytes / bp
    cpu = calibration.get('cpu') or {}
    slow, fast = 0.0, 0.0
    methods = []
    if fraction < 1:
        bandwidth, throughput = cpu.get('ram_bytes_s'), cpu.get('q4_parameters_s')
        if not positive(bandwidth) or not positive(throughput):
            return missing('CPU memory and matrix calibration are unavailable.')
        # Proxy sensitivity envelope, not a statistical confidence interval.
        memory_time = traffic * (1 - fraction) / bandwidth
        compute_time = parameters * (1 - fraction) / throughput
        slow += max(memory_time / 0.35, compute_time / 1.0)
        fast += memory_time / 0.8  # optimized fused kernels may greatly outperform our unpacking proxy
        methods.append('CPU memory + synthetic Q4 matrix proxy')
    if fraction:
        measured = calibration.get('gpus', {}).get(gpu['uuid'] if gpu else '', {})
        bandwidth = measured.get('vram_bytes_s')
        if not positive(bandwidth):
            return missing('Selected GPU has no VRAM bandwidth calibration.')
        gpu_time = traffic * fraction / bandwidth
        # Broad bandwidth-only envelope; no measured GPU inference kernel efficiency.
        slow += gpu_time / 0.10
        fast += gpu_time / 0.55
        methods.append('GPU bandwidth proxy; compute unmeasured')
        if fraction < 1:
            link = min(measured.get('h2d_bytes_s', 0), measured.get('d2h_bytes_s', 0))
            latency = measured.get('transfer_latency_s')
            if not positive(link) or not positive(latency):
                return missing('CPU–GPU transfer calibration is unavailable.')
            # Approximate one contiguous layer split: activations, not all weights, cross the link.
            overhead = 2 * (max(4096, variant.kv_heads * variant.head_dim * 4) / link + latency)
            slow += overhead * 4
            fast += overhead
    if not positive(slow) or not positive(fast):
        return missing('Calibration cannot produce a finite estimate.')
    low, high = 1 / slow, 1 / fast
    return {'available': True, 'low_tps': round(low, 2), 'high_tps': round(high, 2),
            'confidence': 'low', 'target_status': 'likely_meets' if low >= target else 'likely_below' if high < target else 'borderline',
            'method': '; '.join(methods), 'calibrated_at': calibration['timestamp'],
            'scope': 'Single-session token generation at the selected context; not prompt processing or TTFT.',
            'caveat': 'Unvalidated heuristic range. Synthetic probes are not llama.cpp kernels; actual speed can fall outside this range.'}
