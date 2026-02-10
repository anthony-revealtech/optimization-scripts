"""Shared comparison helpers: file size, inference speed, memory, KS test, JS divergence.

Verification summary (all tests computed as below):
- File size: os.path.getsize (bytes); ratio = quant / fp32.
- Memory: RSS delta from baseline (mem_after - mem_before) per trial; mean ± std.
- Speed: mean(latency_per_run) over trials; latency = elapsed / num_inference_runs (seconds); speedup = fp32_ms / quant_ms.
- KS test: scipy.stats.ks_2samp(original_weights, quantized_weights). For QDQ, quantized = dequantized weights only (scale*(x-zp)).
- JS divergence: histogram both weight arrays into n_bins, normalize to probs p,q; jensenshannon(p,q,base=2); 0=identical, 1=max.
- Cosine similarity: dot(a,b)/(||a||*||b||) on flattened outputs; 1=identical direction.
- SQNR (dB): 10*log10(var(signal)/var(noise)) with signal=fp32 output, noise=fp32-quant; higher = less quantization error.
"""
import gc
import numpy as np
import onnx
import os
from scipy import stats
from scipy.spatial.distance import jensenshannon
from time import time

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Quantized (QDQ) model speed: On CPUs without Intel VNNI, QDQ adds QuantizeLinear/DequantizeLinear
# overhead and can be 2–3x slower than FP32. Parallel execution of graph nodes can reduce overhead.
USE_PARALLEL_FOR_QUANT = True   # use ORT_PARALLEL for quant session
QUANT_INTER_OP_THREADS = 4      # threads for parallel graph execution; 0 = default; increase if quant still slow


def _session_options(for_quantized=False):
    """Build SessionOptions: full graph optimization; optional parallel execution for quantized model."""
    import onnxruntime as ort
    opts = ort.SessionOptions()
    try:
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    except AttributeError:
        pass  # use default if enum missing
    if for_quantized and USE_PARALLEL_FOR_QUANT:
        try:
            opts.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if QUANT_INTER_OP_THREADS > 0:
                opts.inter_op_num_threads = QUANT_INTER_OP_THREADS
        except AttributeError:
            pass
    return opts


def get_memory_mb():
    """Return current process RSS in MB, or 0.0 if psutil is not available."""
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def measure_ram_usage(
    model_fp32_path,
    model_quant_path,
    num_trials=5,
    warmup=5,
    dynamic_size=1,
    feed=None,
):
    """Measure and print RAM usage (RSS delta) for loading and running FP32 vs quantized models.

    Args:
        model_fp32_path: Path to FP32 ONNX model.
        model_quant_path: Path to quantized ONNX model.
        num_trials: Number of trials for mean ± std (default 5).
        warmup: Warmup runs before measuring inference RAM (default 5).
        dynamic_size: Value for dynamic dimensions when building dummy inputs (default 1).
        feed: Optional pre-built feed dict. If None, built from model_fp32_path and dynamic_size.

    Returns:
        Dict with keys fp32_load_mb, fp32_infer_mb, quant_load_mb, quant_infer_mb (each list of floats),
        or None if psutil is not available.
    """
    import onnxruntime as ort

    if not HAS_PSUTIL:
        print("  (install psutil for memory stats: pip install psutil)")
        return None

    model_proto = onnx.load(model_fp32_path)
    if feed is None:
        feed = get_dummy_inputs(model_proto, dynamic_size=dynamic_size)
    opts_fp32 = _session_options(for_quantized=False)
    opts_quant = _session_options(for_quantized=True)

    fp32_load_mb = []
    fp32_infer_mb = []
    quant_load_mb = []
    quant_infer_mb = []

    for _ in range(num_trials):
        gc.collect()
        mem_before = get_memory_mb()
        s = ort.InferenceSession(model_fp32_path, sess_options=opts_fp32, providers=["CPUExecutionProvider"])
        mem_after_load = get_memory_mb()
        for _ in range(warmup):
            s.run(None, feed)
        s.run(None, feed)
        mem_after_infer = get_memory_mb()
        del s
        gc.collect()
        fp32_load_mb.append(mem_after_load - mem_before)
        fp32_infer_mb.append(mem_after_infer - mem_before)

    for _ in range(num_trials):
        gc.collect()
        mem_before = get_memory_mb()
        s = ort.InferenceSession(model_quant_path, sess_options=opts_quant, providers=["CPUExecutionProvider"])
        mem_after_load = get_memory_mb()
        for _ in range(warmup):
            s.run(None, feed)
        s.run(None, feed)
        mem_after_infer = get_memory_mb()
        del s
        gc.collect()
        quant_load_mb.append(mem_after_load - mem_before)
        quant_infer_mb.append(mem_after_infer - mem_before)

    fp32_load_mean, fp32_load_std = np.mean(fp32_load_mb), np.std(fp32_load_mb)
    fp32_infer_mean, fp32_infer_std = np.mean(fp32_infer_mb), np.std(fp32_infer_mb)
    quant_load_mean, quant_load_std = np.mean(quant_load_mb), np.std(quant_load_mb)
    quant_infer_mean, quant_infer_std = np.mean(quant_infer_mb), np.std(quant_infer_mb)

    if fp32_load_mean == 0 and fp32_infer_mean == 0 and quant_load_mean == 0 and quant_infer_mean == 0:
        print("Memory (RAM) consumption: deltas were 0 (RSS may not reflect per-session allocation on this system).")
    else:
        print("Memory (RAM) consumption (mean ± std over {} trials):".format(num_trials))
        print(f"  FP32   after load: {fp32_load_mean:.2f} ± {fp32_load_std:.2f} MB, after inference: {fp32_infer_mean:.2f} ± {fp32_infer_std:.2f} MB")
        print(f"  Quant  after load: {quant_load_mean:.2f} ± {quant_load_std:.2f} MB, after inference: {quant_infer_mean:.2f} ± {quant_infer_std:.2f} MB")

    return {
        "fp32_load_mb": fp32_load_mb,
        "fp32_infer_mb": fp32_infer_mb,
        "quant_load_mb": quant_load_mb,
        "quant_infer_mb": quant_infer_mb,
    }


def get_dummy_inputs(model, dynamic_size=1):
    """Build dummy inputs from ONNX model input spec.

    Args:
        model: ONNX ModelProto.
        dynamic_size: Value used for any dynamic dimension (default 1).
                      Use a larger value (e.g. 256, 1024) to get non-constant outputs.
    """
    dummy = {}
    for inp in model.graph.input:
        name = inp.name
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append(dynamic_size)
        dummy[name] = np.random.randn(*shape).astype(np.float32)
    return dummy


def compare_models(
    model_fp32_path,
    model_quant_path,
    num_inference_runs=50,
    warmup=5,
    num_speed_trials=5,
    num_memory_trials=5,
    dynamic_size=1,
):
    """Compare file size, inference speed, and memory (RAM) consumption.

    Inference speed and memory are averaged over num_speed_trials and num_memory_trials
    respectively; mean ± std is reported.

    dynamic_size: Value for dynamic dimensions when building dummy inputs (default 1).
                  Use the same value as calibration/evaluation (e.g. 256) for consistency.
    """
    import onnxruntime as ort

    size_fp32 = os.path.getsize(model_fp32_path)
    size_quant = os.path.getsize(model_quant_path)
    ratio = size_quant / size_fp32 if size_fp32 else 0
    print("\n--- Model comparison (before KS test) ---")
    print("File size:")
    print(f"  FP32:   {size_fp32 / (1024*1024):.2f} MB")
    print(f"  Quant:  {size_quant / (1024*1024):.2f} MB  (ratio: {ratio:.2%})")

    model_proto = onnx.load(model_fp32_path)
    dummy_inputs = get_dummy_inputs(model_proto, dynamic_size=dynamic_size)
    feed = {k: v for k, v in dummy_inputs.items()}

    def run_inference(session, feed_dict, n_warmup, n_runs):
        for _ in range(n_warmup):
            session.run(None, feed_dict)
        start = time()
        for _ in range(n_runs):
            session.run(None, feed_dict)
        return (time() - start) / n_runs

    # Memory first (before any session exists), so deltas reflect fresh allocation.
    measure_ram_usage(
        model_fp32_path,
        model_quant_path,
        num_trials=num_memory_trials,
        warmup=warmup,
        dynamic_size=dynamic_size,
        feed=feed,
    )
    print()

    # Inference speed: multiple trials, each trial = warmup + timed runs; report mean ± std
    lat_fp32_trials = []
    lat_quant_trials = []
    opts_fp32 = _session_options(for_quantized=False)
    opts_quant = _session_options(for_quantized=True)
    sess_fp32 = ort.InferenceSession(model_fp32_path, sess_options=opts_fp32, providers=["CPUExecutionProvider"])
    for _ in range(num_speed_trials):
        lat_fp32_trials.append(run_inference(sess_fp32, feed, warmup, num_inference_runs))
    del sess_fp32
    sess_quant = ort.InferenceSession(model_quant_path, sess_options=opts_quant, providers=["CPUExecutionProvider"])
    for _ in range(num_speed_trials):
        lat_quant_trials.append(run_inference(sess_quant, feed, warmup, num_inference_runs))
    del sess_quant

    lat_fp32_mean = np.mean(lat_fp32_trials) * 1000
    lat_fp32_std = np.std(lat_fp32_trials) * 1000
    lat_quant_mean = np.mean(lat_quant_trials) * 1000
    lat_quant_std = np.std(lat_quant_trials) * 1000
    speedup = lat_fp32_mean / lat_quant_mean

    print("Inference speed (avg latency per run, mean ± std over {} trials):".format(num_speed_trials))
    print(f"  FP32:   {lat_fp32_mean:.2f} ± {lat_fp32_std:.2f} ms")
    print(f"  Quant:  {lat_quant_mean:.2f} ± {lat_quant_std:.2f} ms  (speedup: {speedup:.2f}x)")
    if speedup < 1.0:
        print("  Note: Quantized model is slower than FP32. On CPUs without Intel VNNI, QDQ adds")
        print("        overhead. Try USE_PARALLEL_FOR_QUANT / QUANT_INTER_OP_THREADS in comparison_test, or use a VNNI-capable CPU.")
    print()


def _initializers_by_name(model):
    """Return dict mapping initializer name -> numpy array."""
    return {init.name: onnx.numpy_helper.to_array(init) for init in model.graph.initializer}


def _dequantized_weights_from_qdq(model):
    """Extract weight initializers that feed DequantizeLinear, dequantize them, return float64 array.

    Compares apples-to-apples: original FP32 weights vs dequantized (scale * (x - zero_point)).
    Excludes scale/zero_point initializers so KS/JS reflect weight distribution only.
    Returns None if no DequantizeLinear nodes (e.g. non-QDQ model); caller should fall back to raw.
    """
    inits = _initializers_by_name(model)
    dq_data = []  # list of dequantized float arrays
    for node in model.graph.node:
        if node.op_type != "DequantizeLinear":
            continue
        # DQ inputs: x, x_scale, x_zero_point (optional)
        if node.input[0] not in inits:
            continue  # data is not an initializer (e.g. activation)
        x = np.asarray(inits[node.input[0]], dtype=np.float64)
        scale = np.asarray(inits[node.input[1]], dtype=np.float64)
        zp_name = node.input[2] if len(node.input) > 2 and node.input[2].strip() else None
        zp = np.asarray(inits[zp_name], dtype=np.float64) if zp_name and zp_name in inits else np.float64(0)
        # Broadcast for per-channel: scale/zp (1-D) along axis 0
        if scale.ndim >= 1 and scale.size > 1:
            axis_shape = (scale.size,) + (1,) * (x.ndim - 1)
            scale = scale.reshape(axis_shape)
            zp = np.asarray(zp, dtype=np.float64)
            zp = zp.reshape(axis_shape) if zp.ndim > 0 else np.broadcast_to(zp, axis_shape)
        dq = (x - zp) * scale
        dq_data.append(dq.flatten())
    if not dq_data:
        return None
    return np.concatenate(dq_data, axis=0)


def _weights_as_float64(model, for_quantized=False):
    """Extract weights as float64 for distribution comparison.

    For FP32/original model: all initializers as float64.
    For quantized (QDQ) model: only dequantized weight tensors (excludes scale/zero_point);
    if for_quantized=True and model has DequantizeLinear, uses dequantized values so KS
    compares original vs dequantized (theoretically closer). Otherwise raw initializers.
    """
    if for_quantized:
        dq = _dequantized_weights_from_qdq(model)
        if dq is not None:
            return dq
    # Fallback: all initializers (legacy behavior for non-QDQ or when DQ not found)
    out = []
    for initializer in model.graph.initializer:
        arr = onnx.numpy_helper.to_array(initializer)
        out.extend(arr.flatten().astype(np.float64))
    return np.array(out, dtype=np.float64)


def ks_test(model_original, model_quantized, approval_max_stat=0.2):
    """Kolmogorov-Smirnov test between original and quantized weight distributions.

    For QDQ quantized models, compares original FP32 weights to *dequantized* weights
    (scale * (x - zero_point)), excluding scale/zero_point initializers, so KS reflects
    how close the effective weight distribution is after quantization (theoretically lower KS).
    """
    original_weights = _weights_as_float64(model_original, for_quantized=False)
    quantized_weights = _weights_as_float64(model_quantized, for_quantized=True)

    print("Original weights (preview):", original_weights[:8], "...")
    print("Quantized weights (dequantized, preview):", quantized_weights[:8], "...")
    print("\n\n")
    ks_statistic, p_value = stats.ks_2samp(original_weights, quantized_weights)
    print(f"KS Statistic: {ks_statistic:.4f}")
    print(f"P-value: {p_value:.4e}")
    print(f"Original weights count: {len(original_weights)}")
    print(f"Quantized weights count: {len(quantized_weights)}")
    approved = ks_statistic <= approval_max_stat
    print("KS test: {} (statistic <= {} required for approval)".format(
        "APPROVED" if approved else "NOT APPROVED", approval_max_stat))


def js_divergence_test(model_original, model_quantized, n_bins=100):
    """Jensen-Shannon divergence between original and quantized weight distributions.

    For QDQ models, quantized side uses dequantized weights only (same as KS test).
    """
    original_weights = _weights_as_float64(model_original, for_quantized=False)
    quantized_weights = _weights_as_float64(model_quantized, for_quantized=True)

    low = min(original_weights.min(), quantized_weights.min())
    high = max(original_weights.max(), quantized_weights.max())
    bins = np.linspace(low, high, n_bins + 1)

    p, _ = np.histogram(original_weights, bins=bins)
    q, _ = np.histogram(quantized_weights, bins=bins)
    p = p.astype(np.float64) / p.sum()
    q = q.astype(np.float64) / q.sum()

    js_div = jensenshannon(p, q, base=2)
    print(f"Jensen-Shannon divergence: {js_div:.6f}  (0 = identical, 1 = max)")
    print(f"  (n_bins={n_bins})")


def output_similarity(model_fp32, model_quant, feed=None, dynamic_size=256):
    """Compute cosine similarity and SQNR between FP32 and quantized model outputs.

    Args:
        model_fp32: FP32 model as onnx.ModelProto or path (str/Path) to .onnx file.
        model_quant: Quantized model as onnx.ModelProto or path (str/Path) to .onnx file.
        feed: Optional dict of input name -> array. If None, built from model_fp32.
        dynamic_size: When feed is None, dynamic dimensions get this size (default 256).
                      Use 256 or 1024 so outputs vary; size 1 often gives constant output.

    Runs both models with the same input and compares outputs element-wise.
    Cosine similarity: 1 = identical direction; SQNR (dB): higher = less quantization noise.
    """
    import onnxruntime as ort

    if not isinstance(model_fp32, onnx.ModelProto):
        model_fp32 = onnx.load(model_fp32)
    if not isinstance(model_quant, onnx.ModelProto):
        model_quant = onnx.load(model_quant)

    if feed is None:
        feed = get_dummy_inputs(model_fp32, dynamic_size=dynamic_size)

    def _session(model):
        if isinstance(model, onnx.ModelProto):
            return ort.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
        return ort.InferenceSession(model, providers=["CPUExecutionProvider"])

    sess_fp32 = _session(model_fp32)
    sess_quant = _session(model_quant)
    fp32_outs = sess_fp32.run(None, feed)
    quant_outs = sess_quant.run(None, feed)

    # Check output count match
    if len(fp32_outs) != len(quant_outs):
        print("--- Output similarity (cosine similarity, SQNR) ---")
        print("  CRITICAL: Output count mismatch (fp32 {} vs quant {}). Models are not equivalent.".format(len(fp32_outs), len(quant_outs)))
        return

    SQNR_GOOD_DB = 20.0   # Good quantization typically >20 dB
    COS_SIM_GOOD = 0.99   # Outputs should be nearly identical

    shape_mismatches = []
    results = []  # (output_idx, n_fp32, n_quant, cos_sim, sqnr_num or None)

    print("--- Output similarity (cosine similarity, SQNR) ---")
    for i, (fp32_out, quant_out) in enumerate(zip(fp32_outs, quant_outs)):
        fp32_out = np.asarray(fp32_out, dtype=np.float64).flatten()
        quant_out = np.asarray(quant_out, dtype=np.float64).flatten()

        n_fp32, n_quant = fp32_out.size, quant_out.size
        if n_fp32 != n_quant:
            shape_mismatches.append((i, n_fp32, n_quant))
            n = min(n_fp32, n_quant)
            if n == 0:
                print(f"  Output {i}: skip (empty; shapes fp32 {n_fp32}, quant {n_quant})")
                results.append((i, n_fp32, n_quant, float("nan"), None))
                continue
            fp32_out = fp32_out[:n].copy()
            quant_out = quant_out[:n].copy()
            shape_note = f" [first {n} elements only; shapes fp32 {n_fp32}, quant {n_quant}]"
        else:
            shape_note = ""

        # Cosine similarity
        norm_fp32 = np.linalg.norm(fp32_out)
        norm_quant = np.linalg.norm(quant_out)
        if norm_fp32 == 0 and norm_quant == 0:
            cos_sim = 1.0
        elif norm_fp32 == 0 or norm_quant == 0:
            cos_sim = float("nan")
        else:
            cos_sim = np.dot(fp32_out, quant_out) / (norm_fp32 * norm_quant)

        # SQNR (Signal-to-Quantization-Noise Ratio) in dB
        noise = fp32_out - quant_out
        var_fp32 = np.var(fp32_out)
        var_noise = np.var(noise)
        eps = 1e-12
        var_fp32_eff = var_fp32 + eps
        var_noise_eff = var_noise + eps
        sqnr = 10 * np.log10(var_fp32_eff / var_noise_eff)

        if var_fp32 == 0 and var_noise == 0:
            sqnr_str = "N/A (constant output)"
            sqnr_num = None
        elif var_fp32 == 0:
            sqnr_str = "N/A (constant signal)"
            sqnr_num = None
        elif not np.isfinite(sqnr):
            sqnr_str = "inf dB" if sqnr > 0 else "-inf dB"
            sqnr_num = sqnr
        else:
            sqnr_str = f"{sqnr:.2f} dB"
            sqnr_num = sqnr

        results.append((i, n_fp32, n_quant, cos_sim, sqnr_num))
        cos_sim_str = f"{cos_sim:.6f}" if not np.isnan(cos_sim) else "nan"
        print(f"  Output {i}: cos_sim = {cos_sim_str}, SQNR = {sqnr_str}{shape_note}")

    # Quality summary: flag critical and warning conditions
    print("")
    if shape_mismatches:
        print("  CRITICAL — Output shape mismatch:")
        for i, n_fp32, n_quant in shape_mismatches:
            print("    Output {}: fp32 {} vs quant {} (quantized model is not functionally equivalent; metrics above are on overlapping prefix only).".format(i, n_fp32, n_quant))
        print("  The quantized model may be broken for actual use. Fix quantization or avoid using this model.")
    poor_sqnr = [(r[0], r[4]) for r in results if r[4] is not None and r[4] < SQNR_GOOD_DB]
    poor_cos = [(r[0], r[3]) for r in results if not np.isnan(r[3]) and r[3] < COS_SIM_GOOD]
    if poor_sqnr:
        print("  WARNING — Poor SQNR (good quantization usually >{} dB):".format(int(SQNR_GOOD_DB)))
        for idx, sqnr in poor_sqnr:
            print("    Output {}: {:.2f} dB (negative = noise > signal)".format(idx, sqnr))
    if poor_cos:
        print("  WARNING — Poor cosine similarity (should be >{}):".format(COS_SIM_GOOD))
        for idx, cs in poor_cos:
            print("    Output {}: {:.4f}".format(idx, cs))
