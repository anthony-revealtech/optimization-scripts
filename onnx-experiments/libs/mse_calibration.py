"""
MSE (Mean Squared Error) minimization calibration for ONNX QDQ models.

ONNX Runtime does not provide built-in MSE calibration. This module:
1. Uses an initial MinMax-quantized QDQ model to get graph structure.
2. Collects activation statistics by running the FP32 model on calibration data
   (with intermediate outputs added to expose activation tensors).
3. For each activation tensor, searches for the clipping threshold that minimizes
   MSE between original and quantized-dequantized values.
4. Computes scale and zero_point from the optimal threshold and updates the QDQ
   model's QuantizeLinear initializers in place.

Symmetric int8: zero_point=0, scale = threshold / 127.
"""

from __future__ import annotations

import os
import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort
from typing import Dict, List, Optional, Tuple, Any


# Number of candidate thresholds to search (percentiles from high to max).
MSE_NUM_CANDIDATES = 100
# Percentile range for threshold search (focus on tail where clipping matters).
MSE_PERCENTILE_START = 99.0
MSE_PERCENTILE_END = 100.0


def find_optimal_mse_threshold(
    activations: np.ndarray,
    num_bits: int = 8,
    symmetric: bool = True,
    percentile_samples: int = MSE_NUM_CANDIDATES,
    p_low: float = MSE_PERCENTILE_START,
    p_high: float = MSE_PERCENTILE_END,
) -> Tuple[float, float, float]:
    """
    Search for the clipping threshold that minimizes MSE after quantize-dequantize.

    Args:
        activations: Float array of activation values (any shape).
        num_bits: Bit width (e.g. 8 for int8).
        symmetric: If True, use symmetric quantization (zero_point=0).
        percentile_samples: Number of candidate thresholds.
        p_low, p_high: Percentile range for candidate thresholds (e.g. 99–100).

    Returns:
        (best_threshold, scale, zero_point).
    """
    flat = np.float64(activations.flatten())
    if flat.size == 0:
        return 1.0, 1.0 / 127.0, 0.0

    abs_flat = np.abs(flat)
    # Candidate thresholds: percentiles of |activations|
    percentiles = np.linspace(p_low, p_high, percentile_samples)
    candidate_thresholds = np.percentile(abs_flat, percentiles)
    # Ensure at least one positive threshold and no duplicates
    candidate_thresholds = np.unique(np.maximum(candidate_thresholds, 1e-8))

    q_max = 2 ** (num_bits - 1) - 1  # 127 for int8, 32767 for int16
    q_min = -q_max - 1 if symmetric else 0  # -128 for int8, -32768 for int16

    best_threshold = float(candidate_thresholds[-1]) if len(candidate_thresholds) else 1.0
    best_mse = np.inf
    best_scale = best_threshold / q_max
    best_zp = 0.0

    for threshold in candidate_thresholds:
        threshold = float(np.maximum(threshold, 1e-8))
        if symmetric:
            scale = threshold / q_max
            zero_point = 0.0
            clipped = np.clip(flat, -threshold, threshold)
            quantized = np.round(clipped / scale).astype(np.int32)
            quantized = np.clip(quantized, q_min, q_max)
            dequantized = quantized.astype(np.float64) * scale
        else:
            # Asymmetric: not used for activations typically
            scale = (2 * threshold) / (2**num_bits - 1)
            zero_point = 0.0
            clipped = np.clip(flat, -threshold, threshold)
            quantized = np.round(clipped / scale + zero_point).astype(np.int32)
            quantized = np.clip(quantized, 0, 255)
            dequantized = (quantized.astype(np.float64) - zero_point) * scale

        mse = np.mean((flat - dequantized) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_threshold = threshold
            best_scale = scale
            best_zp = zero_point

    return best_threshold, float(best_scale), float(best_zp)


def _initializer_names(model: onnx.ModelProto) -> set:
    return {init.name for init in model.graph.initializer}


def get_activation_quantize_inputs_from_qdq(model: onnx.ModelProto) -> List[Tuple[str, str, str]]:
    """
    From a QDQ model, list QuantizeLinear nodes that quantize *activations*
    (i.e. first input is not a weight initializer).

    Returns:
        List of (activation_tensor_name, scale_initializer_name, zero_point_initializer_name).
        zero_point_initializer_name may be "" if the node uses optional 3rd input.
    """
    inits = _initializer_names(model)
    result = []
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        # Inputs: x, y_scale, [y_zero_point]
        if len(node.input) < 2:
            continue
        act_name = node.input[0]
        scale_name = node.input[1]
        zp_name = node.input[2] if len(node.input) > 2 else ""
        # Skip weight quantization: activation is an initializer
        if act_name in inits:
            continue
        result.append((act_name, scale_name, zp_name))
    return result


def add_outputs_to_fp32_model(
    model: onnx.ModelProto,
    extra_output_names: List[str],
) -> onnx.ModelProto:
    """
    Return a copy of the model with extra_output_names added as graph outputs.
    Used to run FP32 and collect intermediate activations. Only names that
    already exist as node outputs or graph inputs are added (no new value_info).
    """
    model_copy = onnx.ModelProto()
    model_copy.CopyFrom(model)
    existing_outputs = {o.name for o in model_copy.graph.output}
    # All node output names in the graph
    node_outputs = set()
    for node in model_copy.graph.node:
        for out in node.output:
            node_outputs.add(out)
    graph_inputs = {inp.name for inp in model_copy.graph.input}
    for name in extra_output_names:
        if name in existing_outputs:
            continue
        if name not in node_outputs and name not in graph_inputs:
            continue
        try:
            vi = helper.make_empty_tensor_value_info(name)
        except AttributeError:
            vi = helper.ValueInfoProto()
            vi.name = name
        model_copy.graph.output.append(vi)
    return model_copy


def collect_activations(
    fp32_model_path: str,
    calibration_batches: List[Dict[str, np.ndarray]],
    activation_tensor_names: List[str],
    dynamic_size: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Run the FP32 model on calibration batches and collect activations for the
    given tensor names. The FP32 model is temporarily modified to expose those
    tensors as outputs.

    Args:
        fp32_model_path: Path to FP32 ONNX model.
        calibration_batches: List of feed dicts (input_name -> array).
        activation_tensor_names: Names of tensors to collect (must be graph input
            or node outputs in the FP32 model).
        dynamic_size: If set, used for shape inference when model has dynamic dims.

    Returns:
        Dict mapping tensor name -> list of arrays (one per batch).
    """
    model = onnx.load(fp32_model_path)
    # Input names that we feed; these we already have, no need to add as outputs
    input_names = {inp.name for inp in model.graph.input}
    # Names we need to fetch that are not the model input
    fetch_names = [n for n in activation_tensor_names if n not in input_names]
    if not fetch_names:
        # Only graph input is quantized; we have it from the batch
        out = {n: [] for n in activation_tensor_names}
        for batch in calibration_batches:
            for name in activation_tensor_names:
                if name in batch:
                    out[name].append(batch[name])
        return out

    # Build model with extra outputs
    try:
        model_with_outputs = add_outputs_to_fp32_model(model, fetch_names)
    except Exception:
        # If shape inference or duplicate output causes issues, fall back to
        # only collecting from inputs
        out = {n: [] for n in activation_tensor_names}
        for batch in calibration_batches:
            for name in activation_tensor_names:
                if name in batch:
                    out[name].append(batch[name])
        return out

    # Save to temp and run with ORT
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        temp_path = f.name
    try:
        onnx.save(model_with_outputs, temp_path)
        sess_options = ort.SessionOptions()
        sess = ort.InferenceSession(
            temp_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        output_names = [o.name for o in sess.get_outputs()]
        collected: Dict[str, List[np.ndarray]] = {n: [] for n in activation_tensor_names}
        for batch in calibration_batches:
            feeds = {k: v.astype(np.float32) for k, v in batch.items()}
            outputs = sess.run(output_names, feeds)
            for name, arr in zip(output_names, outputs):
                if name in collected:
                    collected[name].append(arr.copy())
            # Graph input activations (first Q in QDQ is often the input)
            for name in activation_tensor_names:
                if name in batch and name not in output_names:
                    collected[name].append(batch[name].astype(np.float32))
        return collected
    finally:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass


def compute_mse_scales_per_tensor(
    activations_by_name: Dict[str, List[np.ndarray]],
    num_bits: int = 8,
    symmetric: bool = True,
) -> Dict[str, Tuple[float, float]]:
    """
    For each tensor name, concatenate activations, find optimal MSE threshold,
    and return scale and zero_point.

    Returns:
        Dict mapping tensor name -> (scale, zero_point).
    """
    result = {}
    for name, arrays in activations_by_name.items():
        if not arrays:
            result[name] = (1.0 / 127.0, 0.0)
            continue
        concat = np.concatenate([a.flatten() for a in arrays], axis=0)
        _, scale, zp = find_optimal_mse_threshold(concat, num_bits=num_bits, symmetric=symmetric)
        result[name] = (scale, zp)
    return result


def _replace_initializer_value(model: onnx.ModelProto, name: str, value: np.ndarray) -> None:
    """Replace or add an initializer with the given name and value."""
    dtype = value.dtype
    if dtype == np.float64:
        value = value.astype(np.float32)
    if dtype == np.int64:
        value = value.astype(np.int32)
    for init in model.graph.initializer:
        if init.name == name:
            init.CopyFrom(numpy_helper.from_array(value, name=name))
            return
    model.graph.initializer.append(numpy_helper.from_array(value, name=name))


def apply_mse_scales_to_qdq_model(
    qdq_model_path: str,
    mse_scales: Dict[str, Tuple[float, float]],
    activation_quantize_list: List[Tuple[str, str, str]],
    output_path: str,
) -> None:
    """
    Load the QDQ model, replace scale and zero_point initializers for activation
    QuantizeLinear nodes with MSE-optimized values, and save to output_path.

    mse_scales: activation_tensor_name -> (scale, zero_point).
    activation_quantize_list: from get_activation_quantize_inputs_from_qdq.
    """
    model = onnx.load(qdq_model_path)
    scale_zp_by_act = dict(mse_scales)
    for act_name, scale_init_name, zp_init_name in activation_quantize_list:
        if act_name not in scale_zp_by_act:
            continue
        scale, zp = scale_zp_by_act[act_name]
        _replace_initializer_value(model, scale_init_name, np.array(scale, dtype=np.float32))
        if zp_init_name:
            zp_int = int(round(zp))
            _replace_initializer_value(model, zp_init_name, np.array(zp_int, dtype=np.int32))
    onnx.save(model, output_path)


def run_mse_calibration(
    fp32_model_path: str,
    qdq_model_path: str,
    calibration_batches: List[Dict[str, np.ndarray]],
    output_path: str,
    num_bits: int = 8,
    symmetric: bool = True,
    dynamic_size: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """
    Full MSE calibration pipeline:

    1. Read QDQ model and list activation QuantizeLinear nodes.
    2. Collect activations from FP32 model on calibration_batches.
    3. Compute MSE-optimal scale/zero_point per activation tensor.
    4. Write updated QDQ model to output_path.

    Returns:
        Dict activation_name -> (scale, zero_point) that was applied.
    """
    qdq = onnx.load(qdq_model_path)
    act_quant_list = get_activation_quantize_inputs_from_qdq(qdq)
    if not act_quant_list:
        # No activation Q nodes to override; copy as-is
        import shutil
        shutil.copy(qdq_model_path, output_path)
        return {}

    act_names = [t[0] for t in act_quant_list]
    collected = collect_activations(
        fp32_model_path,
        calibration_batches,
        act_names,
        dynamic_size=dynamic_size,
    )
    mse_scales = compute_mse_scales_per_tensor(
        collected,
        num_bits=num_bits,
        symmetric=symmetric,
    )
    apply_mse_scales_to_qdq_model(
        qdq_model_path,
        mse_scales,
        act_quant_list,
        output_path,
    )
    return mse_scales
