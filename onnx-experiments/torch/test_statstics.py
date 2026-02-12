#!/usr/bin/env python3
"""
Compare original TFLite model (float32) vs quantized (fp16) using the same input.
Reports: KS test, Jensen-Shannon divergence, cosine similarity, MSE per output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

print("Testing if the model is able to load first...")
# We test to see if the model runs correctly first
from ai_edge_litert.interpreter import Interpreter
path = "./aliked-n16_fp16.tflite"

interpreter = Interpreter(model_path=path)
interpreter.allocate_tensors()
print("Model loaded successfully")

# We then proceed to run the tests

try:
    from scipy.stats import ks_2samp
    from scipy.spatial.distance import jensenshannon
except ImportError as e:
    raise ImportError("Install scipy: pip install scipy") from e

from ai_edge_litert.interpreter import Interpreter

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ORIGINAL = SCRIPT_DIR / "aliked-n16.tflite"
MODEL_QUANTIZED = SCRIPT_DIR / "aliked-n16_fp16.tflite"
MODEL_PTH = Path("/Users/antlowhur/Documents/Programming/optimization-scripts/onnx-experiments/torch/aliked-n16.pth")
MODEL_ONNX = Path("/Users/antlowhur/Documents/Programming/optimization-scripts/models/aliked-n16_640x640_512kp/aliked-n16_640x640_512kp.onnx")


def _get_input_shape(interpreter: Interpreter) -> tuple[int, ...]:
    """Return input tensor shape (e.g. (1, 640, 640, 3) for NHWC)."""
    inp = interpreter.get_input_details()[0]
    shape = inp["shape"]
    return tuple(shape)


def _run_model(interpreter: Interpreter, input_data: np.ndarray) -> list[np.ndarray]:
    """Run interpreter with input_data; return list of output arrays."""
    inp_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    interpreter.set_tensor(inp_details[0]["index"], input_data)
    interpreter.invoke()
    return [interpreter.get_tensor(d["index"]) for d in out_details]


def _load_pth_model(pth_path: Path):
    """Load ALIKED-n16 from .pth; return wrapper that outputs (score_map, feature_map)."""
    try:
        from pytorch2tflite import _load_aliked_dense_wrapper
        return _load_aliked_dense_wrapper(pth_path)
    except ImportError:
        pass
    import torch
    try:
        from nets.aliked import ALIKED
    except ImportError as e:
        raise ImportError(
            "ALIKED not found. Add ALIKED repo to PYTHONPATH."
        ) from e

    class _DenseWrapper(torch.nn.Module):
        def __init__(self, aliked):
            super().__init__()
            self.aliked = aliked

        def forward(self, image: torch.Tensor):
            feature_map, score_map = self.aliked.extract_dense_map(image)
            return score_map, feature_map

    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(pth_path, map_location="cpu")
    model = ALIKED(
        model_name="aliked-n16",
        device="cpu",
        top_k=-1,
        scores_th=0.2,
        n_limit=5000,
        load_pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return _DenseWrapper(model)


def _run_pth_model(wrapper, input_nchw: np.ndarray) -> list[np.ndarray]:
    """Run PyTorch wrapper on NCHW input; return [score_map, feature_map] as numpy."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(input_nchw.astype(np.float32))
        score_map, feature_map = wrapper(x)
        return [score_map.numpy(), feature_map.numpy()]


def _run_onnx_model(onnx_path: Path, input_nchw: np.ndarray) -> list[np.ndarray]:
    """Run ONNX model on NCHW input; return list of output arrays (e.g. score_map, feature_map)."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    input_feed = {input_name: input_nchw.astype(np.float32)}
    outputs = sess.run(None, input_feed)
    return [out for out in outputs]


def _to_probability_distribution(x: np.ndarray) -> np.ndarray:
    """Flatten, shift to non-negative, normalize to sum=1 for JS divergence."""
    flat = x.flatten().astype(np.float64)
    flat = flat - flat.min() + 1e-12
    return flat / flat.sum()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    n = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if n < 1e-12:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.dot(a_flat, b_flat) / n)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def compare_outputs(
    name: str,
    orig: np.ndarray,
    quant: np.ndarray,
) -> dict[str, float]:
    """Compute KS, JS divergence, cosine similarity, MSE for one output tensor pair."""
    o_flat = orig.flatten().astype(np.float64)
    q_flat = quant.flatten().astype(np.float64)
    ks_stat, ks_pvalue = ks_2samp(o_flat, q_flat)
    try:
        p_orig = _to_probability_distribution(orig)
        p_quant = _to_probability_distribution(quant)
        # Align length (in case of different sizes, should not happen)
        n = min(len(p_orig), len(p_quant))
        js_div = float(jensenshannon(p_orig[:n], p_quant[:n]))
    except Exception:
        js_div = float("nan")
    cos_sim = _cosine_similarity(orig, quant)
    mse_val = _mse(orig, quant)
    return {
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
        "js_divergence": js_div,
        "cosine_similarity": cos_sim,
        "mse": mse_val,
    }


def main() -> None:
    if not MODEL_ORIGINAL.exists():
        raise FileNotFoundError(f"Original model not found: {MODEL_ORIGINAL}")
    if not MODEL_QUANTIZED.exists():
        raise FileNotFoundError(f"Quantized model not found: {MODEL_QUANTIZED}")

    print("Loading models...")
    interp_orig = Interpreter(model_path=str(MODEL_ORIGINAL))
    interp_orig.allocate_tensors()
    interp_quant = Interpreter(model_path=str(MODEL_QUANTIZED))
    interp_quant.allocate_tensors()

    input_shape = _get_input_shape(interp_orig)
    print(f"Input shape: {input_shape}")
    # Same random input for both (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    input_data = rng.standard_normal(input_shape).astype(np.float32)

    print("Running original model...")
    outs_orig = _run_model(interp_orig, input_data)
    print("Running quantized model...")
    outs_quant = _run_model(interp_quant, input_data)

    if len(outs_orig) != len(outs_quant):
        print(
            f"Warning: output count mismatch (orig={len(outs_orig)}, quant={len(outs_quant)}). "
            "Comparing up to min."
        )
    n_outs = min(len(outs_orig), len(outs_quant))
    out_names = [f"output_{i}" for i in range(n_outs)]

    print("\n" + "=" * 60)
    print("Comparison: original (aliked-n16.tflite) vs quantized (aliked-n16_fp16.tflite)")
    print("=" * 60)

    for i in range(n_outs):
        name = out_names[i]
        o, q = outs_orig[i], outs_quant[i]
        if o.shape != q.shape:
            print(f"\n[{name}] Shape mismatch: orig {o.shape} vs quant {q.shape}")
            continue
        metrics = compare_outputs(name, o, q)
        print(f"\n[{name}] shape {o.shape}")
        print(f"  KS statistic:      {metrics['ks_statistic']:.6f}")
        print(f"  KS p-value:        {metrics['ks_pvalue']:.6f}")
        print(f"  JS divergence:     {metrics['js_divergence']:.6f}")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
        print(f"  MSE:               {metrics['mse']:.6e}")
    print("________________________________________________________")

    # ONNX vs original TFLite (same input: NHWC for TFLite, NCHW for ONNX)
    print("Comparison: ONNX (aliked-n16_640x640_512kp.onnx) vs original TFLite (aliked-n16.tflite)")
    print("=" * 60)
    if not MODEL_ONNX.exists():
        print(f"Skipping: ONNX model not found at {MODEL_ONNX}")
    else:
        # TFLite already ran with input_data (NHWC); ONNX needs NCHW
        if len(input_shape) == 4 and input_shape[-1] in (3, 1):
            input_nchw_onnx = np.transpose(input_data, (0, 3, 1, 2))
        else:
            input_nchw_onnx = input_data
        print("Loading ONNX model...")
        outs_onnx = _run_onnx_model(MODEL_ONNX, input_nchw_onnx)
        # outs_orig already from original TFLite above
        onnx_out_names = ["score_map", "feature_map"]
        n_onnx = min(len(outs_onnx), len(outs_orig))
        for i in range(n_onnx):
            name = onnx_out_names[i] if i < len(onnx_out_names) else f"output_{i}"
            o, q = outs_onnx[i], outs_orig[i]
            if o.shape != q.shape:
                print(f"\n[{name}] Shape mismatch: onnx {o.shape} vs tflite {q.shape}")
                continue
            metrics = compare_outputs(name, o, q)
            print(f"\n[{name}] shape {o.shape}")
            print(f"  KS statistic:      {metrics['ks_statistic']:.6f}")
            print(f"  KS p-value:        {metrics['ks_pvalue']:.6f}")
            print(f"  JS divergence:     {metrics['js_divergence']:.6f}")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  MSE:               {metrics['mse']:.6e}")

    print("________________________________________________________")
    print("Comparison: PyTorch .pth (aliked-n16.pth) vs original TFLite (aliked-n16.tflite)")
    print("=" * 60)
    if not MODEL_PTH.exists():
        print(f"Skipping: .pth not found at {MODEL_PTH}")
    else:
        # TFLite input is NHWC; PyTorch expects NCHW
        if len(input_shape) == 4 and input_shape[-1] in (3, 1):
            input_nchw = np.transpose(input_data, (0, 3, 1, 2))
        else:
            input_nchw = input_data
        print("Loading PyTorch .pth model...")
        wrapper = _load_pth_model(MODEL_PTH)
        print("Running PyTorch model...")
        outs_pth = _run_pth_model(wrapper, input_nchw)
        pth_out_names = ["score_map", "feature_map"]
        n_pth = min(len(outs_pth), len(outs_orig))
        for i in range(n_pth):
            name = pth_out_names[i] if i < len(pth_out_names) else f"output_{i}"
            o, q = outs_pth[i], outs_orig[i]
            if o.shape != q.shape:
                print(f"\n[{name}] Shape mismatch: pth {o.shape} vs tflite {q.shape}")
                continue
            metrics = compare_outputs(name, o, q)
            print(f"\n[{name}] shape {o.shape}")
            print(f"  KS statistic:      {metrics['ks_statistic']:.6f}")
            print(f"  KS p-value:        {metrics['ks_pvalue']:.6f}")
            print(f"  JS divergence:     {metrics['js_divergence']:.6f}")
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
            print(f"  MSE:               {metrics['mse']:.6e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
