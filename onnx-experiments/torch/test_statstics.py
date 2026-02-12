#!/usr/bin/env python3
"""
Compare original TFLite model (float32) vs quantized (fp16) using the same input.
Reports: KS test, Jensen-Shannon divergence, cosine similarity, MSE per output.

For PyTorch .pth comparison, set PYTHONPATH to include the ALIKED repo, e.g.:
  export PYTHONPATH=/path/to/onnx-experiments/torch/ALIKED:$PYTHONPATH
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

try:
    from scipy.stats import ks_2samp
    from scipy.spatial.distance import jensenshannon
except ImportError as e:
    raise ImportError("Install scipy: pip install scipy") from e

from ai_edge_litert.interpreter import Interpreter

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_ORIGINAL = SCRIPT_DIR / "aliked-n16.tflite"
MODEL_QUANTIZED = SCRIPT_DIR / "aliked-n16_fp16.tflite"
MODEL_QUANTIZED_RUNNABLE = SCRIPT_DIR / "aliked-n16_fp16_runnable.tflite"
MODEL_PTH = SCRIPT_DIR / "aliked-n16.pth"
MODEL_ONNX = Path("/Users/antlowhur/Documents/Programming/optimization-scripts/models/aliked-n16_640x640_512kp/aliked-n16_640x640_512kp.onnx")


def _generate_runnable_fp16_if_needed() -> Path | None:
    """If the quantization script exists, run it with --runnable to produce a loadable FP16 model. Returns path or None."""
    quant_script = SCRIPT_DIR / "tflite_quantize_fp16.py"
    if not quant_script.exists() or not MODEL_ORIGINAL.exists():
        return None
    try:
        subprocess.run(
            [sys.executable, str(quant_script), str(MODEL_ORIGINAL), "-o", str(MODEL_QUANTIZED_RUNNABLE), "--runnable"],
            check=True,
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            timeout=120,
        )
        return MODEL_QUANTIZED_RUNNABLE if MODEL_QUANTIZED_RUNNABLE.exists() else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


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


def _channel_count_4d(arr: np.ndarray) -> int:
    """Channel count for 4D tensor: NCHW -> shape[1], NHWC -> shape[3]. Else -1."""
    s = arr.shape
    if len(s) != 4:
        return -1
    if s[1] in (1, 128):
        return int(s[1])
    if s[3] in (1, 128):
        return int(s[3])
    return -1


def _to_nchw_4d(arr: np.ndarray) -> np.ndarray:
    """If array is NHWC (1, H, W, C) with C in (1, 128), return NCHW (1, C, H, W). Else return copy."""
    if arr.ndim != 4:
        return arr
    s = arr.shape
    if s[3] in (1, 128) and s[1] not in (1, 128):
        return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
    return arr


def _align_tflite_outputs_to_pth(
    tflite_outs: list[np.ndarray],
    pth_outs: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Return [score_map, feature_map] from TFLite in NCHW, matching PyTorch order and layout.
    TFLite may have swapped order and/or NHWC; we identify by channel count (1 vs 128).
    """
    if len(tflite_outs) != 2 or len(pth_outs) != 2:
        return tflite_outs
    c0 = _channel_count_4d(tflite_outs[0])
    c1 = _channel_count_4d(tflite_outs[1])
    if c0 == 1 and c1 == 128:
        score_map_tflite, feature_map_tflite = tflite_outs[0], tflite_outs[1]
    elif c0 == 128 and c1 == 1:
        score_map_tflite, feature_map_tflite = tflite_outs[1], tflite_outs[0]
    else:
        return tflite_outs
    return [_to_nchw_4d(score_map_tflite), _to_nchw_4d(feature_map_tflite)]


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
    interp_quant = None
    quant_model_path = MODEL_QUANTIZED
    try:
        interp_quant = Interpreter(model_path=str(MODEL_QUANTIZED))
        interp_quant.allocate_tensors()
    except RuntimeError as e:
        if "failed to prepare" in str(e).lower() or "was not true" in str(e):
            print("Quantized model (pure FP16) could not be loaded on this runtime; generating runnable FP16 variant...")
            runnable_path = _generate_runnable_fp16_if_needed()
            if runnable_path is not None:
                try:
                    interp_quant = Interpreter(model_path=str(runnable_path))
                    interp_quant.allocate_tensors()
                    quant_model_path = runnable_path
                    print("Loaded runnable FP16 model (aliked-n16_fp16_runnable.tflite).")
                except RuntimeError:
                    interp_quant = None
            else:
                print("Could not generate or load runnable quantized model; skipping orig-vs-quant comparison.")
                interp_quant = None
        else:
            raise

    input_shape = _get_input_shape(interp_orig)
    print(f"Input shape: {input_shape}")
    # Same random input for both (fixed seed for reproducibility)
    rng = np.random.default_rng(42)
    input_data = rng.standard_normal(input_shape).astype(np.float32)

    print("Running original model...")
    outs_orig = _run_model(interp_orig, input_data)

    if interp_quant is not None:
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
    else:
        print("Skipping orig-vs-quant comparison (quantized model not loaded).")

    # ONNX vs original TFLite (same input: NHWC for TFLite, NCHW for ONNX)
    print("Comparison: ONNX (aliked-n16_640x640_512kp.onnx) vs original TFLite (aliked-n16.tflite)")
    print("=" * 60)
    if not MODEL_ONNX.exists():
        print(f"Skipping: ONNX model not found at {MODEL_ONNX}")
    else:
        if len(input_shape) == 4 and input_shape[-1] in (3, 1):
            input_nchw_onnx = np.transpose(input_data, (0, 3, 1, 2))
        else:
            input_nchw_onnx = input_data
        print("Loading ONNX model...")
        outs_onnx = _run_onnx_model(MODEL_ONNX, input_nchw_onnx)
        # 512kp ONNX has keypoint outputs (e.g. (512,2), (512,128)), not 4D dense maps
        onnx_dense = (
            len(outs_onnx) == 2
            and outs_onnx[0].ndim == 4
            and outs_onnx[1].ndim == 4
        )
        if not onnx_dense:
            print(
                "Skipping: ONNX has keypoint outputs (not dense maps), so it cannot be compared to TFLite. "
                f"ONNX output shapes: {[o.shape for o in outs_onnx]}. "
                "Use a dense-map ONNX export for this comparison."
            )
        else:
            tflite_aligned = _align_tflite_outputs_to_pth(outs_orig, outs_onnx)
            if len(tflite_aligned) != 2 or outs_onnx[0].shape != tflite_aligned[0].shape or outs_onnx[1].shape != tflite_aligned[1].shape:
                print(f"Skipping: shape mismatch (ONNX: {[o.shape for o in outs_onnx]}, TFLite aligned: {[o.shape for o in tflite_aligned]}).")
            else:
                print("Note: TFLite outputs aligned to ONNX (score_map, feature_map, NCHW).")
                onnx_out_names = ["score_map", "feature_map"]
                for i in range(2):
                    name = onnx_out_names[i] if i < len(onnx_out_names) else f"output_{i}"
                    o, q = outs_onnx[i], tflite_aligned[i]
                    metrics = compare_outputs(name, o, q)
                    print(f"\n[{name}] shape {o.shape}")
                    print(f"  KS statistic:      {metrics['ks_statistic']:.6f}")
                    print(f"  KS p-value:        {metrics['ks_pvalue']:.6f}")
                    print(f"  JS divergence:     {metrics['js_divergence']:.6f}")
                    print(f"  Cosine similarity: {metrics['cosine_similarity']:.6f}")
                    print(f"  MSE:               {metrics['mse']:.6e}")

    print("________________________________________________________")
    print("Comparison: PyTorch .pth (aliked-n16.pth) vs quantized TFLite (aliked-n16_fp16.tflite)")
    print("=" * 60)
    if not MODEL_PTH.exists():
        print(f"Skipping: .pth not found at {MODEL_PTH}")
    elif interp_quant is None:
        print("Skipping: quantized model (aliked-n16_fp16.tflite) not loaded.")
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
        # TFLite (quantized) may have swapped order and/or NHWC; align to [score_map, feature_map] NCHW
        outs_tflite_aligned = _align_tflite_outputs_to_pth(outs_quant, outs_pth)
        print("Note: TFLite (aliked-n16_fp16.tflite) outputs aligned to PyTorch (score_map, feature_map, NCHW) for comparison.")
        pth_out_names = ["score_map", "feature_map"]
        n_pth = min(len(outs_pth), len(outs_tflite_aligned))
        for i in range(n_pth):
            name = pth_out_names[i] if i < len(pth_out_names) else f"output_{i}"
            o, q = outs_pth[i], outs_tflite_aligned[i]
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
