#!/usr/bin/env python3
"""
Full pipeline: .pth -> TFLite (float32) -> FP16 quantized TFLite -> test FP16 -> compare stats.

All four stages are inlined in this script (no subprocess). Takes two paths: input .pth and
output quantized .tflite (e.g. aliked-t16.pth, aliked-t16_fp16.tflite).

  1) Stage 1: .pth -> float32 .tflite (direct litert_torch or PyTorch->ONNX->TFLite)
  2) Stage 2: float32 -> pure FP16 (output path) + runnable FP16 (for stage 4)
  3) Stage 3: verify pure FP16 (fp16 shim + tensor check)
  4) Stage 4: compare original vs quantized on images, write metrics .txt

Usage:
  python3 tf_quantize_full.py aliked-t16.pth aliked-t16_fp16.tflite
  python3 tf_quantize_full.py aliked-t16.pth aliked-t16_fp16.tflite --image-dir /path/to/images
  python3 tf_quantize_full.py aliked-t16.pth out.tflite --skip-step1 --intermediate-tflite /path/to/existing.tflite
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Paths and defaults (edit these to match your setup) ---
JAGR_DATA_BASE = "/Users/antlowhur/Documents/Programming/jagr-data"
JAGR_IMAGE_SUBPATH = "data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data/"
DEFAULT_IMAGE_DIR = os.path.join(JAGR_DATA_BASE, JAGR_IMAGE_SUBPATH)

DEFAULT_METRICS_FILENAME = "original_vs_quantized_metrics.txt"
RUNNABLE_TFLITE_SUFFIX = "_runnable.tflite"
FP16_INTERPRETER_SHIM_MODULE = "fp16_interpreter_shim"

# If False: only produce pure FP16 in stage 2; stage 4 (compare on images) is skipped.
ENABLE_RUNNABLE = True

# Model input size (NCHW: 1, 3, H, W) for export and image prep
MODEL_INPUT_HEIGHT = 640
MODEL_INPUT_WIDTH = 640

# TensorFlow Lite schema (for stage 2)
try:
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.tools import flatbuffer_utils
except ImportError:
    schema_fb = None
    flatbuffer_utils = None


# =============================================================================
# Stage 1: PyTorch (.pth) -> float32 TFLite (from pytorch2tflite.py)
# =============================================================================

def _load_aliked_dense_wrapper(pth_path: Path):
    """Load ALIKED from .pth and return a wrapper that outputs (score_map, feature_map)."""
    import torch
    try:
        from nets.aliked import ALIKED
    except ImportError as e:
        raise ImportError(
            "ALIKED package not found. Clone the repo and add it to PYTHONPATH."
        ) from e

    model_name = pth_path.stem or "aliked-n16"

    class _DenseWrapper(torch.nn.Module):
        def __init__(self, aliked: ALIKED):
            super().__init__()
            self.aliked = aliked

        def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            feature_map, score_map = self.aliked.extract_dense_map(image)
            return score_map, feature_map

    print(f"Loading weights from {pth_path}...")
    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(pth_path, map_location="cpu")

    print(f"Building ALIKED model (model_name={model_name!r}, load_pretrained=False)...")
    model = ALIKED(
        model_name=model_name,
        device="cpu",
        top_k=-1,
        scores_th=0.2,
        n_limit=5000,
        load_pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return _DenseWrapper(model)


def _load_full_aliked(pth_path: Path) -> tuple:
    """Load full ALIKED from .pth (no wrapper). Returns (model, device) for keypoints (forward) and dense (extract_dense_map).
    Aligned with aliked_compare_stats_vid_tf.py so .pth keypoints match (DKD output, same coord transform)."""
    import torch
    try:
        from nets.aliked import ALIKED
    except ImportError as e:
        raise ImportError(
            "ALIKED package not found. Clone the repo and add it to PYTHONPATH."
        ) from e
    model_name = pth_path.stem or "aliked-n16"
    print(f"Loading full ALIKED from {pth_path} (model_name={model_name!r})...")
    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(pth_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ALIKED(
        model_name=model_name,
        device=device,
        top_k=-1,
        scores_th=0.2,
        n_limit=5000,
        load_pretrained=False,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, device


def _run_full_aliked_dense(model, device, input_nchw: np.ndarray) -> list[np.ndarray]:
    """Run full ALIKED extract_dense_map; return [score_map, feature_map] as numpy (same order as dense wrapper)."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(input_nchw.astype(np.float32)).to(device)
        feature_map, score_map = model.extract_dense_map(x)
        return [score_map.cpu().numpy(), feature_map.cpu().numpy()]


def _get_pth_keypoints_from_full_aliked(
    model,
    device,
    input_nchw: np.ndarray,
    orig_h: int,
    orig_w: int,
    pad_left: int,
    pad_top: int,
    max_dim: int,
) -> np.ndarray:
    """Get keypoints from full ALIKED forward (DKD), in original image coords. Same formula as aliked_compare_stats_vid_tf.py."""
    import torch
    _orig_sync = None
    if not torch.cuda.is_available():
        _orig_sync = getattr(torch.cuda, "synchronize", None)
        if _orig_sync is not None:
            torch.cuda.synchronize = lambda: None
    try:
        with torch.no_grad():
            x = torch.from_numpy(input_nchw.astype(np.float32)).to(device)
            out = model(x)
    except AttributeError as e:
        if "get_patches_forward" in str(e) or "custom_ops" in str(e):
            raise RuntimeError(
                "ALIKED PyTorch keypoints require the custom_ops extension. "
                "Build it from the ALIKED repo or use TFLite-only comparison."
            ) from e
        raise
    finally:
        if _orig_sync is not None:
            torch.cuda.synchronize = _orig_sync
    if isinstance(out, dict):
        kpts_norm = out.get("keypoints", out.get("kp", out.get("keypoint")))
    else:
        kpts_norm = out[0] if len(out) > 0 else None
    if kpts_norm is None:
        return np.zeros((0, 2), dtype=np.float64)
    if isinstance(kpts_norm, (list, tuple)):
        kpts_norm = kpts_norm[0] if kpts_norm else None
    if kpts_norm is None:
        return np.zeros((0, 2), dtype=np.float64)
    kpts_norm = kpts_norm.cpu().numpy()
    if len(kpts_norm.shape) == 3:
        kpts_norm = kpts_norm[0]
    kpts_px = kpts_norm.astype(np.float64).copy()
    kpts_px[:, 0] = (kpts_px[:, 0] + 1) * 0.5 * max_dim - pad_left
    kpts_px[:, 1] = (kpts_px[:, 1] + 1) * 0.5 * max_dim - pad_top
    valid = (
        (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < orig_w)
        & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < orig_h)
    )
    return kpts_px[valid]


def _patch_deform_conv2d_for_litert() -> None:
    """Replace deform_conv2d with conv2d fallback so litert_torch can convert."""
    import torch
    import torch.nn.functional as F
    import torchvision.ops

    def _deform_conv2d_fallback(
        input: torch.Tensor,
        offset: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
        dilation: tuple[int, int] = (1, 1),
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.conv2d(input, weight, bias, stride, padding, dilation)

    torchvision.ops.deform_conv2d = _deform_conv2d_fallback


def _convert_pytorch_to_tflite_direct(pth_path: Path, tflite_path: Path) -> bool:
    """Convert PyTorch -> TFLite via litert_torch. Returns True on success."""
    _patch_deform_conv2d_for_litert()
    try:
        import litert_torch
    except ImportError:
        return False
    import torch

    print("Using direct PyTorch -> TFLite (litert_torch), no ONNX.")
    wrapper = _load_aliked_dense_wrapper(pth_path)
    sample_inputs = (torch.randn(1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH),)
    print(f"Converting (input shape [1, 3, {MODEL_INPUT_HEIGHT}, {MODEL_INPUT_WIDTH}])...")
    try:
        edge_model = litert_torch.convert(wrapper.eval(), sample_inputs)
    except Exception as e:
        print(f"litert_torch conversion failed: {e}", file=sys.stderr)
        return False
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(tflite_path))
    print(f"TFLite saved to {tflite_path}")
    return True


def _export_pytorch_to_onnx(pth_path: Path, onnx_path: Path) -> None:
    """Export ALIKED dense wrapper to ONNX."""
    import torch
    wrapper = _load_aliked_dense_wrapper(pth_path)
    dummy = torch.randn(1, 3, MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH)
    print(f"Exporting ONNX to {onnx_path}...")
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["score_map", "feature_map"],
        dynamic_axes={
            "image": {0: "batch"},
            "score_map": {0: "batch"},
            "feature_map": {0: "batch"},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    print("ONNX export done.")


def _convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> None:
    """Convert ONNX -> TFLite via onnx2tf."""
    try:
        import onnx2tf
    except ImportError as e:
        raise ImportError("onnx2tf not found. Install with: pip3 install onnx2tf tensorflow") from e

    out_dir = tflite_path.with_suffix("")
    print(f"Converting ONNX to TFLite: {onnx_path} -> {tflite_path}...")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    found = list(out_dir.rglob("*.tflite"))
    if found:
        shutil.move(str(found[0]), str(tflite_path))
        print(f"TFLite saved to {tflite_path}")
    else:
        import tensorflow as tf
        converter = tf.lite.TFLiteConverter.from_saved_model(str(out_dir))
        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f"TFLite saved to {tflite_path}")
    shutil.rmtree(out_dir, ignore_errors=True)


def _step1_pth_to_tflite(pth_path: Path, intermediate_tflite: Path, use_onnx: bool = False) -> None:
    """Stage 1: .pth -> float32 .tflite (direct or ONNX path)."""
    print("[Step 1] .pth -> float32 TFLite" + (" (ONNX path)" if use_onnx else ""))
    if not use_onnx and _convert_pytorch_to_tflite_direct(pth_path, intermediate_tflite):
        return
    if not use_onnx:
        print("Direct conversion failed; falling back to PyTorch -> ONNX -> TFLite...", flush=True)
    onnx_path = intermediate_tflite.with_suffix(".onnx")
    if onnx_path == pth_path:
        onnx_path = pth_path.with_suffix(".onnx")
    _export_pytorch_to_onnx(pth_path, onnx_path)
    _convert_onnx_to_tflite(onnx_path, intermediate_tflite)
    onnx_path.unlink(missing_ok=True)
    print("Removed intermediate ONNX file.")


# =============================================================================
# Stage 2: float32 TFLite -> FP16 (from tflite_quantize_fp16.py)
# =============================================================================

def _read_tflite_model(path: Path) -> "schema_fb.ModelT":
    """Load a TFLite file into a mutable model object."""
    if flatbuffer_utils is None:
        raise ImportError("tensorflow required for stage 2. Install with: pip3 install tensorflow")
    data = path.read_bytes()
    model = flatbuffer_utils.read_model_from_bytearray(bytearray(data))
    return model


def _weight_tensor_indices(model: "schema_fb.ModelT") -> set[tuple[int, int]]:
    """(sg_idx, ten_idx) for tensors that have a non-empty buffer (weights)."""
    result = set()
    for sg_idx, subgraph in enumerate(model.subgraphs):
        if subgraph.tensors is None:
            continue
        for ten_idx, tensor in enumerate(subgraph.tensors):
            if tensor.type != schema_fb.TensorType.FLOAT32:
                continue
            buf_idx = tensor.buffer
            if buf_idx <= 0 or buf_idx >= len(model.buffers):
                continue
            buf = model.buffers[buf_idx]
            if buf.data is None or len(buf.data) == 0:
                continue
            result.add((sg_idx, ten_idx))
    return result


def _convert_weights_to_float16(model: "schema_fb.ModelT") -> set[tuple[int, int]]:
    """In-place: convert weight buffers and tensor types to FLOAT16."""
    weight_indices = _weight_tensor_indices(model)
    converted_buffers: set[int] = set()
    for sg_idx, subgraph in enumerate(model.subgraphs):
        if subgraph.tensors is None:
            continue
        for ten_idx in [ti for (s, ti) in weight_indices if s == sg_idx]:
            tensor = subgraph.tensors[ten_idx]
            buf_idx = tensor.buffer
            if buf_idx in converted_buffers:
                tensor.type = schema_fb.TensorType.FLOAT16
                continue
            buffer = model.buffers[buf_idx]
            if buffer.data is None or len(buffer.data) == 0:
                continue
            n_bytes = len(buffer.data)
            if n_bytes % 4 != 0:
                continue
            arr_f32 = np.frombuffer(buffer.data, dtype=np.float32)
            arr_f16 = arr_f32.astype(np.float16)
            new_data = bytearray(arr_f16.tobytes())
            buffer.data = new_data
            buffer.size = len(new_data)
            buffer.offset = 0
            tensor.type = schema_fb.TensorType.FLOAT16
            converted_buffers.add(buf_idx)
    return weight_indices


def _get_or_add_cast_opcode_index(model: "schema_fb.ModelT") -> int:
    """Return operator code index for CAST; add to model if missing."""
    for i, op_code in enumerate(model.operatorCodes):
        code = max(op_code.builtinCode, op_code.deprecatedBuiltinCode)
        if code == schema_fb.BuiltinOperator.CAST:
            return i
    new_code = schema_fb.OperatorCodeT()
    new_code.builtinCode = schema_fb.BuiltinOperator.CAST
    new_code.deprecatedBuiltinCode = 0
    model.operatorCodes.append(new_code)
    return len(model.operatorCodes) - 1


def _insert_cast_fp16_to_fp32_ops(model: "schema_fb.ModelT", fp16_tensors: set[tuple[int, int]]) -> None:
    """Add CAST (FLOAT16->FLOAT32) for each FP16 weight so activations stay FP32 (runnable on CPU)."""
    if not fp16_tensors:
        return
    cast_opcode_idx = _get_or_add_cast_opcode_index(model)
    for sg_idx, subgraph in enumerate(model.subgraphs):
        if subgraph.tensors is None or subgraph.operators is None:
            continue
        fp16_in_sg = {ten_idx for (s, ten_idx) in fp16_tensors if s == sg_idx}
        if not fp16_in_sg:
            continue
        old_to_new: dict[int, int] = {}
        for old_idx in sorted(fp16_in_sg):
            old_tensor = subgraph.tensors[old_idx]
            new_tensor = schema_fb.TensorT()
            new_tensor.shape = list(old_tensor.shape) if getattr(old_tensor, "shape", None) is not None else []
            new_tensor.type = schema_fb.TensorType.FLOAT32
            new_tensor.buffer = 0
            name = old_tensor.name
            if name is None:
                name = ""
            elif isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            new_tensor.name = str(name) + "_cast_fp32"
            subgraph.tensors.append(new_tensor)
            new_idx = len(subgraph.tensors) - 1
            old_to_new[old_idx] = new_idx
            cast_op = schema_fb.OperatorT()
            cast_op.opcodeIndex = cast_opcode_idx
            cast_op.inputs = [old_idx]
            cast_op.outputs = [new_idx]
            cast_op.builtinOptions = schema_fb.CastOptionsT()
            cast_op.builtinOptions.inDataType = schema_fb.TensorType.FLOAT16
            cast_op.builtinOptions.outDataType = schema_fb.TensorType.FLOAT32
            cast_op.builtinOptionsType = schema_fb.BuiltinOptions.CastOptions
            subgraph.operators.append(cast_op)
        for op in subgraph.operators:
            if op.opcodeIndex == cast_opcode_idx or op.inputs is None:
                continue
            for i in range(len(op.inputs)):
                if op.inputs[i] in old_to_new:
                    op.inputs[i] = old_to_new[op.inputs[i]]


def _set_all_float32_tensors_to_float16(model: "schema_fb.ModelT") -> None:
    """Set every FLOAT32 tensor to FLOAT16 (pure FP16 model)."""
    F32 = schema_fb.TensorType.FLOAT32
    F16 = schema_fb.TensorType.FLOAT16
    for subgraph in model.subgraphs or []:
        if subgraph.tensors is None:
            continue
        for tensor in subgraph.tensors:
            if tensor.type == F32:
                tensor.type = F16


def _quantize_tflite_to_fp16(input_path: Path, output_path: Path, pure_fp16: bool) -> None:
    """Load float32 TFLite, convert to float16 (pure or runnable), write to output_path."""
    print(f"Loading {input_path}...")
    model = _read_tflite_model(input_path)
    print("Converting weight buffers to float16...")
    fp16_tensors = _convert_weights_to_float16(model)
    if pure_fp16:
        print("Setting all tensors to float16 (pure FP16)...")
        _set_all_float32_tensors_to_float16(model)
    else:
        print("Inserting CAST (FLOAT16->FLOAT32) for weights so activations stay float32 (runnable on CPU)...")
        _insert_cast_fp16_to_fp32_ops(model, fp16_tensors)
    if sys.byteorder == "big":
        flatbuffer_utils.byte_swap_tflite_model_obj(model, "big", "little")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flatbuffer_utils.write_model(model, str(output_path))
    print(f"Saved FP16 TFLite to {output_path}")
    in_size = input_path.stat().st_size
    out_size = output_path.stat().st_size
    print(f"Size: {in_size / 1024:.1f} KB -> {out_size / 1024:.1f} KB ({100 * out_size / in_size:.0f}%)")


def _step2_quantize_fp16(
    intermediate_tflite: Path,
    output_tflite: Path,
    runnable_tflite: Path | None,
) -> None:
    """Stage 2: float32 -> pure FP16 (output path). If runnable enabled, also write runnable FP16 for stage 4."""
    print("[Step 2] float32 -> pure FP16")
    _quantize_tflite_to_fp16(intermediate_tflite, output_tflite, pure_fp16=True)
    if runnable_tflite is not None:
        print("[Step 2] float32 -> runnable FP16 (for comparison)")
        _quantize_tflite_to_fp16(intermediate_tflite, runnable_tflite, pure_fp16=False)


# =============================================================================
# Stage 3: test_fp16 – verify pure FP16 (from test_fp16.py)
# =============================================================================

def _step3_test_fp16(output_tflite: Path) -> None:
    """Stage 3: load model (with FP16 shim if needed), verify no FP32 tensors."""
    print("[Step 3] test_fp16: verify pure FP16")
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    try:
        shim = importlib.import_module(FP16_INTERPRETER_SHIM_MODULE)
        shim.patch_interpreter()
    except ImportError as e:
        print(f"Warning: {FP16_INTERPRETER_SHIM_MODULE} not found ({e}); trying default interpreter.", file=sys.stderr)
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=str(output_tflite))
    interpreter.allocate_tensors()
    tensor_details = interpreter.get_tensor_details()
    fp32_tensors = [t for t in tensor_details if t["dtype"] == np.float32]
    if fp32_tensors:
        print("❌ FP32 tensors detected:", file=sys.stderr)
        for t in fp32_tensors:
            print(f'  - {t["name"]}, index={t["index"]}, shape={t["shape"]}', file=sys.stderr)
        raise RuntimeError("Model contains FP32 tensors. Not pure FP16.")
    print("✅ Model is pure FP16 (no FP32 tensors detected).")


# =============================================================================
# Stage 4: compare original vs quantized on images (aliked_compare_stats_vid style)
# =============================================================================

def _get_image_files(directory: Path) -> list[Path]:
    """Sorted image files."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    files = []
    for ext in exts:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def _prepare_image_nhwc(
    image_bgr: np.ndarray,
    input_h: int = MODEL_INPUT_HEIGHT,
    input_w: int = MODEL_INPUT_WIDTH,
) -> np.ndarray:
    """Prepare image for TFLite: BGR->RGB, pad to square, resize, normalize [0,1], NHWC (1,H,W,3)."""
    import cv2
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0
    if orig_h < orig_w:
        pad_top = (orig_w - orig_h) // 2
        pad_bottom = orig_w - orig_h - pad_top
    elif orig_h > orig_w:
        pad_left = (orig_h - orig_w) // 2
        pad_right = orig_h - orig_w - pad_left
    if pad_top or pad_bottom or pad_left or pad_right:
        image_rgb = np.pad(
            image_rgb,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    image_resized = cv2.resize(image_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image_float = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_float, axis=0)


def _run_tflite(interpreter, input_data: np.ndarray) -> list[np.ndarray]:
    """Run TFLite interpreter; return list of output arrays. input_data is NCHW or NHWC per model."""
    inp = interpreter.get_input_details()[0]
    interpreter.set_tensor(inp["index"], input_data)
    interpreter.invoke()
    return [interpreter.get_tensor(d["index"]).copy() for d in interpreter.get_output_details()]


def _run_pth_model(wrapper, input_nchw: np.ndarray) -> list[np.ndarray]:
    """Run PyTorch wrapper on NCHW input; return [score_map, feature_map] as numpy."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(input_nchw.astype(np.float32))
        score_map, feature_map = wrapper(x)
        return [score_map.numpy(), feature_map.numpy()]


# Score map has 1 channel; feature map can be 64 (e.g. aliked-t16) or 128 (e.g. aliked-n16).
SCORE_CHANNELS = (1,)
FEATURE_CHANNELS = (64, 128)

# Keypoint extraction from score_map and drawing (stage 4)
# Match aliked_compare_stats_vid_tf.py: .pth uses full ALIKED keypoints (DKD), TFLite uses top-k from score_map.
MATCH_RADIUS_PX = 4.0
KEYPOINT_TOP_K = 500
KEYPOINT_TOP_K_TFLITE_VIDTF = 512  # same as aliked_compare_stats_vid_tf.py for TFLite map format
KEYPOINT_THRESHOLD = 0.01
KEYPOINT_NMS_SIZE = 5
DRAW_RADIUS = 10
DRAW_OUTLINE_WIDTH = 4


def _channel_count_4d(arr: np.ndarray) -> int:
    """Channel count for 4D: NCHW -> shape[1], NHWC -> shape[3]. Else -1."""
    s = arr.shape
    if len(s) != 4:
        return -1
    if s[1] in (*SCORE_CHANNELS, *FEATURE_CHANNELS):
        return int(s[1])
    if s[3] in (*SCORE_CHANNELS, *FEATURE_CHANNELS):
        return int(s[3])
    return -1


def _to_nchw_4d(arr: np.ndarray) -> np.ndarray:
    """If NHWC (1, H, W, C) with C in score/feature channel set, return NCHW. Else return copy."""
    if arr.ndim != 4:
        return arr
    s = arr.shape
    if s[3] in (*SCORE_CHANNELS, *FEATURE_CHANNELS) and s[1] not in (*SCORE_CHANNELS, *FEATURE_CHANNELS):
        return np.ascontiguousarray(np.transpose(arr, (0, 3, 1, 2)))
    return arr.copy()


def _align_tflite_outputs_to_pth(tflite_outs: list[np.ndarray], pth_outs: list[np.ndarray]) -> list[np.ndarray]:
    """Return [score_map, feature_map] from TFLite in same order/layout as PyTorch (NCHW)."""
    if len(tflite_outs) != 2 or len(pth_outs) != 2:
        return tflite_outs
    c0, c1 = _channel_count_4d(tflite_outs[0]), _channel_count_4d(tflite_outs[1])
    if c0 in SCORE_CHANNELS and c1 in FEATURE_CHANNELS:
        sm, fm = tflite_outs[0], tflite_outs[1]
    elif c0 in FEATURE_CHANNELS and c1 in SCORE_CHANNELS:
        sm, fm = tflite_outs[1], tflite_outs[0]
    else:
        return tflite_outs
    return [_to_nchw_4d(sm), _to_nchw_4d(fm)]


def _resize_4d_to_match(arr: np.ndarray, target_shape: tuple, use_cv2: bool = True) -> np.ndarray:
    """Resize 4D array (NCHW) to target_shape (N,C,H,W); spatial dims only. Returns float32."""
    if arr.shape == target_shape:
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    import cv2
    n, c, h, w = arr.shape
    N, C, H, W = target_shape
    if n != N or c != C:
        return arr.astype(np.float32)
    out = np.empty((N, C, H, W), dtype=np.float32)
    for b in range(n):
        for ch in range(c):
            out[b, ch] = cv2.resize(
                arr[b, ch].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR
            )
    return out


def _score_map_to_keypoints(
    score_map_2d: np.ndarray,
    top_k: int = KEYPOINT_TOP_K,
    threshold: float = KEYPOINT_THRESHOLD,
    nms_size: int = KEYPOINT_NMS_SIZE,
) -> np.ndarray:
    """Extract keypoint (x, y) positions from score map [H, W]. Returns (N, 2) in model coords (x, y)."""
    try:
        from scipy.ndimage import maximum_filter
    except ImportError:
        maximum_filter = None
    sm = np.asarray(score_map_2d, dtype=np.float32)
    if sm.ndim != 2:
        return np.zeros((0, 2), dtype=np.float64)
    above = sm >= threshold
    if maximum_filter is not None and nms_size > 1:
        local_max = sm == maximum_filter(sm, size=nms_size, mode="constant", cval=0)
        above = above & local_max
    ys, xs = np.where(above)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    scores = sm[ys, xs]
    order = np.argsort(-scores)[:top_k]
    xs, ys = xs[order].astype(np.float64), ys[order].astype(np.float64)
    return np.column_stack((xs, ys))


def _keypoints_model_to_orig(
    kpts_xy: np.ndarray,
    input_h: int,
    input_w: int,
    pad_left: int,
    pad_top: int,
    max_dim: int,
) -> np.ndarray:
    """Map keypoints from model grid [0..input_w, 0..input_h] to original image pixel coords. kpts_xy (N,2) x,y."""
    if len(kpts_xy) == 0:
        return kpts_xy.copy()
    x_m, y_m = kpts_xy[:, 0], kpts_xy[:, 1]
    x_pad = x_m * max_dim / max(input_w, 1)
    y_pad = y_m * max_dim / max(input_h, 1)
    x_orig = x_pad - pad_left
    y_orig = y_pad - pad_top
    return np.column_stack((x_orig, y_orig))


def _score_map_to_keypoints_vid_tf_style(
    score_map_2d: np.ndarray,
    orig_w: int,
    orig_h: int,
    top_k: int = KEYPOINT_TOP_K_TFLITE_VIDTF,
) -> np.ndarray:
    """Top-k by score from score map, map to original image coords. Same as aliked_compare_stats_vid_tf.py TFLite path.
    Uses (grid_x + 0.5) * orig_w / map_w and (grid_y + 0.5) * orig_h / map_h; no threshold/NMS."""
    sm = np.asarray(score_map_2d, dtype=np.float32)
    if sm.ndim != 2:
        return np.zeros((0, 2), dtype=np.float64)
    map_h, map_w = sm.shape
    n_kp = min(top_k, sm.size)
    if n_kp <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    flat = sm.ravel()
    idx = np.argpartition(flat, -n_kp)[-n_kp:]
    idx = idx[np.argsort(-flat[idx])]
    grid_ij = np.unravel_index(idx, (map_h, map_w))
    grid_y = np.asarray(grid_ij[0], dtype=np.float64)
    grid_x = np.asarray(grid_ij[1], dtype=np.float64)
    scale_x = orig_w / max(map_w, 1)
    scale_y = orig_h / max(map_h, 1)
    kpts_px = np.column_stack(((grid_x + 0.5) * scale_x, (grid_y + 0.5) * scale_y))
    valid = (
        (kpts_px[:, 0] >= 0) & (kpts_px[:, 0] < orig_w)
        & (kpts_px[:, 1] >= 0) & (kpts_px[:, 1] < orig_h)
    )
    return kpts_px[valid]


def _keypoint_spatial_overlap(
    keypoints_a: np.ndarray,
    keypoints_b: np.ndarray,
    radius_px: float = MATCH_RADIUS_PX,
) -> dict:
    """One-to-one overlap within radius. Returns dict with matched_a_indices, matched_b_indices, etc."""
    keypoints_a = np.asarray(keypoints_a, dtype=np.float64)
    keypoints_b = np.asarray(keypoints_b, dtype=np.float64)
    num_a, num_b = len(keypoints_a), len(keypoints_b)
    if num_a == 0 or num_b == 0:
        return {
            "num_a": num_a,
            "num_b": num_b,
            "matched_a_indices": [],
            "matched_b_indices": [],
        }
    dx = keypoints_a[:, np.newaxis, 0] - keypoints_b[np.newaxis, :, 0]
    dy = keypoints_a[:, np.newaxis, 1] - keypoints_b[np.newaxis, :, 1]
    dist = np.sqrt(dx * dx + dy * dy)
    within_radius = dist <= radius_px
    i_a, i_b = np.where(within_radius)
    used_a, used_b = [], []
    if len(i_a) > 0:
        d_flat = dist[i_a, i_b]
        order = np.argsort(d_flat)
        used_a_set, used_b_set = set(), set()
        for idx in order:
            ia, ib = int(i_a[idx]), int(i_b[idx])
            if ia not in used_a_set and ib not in used_b_set:
                used_a_set.add(ia)
                used_b_set.add(ib)
                used_a.append(ia)
                used_b.append(ib)
    return {
        "num_a": num_a,
        "num_b": num_b,
        "matched_a_indices": used_a,
        "matched_b_indices": used_b,
    }


def _draw_point_with_outline(
    frame: np.ndarray,
    x: float,
    y: float,
    color: tuple,
    radius: int = DRAW_RADIUS,
    outline_width: int = DRAW_OUTLINE_WIDTH,
) -> None:
    import cv2
    xi, yi = int(x), int(y)
    cv2.circle(frame, (xi, yi), radius + outline_width, (0, 0, 0), outline_width)
    cv2.circle(frame, (xi, yi), radius, color, -1)


def _draw_overlap_classified(
    frame: np.ndarray,
    kpts_original: np.ndarray,
    kpts_quantized: np.ndarray,
    overlap: dict,
) -> np.ndarray:
    """Green=overlap, blue=original-only, red=quantized-only (BGR)."""
    matched_a = set(overlap.get("matched_a_indices", []))
    matched_b = set(overlap.get("matched_b_indices", []))
    num_a = len(kpts_original)
    num_b = len(kpts_quantized)
    for i in matched_a:
        x, y = kpts_original[i][0], kpts_original[i][1]
        _draw_point_with_outline(frame, x, y, (0, 255, 0))
    for i in range(num_a):
        if i not in matched_a:
            x, y = kpts_original[i][0], kpts_original[i][1]
            _draw_point_with_outline(frame, x, y, (255, 0, 0))
    for j in range(num_b):
        if j not in matched_b:
            x, y = kpts_quantized[j][0], kpts_quantized[j][1]
            _draw_point_with_outline(frame, x, y, (0, 0, 255))
    return frame


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    n = np.linalg.norm(a_flat) * np.linalg.norm(b_flat)
    if n < 1e-12:
        return 1.0 if np.allclose(a_flat, b_flat) else 0.0
    return float(np.dot(a_flat, b_flat) / n)


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# Default FPS and fourcc for step 4 comparison AVI video
VIDEO_FPS = 10.0
VIDEO_FOURCC_AVI = "MJPG"  # MJPEG for .avi compatibility


def _step4_compare_stats(
    pth_path: Path,
    runnable_tflite: Path,
    image_dir: Path,
    metrics_path: Path,
    video_output_path: Path | None = None,
) -> None:
    """Stage 4: compare keypoints from model A (.pth) vs model B (.tflite FP16).

    - Keypoints from model A: aliked-t16.pth (original).
    - Keypoints from model B: aliked-t16_fp16.tflite / runnable variant (quantized).
    We compute overlapping points percentage and plot: green = overlap, blue = unique to A,
    red = unique to B. We never compare the same model to itself (A and B are always different).
    """
    # Safeguard: never compare the same model to itself (different paths and types).
    pth_path = pth_path.resolve()
    runnable_tflite = runnable_tflite.resolve()
    if pth_path == runnable_tflite:
        raise ValueError(
            "Step 4 must compare two different models (.pth vs .tflite). "
            "Same path was given for both. Refusing to compare model to itself."
        )
    if pth_path.suffix.lower() != ".pth" or runnable_tflite.suffix.lower() != ".tflite":
        raise ValueError(
            "Step 4 expects model A as .pth and model B as .tflite. "
            f"Got A={pth_path.suffix!r} B={runnable_tflite.suffix!r}."
        )
    try:
        import cv2
    except ImportError as e:
        print(f"[Step 4] Skipping: opencv-python required ({e})", file=sys.stderr)
        return
    import tensorflow as tf
    print("[Step 4] Compare keypoints: model A (.pth) vs model B (.tflite FP16)")
    print(f"  Model A (original):   {pth_path}")
    print(f"  Model B (quantized): {runnable_tflite}")
    print("Loading model A (.pth) as full ALIKED (keypoints from DKD, dense from extract_dense_map)...")
    full_aliked, pth_device = _load_full_aliked(pth_path)
    interp_quant = tf.lite.Interpreter(model_path=str(runnable_tflite))
    interp_quant.allocate_tensors()
    input_shape = tuple(interp_quant.get_input_details()[0]["shape"])
    if len(input_shape) == 4:
        # NCHW: (1, 3, H, W) -> H,W from dims 2,3; NHWC: (1, H, W, 3) -> H,W from dims 1,2
        if input_shape[1] == 3:
            input_h, input_w = int(input_shape[2]), int(input_shape[3])
            input_is_nchw = True
        else:
            input_h, input_w = int(input_shape[1]), int(input_shape[2])
            input_is_nchw = False
    else:
        input_h, input_w = MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH
        input_is_nchw = False
    image_files = _get_image_files(image_dir)
    if not image_files:
        raise ValueError(f"No image files in {image_dir}")
    total_images = len(image_files)
    print(f"Found {total_images} images in {image_dir}")
    print("Processing images... Press 'q' during display to stop.\n")
    if video_output_path is None:
        video_output_path = metrics_path.parent / "comparison_points.avi"
    video_output_path = video_output_path.resolve()
    video_output_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer: cv2.VideoWriter | None = None
    video_size: tuple[int, int] | None = None  # (width, height)
    shape_mismatch_warned = False
    sum_mse_score = 0.0
    sum_mse_feat = 0.0
    sum_cos_score = 0.0
    sum_cos_feat = 0.0
    sum_overlap_pct = 0.0
    n_processed = 0
    per_image_lines = []
    for idx, image_path in enumerate(image_files):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        input_nhwc = _prepare_image_nhwc(frame, input_h, input_w)
        if input_is_nchw:
            input_data = np.transpose(input_nhwc, (0, 3, 1, 2)).astype(np.float32)
        else:
            input_data = input_nhwc.astype(np.float32)
        # Original: .pth dense from extract_dense_map, keypoints from full forward (DKD)
        outs_pth = _run_full_aliked_dense(full_aliked, pth_device, input_data)
        # Quantized: FP16 TFLite; align to [score_map, feature_map] order/layout
        outs_tflite = _run_tflite(interp_quant, input_data)
        outs_tflite = _align_tflite_outputs_to_pth(outs_tflite, outs_pth)
        if len(outs_pth) != 2 or len(outs_tflite) != 2:
            if not shape_mismatch_warned:
                print(f"Warning: skipping images - output count mismatch (pth={len(outs_pth)}, tflite={len(outs_tflite)})", file=sys.stderr)
                shape_mismatch_warned = True
            continue
        so, fo = outs_pth[0], outs_pth[1]
        sq, fq = outs_tflite[0], outs_tflite[1]
        if so.shape != sq.shape or fo.shape != fq.shape:
            if not shape_mismatch_warned:
                print(f"Note: .pth vs TFLite output shape mismatch; resizing TFLite to match .pth (pth score={so.shape} feat={fo.shape}, tflite score={sq.shape} feat={fq.shape})", file=sys.stderr)
                shape_mismatch_warned = True
            sq = _resize_4d_to_match(sq, so.shape)
            fq = _resize_4d_to_match(fq, fo.shape)
        mse_s = _mse(so, sq)
        mse_f = _mse(fo, fq)
        cos_s = _cosine_similarity(so, sq)
        cos_f = _cosine_similarity(fo, fq)
        sum_mse_score += mse_s
        sum_mse_feat += mse_f
        sum_cos_score += cos_s
        sum_cos_feat += cos_f
        n_processed += 1
        # Keypoints: .pth from full ALIKED (DKD), TFLite from top-k score map (same as aliked_compare_stats_vid_tf.py)
        orig_h, orig_w = frame.shape[:2]
        max_dim = max(orig_h, orig_w)
        pad_left = (max_dim - orig_w) // 2 if orig_w < max_dim else 0
        pad_top = (max_dim - orig_h) // 2 if orig_h < max_dim else 0
        score_tflite_2d = sq[0, 0] if sq.ndim >= 2 else sq.squeeze()
        kpts_orig = _get_pth_keypoints_from_full_aliked(
            full_aliked, pth_device, input_data, orig_h, orig_w, pad_left, pad_top, max_dim
        )
        kpts_quant = _score_map_to_keypoints_vid_tf_style(
            score_tflite_2d, orig_w, orig_h, top_k=KEYPOINT_TOP_K_TFLITE_VIDTF
        )
        overlap = _keypoint_spatial_overlap(kpts_orig, kpts_quant, radius_px=MATCH_RADIUS_PX)
        num_a = overlap.get("num_a", 0)
        num_b = overlap.get("num_b", 0)
        n_overlap = len(overlap.get("matched_a_indices", []))
        total_kp = num_a + num_b
        overlap_pct = (100.0 * 2 * n_overlap / total_kp) if total_kp > 0 else 0.0
        sum_overlap_pct += overlap_pct
        line = (
            f"image={image_path.name} score_map_mse={mse_s:.6e} feature_map_mse={mse_f:.6e} "
            f"score_map_cosine={cos_s:.4f} feature_map_cosine={cos_f:.4f} "
            f"keypoint_overlap_pct={overlap_pct:.2f} num_a={num_a} num_b={num_b} n_matched={n_overlap}"
        )
        per_image_lines.append(line)

        # Show frame with keypoints: green=overlap, blue=model A only, red=model B only
        vis = frame.copy()
        _draw_overlap_classified(vis, kpts_orig, kpts_quant, overlap)
        font = cv2.FONT_HERSHEY_SIMPLEX
        n_orig_only = num_a - n_overlap
        n_quant_only = num_b - n_overlap
        cv2.putText(vis, f"Green: overlap ({n_overlap}) | overlap % = {overlap_pct:.1f}%", (10, 28), font, 0.6, (0, 255, 0), 2)
        cv2.putText(vis, f"Blue: model A only ({n_orig_only})", (10, 52), font, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f"Red: model B only ({n_quant_only})", (10, 76), font, 0.6, (0, 0, 255), 2)
        cv2.putText(vis, f"score_cos={cos_s:.3f} feat_cos={cos_f:.3f}", (10, 100), font, 0.5, (255, 255, 255), 1)
        cv2.putText(vis, f"{image_path.name} ({n_processed}/{total_images})", (10, vis.shape[0] - 10), font, 0.6, (255, 255, 255), 1)
        # Write frame to AVI video
        h, w = vis.shape[:2]
        if video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC_AVI)
            video_writer = cv2.VideoWriter(str(video_output_path), fourcc, VIDEO_FPS, (w, h))
            video_size = (w, h)
            print(f"Writing comparison video to {video_output_path}")
        if (w, h) != video_size and video_size is not None:
            vis_write = cv2.resize(vis, video_size, interpolation=cv2.INTER_LINEAR)
        else:
            vis_write = vis
        video_writer.write(vis_write)
        cv2.imshow(".pth vs .tflite FP16 (press q to stop)", vis)
        if (n_processed % 10 == 0 or n_processed == total_images) and n_processed > 0:
            print(f"Processed {n_processed}/{total_images} - {image_path.name} - score_cos={cos_s:.3f} feat_cos={cos_f:.3f}")
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Stopped by user.")
            break

    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
        print(f"Saved comparison video to {video_output_path}")
    if n_processed == 0:
        print("No images processed in step 4.", file=sys.stderr)
        return
    n = n_processed
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        f.write("Step 4: Keypoint comparison — model A (.pth) vs model B (.tflite FP16). Never same model vs itself.\n")
        f.write("Dense (score_map, feature_map) + keypoint overlap %% and per-point classification (overlap / A only / B only).\n")
        f.write("=" * 72 + "\n\n")
        f.write("Models compared (two different models only):\n")
        f.write(f"  Model A (original):   {pth_path}\n")
        f.write(f"  Model B (quantized):  {runnable_tflite}\n\n")
        f.write(f"Comparison video (AVI): {video_output_path}\n")
        f.write(f"Image directory: {image_dir}\n")
        f.write(f"Images processed: {n}\n\n")
        f.write("--- Aggregate (averages) ---\n")
        f.write(f"  score_map   MSE: {sum_mse_score / n:.6e}  cosine_sim: {sum_cos_score / n:.4f}\n")
        f.write(f"  feature_map MSE: {sum_mse_feat / n:.6e}  cosine_sim: {sum_cos_feat / n:.4f}\n")
        f.write(f"  keypoint overlap %% (2*matched/(num_a+num_b)): {sum_overlap_pct / n:.2f}%%\n\n")
        f.write("--- Per-image ---\n\n")
        for line in per_image_lines:
            f.write(line + "\n\n")
    print(f"Step 4 done. Metrics saved to {metrics_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full pipeline: .pth -> TFLite -> FP16 quantized -> test -> compare on images (all in one script).",
    )
    parser.add_argument("input_pth", type=Path, help="Input .pth model (e.g. aliked-t16.pth)")
    parser.add_argument("output_tflite", type=Path, help="Output quantized .tflite (e.g. aliked-t16_fp16.tflite)")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help=f"Image directory for step 4 (default: {DEFAULT_IMAGE_DIR}). If missing, step 4 is skipped.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help=f"Path for step 4 metrics .txt (default: script_dir/{DEFAULT_METRICS_FILENAME})",
    )
    parser.add_argument(
        "--video-output",
        type=Path,
        default=None,
        help="Path for step 4 comparison AVI video (default: next to metrics file as comparison_points.avi).",
    )
    parser.add_argument("--skip-step1", action="store_true", help="Skip step 1; use existing float32 .tflite.")
    parser.add_argument(
        "--intermediate-tflite",
        type=Path,
        default=None,
        help="Path to existing float32 .tflite when using --skip-step1.",
    )
    parser.add_argument(
        "--use-onnx",
        action="store_true",
        help="Use PyTorch -> ONNX -> TFLite for step 1 (no litert_torch).",
    )
    args = parser.parse_args()
    input_pth = args.input_pth.resolve()
    output_tflite = args.output_tflite.resolve()
    intermediate_tflite = output_tflite.parent / (input_pth.stem + ".tflite")
    runnable_tflite = (output_tflite.parent / (output_tflite.stem + RUNNABLE_TFLITE_SUFFIX)) if ENABLE_RUNNABLE else None

    if args.skip_step1:
        if args.intermediate_tflite is None:
            print("Error: --skip-step1 requires --intermediate-tflite", file=sys.stderr)
            sys.exit(1)
        intermediate_tflite = args.intermediate_tflite.resolve()
        if not intermediate_tflite.exists():
            print(f"Error: intermediate .tflite not found: {intermediate_tflite}", file=sys.stderr)
            sys.exit(1)
        print("[Step 1] Skipped (using existing intermediate TFLite).")
    else:
        if not input_pth.exists():
            print(f"Error: input .pth not found: {input_pth}", file=sys.stderr)
            sys.exit(1)
        _step1_pth_to_tflite(input_pth, intermediate_tflite, use_onnx=args.use_onnx)

    _step2_quantize_fp16(intermediate_tflite, output_tflite, runnable_tflite)
    _step3_test_fp16(output_tflite)

    image_dir = Path(DEFAULT_IMAGE_DIR) if args.image_dir is None else args.image_dir.resolve()
    if runnable_tflite is not None and image_dir.exists() and image_dir.is_dir():
        metrics_path = SCRIPT_DIR / DEFAULT_METRICS_FILENAME if args.metrics_output is None else args.metrics_output.resolve()
        video_path = args.video_output.resolve() if args.video_output is not None else None
        _step4_compare_stats(input_pth, runnable_tflite, image_dir, metrics_path, video_output_path=video_path)
    elif runnable_tflite is None:
        print("[Step 4] Skipping (runnable disabled; set ENABLE_RUNNABLE = True to enable comparison on images).")
    else:
        print(f"[Step 4] Skipping (image dir not found or not a directory: {image_dir})")
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
