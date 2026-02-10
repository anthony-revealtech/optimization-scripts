from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
)
import numpy as np
import onnx
import os
import sys
from io import StringIO
from glob import glob

import libs.comparison_test as comparison_test
from libs.mse_calibration import run_mse_calibration

try:
    from PIL import Image
except ImportError:
    Image = None
'''
model_fp32 = '../models/lightglue_1024kp.onnx'
model_quant = '../models/lightglue_1024kp_quantized_adaptive.onnx'
'''
model_fp32 = '../models/aliked-n16_640x640_512kp/aliked-n16_640x640_512kp.onnx'
model_quant = '../models/aliked-n16_640x640_512kp/aliked-n16_640x640_512kp_INT8.onnx'




report_path = 'adaptive_quantization_report.txt'

# Same input size for calibration and evaluation so SQNR and output shapes are comparable.
DYNAMIC_SIZE = 256
# KS test approves when statistic <= this (adaptive Q often has slightly different weight CDF).
KS_APPROVAL_MAX_STAT = 0.25
# Hardcoded: "int8" or "int16" — controls quantization bit width and report label.
# Using Int16 for better precision and lower KS statistic
QUANT_BITS = "Int8"

# Map QUANT_BITS to activation/weight types and report label.
_use_int16 = str(QUANT_BITS).strip().lower() == "int16"
activation_type = QuantType.QInt16 if _use_int16 else QuantType.QInt8
weight_type = QuantType.QInt16 if _use_int16 else QuantType.QInt8
inference_mode_label = "Int16" if _use_int16 else "Int8"

# QDQ format; MatMulInteger is int8-only so int16 runtime will dequantize to float.
quant_format = QuantFormat.QDQ

# Calibration dataset directory (images). Use real data for better int8 accuracy; None = random data only.
CALIBRATION_DATA_DIR = "/Users/antlowhur/Documents/Programming/jagr-data/data/vanafi_polygon_6_18_2020_300msq_121m_altitude/data"

# Calibration method: "MinMax" (built-in) or "MSE" (custom MSE minimization of quantization error).
CALIBRATION_METHOD = "MSE"


def _get_calibration_shapes(model_path, dynamic_size=1):
    """Get fixed input shapes from model (dynamic dims → dynamic_size)."""
    model = onnx.load(model_path)
    shapes = {}
    for inp in model.graph.input:
        name = inp.name
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_value:
                shape.append(d.dim_value)
            else:
                shape.append(dynamic_size)
        shapes[name] = tuple(shape)
    return shapes


def _load_calibration_images(calibration_data_dir, shapes, max_samples=None):
    """
    Load images from directory and preprocess to match model input shapes.
    Returns list of dicts: [{input_name: np.ndarray (float32)}, ...].
    Expects single input with shape (N, C, H, W); images are resized to (H, W) and converted to NCHW float32 in [0, 1].
    """
    if Image is None or not calibration_data_dir or not os.path.isdir(calibration_data_dir):
        return []
    # Discover image paths (JPG, PNG, etc.)
    exts = ("*.jpg", "*.jpeg", "*.JPG", "*.JPEG", "*.png", "*.PNG")
    paths = []
    for ext in exts:
        paths.extend(glob(os.path.join(calibration_data_dir, ext)))
    paths = sorted(paths)
    if max_samples is not None:
        paths = paths[:max_samples]
    if not paths:
        return []

    # Get target shape from first input (assume single input with NCHW)
    input_name = next(iter(shapes))
    shape = shapes[input_name]
    if len(shape) != 4:
        return []
    _, c, h, w = shape
    if c != 3:
        return []

    samples = []
    for path in paths:
        try:
            img = Image.open(path)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
            # Resize to (h, w)
            if (img.shape[0], img.shape[1]) != (h, w):
                img_pil = Image.fromarray((img * 255).astype(np.uint8))
                img_pil = img_pil.resize((w, h), Image.BILINEAR)
                img = np.array(img_pil, dtype=np.float32) / 255.0
            # HWC -> NCHW
            img = img.transpose(2, 0, 1)
            img = np.expand_dims(img, axis=0).astype(np.float32)
            samples.append({input_name: img})
        except Exception:
            continue
    return samples


class AdaptiveCalibrationDataReader(CalibrationDataReader):
    """Calibration reader: uses real images from calibration_data_dir when provided and Pillow is available; otherwise uses model input spec with random data."""

    def __init__(self, model_path, num_samples=10, dynamic_size=1, calibration_data_dir=None):
        self.num_samples = num_samples
        self.current = 0
        self.shapes = _get_calibration_shapes(model_path, dynamic_size)
        self.calibration_data_dir = calibration_data_dir
        self.image_samples = _load_calibration_images(
            calibration_data_dir, self.shapes, max_samples=num_samples
        )
        self.use_images = len(self.image_samples) > 0

    def get_next(self):
        if self.current >= self.num_samples:
            return None

        if self.use_images and self.current < len(self.image_samples):
            out = self.image_samples[self.current]
        else:
            out = {
                name: np.random.randn(*shape).astype(np.float32)
                for name, shape in self.shapes.items()
            }
        self.current += 1
        return out

    def rewind(self):
        self.current = 0


# Adaptive quantization: FP32 → int8 or int16 (from QUANT_BITS), QDQ format.
# Calibration: MinMax (built-in) or MSE (custom MSE minimization of quantization error).
# Use fewer samples if you hit OOM (e.g. 50–100); 500 can need 10+ GB RAM with MSE.
num_calib_samples = 100
calib_reader = AdaptiveCalibrationDataReader(
    model_fp32,
    num_samples=num_calib_samples,
    dynamic_size=DYNAMIC_SIZE,
    calibration_data_dir=CALIBRATION_DATA_DIR,
)

if CALIBRATION_METHOD.upper() == "MSE":
    # MSE path: 1) Quantize with MinMax to get QDQ structure; 2) Collect activations and optimize scales.
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        qdq_minmax_path = f.name
    try:
        quantize_static(
            model_input=model_fp32,
            model_output=qdq_minmax_path,
            calibration_data_reader=calib_reader,
            calibrate_method=CalibrationMethod.MinMax,
            quant_format=quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=False,
            extra_options={
                "CalibMovingAverage": True,
                "CalibMovingAverageConstant": 0.5,
                **({"ActivationSymmetric": True} if not _use_int16 else {}),
            },
        )
        # Build calibration batches (same data as reader)
        calib_reader.rewind()
        calibration_batches = []
        while True:
            batch = calib_reader.get_next()
            if batch is None:
                break
            calibration_batches.append(batch)
        mse_scales = run_mse_calibration(
            fp32_model_path=model_fp32,
            qdq_model_path=qdq_minmax_path,
            calibration_batches=calibration_batches,
            output_path=model_quant,
            num_bits=8 if not _use_int16 else 16,
            symmetric=True,
            dynamic_size=DYNAMIC_SIZE,
        )
        print(f"MSE calibration complete. Updated {len(mse_scales)} activation scale(s).")
    finally:
        try:
            os.unlink(qdq_minmax_path)
        except FileNotFoundError:
            pass
else:
    quantize_static(
        model_input=model_fp32,
        model_output=model_quant,
        calibration_data_reader=calib_reader,
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=False,
        extra_options={
            "CalibMovingAverage": True,
            "CalibMovingAverageConstant": 0.5,
            **({"ActivationSymmetric": True} if not _use_int16 else {}),
        },
    )
    print("Adaptive quantization complete!")




# Run comparison first (before loading ONNX protos) so FP32/Quant memory deltas
# are comparable across adaptive vs static scripts; same baseline = post-quant only.
header = f"Adaptive Quantization (inference {inference_mode_label})\n"
old_stdout = sys.stdout
sys.stdout = buf = StringIO()
try:
    comparison_test.compare_models(model_fp32, model_quant, dynamic_size=DYNAMIC_SIZE)
    # Load models only for KS and JS tests
    model_original = onnx.load(model_fp32)
    model_quantized = onnx.load(model_quant)
    print("\n--- KS Test: Comparing Weight Distributions ---")
    comparison_test.ks_test(model_original, model_quantized, approval_max_stat=KS_APPROVAL_MAX_STAT)
    print("\n--- Jensen-Shannon Divergence: Weight Distributions ---")
    comparison_test.js_divergence_test(model_original, model_quantized)
    print("\n--- Output similarity (cosine similarity, SQNR) ---")
    comparison_test.output_similarity(model_fp32, model_quant, dynamic_size=DYNAMIC_SIZE)
finally:
    sys.stdout = old_stdout

report = header + buf.getvalue()
print(report)
with open(report_path, "w") as f:
    f.write(report)
print(f"Report saved to {report_path}")

