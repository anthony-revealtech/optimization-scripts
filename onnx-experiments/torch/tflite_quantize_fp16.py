#!/usr/bin/env python3
"""
Adaptive static quantization: convert a float32 TFLite model to float16.

Reduces model size by ~50% with minimal accuracy impact. The quantized model
still accepts float32 inputs/outputs at inference; weights are stored as float16.

Requires: tensorflow (for flatbuffer_utils and schema).

Usage:
  python3.11 tflite_quantize_fp16.py aliked-n16.tflite -o aliked-n16_fp16.tflite
  python3.11 tflite_quantize_fp16.py model.tflite   # writes model_fp16.tflite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# TensorFlow Lite schema and flatbuffer utils (bundled with tensorflow)
try:
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.tools import flatbuffer_utils
except ImportError as e:
    raise ImportError("This script requires tensorflow. Install with: pip3 install tensorflow") from e


def _read_tflite_model(path: Path) -> "schema_fb.ModelT":
    """Load a TFLite file into a mutable model object (ModelT)."""
    data = path.read_bytes()
    model_bytearray = bytearray(data)
    model = flatbuffer_utils.read_model_from_bytearray(model_bytearray)
    return model


def _convert_model_float32_to_float16(model: "schema_fb.ModelT") -> None:
    """
    In-place: convert all FLOAT32 tensors and their buffers to FLOAT16.
    Buffers are converted once; all tensors that reference a converted buffer
    are updated to type FLOAT16.
    """
    buffers_converted = set()
    for subgraph in model.subgraphs:
        if subgraph.tensors is None:
            continue
        for tensor in subgraph.tensors:
            if tensor.type != schema_fb.TensorType.FLOAT32:
                continue
            buf_idx = tensor.buffer
            if buf_idx <= 0 or buf_idx >= len(model.buffers):
                continue
            if buf_idx in buffers_converted:
                tensor.type = schema_fb.TensorType.FLOAT16
                continue
            buffer = model.buffers[buf_idx]
            if buffer.data is None or len(buffer.data) == 0:
                continue
            # Buffer length must be divisible by 4 (float32)
            n_f32 = len(buffer.data) // 4
            if n_f32 * 4 != len(buffer.data):
                continue
            arr_f32 = np.frombuffer(buffer.data, dtype=np.float32)
            arr_f16 = arr_f32.astype(np.float16)
            buffer.data = bytearray(arr_f16.tobytes())
            tensor.type = schema_fb.TensorType.FLOAT16
            buffers_converted.add(buf_idx)


def quantize_tflite_to_fp16(input_path: Path, output_path: Path) -> None:
    """Load float32 TFLite, quantize to float16, and write the result."""
    print(f"Loading {input_path}...")
    model = _read_tflite_model(input_path)
    print("Converting float32 -> float16...")
    _convert_model_float32_to_float16(model)
    if sys.byteorder == "big":
        flatbuffer_utils.byte_swap_tflite_model_obj(model, "big", "little")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flatbuffer_utils.write_model(model, str(output_path))
    print(f"Saved FP16 TFLite to {output_path}")
    in_size = input_path.stat().st_size
    out_size = output_path.stat().st_size
    print(f"Size: {in_size / 1024:.1f} KB -> {out_size / 1024:.1f} KB ({100 * out_size / in_size:.0f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantize a float32 TFLite model to float16 (adaptive static)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input .tflite file (float32)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .tflite file (default: input path with _fp16 before .tflite)",
    )
    args = parser.parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Error: not found: {input_path}", file=sys.stderr)
        sys.exit(1)
    if args.output is not None:
        output_path = args.output.resolve()
    else:
        output_path = input_path.with_name(
            input_path.stem + "_fp16" + input_path.suffix
        )
    quantize_tflite_to_fp16(input_path, output_path)


if __name__ == "__main__":
    main()
