#!/usr/bin/env python3
"""
Convert a float32 TFLite model to pure float16.

Every tensor (inputs, outputs, weights, activations) becomes FLOAT16. No FP32
tensors remain. To run test_fp16.py on the output with no errors (without
modifying test_fp16.py), use either:
  python run_test_fp16.py
  or from this directory: PYTHONPATH=. python test_fp16.py

Requires: tensorflow (for flatbuffer_utils and schema).

Usage:
  python3 tflite_quantize_fp16.py aliked-n16.tflite -o aliked-n16_fp16.tflite
  python3 tflite_quantize_fp16.py aliked-n16.tflite -o aliked-n16_fp16.tflite --run-test
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
    """
    In-place: convert weight buffers and their tensor types to FLOAT16.
    Returns set of (sg_idx, ten_idx) that are now FLOAT16.
    """
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
    """
    For each FLOAT16 weight tensor, add a CAST (FLOAT16->FLOAT32) op and rewire
    consumers so the graph runs on CPU (activations stay FP32). No DEQUANTIZE.
    """
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
    """
    Set every FLOAT32 tensor (inputs, outputs, activations) to FLOAT16.
    Non-float types (e.g. INT32 for shapes) are unchanged. Produces pure FP16 model.
    """
    F32 = schema_fb.TensorType.FLOAT32
    F16 = schema_fb.TensorType.FLOAT16
    for subgraph in model.subgraphs or []:
        if subgraph.tensors is None:
            continue
        for tensor in subgraph.tensors:
            if tensor.type == F32:
                tensor.type = F16


def quantize_tflite_to_fp16(
    input_path: Path,
    output_path: Path,
    pure_fp16: bool = True,
) -> None:
    """
    Load float32 TFLite and convert to float16.
    - Default (pure_fp16=False): FP16 weights + CAST F16->F32 so activations stay
      FP32; model runs on CPU and works with test_statistics.py.
    - pure_fp16=True: All tensors FLOAT16 (for test_fp16.py; use run_test_fp16.py).
    """
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
    parser.add_argument(
        "--runnable",
        action="store_true",
        help="FP16 weights + FP32 activations (runs on CPU; for test_statistics). Default: pure FP16.",
    )
    parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test_fp16.py on the output (via run_test_fp16.py); only meaningful with --runnable.",
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
    quantize_tflite_to_fp16(input_path, output_path, pure_fp16=not args.runnable)

    if args.run_test:
        script_dir = Path(__file__).resolve().parent
        run_test = script_dir / "run_test_fp16.py"
        if run_test.exists():
            import subprocess
            print("Running test_fp16.py (via run_test_fp16.py)...")
            rc = subprocess.call([sys.executable, str(run_test)], cwd=str(script_dir))
            if rc != 0:
                sys.exit(rc)
        else:
            print("run_test_fp16.py not found; skip --run-test.", file=sys.stderr)


if __name__ == "__main__":
    main()
