"""
Convert an ONNX model to TFLite.
Saves the .tflite file in the same directory as the original model.

Uses onnx2tf (ONNX → TensorFlow/TFLite). Install with:
  pip install onnx2tf
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
DEFAULT_MODEL = "../models/aliked-n16_640x640_512kp/aliked-n16_640x640_512kp.onnx"


def _find_tflite_in_dir(dir_path: str) -> str | None:
    """Return path to the first .tflite file under dir_path, or None."""
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            if name.endswith(".tflite"):
                return os.path.join(root, name)
    return None


def to_tflite(model_path: str, output_path: str | None = None) -> str:
    """
    Convert an ONNX model to TFLite and save.

    Args:
        model_path: Path to the ONNX model.
        output_path: Path for the output .tflite file. If None, saves in the
            same directory as model_path with the same base name and .tflite extension.

    Returns:
        Path to the saved .tflite file.
    """
    model_path = os.path.abspath(model_path)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if output_path is None:
        base, _ext = os.path.splitext(model_path)
        output_path = f"{base}.tflite"
    else:
        output_path = os.path.abspath(output_path)

    try:
        import onnx2tf
    except ImportError:
        raise SystemExit(
            "onnx2tf is required. Install with: pip install onnx2tf"
        ) from None

    with tempfile.TemporaryDirectory(prefix="onnx2tf_") as tmpdir:
        # onnx2tf writes to an output folder. Try -o then fallback to default output location.
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        cmd = [sys.executable, "-m", "onnx2tf", "-i", model_path, "-o", tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr or result.stdout, file=sys.stderr)
            raise SystemExit(f"onnx2tf failed with code {result.returncode}")

        tflite_src = _find_tflite_in_dir(tmpdir)
        if not tflite_src:
            for name in [f"{base_name}.tflite", "model.tflite", "saved_model.tflite"]:
                candidate = os.path.join(tmpdir, name)
                if os.path.isfile(candidate):
                    tflite_src = candidate
                    break
        if not tflite_src:
            for root, _dirs, files in os.walk(tmpdir):
                for f in files:
                    if f.endswith(".tflite"):
                        tflite_src = os.path.join(root, f)
                        break
                if tflite_src:
                    break
        if not tflite_src:
            for root, _dirs, files in os.walk(tmpdir):
                for f in files:
                    print(os.path.join(root, f), file=sys.stderr)
            raise SystemExit("onnx2tf did not produce a .tflite file")

        shutil.copy2(tflite_src, output_path)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to TFLite")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Path to ONNX model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output .tflite path (default: same dir as model, same base name with .tflite)",
    )
    args = parser.parse_args()

    model_path = os.path.normpath(args.model)
    out = to_tflite(model_path, output_path=args.output)
    print(f"Saved TFLite model to {out}")


if __name__ == "__main__":
    main()
