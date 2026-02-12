#!/usr/bin/env python3
"""
Convert ALIKED-n16 PyTorch (.pth) model to TFLite.

Tries direct PyTorch -> TFLite (litert_torch) first; if that fails (e.g. ALIKED's
deform_conv2d is unsupported), falls back to PyTorch -> ONNX -> TFLite so a .tflite
file is always produced when possible.

  Recommended venv for ALIKED (has onnx2tf for fallback):
    python3.11 -m venv .venv-p2t-onnx && source .venv-p2t-onnx/bin/activate
    pip3 install -r requirements-pytorch2tflite-onnx.txt
    export PYTHONPATH=/home/anthony/Documents/Programming/reveal/ALIKED:$PYTHONPATH
    python3.11 pytorch2tflite.py

  Optional: .venv-p2t with litert-torch + torchvision for direct path; add
  onnx2tf and tensorflow to the same venv if you want fallback (may have dep conflicts).

Usage:
  python3.11 pytorch2tflite.py                    # try direct, then ONNX fallback
  python3.11 pytorch2tflite.py --use-onnx         # skip direct, use ONNX only
  python3.11 pytorch2tflite.py --onnx model.onnx  # existing ONNX -> TFLite



  python3.11 tflite_quantize_fp16.py aliked-n16.tflite -o aliked-n16_fp16.tflite
  python3.11 tflite_quantize_fp16.py model.tflite   # writes model_fp16.tflite
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Default paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PTH = SCRIPT_DIR / "aliked-n16.pth"
DEFAULT_ONNX = SCRIPT_DIR / "aliked-n16.onnx"
DEFAULT_TFLITE = SCRIPT_DIR / "aliked-n16.tflite"

# Input shape (NCHW: 1, 3, H, W)
INPUT_HEIGHT = 640
INPUT_WIDTH = 640


def _load_aliked_dense_wrapper(pth_path: Path):
    """Load ALIKED-n16 from .pth and return a wrapper that outputs (score_map, feature_map)."""
    import torch

    try:
        from nets.aliked import ALIKED
    except ImportError as e:
        raise ImportError(
            "ALIKED package not found. Clone the repo and add it to PYTHONPATH:\n"
            "  git clone https://github.com/Shiaoming/ALIKED\n"
            "  export PYTHONPATH=/home/anthony/Documents/Programming/reveal/ALIKED:$PYTHONPATH"
        ) from e

    class _DenseWrapper(torch.nn.Module):
        """Exports only score_map and feature_map."""

        def __init__(self, aliked: ALIKED):
            super().__init__()
            self.aliked = aliked

        def forward(
            self, image: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            feature_map, score_map = self.aliked.extract_dense_map(image)
            return score_map, feature_map

    print(f"Loading weights from {pth_path}...")
    try:
        state_dict = torch.load(pth_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(pth_path, map_location="cpu")

    print("Building ALIKED-n16 model (load_pretrained=False)...")
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


def _patch_deform_conv2d_for_litert() -> None:
    """
    Replace torchvision.ops.deform_conv2d with a fallback that uses F.conv2d
    (same weights, zero offset) so litert_torch can convert the graph.
    Call before importing ALIKED in the direct-conversion path.
    """
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
        # Equivalent to deform_conv2d with zero offset: standard conv2d.
        return F.conv2d(input, weight, bias, stride, padding, dilation)

    torchvision.ops.deform_conv2d = _deform_conv2d_fallback


def convert_pytorch_to_tflite_direct(pth_path: Path, tflite_path: Path) -> bool:
    """
    Convert PyTorch model directly to TFLite using litert_torch (no ONNX).
    Returns True on success, False if litert_torch is missing or conversion fails.
    """
    try:
        import litert_torch
    except ImportError:
        return False

    import torch

    # Use regular conv2d instead of deform_conv2d so litert_torch can lower the graph.
    _patch_deform_conv2d_for_litert()

    print("Using direct PyTorch -> TFLite (litert_torch), no ONNX.")
    wrapper = _load_aliked_dense_wrapper(pth_path)
    sample_inputs = (torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH),)
    print(f"Converting (input shape [1, 3, {INPUT_HEIGHT}, {INPUT_WIDTH}])...")
    try:
        edge_model = litert_torch.convert(wrapper.eval(), sample_inputs)
    except Exception as e:
        err = str(e)
        print(f"litert_torch conversion failed: {e}", file=sys.stderr)
        if "deform_conv2d" in err or "Lowering not found" in err:
            print("(ALIKED uses deform_conv2d, unsupported by litert_torch; will try ONNX path.)", file=sys.stderr)
        return False
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    edge_model.export(str(tflite_path))
    print(f"TFLite saved to {tflite_path}")
    return True


def export_pytorch_to_onnx(pth_path: Path, onnx_path: Path) -> None:
    """Export ALIKED dense wrapper to ONNX."""
    import torch

    wrapper = _load_aliked_dense_wrapper(pth_path)
    dummy = torch.randn(1, 3, INPUT_HEIGHT, INPUT_WIDTH)
    print(f"Exporting ONNX (input shape [1, 3, {INPUT_HEIGHT}, {INPUT_WIDTH}]) to {onnx_path}...")
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


def convert_onnx_to_tflite(onnx_path: Path, tflite_path: Path) -> None:
    """Convert ONNX model to TFLite using onnx2tf."""
    try:
        import onnx2tf
    except ImportError as e:
        raise ImportError(
            "onnx2tf not found. Install with: pip3 install onnx2tf tensorflow"
        ) from e

    out_dir = tflite_path.with_suffix("")  # e.g. aliked-n16/
    print(f"Converting ONNX to TFLite: {onnx_path} -> {tflite_path}...")
    onnx2tf.convert(
        input_onnx_file_path=str(onnx_path),
        output_folder_path=str(out_dir),
        copy_onnx_input_output_names_to_tflite=True,
        non_verbose=True,
    )
    # onnx2tf writes saved_model.pb and *_float32.tflite (and optionally *_float16.tflite) into output_folder_path
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ALIKED-n16 PyTorch (.pth) to TFLite."
    )
    parser.add_argument(
        "--pth",
        type=Path,
        default=DEFAULT_PTH,
        help=f"Path to aliked-n16 .pth weights (default: {DEFAULT_PTH})",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="If set, skip PyTorch export and convert this ONNX file to TFLite.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_TFLITE,
        help=f"Output .tflite path (default: {DEFAULT_TFLITE})",
    )
    parser.add_argument(
        "--keep-onnx",
        action="store_true",
        help="Keep intermediate ONNX file when using --use-onnx (default: remove it).",
    )
    parser.add_argument(
        "--use-onnx",
        action="store_true",
        help="Force PyTorch -> ONNX -> TFLite instead of direct PyTorch -> TFLite.",
    )
    args = parser.parse_args()

    tflite_path = Path(args.output).resolve()

    if args.onnx is not None:
        onnx_path = Path(args.onnx).resolve()
        if not onnx_path.exists():
            print(f"Error: ONNX file not found: {onnx_path}", file=sys.stderr)
            sys.exit(1)
        convert_onnx_to_tflite(onnx_path, tflite_path)
        print("Done.")
        return

    pth_path = Path(args.pth).resolve()
    if not pth_path.exists():
        print(f"Error: .pth file not found: {pth_path}", file=sys.stderr)
        sys.exit(1)

    if not args.use_onnx:
        if convert_pytorch_to_tflite_direct(pth_path, tflite_path):
            print("Done.")
            return
        print("Direct conversion failed; falling back to PyTorch -> ONNX -> TFLite...", flush=True)

    # ONNX path: always used when --use-onnx, or as fallback when direct fails
    onnx_path = Path(args.output).with_suffix(".onnx")
    if onnx_path == pth_path:
        onnx_path = pth_path.with_suffix(".onnx")
    try:
        export_pytorch_to_onnx(pth_path, onnx_path)
        convert_onnx_to_tflite(onnx_path, tflite_path)
    except ImportError as e:
        err_str = str(e)
        if "onnx2tf" in err_str or "tensorflow" in err_str.lower():
            print(
                "Error: ONNX fallback requires onnx2tf and tensorflow. Install them:\n"
                "  pip3 install -r requirements-pytorch2tflite-onnx.txt\n"
                "  or activate the .venv-p2t-onnx venv and run again.",
                file=sys.stderr,
            )
            sys.exit(1)
        raise
    if not args.keep_onnx:
        onnx_path.unlink(missing_ok=True)
        print("Removed intermediate ONNX file.")
    print("Done.")


if __name__ == "__main__":
    main()
