"""
03_export_to_onnx.py
=====================
Export a trained SimpleDetector checkpoint to ONNX format and verify it.

Steps performed
---------------
  1. Load model and checkpoint from 02_pytorch_detector.py
  2. Run a dummy forward pass to warm up and check outputs
  3. Export with torch.onnx.export (opset=12, static shapes by default)
  4. Optionally simplify with onnxsim
  5. Verify with onnxruntime — compare outputs numerically against PyTorch
  6. Print input/output names, shapes, and file size

Usage
-----
  python 03_export_to_onnx.py --checkpoint ./runs/best.pth --output model.onnx
  python 03_export_to_onnx.py --checkpoint ./runs/best.pth --output model.onnx --simplify
  python 03_export_to_onnx.py --checkpoint ./runs/best.pth --output model.onnx --dynamic_batch
"""

import argparse
import os
import sys

import numpy as np
import torch

# Import model definition from the training script
sys.path.insert(0, os.path.dirname(__file__))
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "pytorch_detector",
    os.path.join(os.path.dirname(__file__), "02_pytorch_detector.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SimpleDetector = _mod.SimpleDetector

# Optional imports — checked at runtime
try:
    import onnx
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

try:
    from onnxsim import simplify as onnxsim_simplify
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_model(model, dummy_input, output_path, dynamic_batch=False, opset=12):
    """Run torch.onnx.export with sensible defaults for OpenCV DNN."""
    model.eval()

    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'cls_scores': {0: 'batch_size'},
            'boxes': {0: 'batch_size'},
        }

    print(f"Exporting to {output_path}  (opset={opset}, dynamic_batch={dynamic_batch}) ...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=['input'],
        output_names=['cls_scores', 'boxes'],
        dynamic_axes=dynamic_axes if dynamic_axes else None,
        export_params=True,
        do_constant_folding=True,
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Exported. File size: {size_mb:.2f} MB")


# ---------------------------------------------------------------------------
# Simplification
# ---------------------------------------------------------------------------

def simplify_model(input_path, output_path):
    """Run onnx-simplifier on the exported model."""
    if not HAS_ONNX:
        print("  [skip] onnx not installed. pip install onnx")
        return False
    if not HAS_ONNXSIM:
        print("  [skip] onnxsim not installed. pip install onnxsim")
        return False

    print(f"Simplifying {input_path} -> {output_path} ...")
    model_onnx = onnx.load(input_path)
    model_simplified, ok = onnxsim_simplify(model_onnx)
    if not ok:
        print("  WARNING: simplification did not fully succeed; using original.")
        model_simplified = model_onnx
    onnx.save(model_simplified, output_path)
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"  Simplified. File size: {size_mb:.2f} MB")
    return True


# ---------------------------------------------------------------------------
# ONNX inspection
# ---------------------------------------------------------------------------

def inspect_onnx(path):
    """Print input/output names and shapes using onnx or onnxruntime."""
    if HAS_ORT:
        sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        print("\nONNX model inputs:")
        for inp in sess.get_inputs():
            print(f"  name={inp.name!r}  shape={inp.shape}  dtype={inp.type}")
        print("ONNX model outputs:")
        for out in sess.get_outputs():
            print(f"  name={out.name!r}  shape={out.shape}  dtype={out.type}")
        return sess
    elif HAS_ONNX:
        model_onnx = onnx.load(path)
        onnx.checker.check_model(model_onnx)
        print("  ONNX model is valid (onnx.checker passed).")
        return None
    else:
        print("  [skip] neither onnx nor onnxruntime installed.")
        return None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_outputs(model, dummy_input, onnx_path, atol=1e-4, rtol=1e-4):
    """Compare PyTorch and ONNX Runtime outputs numerically."""
    if not HAS_ORT:
        print("  [skip] onnxruntime not installed. pip install onnxruntime")
        return

    print("\nVerifying outputs (PyTorch vs ONNX Runtime) ...")
    model.eval()
    with torch.no_grad():
        pt_cls, pt_boxes = model(dummy_input)
    pt_cls_np = pt_cls.cpu().numpy()
    pt_boxes_np = pt_boxes.cpu().numpy()

    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    ort_cls, ort_boxes = sess.run(None, {input_name: dummy_input.cpu().numpy()})

    cls_ok = np.allclose(pt_cls_np, ort_cls, atol=atol, rtol=rtol)
    boxes_ok = np.allclose(pt_boxes_np, ort_boxes, atol=atol, rtol=rtol)

    print(f"  cls_scores match: {cls_ok}  "
          f"(max diff={np.abs(pt_cls_np - ort_cls).max():.2e})")
    print(f"  boxes match:      {boxes_ok}  "
          f"(max diff={np.abs(pt_boxes_np - ort_boxes).max():.2e})")

    if cls_ok and boxes_ok:
        print("  Verification PASSED.")
    else:
        print("  WARNING: outputs differ beyond tolerance. Check model and export flags.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Export SimpleDetector to ONNX')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to .pth checkpoint from 02_pytorch_detector.py')
    parser.add_argument('--output', default='model.onnx',
                        help='Output ONNX file path (default: model.onnx)')
    parser.add_argument('--num_classes', type=int, default=None,
                        help='Override num_classes (auto-detected from checkpoint if omitted)')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12; OpenCV DNN supports up to 13)')
    parser.add_argument('--simplify', action='store_true',
                        help='Run onnx-simplifier after export')
    parser.add_argument('--dynamic_batch', action='store_true',
                        help='Export with dynamic batch dimension (not recommended for OpenCV DNN)')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size (default: 224)')
    args = parser.parse_args()

    device = torch.device('cpu')  # export always on CPU for portability

    # --- Load checkpoint ---
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    num_classes = args.num_classes or ckpt.get('num_classes', 3)
    print(f"  num_classes={num_classes}")

    model = SimpleDetector(num_classes=num_classes)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model.to(device)

    # --- Dummy forward pass ---
    dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
    with torch.no_grad():
        cls_out, box_out = model(dummy)
    print(f"Dummy forward pass OK. cls_out={cls_out.shape}  box_out={box_out.shape}")

    # --- Export ---
    export_model(model, dummy, args.output, dynamic_batch=args.dynamic_batch, opset=args.opset)

    # --- Simplify ---
    final_path = args.output
    if args.simplify:
        base, ext = os.path.splitext(args.output)
        simplified_path = base + '_simplified' + ext
        ok = simplify_model(args.output, simplified_path)
        if ok:
            final_path = simplified_path

    # --- Inspect ---
    inspect_onnx(final_path)

    # --- Verify ---
    verify_outputs(model, dummy, final_path)

    print(f"\nFinal ONNX model: {final_path}")
    print("Use this file with 04_opencv_inference.py")


if __name__ == '__main__':
    main()
