"""
02_train_yolov8.py — Train a YOLOv8 model on a custom dataset, then export to ONNX.

Usage:
    python 02_train_yolov8.py --data synthetic_dataset/data.yaml --model yolov8n --epochs 50

    # Train on GPU 0 with a larger image size:
    python 02_train_yolov8.py --data data.yaml --model yolov8s --epochs 100 --imgsz 640 --device 0

Requirements:
    pip install ultralytics

The script:
  1. Loads (or downloads) the specified pretrained YOLOv8 model.
  2. Trains on the dataset described by --data.
  3. Prints final validation mAP results.
  4. Exports the best checkpoint to ONNX (opset 12, simplified).
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# Check ultralytics is installed
# ---------------------------------------------------------------------------
def check_ultralytics():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("ERROR: ultralytics is not installed.")
        print("Install it with:")
        print("  pip install ultralytics")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 on a custom dataset and export to ONNX."
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to data.yaml (e.g. synthetic_dataset/data.yaml)"
    )
    parser.add_argument(
        "--model", default="yolov8n",
        choices=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
        help="YOLOv8 model variant (default: yolov8n)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (default: 50)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Input image size (default: 640)"
    )
    parser.add_argument(
        "--batch", type=int, default=-1,
        help="Batch size; -1 = auto-detect (default: -1)"
    )
    parser.add_argument(
        "--device", default="",
        help="Device: '' = auto, '0' = GPU 0, 'cpu' = force CPU (default: auto)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="DataLoader worker threads; use 0 on Windows (default: 4)"
    )
    parser.add_argument(
        "--patience", type=int, default=30,
        help="Early-stopping patience in epochs (default: 30)"
    )
    parser.add_argument(
        "--project", default="runs/detect",
        help="Output project directory (default: runs/detect)"
    )
    parser.add_argument(
        "--name", default="train",
        help="Run name subdirectory (default: train)"
    )
    parser.add_argument(
        "--lr0", type=float, default=0.01,
        help="Initial learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--no-pretrained", action="store_true",
        help="Disable pretrained COCO weights (not recommended)"
    )
    parser.add_argument(
        "--no-export", action="store_true",
        help="Skip ONNX export after training"
    )
    parser.add_argument(
        "--opset", type=int, default=12,
        help="ONNX opset version for export (default: 12)"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args):
    from ultralytics import YOLO

    # Resolve data.yaml to absolute path so ultralytics can find it regardless
    # of the working directory inside runs/
    data_path = os.path.abspath(args.data)
    if not os.path.isfile(data_path):
        print(f"ERROR: data.yaml not found at: {data_path}")
        print("Run 01_generate_dataset.py first, or provide the correct --data path.")
        sys.exit(1)

    # Load pretrained or untrained model
    # YOLOv8 downloads the pretrained .pt automatically on first use
    model_id = f"{args.model}.pt" if not args.no_pretrained else f"{args.model}.yaml"
    print(f"\nLoading model: {model_id}")
    model = YOLO(model_id)

    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)
    print(f"  data    : {data_path}")
    print(f"  model   : {model_id}")
    print(f"  epochs  : {args.epochs}")
    print(f"  imgsz   : {args.imgsz}")
    print(f"  batch   : {args.batch} ({'auto' if args.batch == -1 else 'fixed'})")
    print(f"  device  : {args.device if args.device else 'auto'}")
    print(f"  project : {args.project}/{args.name}")
    print()

    results = model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        lr0=args.lr0,
        lrf=0.01,               # final LR = lr0 * lrf (cosine decay)
        warmup_epochs=3,        # linear warmup from 0 to lr0
        mosaic=1.0,             # mosaic augmentation probability
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
        pretrained=not args.no_pretrained,
        plots=True,             # save training plots
        save=True,
        val=True,
        verbose=True,
        seed=42,
    )

    return model, results


# ---------------------------------------------------------------------------
# Results reporting
# ---------------------------------------------------------------------------
def print_results(results):
    print("\n" + "=" * 60)
    print("Training complete — final validation metrics")
    print("=" * 60)
    try:
        # results.results_dict is a dict of metric name -> final value
        metrics = results.results_dict
        keys_of_interest = [
            "metrics/precision(B)",
            "metrics/recall(B)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ]
        for k in keys_of_interest:
            if k in metrics:
                print(f"  {k:35s}: {metrics[k]:.4f}")
    except Exception:
        # Fallback: just print the raw results object
        print(results)
    print()


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------
def export_onnx(model, save_dir, opset):
    """Export the model's best.pt to ONNX."""
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.isfile(best_pt):
        print(f"WARNING: best.pt not found at {best_pt}, skipping ONNX export.")
        return None

    print(f"\nExporting best checkpoint to ONNX (opset={opset}, simplified)...")
    # Re-load best.pt for export
    from ultralytics import YOLO as _YOLO
    best_model = _YOLO(best_pt)
    export_path = best_model.export(
        format="onnx",
        opset=opset,
        simplify=True,
        dynamic=False,
    )
    print(f"ONNX model saved to: {export_path}")

    # Quick sanity check with onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(export_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dummy = np.zeros(inp.shape, dtype=np.float32)
        outs = sess.run(None, {inp.name: dummy})
        print(f"ONNX sanity check passed — output shape: {outs[0].shape}")
    except ImportError:
        print("(onnxruntime not installed — skipping sanity check)")
    except Exception as e:
        print(f"WARNING: ONNX sanity check failed: {e}")

    return export_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    check_ultralytics()
    args = parse_args()

    model, results = train(args)
    print_results(results)

    # Locate the run output directory
    save_dir = str(results.save_dir) if hasattr(results, "save_dir") else \
               os.path.join(args.project, args.name)

    print(f"Results saved to: {save_dir}")
    print(f"  Best weights : {os.path.join(save_dir, 'weights', 'best.pt')}")
    print(f"  Last weights : {os.path.join(save_dir, 'weights', 'last.pt')}")
    print(f"  Training plots: {save_dir}/results.png")

    if not args.no_export:
        onnx_path = export_onnx(model, save_dir, args.opset)
        if onnx_path:
            print(f"\nNext step — run inference:")
            print(f"  python 03_inference_opencv.py --model {onnx_path} --source path/to/image.jpg")
    else:
        print("\nONNX export skipped (--no-export).")

    print("\nDone.")


if __name__ == "__main__":
    main()
