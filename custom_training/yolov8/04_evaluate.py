"""
04_evaluate.py — Evaluate a YOLOv8 ONNX model on a validation set.

Computes:
  - Per-class precision and recall
  - mAP@50 (IoU threshold = 0.50)
  - Confusion matrix (predicted class vs ground-truth class)
  - PR curve per class (saved as PNG or printed as text table)

Usage:
    python 04_evaluate.py \\
        --model runs/detect/train/weights/best.onnx \\
        --data  synthetic_dataset/data.yaml

    # Adjust IoU and confidence thresholds:
    python 04_evaluate.py --model best.onnx --data data.yaml --iou 0.5 --conf 0.25

Requirements:
    pip install opencv-python numpy pyyaml
    pip install matplotlib   # optional — for PR curve plots
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a YOLOv8 ONNX model on a validation dataset."
    )
    parser.add_argument("--model",  required=True, help="Path to ONNX model")
    parser.add_argument("--data",   required=True, help="Path to data.yaml")
    parser.add_argument("--conf",   type=float, default=0.001,
                        help="Confidence threshold for detection (default: 0.001 to maximize recall)")
    parser.add_argument("--nms",    type=float, default=0.6,
                        help="NMS IoU threshold (default: 0.6)")
    parser.add_argument("--iou",    type=float, default=0.5,
                        help="IoU threshold to count a detection as TP (default: 0.5)")
    parser.add_argument("--imgsz",  type=int, default=0,
                        help="Model input size override (default: auto)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PR curve plots (print text table instead)")
    parser.add_argument("--output-dir", default="eval_results",
                        help="Directory to save PR curve plots (default: eval_results)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_yaml(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    base = data.get("path", os.path.dirname(os.path.abspath(yaml_path)))
    val_rel = data.get("val", "images/val")
    val_img_dir = os.path.join(base, val_rel)
    class_names = data.get("names", [])
    nc = data.get("nc", len(class_names))
    return val_img_dir, class_names, nc


def get_val_pairs(val_img_dir):
    """
    Yield (image_path, label_path) pairs for validation images.
    Labels are in parallel labels/ directory mirroring images/.
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    label_dir = val_img_dir.replace("images", "labels")
    pairs = []
    for fname in sorted(os.listdir(val_img_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in img_extensions:
            continue
        img_path = os.path.join(val_img_dir, fname)
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(label_dir, lbl_name)
        pairs.append((img_path, lbl_path))
    return pairs


def load_ground_truth(lbl_path, img_w, img_h):
    """
    Parse a YOLO label file.
    Returns list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    """
    gt = []
    if not os.path.isfile(lbl_path):
        return gt
    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cid, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), \
                                    float(parts[3]), float(parts[4])
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            gt.append((cid, x1, y1, x2, y2))
    return gt


# ---------------------------------------------------------------------------
# Model inference helpers (mirrors 03_inference_opencv.py)
# ---------------------------------------------------------------------------

def get_input_size(model_path, fallback=640):
    try:
        import onnx
        m = onnx.load(model_path)
        inp = m.graph.input[0]
        h = inp.type.tensor_type.shape.dim[2].dim_value
        w = inp.type.tensor_type.shape.dim[3].dim_value
        if h > 0 and w > 0:
            return h, w
    except Exception:
        pass
    return fallback, fallback


def detect_num_classes_from_onnx(model_path):
    try:
        import onnx
        m = onnx.load(model_path)
        d = m.graph.output[0].type.tensor_type.shape.dim[1].dim_value
        return d - 4 if d > 4 else 80
    except Exception:
        return 80


def infer(net, frame, input_h, input_w, nc, conf_thr, nms_thr):
    orig_h, orig_w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (input_w, input_h)),
        1.0 / 255.0, (input_w, input_h), (0, 0, 0), swapRB=True, crop=False
    )
    net.setInput(blob)
    raw = net.forward()          # (1, 4+nc, 8400)
    preds = raw[0].T              # (8400, 4+nc)

    boxes_cxcywh = preds[:, :4]
    class_scores  = preds[:, 4:]
    confs         = class_scores.max(axis=1)
    cids          = class_scores.argmax(axis=1)

    mask = confs > conf_thr
    if not np.any(mask):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes_cxcywh = boxes_cxcywh[mask]
    confs         = confs[mask]
    cids          = cids[mask].astype(int)

    sx = orig_w / input_w
    sy = orig_h / input_h
    cx = boxes_cxcywh[:, 0] * sx
    cy = boxes_cxcywh[:, 1] * sy
    bw = boxes_cxcywh[:, 2] * sx
    bh = boxes_cxcywh[:, 3] * sy
    x1, y1, x2, y2 = cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    nms_in = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])]
              for b in boxes_xyxy]
    idxs = cv2.dnn.NMSBoxes(nms_in, confs.tolist(), conf_thr, nms_thr)
    if len(idxs) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    idxs = np.array(idxs).flatten()
    return boxes_xyxy[idxs], confs[idxs], cids[idxs]


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(box_a, box_b):
    """box_a, box_b: (x1,y1,x2,y2)"""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, box_a[2]-box_a[0]) * max(0, box_a[3]-box_a[1])
    area_b = max(0, box_b[2]-box_b[0]) * max(0, box_b[3]-box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-image matching — returns TP/FP/FN contributions per class
# ---------------------------------------------------------------------------

def match_detections(gt_list, det_boxes, det_confs, det_cids, iou_threshold, nc):
    """
    For each detection, determine TP or FP (greedy, highest-conf first).
    Returns:
        tp_fp: list of (conf, is_tp, class_id) for each detection
        per_class_gt_count: dict class_id -> count of GT boxes
    """
    per_class_gt_count = {}
    for cid, *_ in gt_list:
        per_class_gt_count[cid] = per_class_gt_count.get(cid, 0) + 1

    if len(det_boxes) == 0:
        return [], per_class_gt_count

    # Sort detections by confidence descending
    order = np.argsort(-det_confs)
    gt_matched = [False] * len(gt_list)
    tp_fp = []

    for idx in order:
        conf  = float(det_confs[idx])
        cid   = int(det_cids[idx])
        dbox  = det_boxes[idx]
        best_iou = iou_threshold - 1e-9
        best_gt  = -1
        for gi, (gt_cid, gx1, gy1, gx2, gy2) in enumerate(gt_list):
            if gt_cid != cid or gt_matched[gi]:
                continue
            v = iou(dbox, (gx1, gy1, gx2, gy2))
            if v > best_iou:
                best_iou = v
                best_gt  = gi
        if best_gt >= 0:
            gt_matched[best_gt] = True
            tp_fp.append((conf, 1, cid))  # TP
        else:
            tp_fp.append((conf, 0, cid))  # FP

    return tp_fp, per_class_gt_count


# ---------------------------------------------------------------------------
# Precision-Recall and AP computation
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """Compute area under PR curve using the 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p_at_t = precisions[recalls >= t]
        ap += (p_at_t.max() if len(p_at_t) > 0 else 0.0)
    return ap / 11.0


def compute_per_class_metrics(all_tp_fp, all_gt_counts, nc):
    """
    Compute precision, recall, AP per class.
    Returns:
        results: list of dict per class
    """
    results = []
    for c in range(nc):
        class_dets = [(conf, tp) for conf, tp, cid in all_tp_fp if cid == c]
        n_gt = all_gt_counts.get(c, 0)

        if n_gt == 0 and len(class_dets) == 0:
            results.append({"class_id": c, "precision": 0.0, "recall": 0.0, "ap50": 0.0,
                             "n_gt": 0, "n_det": 0,
                             "recall_curve": np.array([0.0]), "prec_curve": np.array([0.0])})
            continue

        if len(class_dets) == 0:
            results.append({"class_id": c, "precision": 0.0, "recall": 0.0, "ap50": 0.0,
                             "n_gt": n_gt, "n_det": 0,
                             "recall_curve": np.array([0.0]), "prec_curve": np.array([0.0])})
            continue

        # Sort by confidence descending
        class_dets.sort(key=lambda x: -x[0])
        tp_arr = np.array([tp for _, tp in class_dets])
        cum_tp = np.cumsum(tp_arr)
        cum_fp = np.cumsum(1 - tp_arr)

        recall_curve    = cum_tp / (n_gt + 1e-9)
        precision_curve = cum_tp / (cum_tp + cum_fp + 1e-9)

        # AP@50
        ap50 = compute_ap(recall_curve, precision_curve)

        # Final precision/recall at max F1
        f1 = 2 * precision_curve * recall_curve / (precision_curve + recall_curve + 1e-9)
        best_idx = f1.argmax()

        results.append({
            "class_id":     c,
            "precision":    float(precision_curve[best_idx]),
            "recall":       float(recall_curve[best_idx]),
            "ap50":         float(ap50),
            "n_gt":         n_gt,
            "n_det":        len(class_dets),
            "recall_curve":    recall_curve,
            "prec_curve":      precision_curve,
        })
    return results


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def build_confusion_matrix(all_tp_fp, nc):
    """
    Build a simplified (pred_class x gt_class) confusion matrix.
    Background class = nc (for FP predictions not matching any GT).
    """
    cm = np.zeros((nc + 1, nc + 1), dtype=int)
    for conf, is_tp, cid in all_tp_fp:
        if is_tp:
            cm[cid][cid] += 1   # TP: predicted correctly
        else:
            cm[cid][nc] += 1    # FP: predicted class, no matching GT
    return cm


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(results, class_names):
    header = f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'AP@50':>10} {'#GT':>6} {'#Det':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    ap_vals = []
    for r in results:
        name = class_names[r["class_id"]] if r["class_id"] < len(class_names) else str(r["class_id"])
        print(f"{name:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['ap50']:>10.4f} {r['n_gt']:>6d} {r['n_det']:>6d}")
        if r["n_gt"] > 0:
            ap_vals.append(r["ap50"])
    print("-" * len(header))
    map50 = np.mean(ap_vals) if ap_vals else 0.0
    print(f"{'mAP@50':<20} {'':>10} {'':>10} {map50:>10.4f}")
    print("=" * len(header))
    return map50


def print_confusion_matrix(cm, class_names):
    nc = len(class_names)
    print("\nConfusion Matrix (rows=predicted, cols=ground-truth; last row/col = background):")
    header = "         " + "  ".join(f"{n[:6]:>6}" for n in class_names) + "   BG"
    print(header)
    for i in range(nc + 1):
        row_name = class_names[i] if i < nc else "BG"
        row = "  ".join(f"{cm[i][j]:>6d}" for j in range(nc + 1))
        print(f"{row_name[:8]:<8} {row}")


def plot_pr_curves(results, class_names, output_dir):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Printing PR values as text instead.")
        _print_pr_text(results, class_names)
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, r in enumerate(results):
        if r["n_gt"] == 0:
            continue
        name = class_names[r["class_id"]] if r["class_id"] < len(class_names) else str(r["class_id"])
        color = colors[i % len(colors)]
        ax.plot(r["recall_curve"], r["prec_curve"],
                label=f"{name} (AP50={r['ap50']:.3f})", color=color, linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve (IoU@0.50)", fontsize=13)
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, "PR_curve.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"PR curve saved to: {out_path}")


def _print_pr_text(results, class_names):
    for r in results:
        if r["n_gt"] == 0:
            continue
        name = class_names[r["class_id"]] if r["class_id"] < len(class_names) else str(r["class_id"])
        print(f"\nPR values for class '{name}':")
        print(f"  {'Recall':>8}  {'Precision':>10}")
        step = max(1, len(r["recall_curve"]) // 10)
        for rec, prec in zip(r["recall_curve"][::step], r["prec_curve"][::step]):
            print(f"  {rec:>8.4f}  {prec:>10.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.data):
        print(f"ERROR: data.yaml not found: {args.data}")
        sys.exit(1)

    val_img_dir, class_names, nc = load_data_yaml(args.data)
    pairs = get_val_pairs(val_img_dir)
    print(f"Validation images: {len(pairs)}  classes: {nc}  names: {class_names}")

    if args.imgsz > 0:
        input_h = input_w = args.imgsz
    else:
        input_h, input_w = get_input_size(args.model)
    print(f"Model input size: {input_h}x{input_w}\n")

    net = cv2.dnn.readNetFromONNX(args.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    all_tp_fp = []
    all_gt_counts = {}  # class_id -> total count across dataset

    t0 = time.time()
    for n, (img_path, lbl_path) in enumerate(pairs):
        if n % 20 == 0:
            print(f"  Processing {n+1}/{len(pairs)}...", end="\r", flush=True)

        frame = cv2.imread(img_path)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt_list = load_ground_truth(lbl_path, w, h)

        det_boxes, det_confs, det_cids = infer(
            net, frame, input_h, input_w, nc, args.conf, args.nms
        )

        tp_fp, gt_counts = match_detections(
            gt_list, det_boxes, det_confs, det_cids, args.iou, nc
        )
        all_tp_fp.extend(tp_fp)
        for cid, count in gt_counts.items():
            all_gt_counts[cid] = all_gt_counts.get(cid, 0) + count

    elapsed = time.time() - t0
    print(f"\nInference complete: {len(pairs)} images in {elapsed:.1f}s "
          f"({len(pairs)/elapsed:.1f} img/s)")

    # Per-class metrics
    results = compute_per_class_metrics(all_tp_fp, all_gt_counts, nc)
    map50 = print_metrics_table(results, class_names)

    # Confusion matrix
    cm = build_confusion_matrix(all_tp_fp, nc)
    print_confusion_matrix(cm, class_names)

    # PR curves
    if not args.no_plot:
        plot_pr_curves(results, class_names, args.output_dir)
    else:
        _print_pr_text(results, class_names)

    print(f"\nmAP@50 = {map50:.4f}")


if __name__ == "__main__":
    main()
