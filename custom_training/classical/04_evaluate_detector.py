"""
04_evaluate_detector.py
-----------------------
Evaluate a trained HOG+SVM detector on a labelled test set.

Annotation format (CSV, one row per ground-truth box)
-----------------------------------------------------
    filename,x1,y1,x2,y2

Example annotations.csv:
    img_000.png,10,12,55,58
    img_000.png,70,80,130,140
    img_001.png,5,5,60,60

Usage
-----
    python 04_evaluate_detector.py \
        --model    detector.xml \
        --test-dir test_images/ \
        --annotations annotations.csv \
        [--threshold 0.3] \
        [--iou-threshold 0.5] \
        [--scale 1.05] \
        [--win-stride 8]

Outputs
-------
  • Per-image detection results (stdout)
  • Summary metrics: precision, recall, F1, mAP
  • Precision-Recall curve (matplotlib if available, else ASCII table)
  • results_summary.txt  — machine-readable JSON of all metrics
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HOG configuration — must match training
# ---------------------------------------------------------------------------

HOG_WIN_SIZE    = (64, 64)
HOG_BLOCK_SIZE  = (16, 16)
HOG_BLOCK_STRIDE= (8, 8)
HOG_CELL_SIZE   = (8, 8)
HOG_N_BINS      = 9


def build_hog(win_size=(64, 64)):
    return cv2.HOGDescriptor(
        win_size, HOG_BLOCK_SIZE, HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE, HOG_N_BINS,
    )


# ---------------------------------------------------------------------------
# NMS (duplicated here so script is self-contained)
# ---------------------------------------------------------------------------

def nms(boxes, scores, iou_threshold=0.4):
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        iw = np.maximum(0.0, xx2 - xx1 + 1)
        ih = np.maximum(0.0, yy2 - yy1 + 1)
        inter = iw * ih
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou < iou_threshold]
    return kept


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def run_detector(img_gray, svm, hog, win_size, win_stride,
                 scale_factor, threshold, max_det=500):
    W, H = win_size
    stride = win_stride
    h_img, w_img = img_gray.shape[:2]
    boxes, scores = [], []
    current_scale = 1.0

    while True:
        nw = int(w_img / current_scale)
        nh = int(h_img / current_scale)
        if nw < W or nh < H:
            break
        img_s = cv2.resize(img_gray, (nw, nh))

        for y in range(0, nh - H + 1, stride):
            for x in range(0, nw - W + 1, stride):
                crop = img_s[y:y + H, x:x + W]
                feat = hog.compute(crop, winStride=(W, H), padding=(0, 0))
                feat = feat.ravel().reshape(1, -1).astype(np.float32)
                _, s = svm.predict(feat, flags=cv2.ml.StatModel_RAW_OUTPUT)
                sc = float(s[0, 0])
                if sc > threshold:
                    x1 = int(x * current_scale)
                    y1 = int(y * current_scale)
                    x2 = int((x + W) * current_scale)
                    y2 = int((y + H) * current_scale)
                    boxes.append((x1, y1, x2, y2))
                    scores.append(sc)
                    if len(boxes) >= max_det:
                        break
            if len(boxes) >= max_det:
                break
        current_scale *= scale_factor

    return boxes, scores


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(boxA, boxB):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    areaB = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return inter / (areaA + areaB - inter)


# ---------------------------------------------------------------------------
# Precision-Recall computation
# ---------------------------------------------------------------------------

def compute_pr_curve(all_detections, ground_truths, iou_thresh=0.5):
    """
    all_detections : list of dicts {image, box, score}
    ground_truths  : dict {image_name: [box, ...]}

    Returns arrays: precision, recall, thresholds, AP
    """
    # Sort by descending score
    all_detections = sorted(all_detections, key=lambda d: -d["score"])

    # Count total ground-truth boxes
    n_gt = sum(len(v) for v in ground_truths.values())
    if n_gt == 0:
        return np.array([]), np.array([]), np.array([]), 0.0

    # Track which GT boxes have been matched
    matched_gt = defaultdict(lambda: defaultdict(bool))
    # matched_gt[image_name][gt_idx] = True/False

    tp_cumul = []
    fp_cumul = []
    tp_total = 0
    fp_total = 0

    for det in all_detections:
        img_name = det["image"]
        det_box  = det["box"]
        gt_boxes = ground_truths.get(img_name, [])

        best_iou = 0.0
        best_j   = -1
        for j, gt_box in enumerate(gt_boxes):
            ov = iou(det_box, gt_box)
            if ov > best_iou:
                best_iou = ov
                best_j   = j

        if best_iou >= iou_thresh and best_j >= 0 and \
                not matched_gt[img_name][best_j]:
            tp_total += 1
            matched_gt[img_name][best_j] = True
        else:
            fp_total += 1

        tp_cumul.append(tp_total)
        fp_cumul.append(fp_total)

    tp_arr = np.array(tp_cumul, dtype=float)
    fp_arr = np.array(fp_cumul, dtype=float)

    precision = tp_arr / (tp_arr + fp_arr + 1e-9)
    recall    = tp_arr / (n_gt + 1e-9)
    thresholds = np.array([d["score"] for d in all_detections])

    # AP = area under PR curve (interpolated at 11 recall levels)
    ap = 0.0
    for r_thresh in np.linspace(0, 1, 11):
        mask = recall >= r_thresh
        if mask.any():
            ap += precision[mask].max()
    ap /= 11.0

    return precision, recall, thresholds, ap


# ---------------------------------------------------------------------------
# ASCII PR table
# ---------------------------------------------------------------------------

def ascii_pr_table(precision, recall, thresholds, n_rows=20):
    """Print a compact ASCII representation of the PR curve."""
    if len(precision) == 0:
        print("  (no detections)")
        return

    step = max(1, len(precision) // n_rows)
    indices = list(range(0, len(precision), step))
    if indices[-1] != len(precision) - 1:
        indices.append(len(precision) - 1)

    print(f"  {'Threshold':>10}  {'Recall':>8}  {'Precision':>10}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*10}")
    for i in indices:
        thresh = thresholds[i] if i < len(thresholds) else float("nan")
        print(f"  {thresh:>10.4f}  {recall[i]:>8.4f}  {precision[i]:>10.4f}")


# ---------------------------------------------------------------------------
# Matplotlib PR curve
# ---------------------------------------------------------------------------

def plot_pr_curve(precision, recall, ap, output_path="pr_curve.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 5))
        plt.step(recall, precision, where="post", color="steelblue", lw=2)
        plt.fill_between(recall, precision, alpha=0.15, color="steelblue",
                         step="post")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve  (AP = {ap:.4f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close()
        print(f"  PR curve saved → {output_path}")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Annotation loader
# ---------------------------------------------------------------------------

def load_annotations(csv_path):
    """
    Returns dict: {filename: [(x1,y1,x2,y2), ...]}
    """
    gt = defaultdict(list)
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 5:
                print(f"  [warn] skipping malformed row: {row}")
                continue
            fname = row[0].strip()
            try:
                x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            except ValueError:
                # Header row?
                continue
            gt[fname].append((x1, y1, x2, y2))
    return dict(gt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate HOG+SVM detector with PR curve",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",       default="detector.xml")
    parser.add_argument("--test-dir",    required=True, help="Directory of test images")
    parser.add_argument("--annotations", required=True,
                        help="CSV annotation file (filename,x1,y1,x2,y2)")
    parser.add_argument("--threshold",   type=float, default=0.3,
                        help="SVM detection threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for TP matching")
    parser.add_argument("--scale",       type=float, default=1.05)
    parser.add_argument("--win-stride",  type=int,   default=8)
    parser.add_argument("--win-size",    type=int,   default=64)
    parser.add_argument("--nms-iou",     type=float, default=0.4)
    parser.add_argument("--output",      default="results_summary.json",
                        help="JSON file for machine-readable results")
    parser.add_argument("--pr-curve",    default="pr_curve.png",
                        help="Output path for PR curve plot")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load model
    if not os.path.exists(args.model):
        sys.exit(f"ERROR: model not found: {args.model}")
    svm = cv2.ml.SVM.load(args.model)
    print(f"Loaded SVM: {svm.getSupportVectors().shape[0]} support vectors")

    win_sz = (args.win_size, args.win_size)
    hog    = build_hog(win_sz)

    # Load annotations
    if not os.path.exists(args.annotations):
        sys.exit(f"ERROR: annotations file not found: {args.annotations}")
    ground_truths = load_annotations(args.annotations)
    n_gt_total = sum(len(v) for v in ground_truths.values())
    print(f"Loaded {n_gt_total} ground-truth boxes across "
          f"{len(ground_truths)} annotated images")

    # Collect test images
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    test_images = [
        f for f in sorted(os.listdir(args.test_dir))
        if os.path.splitext(f)[1].lower() in exts
    ]
    if not test_images:
        sys.exit(f"ERROR: no images found in {args.test_dir}")
    print(f"Test images: {len(test_images)}")
    print()

    # ---- Run detector on all test images -----------------------------------
    all_dets: list[dict] = []
    total_time = 0.0

    for fname in test_images:
        img_path = os.path.join(args.test_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [skip] {fname}")
            continue

        t0 = time.perf_counter()
        boxes, scores = run_detector(
            img, svm, hog, win_sz,
            args.win_stride, args.scale, args.threshold,
        )
        elapsed = time.perf_counter() - t0
        total_time += elapsed

        # Apply NMS
        if boxes:
            boxes_arr  = np.array(boxes, dtype=np.int32)
            scores_arr = np.array(scores, dtype=np.float32)
            keep       = nms(boxes_arr, scores_arr, args.nms_iou)
            boxes  = [boxes[i]  for i in keep]
            scores = [scores[i] for i in keep]

        n_gt = len(ground_truths.get(fname, []))
        print(f"  {fname:40s}  gt={n_gt}  det={len(boxes)}  "
              f"{elapsed*1000:.1f}ms")

        for box, score in zip(boxes, scores):
            all_dets.append({"image": fname, "box": box, "score": score})

    avg_time = total_time / max(len(test_images), 1) * 1000
    print(f"\nAverage inference time: {avg_time:.1f}ms/image")

    # ---- PR curve and AP ---------------------------------------------------
    print("\nComputing Precision-Recall curve ...")
    precision, recall, thresholds, ap = compute_pr_curve(
        all_dets, ground_truths, iou_thresh=args.iou_threshold
    )

    # ---- Metrics at chosen threshold ---------------------------------------
    tp = fp = fn = 0
    # Recompute at specific threshold
    for img_name, gt_boxes in ground_truths.items():
        img_dets = [d for d in all_dets
                    if d["image"] == img_name and d["score"] >= args.threshold]
        matched = [False] * len(gt_boxes)
        for det in img_dets:
            best_ov, best_j = 0.0, -1
            for j, gt_box in enumerate(gt_boxes):
                ov = iou(det["box"], gt_box)
                if ov > best_ov:
                    best_ov, best_j = ov, j
            if best_ov >= args.iou_threshold and not matched[best_j]:
                tp += 1
                matched[best_j] = True
            else:
                fp += 1
        fn += matched.count(False)

    prec_at_thresh = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_at_thresh  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_at_thresh   = (2 * prec_at_thresh * rec_at_thresh /
                      (prec_at_thresh + rec_at_thresh + 1e-9))

    print(f"\n=== Results @ threshold={args.threshold:.2f}, "
          f"IoU>={args.iou_threshold:.2f} ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"  Precision : {prec_at_thresh:.4f}")
    print(f"  Recall    : {rec_at_thresh:.4f}")
    print(f"  F1        : {f1_at_thresh:.4f}")
    print(f"  AP (mAP)  : {ap:.4f}")
    print(f"  Avg time  : {avg_time:.1f}ms/image")

    # ---- Plot / print PR curve ---------------------------------------------
    print("\nPrecision-Recall curve:")
    if len(precision):
        plotted = plot_pr_curve(precision, recall, ap, args.pr_curve)
        if not plotted:
            print("  (matplotlib not available — ASCII table below)")
            ascii_pr_table(precision, recall, thresholds)
    else:
        print("  (no detections — nothing to plot)")

    # ---- Save JSON summary -------------------------------------------------
    summary = {
        "threshold": args.threshold,
        "iou_threshold": args.iou_threshold,
        "n_test_images": len(test_images),
        "n_gt_boxes": n_gt_total,
        "n_detections": len(all_dets),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec_at_thresh, 6),
        "recall":    round(rec_at_thresh,  6),
        "f1":        round(f1_at_thresh,   6),
        "ap":        round(float(ap),      6),
        "avg_inference_ms": round(avg_time, 2),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.output}")


if __name__ == "__main__":
    main()
