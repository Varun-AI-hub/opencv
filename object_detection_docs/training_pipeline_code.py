"""
training_pipeline_code.py
=========================
Complete demonstration of:
  1. HOG + SVM training pipeline (synthetic data, sliding window, NMS)
  2. YOLOv8 ONNX inference helper class (ready for real models)
  3. IoU and mAP computation utilities
  4. Data augmentation utilities

Requirements:
    pip install opencv-python numpy

Optional (for ONNX inference verification):
    pip install onnxruntime
"""

import cv2
import numpy as np
import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# =============================================================================
# SECTION 1: Synthetic Data Generation
# =============================================================================

def generate_positive_sample(size: Tuple[int, int] = (64, 128)) -> np.ndarray:
    """
    Generate a synthetic positive sample: white circle on dark background.
    Simulates a simple object the detector will learn to find.
    """
    img = np.zeros((size[1], size[0]), dtype=np.uint8)
    cx, cy = size[0] // 2, size[1] // 2
    radius = min(size) // 3
    cv2.circle(img, (cx, cy), radius, 255, -1)
    # Add slight noise to avoid trivial memorization
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


def generate_negative_sample(size: Tuple[int, int] = (64, 128)) -> np.ndarray:
    """
    Generate a synthetic negative sample: random Gaussian noise.
    """
    img = np.random.randint(0, 200, (size[1], size[0]), dtype=np.uint8)
    return img


def generate_dataset(n_pos: int = 300, n_neg: int = 600,
                     win_size: Tuple[int, int] = (64, 128)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced synthetic dataset of positives (label=1) and negatives (label=0).

    Returns:
        X: float32 array of shape (n_samples, feature_dim)
        y: int32 array of shape (n_samples,)
    """
    hog = _build_hog_descriptor(win_size)
    X, y = [], []

    print(f"[Dataset] Generating {n_pos} positives and {n_neg} negatives ...")
    for _ in range(n_pos):
        img = generate_positive_sample(win_size)
        X.append(hog.compute(img).flatten())
        y.append(1)

    for _ in range(n_neg):
        img = generate_negative_sample(win_size)
        X.append(hog.compute(img).flatten())
        y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# =============================================================================
# SECTION 2: HOG Descriptor Configuration
# =============================================================================

WIN_SIZE     = (64, 128)
BLOCK_SIZE   = (16, 16)
BLOCK_STRIDE = (8, 8)
CELL_SIZE    = (8, 8)
NBINS        = 9


def _build_hog_descriptor(win_size: Tuple[int, int] = WIN_SIZE) -> cv2.HOGDescriptor:
    """
    Build a HOG descriptor with standard parameters.
    Feature vector length for WIN_SIZE=(64,128): 3780 dimensions.
    """
    return cv2.HOGDescriptor(win_size, BLOCK_SIZE, BLOCK_STRIDE, CELL_SIZE, NBINS)


# =============================================================================
# SECTION 3: HOG + SVM Training Pipeline
# =============================================================================

def train_hog_svm(
    X: np.ndarray,
    y: np.ndarray,
    svm_type: int = cv2.ml.SVM_C_SVC,
    kernel: int = cv2.ml.SVM_RBF,
    auto_tune: bool = True,
    C: float = 1.0,
    gamma: float = 0.001
) -> cv2.ml.SVM:
    """
    Train an SVM on HOG feature vectors.

    Args:
        X: Feature matrix (n_samples, n_features), float32.
        y: Label vector (n_samples,), int32. Classes: 0 (negative), 1 (positive).
        svm_type: SVM formulation — cv2.ml.SVM_C_SVC (soft-margin) or SVM_NU_SVC.
        kernel: Kernel type — SVM_RBF, SVM_LINEAR, SVM_POLY.
        auto_tune: If True, use grid search for C and gamma (slower, better).
        C: Regularization parameter (used if auto_tune=False).
        gamma: RBF kernel bandwidth (used if auto_tune=False).

    Returns:
        Trained cv2.ml.SVM instance.
    """
    svm = cv2.ml.SVM_create()
    svm.setType(svm_type)
    svm.setKernel(kernel)
    svm.setTermCriteria((
        cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
        1000, 1e-6
    ))

    train_data = cv2.ml.TrainData.create(X, cv2.ml.ROW_SAMPLE, y)

    if auto_tune:
        print("[SVM] Auto-tuning hyperparameters with grid search ...")
        svm.trainAuto(train_data)
        print(f"[SVM] Best C={svm.getC():.4f}, gamma={svm.getGamma():.6f}")
    else:
        svm.setC(C)
        svm.setGamma(gamma)
        svm.train(train_data)

    # Evaluate on training data
    _, preds = svm.predict(X)
    acc = np.mean(preds.flatten().astype(np.int32) == y) * 100
    print(f"[SVM] Training accuracy: {acc:.2f}%")

    return svm


def save_svm(svm: cv2.ml.SVM, path: str) -> None:
    """Save trained SVM to XML file."""
    svm.save(path)
    print(f"[SVM] Model saved to {path}")


def load_svm(path: str) -> cv2.ml.SVM:
    """Load SVM from XML file."""
    svm = cv2.ml.SVM_load(path)
    print(f"[SVM] Model loaded from {path}")
    return svm


# =============================================================================
# SECTION 4: Sliding Window Detector
# =============================================================================

def sliding_window_generator(image: np.ndarray, step: int = 8,
                              win_size: Tuple[int, int] = WIN_SIZE):
    """
    Yields (x, y, window_patch) for each sliding window position.
    win_size is (width, height).
    """
    w, h = win_size
    for y in range(0, image.shape[0] - h + 1, step):
        for x in range(0, image.shape[1] - w + 1, step):
            yield x, y, image[y:y + h, x:x + w]


def image_pyramid(image: np.ndarray, scale_factor: float = 1.25,
                  min_size: Tuple[int, int] = WIN_SIZE):
    """
    Yields (scale, resized_image) pairs from original down to min_size.
    """
    current = image.copy()
    current_scale = 1.0
    while True:
        h, w = current.shape[:2]
        if h < min_size[1] or w < min_size[0]:
            break
        yield current_scale, current
        new_w = int(w / scale_factor)
        new_h = int(h / scale_factor)
        current = cv2.resize(current, (new_w, new_h))
        current_scale /= scale_factor


def detect_with_hog_svm(
    image: np.ndarray,
    svm: cv2.ml.SVM,
    win_size: Tuple[int, int] = WIN_SIZE,
    step: int = 8,
    scale_factor: float = 1.25,
    nms_thresh: float = 0.4
) -> List[Tuple[int, int, int, int]]:
    """
    Run the full HOG+SVM sliding window detector.

    Args:
        image: BGR image.
        svm: Trained SVM.
        win_size: Detection window (width, height).
        step: Sliding window stride in pixels.
        scale_factor: Pyramid downscale factor per level.
        nms_thresh: IoU threshold for NMS.

    Returns:
        List of (x1, y1, x2, y2) bounding boxes after NMS.
    """
    hog = _build_hog_descriptor(win_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raw_boxes: List[List[int]] = []
    orig_h, orig_w = gray.shape[:2]

    for sc, scaled_img in image_pyramid(gray, scale_factor=scale_factor, min_size=win_size):
        for x, y, window in sliding_window_generator(scaled_img, step=step, win_size=win_size):
            feat = hog.compute(window).flatten().reshape(1, -1).astype(np.float32)
            _, pred = svm.predict(feat)
            if int(pred[0, 0]) == 1:
                # Map back to original image coordinates
                x1 = int(x / sc)
                y1 = int(y / sc)
                x2 = int((x + win_size[0]) / sc)
                y2 = int((y + win_size[1]) / sc)
                # Clip to image
                x1, x2 = max(0, x1), min(orig_w, x2)
                y1, y2 = max(0, y1), min(orig_h, y2)
                raw_boxes.append([x1, y1, x2 - x1, y2 - y1])

    if not raw_boxes:
        return []

    # NMS
    weights = [1.0] * len(raw_boxes)
    keep = cv2.dnn.NMSBoxes(
        raw_boxes, weights,
        score_threshold=0.0,
        nms_threshold=nms_thresh
    )
    result = []
    if len(keep) > 0:
        for k in keep.flatten():
            x, y, w, h = raw_boxes[k]
            result.append((x, y, x + w, y + h))
    return result


# =============================================================================
# SECTION 5: YOLOv8 ONNX Inference Helper
# =============================================================================

class YOLOv8ONNXDetector:
    """
    OpenCV DNN-based YOLOv8 ONNX inference.

    Usage:
        detector = YOLOv8ONNXDetector("best.onnx", ["cat", "dog"], conf_thresh=0.25)
        dets = detector.detect(image)
        annotated = detector.draw(image, dets)

    Export your model with:
        from ultralytics import YOLO
        YOLO("best.pt").export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)

    Output tensor format: [1, 4+num_classes, 8400] for imgsz=640.
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        imgsz: int = 640,
        use_cuda: bool = False
    ):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz

        self.net = cv2.dnn.readNetFromONNX(model_path)
        if use_cuda:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        print(f"[YOLOv8] Loaded {model_path} | {self.num_classes} classes | "
              f"conf={conf_thresh} iou={iou_thresh} imgsz={imgsz}")

    def _letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, float, int, int]:
        """
        Resize with aspect-ratio preservation and pad to self.imgsz x self.imgsz.
        Returns (padded_image, scale, pad_left, pad_top).
        """
        h, w = image.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_top  = (self.imgsz - new_h) // 2
        pad_left = (self.imgsz - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
        return canvas, scale, pad_left, pad_top

    def detect(self, image: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Run detection on a BGR image.

        Returns:
            List of (x1, y1, x2, y2, confidence, class_id).
        """
        letterboxed, scale, pad_left, pad_top = self._letterbox(image)
        blob = cv2.dnn.blobFromImage(
            letterboxed, scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            mean=(0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        raw = self.net.forward()     # shape: [1, 4+N, 8400]
        return self._parse_output(raw[0], image.shape, scale, pad_left, pad_top)

    def _parse_output(self, output: np.ndarray, orig_shape: Tuple,
                      scale: float, pad_left: int, pad_top: int
                      ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Parse raw network output.
        output shape: [4+num_classes, 8400]
        """
        preds = output.T                         # [8400, 4+N]
        boxes_raw = preds[:, :4]                 # cx, cy, w, h in letterbox pixels
        class_scores = preds[:, 4:]              # [8400, N]

        class_ids   = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences >= self.conf_thresh
        boxes_raw   = boxes_raw[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(boxes_raw) == 0:
            return []

        # cx,cy,w,h -> x1,y1,x2,y2 in letterbox space
        x1 = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
        y1 = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
        x2 = boxes_raw[:, 0] + boxes_raw[:, 2] / 2
        y2 = boxes_raw[:, 1] + boxes_raw[:, 3] / 2

        # Unpad and unscale to original image space
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top)  / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top)  / scale

        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)

        results = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            nms_boxes = np.stack(
                [x1[idx], y1[idx], x2[idx] - x1[idx], y2[idx] - y1[idx]], axis=1
            ).tolist()
            nms_scores = confidences[idx].tolist()
            keep = cv2.dnn.NMSBoxes(
                nms_boxes, nms_scores, self.conf_thresh, self.iou_thresh
            )
            if len(keep) > 0:
                for k in keep.flatten():
                    results.append((
                        int(x1[idx[k]]), int(y1[idx[k]]),
                        int(x2[idx[k]]), int(y2[idx[k]]),
                        float(confidences[idx[k]]),
                        int(cls)
                    ))
        return results

    def draw(self, image: np.ndarray,
             detections: List[Tuple[int, int, int, int, float, int]]) -> np.ndarray:
        """Draw detections on a copy of image."""
        out = image.copy()
        colors = [
            (0, 255, 0), (255, 0, 0), (0, 0, 255),
            (255, 165, 0), (128, 0, 128), (0, 255, 255)
        ]
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            color = colors[cls_id % len(colors)]
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out


# =============================================================================
# SECTION 6: IoU and mAP Evaluation Utilities
# =============================================================================

def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union for two [x1, y1, x2, y2] boxes.
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using all-point (COCO-style) interpolation.

    Args:
        recalls:    Array of recall values sorted ascending.
        precisions: Array of precision values matching recalls.

    Returns:
        AP as a float in [0, 1].
    """
    # Add boundary sentinels
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])

    # Envelope: make precision monotonically non-increasing right-to-left
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate: sum up rectangular areas at recall breakpoints
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def compute_pr_curve(
    pred_boxes: List[List[float]],
    pred_scores: List[float],
    gt_boxes: List[List[float]],
    iou_thresh: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Precision-Recall curve and AP for a single class over one image set.

    Args:
        pred_boxes:  List of predicted [x1, y1, x2, y2].
        pred_scores: List of confidence scores (same order).
        gt_boxes:    List of ground-truth [x1, y1, x2, y2].
        iou_thresh:  IoU threshold to consider a detection correct.

    Returns:
        (recalls, precisions, ap): Arrays sorted by recall, and scalar AP.
    """
    if len(gt_boxes) == 0:
        return np.array([0.0]), np.array([0.0]), 0.0

    # Sort predictions by descending confidence
    order = np.argsort(pred_scores)[::-1]
    pred_boxes  = [pred_boxes[i]  for i in order]

    matched = set()
    tp_arr, fp_arr = [], []

    for pb in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou = iou
                best_idx = gi
        if best_iou >= iou_thresh and best_idx >= 0:
            tp_arr.append(1); fp_arr.append(0)
            matched.add(best_idx)
        else:
            tp_arr.append(0); fp_arr.append(1)

    tp_cum = np.cumsum(tp_arr).astype(float)
    fp_cum = np.cumsum(fp_arr).astype(float)
    n_gt   = len(gt_boxes)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
    ap = compute_ap(recalls, precisions)
    return recalls, precisions, ap


def compute_map(
    predictions_per_class: Dict[str, Tuple[List, List]],
    gt_per_class: Dict[str, List],
    iou_thresh: float = 0.5
) -> Dict[str, float]:
    """
    Compute per-class AP and mean AP at a single IoU threshold.

    Args:
        predictions_per_class: {class_name: ([boxes], [scores])}
        gt_per_class:          {class_name: [boxes]}
        iou_thresh:            IoU threshold.

    Returns:
        Dict with per-class APs and 'mAP' key for the mean.
    """
    results = {}
    for cls, gt_boxes in gt_per_class.items():
        pred_boxes, pred_scores = predictions_per_class.get(cls, ([], []))
        _, _, ap = compute_pr_curve(pred_boxes, pred_scores, gt_boxes, iou_thresh)
        results[cls] = ap

    results["mAP"] = float(np.mean(list(results.values()))) if results else 0.0
    return results


def compute_map50_95(
    predictions_per_class: Dict[str, Tuple[List, List]],
    gt_per_class: Dict[str, List]
) -> float:
    """
    Compute mAP@[0.50:0.05:0.95] (COCO primary metric).
    """
    thresholds = np.arange(0.5, 1.0, 0.05)
    map_values = []
    for t in thresholds:
        r = compute_map(predictions_per_class, gt_per_class, iou_thresh=round(t, 2))
        map_values.append(r["mAP"])
    return float(np.mean(map_values))


# =============================================================================
# SECTION 7: Data Augmentation Utilities
# =============================================================================

def augment_flip_horizontal(image: np.ndarray, boxes: Optional[List] = None):
    """
    Horizontally flip image and optionally its YOLO-format boxes.

    Args:
        image: BGR image.
        boxes: List of [class_id, cx, cy, w, h] normalized. If None, skipped.

    Returns:
        (flipped_image, flipped_boxes)
    """
    flipped = cv2.flip(image, 1)
    if boxes is None:
        return flipped, None
    new_boxes = []
    for box in boxes:
        cls, cx, cy, w, h = box
        new_boxes.append([cls, 1.0 - cx, cy, w, h])
    return flipped, new_boxes


def augment_brightness(image: np.ndarray, delta: float = 0.3) -> np.ndarray:
    """
    Randomly adjust brightness in HSV space.

    Args:
        image: BGR image.
        delta: Maximum fractional change (e.g., 0.3 means ±30%).

    Returns:
        Brightness-adjusted BGR image.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = 1.0 + random.uniform(-delta, delta)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_random_crop(
    image: np.ndarray,
    boxes: Optional[List] = None,
    crop_fraction: float = 0.8
):
    """
    Random crop keeping at least crop_fraction of each side.

    Args:
        image: BGR image.
        boxes: List of [class_id, cx, cy, w, h] normalized.
        crop_fraction: Minimum fraction of image to keep (0.7-0.95 typical).

    Returns:
        (cropped_image, updated_boxes) — boxes outside crop are removed,
        boxes partially inside are clipped.
    """
    h, w = image.shape[:2]
    crop_h = int(h * random.uniform(crop_fraction, 1.0))
    crop_w = int(w * random.uniform(crop_fraction, 1.0))
    top    = random.randint(0, h - crop_h)
    left   = random.randint(0, w - crop_w)

    cropped = image[top:top + crop_h, left:left + crop_w]

    if boxes is None:
        return cropped, None

    new_boxes = []
    for box in boxes:
        cls, cx, cy, bw, bh = box
        # Convert to absolute coords
        abs_cx = cx * w
        abs_cy = cy * h
        abs_bw = bw * w
        abs_bh = bh * h
        x1 = abs_cx - abs_bw / 2
        y1 = abs_cy - abs_bh / 2
        x2 = abs_cx + abs_bw / 2
        y2 = abs_cy + abs_bh / 2

        # Clip to crop region
        x1c = max(x1, left) - left
        y1c = max(y1, top)  - top
        x2c = min(x2, left + crop_w) - left
        y2c = min(y2, top  + crop_h) - top

        if x2c <= x1c or y2c <= y1c:
            continue  # Box fully outside crop

        # Back to normalized YOLO format relative to new image size
        ncx = (x1c + x2c) / 2 / crop_w
        ncy = (y1c + y2c) / 2 / crop_h
        nw  = (x2c - x1c) / crop_w
        nh  = (y2c - y1c) / crop_h
        new_boxes.append([cls, ncx, ncy, nw, nh])

    return cropped, new_boxes


def augment_hsv(image: np.ndarray,
                hgain: float = 0.015,
                sgain: float = 0.7,
                vgain: float = 0.4) -> np.ndarray:
    """
    Random HSV color jitter (matches Ultralytics default augmentation).

    Args:
        image: BGR image.
        hgain: Hue shift fraction.
        sgain: Saturation multiplier range.
        vgain: Value multiplier range.

    Returns:
        Jittered BGR image.
    """
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lut_h = np.arange(0, 256, dtype=np.int16)
    lut_s = np.arange(0, 256, dtype=np.int16)
    lut_v = np.arange(0, 256, dtype=np.int16)

    lut_h = ((lut_h * r[0]) % 180).clip(0, 179).astype(np.uint8)
    lut_s = (lut_s * r[1]).clip(0, 255).astype(np.uint8)
    lut_v = (lut_v * r[2]).clip(0, 255).astype(np.uint8)

    h, s, v = cv2.split(hsv)
    h = cv2.LUT(h, lut_h)
    s = cv2.LUT(s, lut_s)
    v = cv2.LUT(v, lut_v)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


# =============================================================================
# SECTION 8: Full Demo (HOG+SVM end-to-end)
# =============================================================================

def run_hog_svm_demo(save_model_path: str = "/tmp/hog_svm_demo.xml"):
    """
    End-to-end demo:
      1. Generate synthetic data
      2. Train SVM
      3. Save model
      4. Reload model
      5. Test detection on a synthetic test image
      6. Display result (or save if no display)
    """
    print("=" * 60)
    print("HOG + SVM Training Pipeline Demo")
    print("=" * 60)

    # --- Step 1: Generate data ---
    X, y = generate_dataset(n_pos=400, n_neg=800, win_size=WIN_SIZE)
    print(f"[Dataset] X.shape={X.shape}, y.shape={y.shape}")

    # --- Step 2: Train ---
    svm = train_hog_svm(X, y, auto_tune=False, C=1.0, gamma=0.001)

    # --- Step 3: Save ---
    save_svm(svm, save_model_path)

    # --- Step 4: Reload ---
    svm_loaded = load_svm(save_model_path)

    # --- Step 5: Synthetic test image with a circle embedded ---
    test_img = np.random.randint(50, 150, (300, 400, 3), dtype=np.uint8)
    # Plant a detectable "object" (bright circle)
    cv2.circle(test_img, (200, 150), 45, (220, 220, 220), -1)

    print("[Detect] Running sliding window detection on test image ...")
    detections = detect_with_hog_svm(
        test_img, svm_loaded, win_size=WIN_SIZE, step=16, scale_factor=1.3
    )
    print(f"[Detect] Found {len(detections)} detection(s): {detections}")

    # Draw results
    result_img = test_img.copy()
    for (x1, y1, x2, y2) in detections:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save output
    out_path = "/tmp/hog_svm_result.jpg"
    cv2.imwrite(out_path, result_img)
    print(f"[Output] Result saved to {out_path}")

    print("=" * 60)
    print("Demo complete.")
    print("=" * 60)


def run_map_demo():
    """
    Quick demonstration of mAP computation with random predictions.
    """
    print("\n--- mAP Evaluation Demo ---")
    random.seed(42)
    np.random.seed(42)

    # Fake ground-truth and prediction boxes for 2 classes
    gt = {
        "cat": [[10, 10, 100, 100], [200, 150, 350, 280]],
        "dog": [[50, 60, 180, 200]],
    }
    preds = {
        "cat": (
            [[12, 8, 98, 102], [300, 200, 400, 300], [15, 15, 105, 105]],
            [0.95, 0.30, 0.70]
        ),
        "dog": (
            [[55, 65, 175, 195]],
            [0.88]
        ),
    }

    results = compute_map(preds, gt, iou_thresh=0.5)
    for k, v in results.items():
        print(f"  {k:10s}: AP={v:.4f}" if k != "mAP" else f"  {'mAP@50':10s}: {v:.4f}")

    map50_95 = compute_map50_95(preds, gt)
    print(f"  {'mAP@50-95':10s}: {map50_95:.4f}")


if __name__ == "__main__":
    # Run HOG+SVM pipeline demo
    run_hog_svm_demo()

    # Run mAP evaluation demo
    run_map_demo()

    # --- YOLOv8 ONNX usage note ---
    print("\n--- YOLOv8 ONNX Detector Usage ---")
    print("To use YOLOv8ONNXDetector with a real model:")
    print("  1. Export: YOLO('best.pt').export(format='onnx', opset=12, simplify=True)")
    print("  2. Instantiate: detector = YOLOv8ONNXDetector('best.onnx', class_names)")
    print("  3. Detect:  dets = detector.detect(cv2.imread('image.jpg'))")
    print("  4. Draw:    annotated = detector.draw(img, dets)")
