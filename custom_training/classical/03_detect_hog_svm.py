"""
03_detect_hog_svm.py
--------------------
Run the trained HOG+SVM detector on a still image, a directory of images,
or a live webcam feed.

Usage
-----
    # Single image
    python 03_detect_hog_svm.py --model detector.xml --image photo.jpg

    # All images in a directory
    python 03_detect_hog_svm.py --model detector.xml --image-dir test_images/

    # Webcam (press 'q' to quit)
    python 03_detect_hog_svm.py --model detector.xml --webcam

    # Tune detection sensitivity
    python 03_detect_hog_svm.py --model detector.xml --image photo.jpg \
        --threshold 0.5 --scale 1.05 --win-stride 8

The script performs:
  1. Load SVM model from XML
  2. Build HOG descriptor with the same parameters used during training
  3. Multi-scale sliding-window detection (manual scale pyramid)
  4. Non-Maximum Suppression (NMS) with configurable IoU threshold
  5. Draw bounding boxes and confidence scores
  6. Report per-frame inference time
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HOG configuration — must match 02_train_hog_svm.py
# ---------------------------------------------------------------------------

HOG_WIN_SIZE    = (64, 64)
HOG_BLOCK_SIZE  = (16, 16)
HOG_BLOCK_STRIDE= (8, 8)
HOG_CELL_SIZE   = (8, 8)
HOG_N_BINS      = 9


def build_hog(win_size: tuple[int, int] = HOG_WIN_SIZE) -> cv2.HOGDescriptor:
    return cv2.HOGDescriptor(
        win_size,
        HOG_BLOCK_SIZE,
        HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE,
        HOG_N_BINS,
    )


# ---------------------------------------------------------------------------
# NMS
# ---------------------------------------------------------------------------

def nms(boxes: np.ndarray, scores: np.ndarray,
        iou_threshold: float = 0.4) -> list[int]:
    """
    Non-Maximum Suppression.

    Parameters
    ----------
    boxes : (N, 4) int array — [x1, y1, x2, y2]
    scores: (N,)   float array
    iou_threshold : float

    Returns
    -------
    List of kept indices sorted by descending score.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = boxes[:, 2].astype(float)
    y2 = boxes[:, 3].astype(float)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = np.argsort(scores)[::-1]
    kept: list[int] = []

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

        inter_w = np.maximum(0.0, xx2 - xx1 + 1)
        inter_h = np.maximum(0.0, yy2 - yy1 + 1)
        inter   = inter_w * inter_h
        iou     = inter / (areas[i] + areas[rest] - inter)

        order = rest[iou < iou_threshold]

    return kept


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

def detect(
    img_gray: np.ndarray,
    svm: cv2.ml.SVM,
    hog: cv2.HOGDescriptor,
    win_size: tuple[int, int],
    win_stride: int,
    scale_factor: float,
    threshold: float,
    max_detections: int = 200,
) -> tuple[list[tuple[int, int, int, int]], list[float]]:
    """
    Multi-scale sliding-window HOG+SVM detector.

    Returns
    -------
    boxes  : list of (x1, y1, x2, y2) in original image coordinates
    scores : list of SVM decision values (higher = more confident)
    """
    W, H = win_size
    stride = win_stride
    h_img, w_img = img_gray.shape[:2]

    all_boxes: list[tuple[int, int, int, int]] = []
    all_scores: list[float] = []

    current_scale = 1.0
    while True:
        new_w = int(w_img / current_scale)
        new_h = int(h_img / current_scale)
        if new_w < W or new_h < H:
            break

        img_scaled = cv2.resize(img_gray, (new_w, new_h))

        for y in range(0, new_h - H + 1, stride):
            for x in range(0, new_w - W + 1, stride):
                crop = img_scaled[y:y + H, x:x + W]
                feat = hog.compute(
                    crop,
                    winStride=(W, H),
                    padding=(0, 0),
                ).ravel().reshape(1, -1).astype(np.float32)

                # Get raw decision score
                _, score_mat = svm.predict(feat, flags=cv2.ml.StatModel_RAW_OUTPUT)
                score = float(score_mat[0, 0])

                if score > threshold:
                    # Map back to original image coordinates
                    x1 = int(x * current_scale)
                    y1 = int(y * current_scale)
                    x2 = int((x + W) * current_scale)
                    y2 = int((y + H) * current_scale)
                    all_boxes.append((x1, y1, x2, y2))
                    all_scores.append(score)

                    if len(all_boxes) >= max_detections:
                        break
            if len(all_boxes) >= max_detections:
                break

        current_scale *= scale_factor

    return all_boxes, all_scores


def detect_and_draw(
    img_bgr: np.ndarray,
    svm: cv2.ml.SVM,
    hog: cv2.HOGDescriptor,
    win_size: tuple[int, int],
    win_stride: int = 8,
    scale_factor: float = 1.05,
    threshold: float = 0.3,
    nms_iou: float = 0.4,
) -> tuple[np.ndarray, int, float]:
    """
    Run detection on a BGR image, draw results, return annotated image,
    detection count, and inference time (seconds).
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    t0 = time.perf_counter()
    boxes, scores = detect(
        img_gray, svm, hog, win_size,
        win_stride, scale_factor, threshold,
    )
    t_detect = time.perf_counter() - t0

    # Apply NMS
    n_raw = len(boxes)
    if boxes:
        boxes_arr  = np.array(boxes, dtype=np.int32)
        scores_arr = np.array(scores, dtype=np.float32)
        keep       = nms(boxes_arr, scores_arr, nms_iou)
        boxes      = [boxes[i] for i in keep]
        scores     = [scores[i] for i in keep]

    out = img_bgr.copy()
    for (x1, y1, x2, y2), score in zip(boxes, scores):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{score:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1,
                    cv2.LINE_AA)

    # HUD overlay
    info = (f"det={len(boxes)} raw={n_raw} "
            f"t={t_detect*1000:.1f}ms thr={threshold:.2f}")
    cv2.putText(out, info, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    return out, len(boxes), t_detect


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="HOG+SVM detector — image / directory / webcam",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",     help="Path to a single input image")
    src.add_argument("--image-dir", help="Directory of images to process")
    src.add_argument("--webcam",    action="store_true", help="Use webcam")

    parser.add_argument("--model",      default="detector.xml",
                        help="Trained SVM model (.xml)")
    parser.add_argument("--win-size",   type=int, default=64,
                        help="Detection window size (square, pixels)")
    parser.add_argument("--win-stride", type=int, default=8,
                        help="Sliding window stride (pixels)")
    parser.add_argument("--scale",      type=float, default=1.05,
                        help="Scale pyramid factor (1.01–1.5)")
    parser.add_argument("--threshold",  type=float, default=0.3,
                        help="SVM decision threshold")
    parser.add_argument("--nms-iou",    type=float, default=0.4,
                        help="NMS IoU overlap threshold")
    parser.add_argument("--output-dir", default=None,
                        help="Save annotated images here (image-dir mode)")
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open GUI windows (useful on headless servers)")
    return parser.parse_args()


def load_svm(model_path: str) -> cv2.ml.SVM:
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: model not found: {model_path}")
    svm = cv2.ml.SVM.load(model_path)
    print(f"Loaded SVM from {model_path}  "
          f"(SVs: {svm.getSupportVectors().shape[0]})")
    return svm


def main():
    args = parse_args()

    svm     = load_svm(args.model)
    win_sz  = (args.win_size, args.win_size)
    hog     = build_hog(win_sz)

    kwargs = dict(
        win_size    = win_sz,
        win_stride  = args.win_stride,
        scale_factor= args.scale,
        threshold   = args.threshold,
        nms_iou     = args.nms_iou,
    )

    # ---- Webcam mode -------------------------------------------------------
    if args.webcam:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            sys.exit("ERROR: cannot open webcam")
        print("Press 'q' to quit, '+'/'-' to adjust threshold.")
        threshold = args.threshold

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            out, n_det, t_inf = detect_and_draw(
                frame, svm, hog, **{**kwargs, "threshold": threshold}
            )
            if not args.no_display:
                cv2.imshow("HOG+SVM Detector  [q=quit  +/-=threshold]", out)

            print(f"\r  detections={n_det}  time={t_inf*1000:.1f}ms  "
                  f"threshold={threshold:.2f}  ", end="", flush=True)

            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("+"):
                    threshold = min(threshold + 0.1, 5.0)
                elif key == ord("-"):
                    threshold = max(threshold - 0.1, -5.0)

        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print()
        return

    # ---- Image / directory mode --------------------------------------------
    if args.image:
        image_paths = [args.image]
    else:
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if os.path.splitext(f)[1].lower() in exts
        ]
        if not image_paths:
            sys.exit(f"ERROR: no images found in {args.image_dir}")
        print(f"Found {len(image_paths)} images in {args.image_dir}")

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    total_det = 0
    total_time = 0.0

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] cannot read {path}")
            continue

        out, n_det, t_inf = detect_and_draw(img, svm, hog, **kwargs)
        total_det  += n_det
        total_time += t_inf

        print(f"  {os.path.basename(path):40s}  "
              f"detections={n_det}  time={t_inf*1000:.1f}ms")

        if args.output_dir:
            out_path = os.path.join(args.output_dir,
                                    os.path.basename(path))
            cv2.imwrite(out_path, out)

        if not args.no_display and len(image_paths) == 1:
            cv2.imshow(f"HOG+SVM  [{os.path.basename(path)}]", out)
            print("Press any key to close ...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    if len(image_paths) > 1:
        avg_t = total_time / len(image_paths) * 1000
        print(f"\nSummary: {total_det} total detections across "
              f"{len(image_paths)} images,  avg {avg_t:.1f}ms/frame")


if __name__ == "__main__":
    main()
