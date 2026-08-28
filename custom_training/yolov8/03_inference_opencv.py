"""
03_inference_opencv.py — YOLOv8 ONNX inference using OpenCV DNN.

Supports:
  - Single image
  - Video file
  - Webcam (source=0)

Usage:
    # Image
    python 03_inference_opencv.py --model best.onnx --source image.jpg

    # Video
    python 03_inference_opencv.py --model best.onnx --source video.mp4

    # Webcam
    python 03_inference_opencv.py --model best.onnx --source 0

    # Adjust thresholds
    python 03_inference_opencv.py --model best.onnx --source image.jpg \\
        --conf 0.4 --nms 0.45

    # Save output instead of displaying
    python 03_inference_opencv.py --model best.onnx --source image.jpg --output result.jpg

Requirements:
    pip install opencv-python numpy

Notes:
    - The ONNX model must have been exported with static input shape, e.g.
      model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
    - This script auto-detects input size from the ONNX graph (or use --imgsz).
    - Class names are read from --names or default to numeric labels.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Default colors (BGR) for up to 20 classes
# ---------------------------------------------------------------------------
PALETTE = [
    (56,  56,  255), (151, 157, 255), (31,  112, 255), (29,  178, 255),
    (49,  210, 207), (10,  249, 72),  (23,  204, 146), (134, 219, 61),
    (52,  147, 26),  (187, 212, 0),   (168, 153, 44),  (255, 194, 0),
    (255, 162, 0),   (255, 108, 23),  (255, 42,  123),  (226, 11,  227),
    (148, 0,   240), (112, 0,   224), (64,  0,   224),  (0,   0,   255),
]


def get_color(class_id):
    return PALETTE[class_id % len(PALETTE)]


# ---------------------------------------------------------------------------
# ONNX model loader
# ---------------------------------------------------------------------------

def load_model(model_path, backend=cv2.dnn.DNN_BACKEND_OPENCV,
               target=cv2.dnn.DNN_TARGET_CPU):
    if not os.path.isfile(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        sys.exit(1)
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(backend)
    net.setPreferableTarget(target)
    return net


def get_input_size(model_path, fallback=640):
    """
    Try to read input spatial size from the ONNX model graph.
    Returns (height, width). Falls back to (fallback, fallback).
    """
    try:
        import onnx
        model = onnx.load(model_path)
        inp = model.graph.input[0]
        shape = inp.type.tensor_type.shape
        h = shape.dim[2].dim_value
        w = shape.dim[3].dim_value
        if h > 0 and w > 0:
            return h, w
    except ImportError:
        pass  # onnx not installed; use fallback
    except Exception:
        pass
    return fallback, fallback


# ---------------------------------------------------------------------------
# Pre/post processing
# ---------------------------------------------------------------------------

def preprocess(frame, input_h, input_w):
    """
    Resize and create a blob from the input frame.
    Returns:
        blob: (1, 3, input_h, input_w) float32 ready for net.setInput()
        scale_x, scale_y: factors to map back to original frame coordinates
    """
    orig_h, orig_w = frame.shape[:2]
    # Simple resize (no letterboxing for clarity; letterboxing improves accuracy slightly)
    resized = cv2.resize(frame, (input_w, input_h))
    blob = cv2.dnn.blobFromImage(
        resized,
        scalefactor=1.0 / 255.0,   # normalize [0,255] -> [0.0,1.0]
        size=(input_w, input_h),
        mean=(0, 0, 0),            # no mean subtraction
        swapRB=True,               # BGR (OpenCV) -> RGB (YOLO expects)
        crop=False,
    )
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h
    return blob, scale_x, scale_y


def postprocess(net_output, scale_x, scale_y, num_classes,
                conf_threshold, nms_threshold):
    """
    Decode YOLOv8 output tensor.

    net_output shape: (1, 4 + num_classes, 8400)
      - 8400 = 80x80 + 40x40 + 20x20 = 6400+1600+400 anchor-free predictions
      - First 4 channels: cx, cy, w, h in *input image* pixel space
      - Remaining channels: class probabilities (no objectness score in YOLOv8)

    Returns:
        boxes_xyxy: np.ndarray of shape (N, 4), pixel coords in original image
        confidences: np.ndarray of shape (N,)
        class_ids: np.ndarray of shape (N,)
    """
    # net_output: (1, 4+nc, 8400)  ->  preds: (8400, 4+nc)
    preds = net_output[0].T   # shape: (8400, 4 + num_classes)

    boxes_cxcywh = preds[:, :4]          # (8400, 4)  cx,cy,w,h in input-px space
    class_scores = preds[:, 4:]          # (8400, num_classes)

    # For each prediction pick the class with highest probability
    confidences = class_scores.max(axis=1)    # (8400,)
    class_ids   = class_scores.argmax(axis=1) # (8400,)

    # Filter by confidence threshold before NMS (fast pre-filter)
    mask = confidences > conf_threshold
    if not np.any(mask):
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    boxes_cxcywh = boxes_cxcywh[mask]
    confidences  = confidences[mask]
    class_ids    = class_ids[mask]

    # Convert cx,cy,w,h (input-pixel space) to x1,y1,x2,y2 (original-pixel space)
    cx = boxes_cxcywh[:, 0] * scale_x
    cy = boxes_cxcywh[:, 1] * scale_y
    bw = boxes_cxcywh[:, 2] * scale_x
    bh = boxes_cxcywh[:, 3] * scale_y

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # NMS — cv2.dnn.NMSBoxes expects [x, y, w, h] format and list of floats
    nms_input_boxes = []
    for b in boxes_xyxy:
        nms_input_boxes.append([float(b[0]), float(b[1]),
                                  float(b[2] - b[0]), float(b[3] - b[1])])

    indices = cv2.dnn.NMSBoxes(
        bboxes=nms_input_boxes,
        scores=confidences.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=nms_threshold,
    )

    if len(indices) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # cv2.dnn.NMSBoxes returns shape (N,1) in older OpenCV, (N,) in newer
    indices = np.array(indices).flatten()

    return boxes_xyxy[indices], confidences[indices], class_ids[indices].astype(int)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_detections(frame, boxes_xyxy, confidences, class_ids, class_names):
    for (x1, y1, x2, y2), conf, cid in zip(boxes_xyxy, confidences, class_ids):
        color = get_color(cid)
        name = class_names[cid] if cid < len(class_names) else str(cid)
        label = f"{name} {conf:.2f}"
        ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (ix1, iy1), (ix2, iy2), color, 2)
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (ix1, iy1 - th - 6), (ix1 + tw + 2, iy1), color, -1)
        cv2.putText(frame, label, (ix1 + 1, iy1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self._fps = 0.0
        self._t = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        inst = 1.0 / max(now - self._t, 1e-9)
        self._fps = self.alpha * inst + (1 - self.alpha) * self._fps
        self._t = now
        return self._fps

    @property
    def fps(self):
        return self._fps


# ---------------------------------------------------------------------------
# Inference on a single frame
# ---------------------------------------------------------------------------

def infer_frame(net, frame, input_h, input_w, num_classes,
                conf_threshold, nms_threshold):
    blob, sx, sy = preprocess(frame, input_h, input_w)
    net.setInput(blob)
    raw = net.forward()   # shape: (1, 4+nc, 8400)
    boxes, confs, cids = postprocess(raw, sx, sy, num_classes,
                                      conf_threshold, nms_threshold)
    return boxes, confs, cids


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_image(net, source, input_h, input_w, num_classes, class_names,
              conf_threshold, nms_threshold, output_path):
    frame = cv2.imread(source)
    if frame is None:
        print(f"ERROR: Cannot read image: {source}")
        sys.exit(1)

    boxes, confs, cids = infer_frame(net, frame, input_h, input_w,
                                      num_classes, conf_threshold, nms_threshold)
    frame = draw_detections(frame, boxes, confs, cids, class_names)

    print(f"Detected {len(boxes)} object(s):")
    for i, (b, c, cid) in enumerate(zip(boxes, confs, cids)):
        name = class_names[cid] if cid < len(class_names) else str(cid)
        print(f"  [{i}] {name:15s}  conf={c:.3f}  "
              f"box=[{b[0]:.0f},{b[1]:.0f},{b[2]:.0f},{b[3]:.0f}]")

    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Result saved to: {output_path}")
    else:
        cv2.imshow("YOLOv8 Inference", frame)
        print("Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(net, source, input_h, input_w, num_classes, class_names,
              conf_threshold, nms_threshold, output_path):
    # source may be int (webcam) or str (file path)
    try:
        src = int(source)
    except (ValueError, TypeError):
        src = source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        sys.exit(1)

    fps_counter = FPSCounter()
    writer = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(output_path, fourcc, cap_fps, (orig_w, orig_h))
        print(f"Writing output to: {output_path}")

    print("Running inference. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confs, cids = infer_frame(net, frame, input_h, input_w,
                                          num_classes, conf_threshold, nms_threshold)
        frame = draw_detections(frame, boxes, confs, cids, class_names)

        current_fps = fps_counter.tick()
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(frame)
        else:
            cv2.imshow("YOLOv8 Inference", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLOv8 ONNX inference using OpenCV DNN."
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to YOLOv8 ONNX model (e.g. runs/detect/train/weights/best.onnx)"
    )
    parser.add_argument(
        "--source", required=True,
        help="Input: path to image file, video file, or webcam index (0, 1, ...)"
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold (default: 0.25)"
    )
    parser.add_argument(
        "--nms", type=float, default=0.45,
        help="NMS IoU threshold (default: 0.45)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=0,
        help="Model input size override (default: auto-detect from ONNX graph)"
    )
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Class names in order (e.g. --names circle rectangle triangle)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Save result to this file instead of displaying (jpg/png for images, mp4 for video)"
    )
    parser.add_argument(
        "--nc", type=int, default=None,
        help="Number of classes (auto-detected from ONNX if possible)"
    )
    return parser.parse_args()


def detect_num_classes(model_path, fallback=80):
    """
    Try to infer number of classes from ONNX output shape.
    YOLOv8 output: (1, 4+nc, 8400) so nc = dim[1] - 4
    """
    try:
        import onnx
        model = onnx.load(model_path)
        out = model.graph.output[0]
        dim1 = out.type.tensor_type.shape.dim[1].dim_value
        if dim1 > 4:
            return dim1 - 4
    except Exception:
        pass
    return fallback


def main():
    args = parse_args()

    # Input size
    if args.imgsz > 0:
        input_h = input_w = args.imgsz
    else:
        input_h, input_w = get_input_size(args.model)
    print(f"Model input size: {input_h}x{input_w}")

    # Number of classes
    num_classes = args.nc if args.nc else detect_num_classes(args.model)
    print(f"Number of classes: {num_classes}")

    # Class names
    if args.names:
        class_names = args.names
    else:
        class_names = [str(i) for i in range(num_classes)]
    print(f"Class names: {class_names}")

    # Load model
    net = load_model(args.model)
    print(f"Model loaded: {args.model}")

    # Determine if source is image or video
    source = args.source
    is_image = False
    try:
        int(source)  # webcam index
    except (ValueError, TypeError):
        # Check extension for image
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        ext = os.path.splitext(source)[1].lower()
        is_image = ext in image_exts

    if is_image:
        run_image(net, source, input_h, input_w, num_classes, class_names,
                  args.conf, args.nms, args.output)
    else:
        run_video(net, source, input_h, input_w, num_classes, class_names,
                  args.conf, args.nms, args.output)


if __name__ == "__main__":
    main()
