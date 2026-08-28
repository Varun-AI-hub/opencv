"""
04_opencv_inference.py
=======================
Run inference with a trained SimpleDetector ONNX model using OpenCV DNN.

Handles:
  - Static image inference with saved output
  - Live webcam inference with real-time FPS display
  - Preprocessing that exactly matches training normalisation
  - Post-processing: parse cls_scores and boxes, apply confidence threshold, draw results
  - Graceful handling of the no-detection case

Usage
-----
  # Single image:
  python 04_opencv_inference.py --model model.onnx --image test.jpg

  # Webcam (press Q to quit):
  python 04_opencv_inference.py --model model.onnx --webcam

  # Save output image:
  python 04_opencv_inference.py --model model.onnx --image test.jpg --save output.jpg

  # Use CUDA backend (requires OpenCV built with CUDA):
  python 04_opencv_inference.py --model model.onnx --webcam --cuda
"""

import argparse
import os
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants matching training normalisation (from 02_pytorch_detector.py)
# ---------------------------------------------------------------------------

IMG_SIZE = 224
CLASS_NAMES = ['circle', 'rectangle', 'triangle']

# ImageNet stats used during training
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Colour palette for bounding boxes (BGR)
BOX_COLORS = [
    (0, 200, 255),   # circle  -> yellow-ish
    (0, 255, 100),   # rectangle -> green
    (255, 100, 50),  # triangle -> blue-ish
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path, use_cuda=False):
    """Load ONNX model into OpenCV DNN and set backend/target."""
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    net = cv2.dnn.readNetFromONNX(model_path)

    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("Backend: CUDA")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("Backend: OpenCV CPU")

    return net


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(img_bgr, img_size=IMG_SIZE):
    """
    Convert a BGR image (as loaded by cv2.imread) to a DNN-ready blob.

    Matches training pipeline:
      1. Resize to img_size x img_size
      2. BGR -> RGB (swapRB=True)
      3. Scale to [0, 1]  (scalefactor=1/255)
      4. Subtract ImageNet mean
      5. Divide by ImageNet std
    """
    # blobFromImage handles steps 1-4 (resize, swapRB, scalefactor, mean subtraction)
    blob = cv2.dnn.blobFromImage(
        img_bgr,
        scalefactor=1.0 / 255.0,
        size=(img_size, img_size),
        mean=MEAN * 255.0,          # blobFromImage expects mean in 0-255 space
        swapRB=True,                # BGR -> RGB
        crop=False,
    )
    # blobFromImage does NOT divide by std; apply manually per channel
    for c in range(3):
        blob[0, c] /= STD[c]

    return blob  # shape: (1, 3, img_size, img_size)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def postprocess(net, orig_img, conf_threshold=0.5):
    """
    Run forward pass and parse SimpleDetector outputs.

    SimpleDetector outputs:
      cls_scores: (1, num_classes) — raw logits
      boxes:      (1, 4)           — normalised (cx, cy, w, h) in [0, 1]

    Returns list of dicts: {'class_id', 'class_name', 'confidence', 'box_xyxy'}
    """
    output_names = ['cls_scores', 'boxes']
    outputs = net.forward(output_names)

    cls_scores = outputs[0]  # (1, num_classes)
    boxes = outputs[1]       # (1, 4)

    H, W = orig_img.shape[:2]
    detections = []

    # Softmax over class dimension
    scores = cls_scores[0]                        # (num_classes,)
    exp_s = np.exp(scores - scores.max())
    probs = exp_s / exp_s.sum()
    class_id = int(np.argmax(probs))
    confidence = float(probs[class_id])

    if confidence < conf_threshold:
        # No detection above threshold — return empty list
        return detections

    cx, cy, bw, bh = boxes[0]
    # Convert normalised (cx,cy,w,h) -> absolute pixel (x1,y1,x2,y2)
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)

    # Clamp to image
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(W - 1, x2)
    y2 = min(H - 1, y2)

    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
    detections.append({
        'class_id': class_id,
        'class_name': class_name,
        'confidence': confidence,
        'box_xyxy': (x1, y1, x2, y2),
    })

    return detections


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def draw_detections(img, detections):
    """Draw bounding boxes and labels on a copy of img. Returns annotated image."""
    out = img.copy()

    if not detections:
        cv2.putText(out, 'No detection', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out

    for det in detections:
        x1, y1, x2, y2 = det['box_xyxy']
        cls_id = det['class_id']
        color = BOX_COLORS[cls_id % len(BOX_COLORS)]
        label = f"{det['class_name']} {det['confidence']:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return out


def draw_fps(img, fps):
    text = f"FPS: {fps:.1f}"
    cv2.putText(img, text, (10, img.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


# ---------------------------------------------------------------------------
# Inference modes
# ---------------------------------------------------------------------------

def infer_image(net, image_path, conf_threshold=0.5, save_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    blob = preprocess(img)
    net.setInput(blob)

    t0 = time.perf_counter()
    detections = postprocess(net, img, conf_threshold)
    elapsed = time.perf_counter() - t0

    annotated = draw_detections(img, detections)

    print(f"Inference time: {elapsed*1000:.1f} ms")
    if detections:
        for d in detections:
            print(f"  {d['class_name']}  conf={d['confidence']:.3f}  box={d['box_xyxy']}")
    else:
        print("  No objects detected above threshold.")

    if save_path:
        cv2.imwrite(save_path, annotated)
        print(f"Saved annotated image: {save_path}")
    else:
        cv2.imshow('Detection', annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def infer_webcam(net, cam_index=0, conf_threshold=0.5):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam index {cam_index}")

    print("Webcam inference running. Press Q to quit.")
    fps_smooth = 0.0
    alpha = 0.1  # EMA smoothing

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        t0 = time.perf_counter()
        blob = preprocess(frame)
        net.setInput(blob)
        detections = postprocess(net, frame, conf_threshold)
        elapsed = time.perf_counter() - t0

        fps = 1.0 / max(elapsed, 1e-6)
        fps_smooth = alpha * fps + (1 - alpha) * fps_smooth

        annotated = draw_detections(frame, detections)
        draw_fps(annotated, fps_smooth)

        cv2.imshow('Detection (Q to quit)', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='OpenCV DNN inference with ONNX model')
    parser.add_argument('--model', required=True, help='Path to .onnx model file')
    parser.add_argument('--image', default='', help='Path to input image')
    parser.add_argument('--webcam', action='store_true', help='Use webcam as input')
    parser.add_argument('--cam_index', type=int, default=0,
                        help='Webcam device index (default: 0)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    parser.add_argument('--save', default='',
                        help='Save annotated image to this path (image mode only)')
    parser.add_argument('--cuda', action='store_true',
                        help='Use CUDA backend (requires OpenCV built with CUDA support)')
    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.error("Specify --image <path> or --webcam")

    net = load_model(args.model, use_cuda=args.cuda)

    if args.image:
        infer_image(net, args.image, conf_threshold=args.conf,
                    save_path=args.save if args.save else None)
    elif args.webcam:
        infer_webcam(net, cam_index=args.cam_index, conf_threshold=args.conf)


if __name__ == '__main__':
    main()
