"""
dnn_detection_code.py
=====================
Comprehensive deep learning object detection with OpenCV's DNN module.

Covers:
  - Generic DNNDetector class (works with any model)
  - YOLOv4 full inference pipeline (decode + NMS)
  - MobileNet-SSD inference pipeline
  - Visualization utilities (bounding boxes, labels, confidence bars)
  - COCO 80-class name list
  - Synthetic pipeline test (no real model files required)

Requirements: opencv-python >= 4.5, numpy
"""

import cv2
import numpy as np
import random
import os
from typing import List, Tuple, Dict, Optional

# ---------------------------------------------------------------------------
# COCO 80-class names
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# MobileNet-SSD VOC 20 classes (+ background at index 0)
SSD_VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


# ---------------------------------------------------------------------------
# Utility: deterministic per-class color
# ---------------------------------------------------------------------------
def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """Return a visually distinct BGR color for each class ID."""
    rng = random.Random(class_id * 42 + 17)
    return (rng.randint(64, 255), rng.randint(64, 255), rng.randint(64, 255))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------
def draw_box_with_label(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    score: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a bounding box and a filled label tag above it."""
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    text = f"{label}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Filled background for label
    tag_y1 = max(y1 - th - baseline - 6, 0)
    cv2.rectangle(image, (x1, tag_y1), (x1 + tw + 4, y1), color, -1)
    # White text on colored background
    cv2.putText(image, text, (x1 + 2, y1 - baseline - 2),
                font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)


def draw_confidence_bar(
    image: np.ndarray,
    x: int, y: int,
    bar_width: int,
    confidence: float,
    color: Tuple[int, int, int] = (0, 255, 0),
    bar_height: int = 6,
) -> None:
    """Draw a horizontal confidence bar below a bounding box."""
    filled = int(bar_width * min(max(confidence, 0.0), 1.0))
    cv2.rectangle(image, (x, y), (x + bar_width, y + bar_height), (60, 60, 60), -1)
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + bar_height), color, -1)


def visualize_detections(
    image: np.ndarray,
    detections: List[Dict],
    class_names: List[str],
    conf_threshold: float = 0.0,
) -> np.ndarray:
    """
    Overlay detections on image.

    Args:
        image: BGR numpy array.
        detections: list of dicts with keys:
            'x1','y1','x2','y2' (int pixel coords)
            'class_id'          (int)
            'score'             (float)
        class_names: list of class label strings.
        conf_threshold: skip detections below this score.

    Returns:
        Annotated copy of image.
    """
    annotated = image.copy()
    for det in detections:
        if det["score"] < conf_threshold:
            continue
        cid = det["class_id"]
        color = get_class_color(cid)
        label = class_names[cid] if cid < len(class_names) else f"cls_{cid}"
        draw_box_with_label(
            annotated,
            det["x1"], det["y1"], det["x2"], det["y2"],
            label, det["score"], color,
        )
        bar_w = max(det["x2"] - det["x1"], 20)
        draw_confidence_bar(
            annotated,
            det["x1"], det["y2"] + 2,
            bar_w, det["score"], color,
        )
    return annotated


# ---------------------------------------------------------------------------
# Generic DNN Detector base class
# ---------------------------------------------------------------------------
class DNNDetector:
    """
    Generic wrapper around cv2.dnn.Net for object detection models.

    Subclass and override `_decode_outputs` to implement model-specific
    post-processing (YOLO, SSD, Faster R-CNN, etc.).
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        input_size: Tuple[int, int] = (416, 416),
        scale: float = 1 / 255.0,
        mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        swap_rb: bool = True,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        class_names: Optional[List[str]] = None,
        backend: int = cv2.dnn.DNN_BACKEND_OPENCV,
        target: int = cv2.dnn.DNN_TARGET_CPU,
    ) -> None:
        self.input_size = input_size
        self.scale = scale
        self.mean = mean
        self.swap_rb = swap_rb
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.class_names = class_names or []

        if model_path and os.path.isfile(model_path):
            if config_path:
                self.net = cv2.dnn.readNet(model_path, config_path)
            else:
                self.net = cv2.dnn.readNet(model_path)
            self.net.setPreferableBackend(backend)
            self.net.setPreferableTarget(target)
            self._output_layers = self._get_output_layer_names()
        else:
            # Allow construction without files for testing
            self.net = None
            self._output_layers = []

    def _get_output_layer_names(self) -> List[str]:
        layer_names = self.net.getLayerNames()
        unconnected = self.net.getUnconnectedOutLayers()
        # OpenCV >= 4.5 returns a 1D array; older versions return 2D
        if hasattr(unconnected, 'flatten'):
            unconnected = unconnected.flatten()
        return [layer_names[i - 1] for i in unconnected]

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Convert image to DNN blob."""
        return cv2.dnn.blobFromImage(
            image,
            scalefactor=self.scale,
            size=self.input_size,
            mean=self.mean,
            swapRB=self.swap_rb,
            crop=False,
        )

    def forward(self, image: np.ndarray):
        """Run forward pass, return raw network outputs."""
        if self.net is None:
            raise RuntimeError("No model loaded. Provide a valid model_path.")
        blob = self.preprocess(image)
        self.net.setInput(blob)
        return self.net.forward(self._output_layers)

    def _decode_outputs(self, outputs, image_h: int, image_w: int) -> List[Dict]:
        """Override in subclasses to implement model-specific decoding."""
        raise NotImplementedError

    def detect(self, image: np.ndarray) -> List[Dict]:
        """End-to-end detection: preprocess → forward → decode."""
        h, w = image.shape[:2]
        outputs = self.forward(image)
        return self._decode_outputs(outputs, h, w)


# ---------------------------------------------------------------------------
# YOLOv4 Detector
# ---------------------------------------------------------------------------
class YOLOv4Detector(DNNDetector):
    """
    Full YOLOv4 inference pipeline.

    Model files:
        yolov4.cfg      — architecture config (Darknet format)
        yolov4.weights  — pretrained weights
        https://github.com/AlexeyAB/darknet

    YOLO output format (per detection head):
        shape: (num_detections, 5 + num_classes)
        columns: [cx_norm, cy_norm, w_norm, h_norm, objectness, cls0, cls1, ...]
    """

    def __init__(
        self,
        cfg_path: str = "yolov4.cfg",
        weights_path: str = "yolov4.weights",
        input_size: Tuple[int, int] = (416, 416),
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=weights_path,
            config_path=cfg_path,
            input_size=input_size,
            scale=1 / 255.0,
            mean=(0.0, 0.0, 0.0),
            swap_rb=True,
            conf_threshold=conf_threshold,
            nms_threshold=nms_threshold,
            class_names=class_names or COCO_CLASSES,
            **kwargs,
        )

    def _decode_outputs(self, outputs, image_h: int, image_w: int) -> List[Dict]:
        """
        Decode YOLO raw outputs into detection dicts.

        Steps:
          1. Each row = [cx, cy, w, h, obj_conf, cls_0, ..., cls_N] (all normalized)
          2. Final confidence = obj_conf * max(cls_probs)
          3. Convert normalized center-form to pixel corner-form
          4. Apply NMS across all scales
        """
        boxes_xywh = []     # [x, y, w, h] in pixels (x,y = top-left corner)
        confidences = []
        class_ids = []

        for output in outputs:
            # output shape: (num_boxes, 5 + num_classes)
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                class_score = float(scores[class_id])
                objectness = float(detection[4])
                confidence = objectness * class_score

                if confidence < self.conf_threshold:
                    continue

                # YOLO outputs are normalized [0, 1] center-form
                cx = detection[0] * image_w
                cy = detection[1] * image_h
                bw = detection[2] * image_w
                bh = detection[3] * image_h

                x = int(cx - bw / 2)
                y = int(cy - bh / 2)

                boxes_xywh.append([x, y, int(bw), int(bh)])
                confidences.append(confidence)
                class_ids.append(class_id)

        if not boxes_xywh:
            return []

        # NMS
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, confidences,
            self.conf_threshold, self.nms_threshold,
        )
        if len(indices) == 0:
            return []

        indices = indices.flatten()
        detections = []
        for i in indices:
            x, y, w, h = boxes_xywh[i]
            detections.append({
                "x1": max(x, 0),
                "y1": max(y, 0),
                "x2": max(x + w, 0),
                "y2": max(y + h, 0),
                "class_id": class_ids[i],
                "score": confidences[i],
            })
        return detections


# ---------------------------------------------------------------------------
# MobileNet-SSD Detector (Caffe)
# ---------------------------------------------------------------------------
class MobileNetSSDDetector(DNNDetector):
    """
    MobileNet-SSD inference pipeline (Caffe model).

    Model files:
        MobileNetSSD_deploy.prototxt
        MobileNetSSD_deploy.caffemodel
        https://github.com/chuanqi305/MobileNet-SSD

    SSD output format:
        shape: (1, 1, N, 7)
        columns: [img_id, class_id, confidence, x1, y1, x2, y2]
        Coordinates are normalized [0, 1]; NMS/softmax applied internally.
    """

    def __init__(
        self,
        prototxt_path: str = "MobileNetSSD_deploy.prototxt",
        caffemodel_path: str = "MobileNetSSD_deploy.caffemodel",
        conf_threshold: float = 0.5,
        class_names: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=caffemodel_path,
            config_path=prototxt_path,
            input_size=(300, 300),
            scale=0.007843,
            mean=(127.5, 127.5, 127.5),
            swap_rb=False,
            conf_threshold=conf_threshold,
            nms_threshold=0.45,
            class_names=class_names or SSD_VOC_CLASSES,
            **kwargs,
        )

    def _decode_outputs(self, outputs, image_h: int, image_w: int) -> List[Dict]:
        """
        Decode SSD DetectionOutput layer.

        The Caffe DetectionOutput layer already applies NMS and softmax,
        so we only need to threshold and scale coordinates.
        """
        detections_raw = outputs[0]  # shape: (1, 1, N, 7)
        detections = []

        for i in range(detections_raw.shape[2]):
            row = detections_raw[0, 0, i]
            confidence = float(row[2])
            if confidence < self.conf_threshold:
                continue

            class_id = int(row[1])
            x1 = int(row[3] * image_w)
            y1 = int(row[4] * image_h)
            x2 = int(row[5] * image_w)
            y2 = int(row[6] * image_h)

            detections.append({
                "x1": max(x1, 0),
                "y1": max(y1, 0),
                "x2": min(x2, image_w),
                "y2": min(y2, image_h),
                "class_id": class_id,
                "score": confidence,
            })
        return detections


# ---------------------------------------------------------------------------
# Synthetic pipeline test
# ---------------------------------------------------------------------------
def create_synthetic_test_image(
    width: int = 416,
    height: int = 416,
) -> np.ndarray:
    """
    Create a simple synthetic test image:
    black background with a bright colored rectangle.

    This lets us verify the blob/forward pipeline shape without real model files.
    """
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Draw a bright green rectangle (simulates an "object")
    x1, y1, x2, y2 = 80, 100, 260, 300
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 50), -1)

    # A red square in the corner
    cv2.rectangle(image, (300, 310), (380, 390), (30, 30, 210), -1)

    # Draw some text
    cv2.putText(image, "Synthetic Test", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    return image


def run_blob_pipeline_test() -> None:
    """
    Verify the blobFromImage → shape pipeline without a real model.

    This is a unit-style test confirming:
      - blobFromImage produces correct NCHW shape
      - synthetic image creation works
      - visualization utilities work
    """
    print("=" * 60)
    print("SYNTHETIC PIPELINE TEST (no model files required)")
    print("=" * 60)

    # 1. Create synthetic image
    image = create_synthetic_test_image(416, 416)
    print(f"[1] Synthetic image shape: {image.shape}")

    # 2. YOLO-style preprocessing
    blob_yolo = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
    )
    print(f"[2] YOLO blob shape (NCHW): {blob_yolo.shape}")
    assert blob_yolo.shape == (1, 3, 416, 416), "YOLO blob shape mismatch"
    assert blob_yolo.min() >= 0.0 and blob_yolo.max() <= 1.0, "YOLO blob value range error"
    print(f"    Value range: [{blob_yolo.min():.4f}, {blob_yolo.max():.4f}]  OK")

    # 3. SSD-style preprocessing
    blob_ssd = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
        swapRB=False,
    )
    print(f"[3] SSD blob shape (NCHW):  {blob_ssd.shape}")
    assert blob_ssd.shape == (1, 3, 300, 300), "SSD blob shape mismatch"
    print(f"    Value range: [{blob_ssd.min():.4f}, {blob_ssd.max():.4f}]  OK")

    # 4. Simulate decoded detections and test visualization
    fake_detections = [
        {"x1": 80, "y1": 100, "x2": 260, "y2": 300, "class_id": 0, "score": 0.92},
        {"x1": 300, "y1": 310, "x2": 380, "y2": 390, "class_id": 2, "score": 0.76},
    ]
    annotated = visualize_detections(image, fake_detections, COCO_CLASSES, conf_threshold=0.5)
    print(f"[4] Visualized {len(fake_detections)} synthetic detections")
    print(f"    Annotated image shape: {annotated.shape}")

    # 5. Test NMS
    boxes_xywh = [[10, 10, 100, 100], [15, 15, 100, 100], [200, 200, 80, 80]]
    scores = [0.9, 0.8, 0.75]
    indices = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.5, 0.4)
    kept = indices.flatten().tolist() if len(indices) > 0 else []
    print(f"[5] NMS test: {len(boxes_xywh)} boxes → {len(kept)} kept after NMS (IoU=0.4)")
    assert 0 in kept, "Highest-confidence box should survive NMS"
    assert 1 not in kept, "Overlapping box should be suppressed"
    assert 2 in kept, "Non-overlapping box should survive NMS"

    # 6. Color coding
    colors = [get_class_color(i) for i in range(5)]
    print(f"[6] Sample class colors: {colors}")

    print()
    print("All pipeline tests passed.")
    return annotated


# ---------------------------------------------------------------------------
# Demo: run YOLOv4 on a real image (requires model files)
# ---------------------------------------------------------------------------
def demo_yolov4(image_path: str, cfg: str, weights: str, names_file: str) -> None:
    """
    Full YOLOv4 demo. Requires model files from:
    https://github.com/AlexeyAB/darknet/releases/tag/darknet_yolo_v4_pre
    """
    if not all(os.path.isfile(p) for p in [image_path, cfg, weights]):
        print("demo_yolov4: one or more files not found, skipping.")
        return

    with open(names_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    detector = YOLOv4Detector(
        cfg_path=cfg,
        weights_path=weights,
        input_size=(416, 416),
        conf_threshold=0.4,
        nms_threshold=0.4,
        class_names=class_names,
    )

    image = cv2.imread(image_path)
    print(f"Running YOLOv4 on {image_path} ({image.shape[1]}x{image.shape[0]}) ...")

    detections = detector.detect(image)
    print(f"Detected {len(detections)} object(s):")
    for d in detections:
        name = class_names[d['class_id']] if d['class_id'] < len(class_names) else "?"
        print(f"  {name:20s}  score={d['score']:.3f}  "
              f"box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")

    annotated = visualize_detections(image, detections, class_names)
    out_path = "yolov4_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated result to: {out_path}")


# ---------------------------------------------------------------------------
# Demo: run MobileNet-SSD on a real image (requires model files)
# ---------------------------------------------------------------------------
def demo_mobilenet_ssd(image_path: str, prototxt: str, caffemodel: str) -> None:
    """
    Full MobileNet-SSD demo. Requires model files from:
    https://github.com/chuanqi305/MobileNet-SSD
    """
    if not all(os.path.isfile(p) for p in [image_path, prototxt, caffemodel]):
        print("demo_mobilenet_ssd: one or more files not found, skipping.")
        return

    detector = MobileNetSSDDetector(
        prototxt_path=prototxt,
        caffemodel_path=caffemodel,
        conf_threshold=0.4,
    )

    image = cv2.imread(image_path)
    print(f"Running MobileNet-SSD on {image_path} ...")

    detections = detector.detect(image)
    print(f"Detected {len(detections)} object(s):")
    for d in detections:
        name = SSD_VOC_CLASSES[d['class_id']] if d['class_id'] < len(SSD_VOC_CLASSES) else "?"
        print(f"  {name:20s}  score={d['score']:.3f}  "
              f"box=({d['x1']},{d['y1']})-({d['x2']},{d['y2']})")

    annotated = visualize_detections(image, detections, SSD_VOC_CLASSES)
    out_path = "ssd_result.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"Saved annotated result to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Always run the self-contained synthetic test
    annotated = run_blob_pipeline_test()

    # Save the synthetic annotated image
    synthetic_out = "synthetic_detection_test.jpg"
    cv2.imwrite(synthetic_out, annotated)
    print(f"\nSynthetic annotated image saved to: {synthetic_out}")

    # Optionally run real model demos if files are present
    # Uncomment and update paths as needed:
    #
    # demo_yolov4(
    #     image_path="test.jpg",
    #     cfg="yolov4.cfg",
    #     weights="yolov4.weights",
    #     names_file="coco.names",
    # )
    #
    # demo_mobilenet_ssd(
    #     image_path="test.jpg",
    #     prototxt="MobileNetSSD_deploy.prototxt",
    #     caffemodel="MobileNetSSD_deploy.caffemodel",
    # )
