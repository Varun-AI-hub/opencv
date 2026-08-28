# Deep Learning Object Detection with OpenCV DNN

## 1. OpenCV DNN Module Overview

OpenCV's `dnn` module provides a unified interface for loading and running inference with neural networks trained in popular frameworks including Darknet, Caffe, TensorFlow, PyTorch (via ONNX), and others.

### Loading Models

```python
import cv2

# Generic loader — auto-detects format in many cases
net = cv2.dnn.readNet("model.weights", "model.cfg")          # Darknet
net = cv2.dnn.readNetFromONNX("model.onnx")                  # ONNX (PyTorch, etc.)
net = cv2.dnn.readNetFromCaffe("model.prototxt", "model.caffemodel")  # Caffe
net = cv2.dnn.readNetFromDarknet("model.cfg", "model.weights")        # Darknet explicit
net = cv2.dnn.readNetFromTensorflow("model.pb", "model.pbtxt")        # TensorFlow
```

### Backend and Target Selection

OpenCV DNN supports multiple computation backends and hardware targets, decoupled from each other:

| Backend constant | Description |
|---|---|
| `cv2.dnn.DNN_BACKEND_DEFAULT` | OpenCV's own implementation |
| `cv2.dnn.DNN_BACKEND_OPENCV` | Explicit OpenCV backend |
| `cv2.dnn.DNN_BACKEND_CUDA` | NVIDIA CUDA (requires OpenCV built with CUDA) |
| `cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE` | Intel OpenVINO |
| `cv2.dnn.DNN_BACKEND_HALIDE` | Halide JIT compilation |

| Target constant | Description |
|---|---|
| `cv2.dnn.DNN_TARGET_CPU` | Host CPU (default) |
| `cv2.dnn.DNN_TARGET_CUDA` | NVIDIA GPU (CUDA) |
| `cv2.dnn.DNN_TARGET_CUDA_FP16` | NVIDIA GPU, half-precision |
| `cv2.dnn.DNN_TARGET_OPENCL` | Any OpenCL device (GPU/iGPU) |
| `cv2.dnn.DNN_TARGET_OPENCL_FP16` | OpenCL half-precision |
| `cv2.dnn.DNN_TARGET_MYRIAD` | Intel Myriad VPU |

```python
# GPU acceleration with CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Or OpenCL for AMD/Intel GPUs
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
```

### Preprocessing with `cv2.dnn.blobFromImage`

`blobFromImage` converts an image (or batch of images) into the 4-D blob format expected by DNN models.

```python
blob = cv2.dnn.blobFromImage(
    image,          # Input: HxWxC BGR numpy array
    scalefactor,    # Pixel scaling factor (e.g. 1/255.0 for [0,1] normalization)
    size,           # Spatial size to resize to, e.g. (416, 416)
    mean,           # Mean subtraction per channel: (B_mean, G_mean, R_mean)
    swapRB,         # If True, swap R and B channels (BGR→RGB), default False
    crop            # If True, center-crop after resize; if False, stretch
)
# Returns: blob of shape (1, C, H, W) — NCHW format
```

**Common presets:**

| Model family | scalefactor | size | mean | swapRB |
|---|---|---|---|---|
| YOLO | `1/255.0` | `(416,416)` or `(608,608)` | `(0,0,0)` | `True` |
| MobileNet-SSD (Caffe) | `0.007843` | `(300,300)` | `(127.5,127.5,127.5)` | `False` |
| Faster R-CNN (ONNX) | `1.0` | variable | `(102.9, 115.9, 122.8)` | `False` |
| EfficientDet | `1/255.0` | `(512,512)` | `(0,0,0)` | `True` |

### Forward Pass and Output Blob Interpretation

```python
net.setInput(blob)

# Single output layer
output = net.forward()

# Multiple output layers (e.g. YOLO's 3 heads)
layer_names = net.getLayerNames()
output_layer_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
outputs = net.forward(output_layer_names)
# outputs is a list of arrays, one per detection head
```

---

## 2. YOLO Family (You Only Look Once)

### Architecture Overview

YOLOv3 and YOLOv4 share the same output structure, differing primarily in their backbone and neck:

- **YOLOv3**: Darknet-53 backbone, no neck, three detection heads.
- **YOLOv4**: CSPDarknet-53 backbone, PANet/SPP neck, three detection heads. Adds Mish activations, mosaic augmentation, CIoU loss.

The network outputs detections at three feature map scales to handle objects of different sizes:
- **13 × 13** — large objects (anchors cover large receptive fields)
- **26 × 26** — medium objects
- **52 × 52** — small objects

### Anchor Boxes

YOLO uses 9 pre-defined anchors (3 per scale), chosen by k-means clustering on COCO bounding box dimensions. For a 416×416 input (COCO defaults):

```
Large scale (13×13):   [(116,90), (156,198), (373,326)]
Medium scale (26×26):  [(30,61),  (62,45),   (59,119)]
Small scale (52×52):   [(10,13),  (16,30),   (33,23)]
```

Each anchor is a `(width, height)` pair in pixels relative to the input image size.

### Output Format

Each detection head outputs a tensor of shape:

```
[batch, grid_h, grid_w, num_anchors * (5 + C)]
```

or equivalently (after reshape):

```
[batch, grid_h * grid_w * num_anchors, 5 + C]
```

where `C` is the number of classes (80 for COCO). The 5 base values per prediction are:
`[tx, ty, tw, th, objectness_score]` followed by `C` class probability logits.

### Decoding Predictions

Raw network outputs must be decoded to absolute bounding box coordinates:

```
bx = sigmoid(tx) + cx      # cx = column index of the grid cell
by = sigmoid(ty) + cy      # cy = row index of the grid cell
bw = pw * exp(tw)           # pw = anchor width (in grid units)
bh = ph * exp(th)           # ph = anchor height (in grid units)
```

- `sigmoid` squashes `tx`, `ty` to [0,1], ensuring the box center stays within the responsible grid cell.
- `exp` allows the box to grow or shrink relative to the anchor.
- Divide `bx/grid_w` and `by/grid_h` to get normalized [0,1] coordinates, then multiply by image dimensions for pixel coordinates.

**Final confidence score:**

```
confidence = sigmoid(objectness) * sigmoid(class_prob[c])
```

Only boxes where `confidence > threshold` are kept.

### Non-Maximum Suppression (NMS)

After decoding, many overlapping boxes remain. NMS removes redundant detections:

1. Sort all boxes by confidence (descending).
2. Keep the highest-confidence box; suppress all boxes with IoU > `nms_threshold` against it.
3. Repeat for remaining boxes.

OpenCV provides `cv2.dnn.NMSBoxes` for this step.

### Full OpenCV YOLOv4 Code Example

```python
import cv2
import numpy as np

# --- Load model ---
net = cv2.dnn.readNetFromDarknet("yolov4.cfg", "yolov4.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

with open("coco.names", "r") as f:
    class_names = [line.strip() for line in f]

# --- Inference ---
image = cv2.imread("image.jpg")
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416),
                              mean=(0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outputs = net.forward(output_layers)  # list of 3 arrays

# --- Decode ---
CONF_THRESHOLD = 0.5
NMS_THRESHOLD  = 0.4

boxes, confidences, class_ids = [], [], []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id]) * float(detection[4])
        if confidence > CONF_THRESHOLD:
            cx = int(detection[0] * w)
            cy = int(detection[1] * h)
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)
            x1 = cx - bw // 2
            y1 = cy - bh // 2
            boxes.append([x1, y1, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

# --- Visualize ---
for i in indices.flatten():
    x, y, bw, bh = boxes[i]
    label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(image, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.putText(image, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

cv2.imwrite("result.jpg", image)
```

### Key YOLO Parameters

| Parameter | Typical value | Effect |
|---|---|---|
| Confidence threshold | 0.25 – 0.5 | Lower = more detections, more false positives |
| NMS threshold (IoU) | 0.3 – 0.5 | Lower = more aggressive suppression |
| Input size | 416×416 / 608×608 | Larger = higher accuracy, slower inference |
| Anchor scale | model-specific | Must match `.cfg` file anchors |

---

## 3. SSD (Single Shot Detector)

### Architecture

SSD performs detection in a single forward pass using feature maps at multiple scales. The typical variant used with OpenCV is **MobileNet-SSD**, which replaces VGG-16 with a lightweight MobileNetV1 or V2 backbone.

**Key components:**
- **Base network**: MobileNetV1/V2, truncated before final classifier.
- **Extra feature layers**: 4–5 additional convolutional layers appended to extract coarser feature maps.
- **Default boxes (anchors)**: Each feature map cell generates boxes with multiple aspect ratios (1:1, 2:1, 1:2, 3:1, 1:3) at two scales. Total ≈ 8732 default boxes for a 300×300 input.

**Detection heads (two parallel outputs per feature map):**
1. **Location head**: 4 values per default box (offset from default box: `[dx, dy, dw, dh]`).
2. **Classification head**: `num_classes + 1` softmax scores per default box (class 0 = background).

### Output Format (Caffe MobileNet-SSD)

After `net.forward()`, the output shape is `(1, 1, N, 7)` where each row is:

```
[batch_id, class_id, confidence, x1_norm, y1_norm, x2_norm, y2_norm]
```

Coordinates are normalized to [0, 1] relative to input image size.

### Full OpenCV MobileNet-SSD Code Example

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

image = cv2.imread("image.jpg")
h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)),
    scalefactor=0.007843,
    size=(300, 300),
    mean=(127.5, 127.5, 127.5),
    swapRB=False
)
net.setInput(blob)
detections = net.forward()   # shape: (1, 1, N, 7)

CONF_THRESHOLD = 0.5
for i in range(detections.shape[2]):
    confidence = float(detections[0, 0, i, 2])
    if confidence > CONF_THRESHOLD:
        class_id = int(detections[0, 0, i, 1])
        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)
        label = f"{CLASSES[class_id]}: {confidence:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

cv2.imwrite("ssd_result.jpg", image)
```

### Softmax Decoding

The Caffe MobileNet-SSD model applies softmax internally, so the output `detections[0,0,i,2]` is already a probability in [0,1]. No manual softmax is needed. NMS is also applied internally by the `DetectionOutput` layer.

---

## 4. Faster R-CNN / R-FCN (via ONNX)

### Two-Stage Detection Pipeline

Faster R-CNN separates detection into two stages:

1. **Region Proposal Network (RPN)**: A small convolutional network sliding over the shared feature map, producing candidate regions ("proposals") likely to contain objects. The RPN outputs objectness scores and bounding box deltas for each anchor.
2. **ROI Pooling → Classification**: Each proposal is cropped from the feature map using ROI Pooling (or ROI Align), then classified and box-refined by fully connected layers.

**Advantages**: Higher accuracy on small and overlapping objects.
**Disadvantages**: Slower inference; two-pass architecture.

**R-FCN (Region-based Fully Convolutional Networks)** replaces the per-ROI FC layers with position-sensitive score maps, greatly improving speed while retaining accuracy.

### Loading via ONNX

PyTorch's `torchvision` library exports Faster R-CNN to ONNX format for use in OpenCV:

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("faster_rcnn_resnet50_coco.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

image = cv2.imread("image.jpg")
h, w = image.shape[:2]

# Faster R-CNN expects float32 input, no normalization beyond mean subtraction
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1.0,
    size=(w, h),           # Keep original size or use a standard size
    mean=(102.9, 115.9, 122.8),  # ImageNet mean (BGR order)
    swapRB=False,
    crop=False
)
net.setInput(blob)

# Output layers depend on ONNX export; typically 'boxes', 'labels', 'scores'
output_layer_names = ["boxes", "labels", "scores"]
boxes_out, labels_out, scores_out = net.forward(output_layer_names)

CONF_THRESHOLD = 0.5
for i in range(scores_out.shape[0]):
    score = float(scores_out[i])
    if score > CONF_THRESHOLD:
        x1, y1, x2, y2 = boxes_out[i].astype(int)
        label = int(labels_out[i])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, f"cls{label}:{score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

cv2.imwrite("frcnn_result.jpg", image)
```

**Note**: The exact output layer names vary by ONNX export version. Inspect available layers with:

```python
for layer_id in net.getUnconnectedOutLayers():
    print(net.getLayerNames()[layer_id - 1])
```

---

## 5. EfficientDet (via ONNX)

### Architecture

EfficientDet introduces two innovations over earlier detectors:

- **BiFPN (Bidirectional Feature Pyramid Network)**: Replaces the standard FPN/PANet with a weighted bi-directional feature fusion network, allowing each resolution to contribute more evenly.
- **Compound Scaling**: Simultaneously scales the backbone depth/width/resolution, BiFPN width/depth, and prediction head using a single compound coefficient φ. Models are named EfficientDet-D0 through D7.

| Variant | Input size | mAP COCO | Params |
|---|---|---|---|
| D0 | 512×512 | 34.6 | 3.9M |
| D1 | 640×640 | 40.5 | 6.6M |
| D2 | 768×768 | 43.0 | 8.1M |
| D4 | 1024×1024 | 49.4 | 20.7M |
| D7 | 1536×1536 | 55.1 | 51.9M |

### Loading and Running in OpenCV

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromONNX("efficientdet_d0_coco.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

image = cv2.imread("image.jpg")
input_size = 512   # D0

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/255.0,
    size=(input_size, input_size),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)
net.setInput(blob)
detections = net.forward()

# Typical EfficientDet ONNX output: (1, num_detections, 7)
# [image_id, y1, x1, y2, x2, score, class_id] — coordinates normalized [0,1]
h, w = image.shape[:2]
CONF_THRESHOLD = 0.4

for det in detections[0]:
    score = float(det[5])
    if score > CONF_THRESHOLD:
        y1 = int(det[1] * h); x1 = int(det[2] * w)
        y2 = int(det[3] * h); x2 = int(det[4] * w)
        cls = int(det[6])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 100), 2)
        cv2.putText(image, f"cls{cls}:{score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

cv2.imwrite("efficientdet_result.jpg", image)
```

---

## 6. Performance Comparison

The table below summarizes commonly referenced metrics on the COCO 2017 validation set. Speeds are approximate and will vary significantly by hardware, input size, and software version.

| Model | Backbone | mAP COCO | Speed (ms) CPU | Model Size (MB) | Input Size |
|---|---|---|---|---|---|
| YOLOv3 | Darknet-53 | 55.3 | ~200 | 237 | 416×416 |
| YOLOv4 | CSPDarknet-53 | 65.7 | ~250 | 245 | 416×416 |
| YOLOv4-tiny | CSPDarknet-tiny | 40.2 | ~29 | 23 | 416×416 |
| MobileNet-SSD v1 | MobileNetV1 | 23.2 | ~25 | 22 | 300×300 |
| MobileNet-SSD v2 | MobileNetV2 | 29.0 | ~31 | 64 | 300×300 |
| Faster R-CNN | ResNet-50-FPN | 42.0 | ~140 | 167 | variable |
| EfficientDet-D0 | EfficientNet-B0 | 34.6 | ~45 | 16 | 512×512 |
| EfficientDet-D4 | EfficientNet-B4 | 49.4 | ~260 | 83 | 1024×1024 |

**Notes:**
- mAP is COCO AP@[0.50:0.95] (primary metric).
- CPU timings are indicative only (measured on Intel Core i7; CUDA/OpenCL can provide 5–20× speedup).
- "Speed" for YOLO variants above refers to GPU inference; CPU is slower.

---

## 7. Visualization Utilities

### Drawing Bounding Boxes with Labels and Scores

```python
def draw_detection(image, box, label, score, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with a label+score tag above it."""
    x, y, w, h = box
    x2, y2 = x + w, y + h
    cv2.rectangle(image, (x, y), (x2, y2), color, thickness)

    text = f"{label}: {score:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    # Background rectangle for text readability
    cv2.rectangle(image,
                  (x, y - text_h - baseline - 5),
                  (x + text_w, y),
                  color, -1)
    cv2.putText(image, text, (x, y - baseline - 3),
                font, font_scale, (0, 0, 0), thickness)
```

### Color-Coding by Class

```python
import random

def get_class_color(class_id, num_classes=80):
    """Deterministic, visually distinct color per class ID."""
    random.seed(class_id * 42 + 17)
    return tuple(random.randint(64, 255) for _ in range(3))
```

### Confidence Bar Overlay

```python
def draw_confidence_bar(image, x, y, width, confidence, color=(0, 255, 0)):
    """Draw a filled confidence bar to the right of the bounding box."""
    bar_height = 8
    max_bar_len = width
    filled_len = int(max_bar_len * confidence)

    bar_x = x
    bar_y = y + 4

    # Background bar (gray)
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + max_bar_len, bar_y + bar_height),
                  (80, 80, 80), -1)
    # Filled portion
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + filled_len, bar_y + bar_height),
                  color, -1)
```

### Complete Visualization Wrapper

```python
def visualize_detections(image, detections, class_names, conf_threshold=0.5):
    """
    detections: list of dicts with keys:
        'box'      : [x, y, w, h] in pixel coordinates
        'class_id' : int
        'score'    : float
    """
    annotated = image.copy()
    for det in detections:
        if det["score"] < conf_threshold:
            continue
        color = get_class_color(det["class_id"], len(class_names))
        label = class_names[det["class_id"]] if det["class_id"] < len(class_names) \
                else f"cls_{det['class_id']}"
        draw_detection(annotated, det["box"], label, det["score"], color)
        draw_confidence_bar(annotated,
                            det["box"][0], det["box"][1] + det["box"][3] + 2,
                            min(det["box"][2], 60), det["score"], color)
    return annotated
```

---

## References

- Redmon, J. & Farhadi, A. (2018). *YOLOv3: An Incremental Improvement*. arXiv:1804.02767
- Bochkovskiy, A., Wang, C.-Y. & Liao, H.-Y. M. (2020). *YOLOv4: Optimal Speed and Accuracy of Object Detection*. arXiv:2004.10934
- Liu, W. et al. (2016). *SSD: Single Shot MultiBox Detector*. ECCV 2016.
- Ren, S. et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015.
- Tan, M., Pang, R. & Le, Q. V. (2020). *EfficientDet: Scalable and Efficient Object Detection*. CVPR 2020.
- OpenCV DNN Module documentation: https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html
