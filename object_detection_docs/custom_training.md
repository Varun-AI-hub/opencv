# Custom Object Detection Training for OpenCV Deployment

This guide walks through the full pipeline: dataset preparation, model training, ONNX export, and running inference with OpenCV's DNN module. It covers modern deep learning approaches (YOLOv8) as well as classical methods (Haar Cascade, HOG+SVM).

---

## 1. Dataset Preparation

### Annotation Formats

**YOLO TXT format** (one `.txt` per image, same filename stem):
```
<class_id> <cx> <cy> <width> <height>
```
All values are normalized to [0, 1] relative to image dimensions.
```
0 0.5156 0.3812 0.1234 0.2100
1 0.7800 0.6100 0.0900 0.1500
```

**COCO JSON format** (single file for the split):
```json
{
  "images": [{"id": 1, "file_name": "img001.jpg", "width": 640, "height": 480}],
  "annotations": [
    {"id": 1, "image_id": 1, "category_id": 1,
     "bbox": [x_min, y_min, width, height], "area": 5000, "iscrowd": 0}
  ],
  "categories": [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
}
```

**Pascal VOC XML format** (one `.xml` per image):
```xml
<annotation>
  <filename>img001.jpg</filename>
  <size><width>640</width><height>480</height><depth>3</depth></size>
  <object>
    <name>cat</name>
    <bndbox>
      <xmin>120</xmin><ymin>80</ymin><xmax>300</xmax><ymax>260</ymax>
    </bndbox>
  </object>
</annotation>
```

### Annotation Tools

| Tool | Formats | Notes |
|------|---------|-------|
| **LabelImg** | YOLO TXT, Pascal VOC | Lightweight desktop app, good for YOLO workflows |
| **CVAT** | COCO, VOC, YOLO, MOT | Web-based, supports video, team collaboration |
| **Roboflow** | All major formats | Cloud-based, built-in augmentation & export |
| **Label Studio** | COCO, YOLO, custom | Self-hostable, flexible UI |

### Recommended Directory Structure

```
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   ├── val/
│   │   └── img100.jpg
│   └── test/
│       └── img200.jpg
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   ├── val/
│   │   └── img100.txt
│   └── test/
│       └── img200.txt
└── data.yaml
```

### data.yaml Configuration

```yaml
# data.yaml — required by Ultralytics YOLOv8
path: /absolute/path/to/dataset   # root dataset dir
train: images/train
val:   images/val
test:  images/test                 # optional

nc: 3                              # number of classes
names:
  0: cat
  1: dog
  2: person
```

### Data Augmentation

Ultralytics handles augmentation automatically during training. Key transforms:

| Augmentation | Parameter | Typical Value |
|---|---|---|
| Horizontal flip | `fliplr` | 0.5 |
| Vertical flip | `flipud` | 0.0 |
| HSV hue shift | `hsv_h` | 0.015 |
| HSV saturation | `hsv_s` | 0.7 |
| HSV value | `hsv_v` | 0.4 |
| Rotation | `degrees` | 0.0 |
| Mosaic | `mosaic` | 1.0 |
| Mixup | `mixup` | 0.0 |
| Copy-paste | `copy_paste` | 0.0 |
| Random crop | `scale` | 0.5 |

### Train / Val / Test Split Strategies

**Random 80/10/10 split** (common baseline):
```python
import os, shutil, random
from pathlib import Path

def split_dataset(src_images, src_labels, dst_root, ratios=(0.8, 0.1, 0.1), seed=42):
    random.seed(seed)
    images = sorted(Path(src_images).glob("*.jpg"))
    random.shuffle(images)
    n = len(images)
    cuts = [int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))]
    splits = {"train": images[:cuts[0]], "val": images[cuts[0]:cuts[1]], "test": images[cuts[1]:]}

    for split, files in splits.items():
        (Path(dst_root) / "images" / split).mkdir(parents=True, exist_ok=True)
        (Path(dst_root) / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, Path(dst_root) / "images" / split / img.name)
            lbl = Path(src_labels) / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy(lbl, Path(dst_root) / "labels" / split / lbl.name)
```

**Stratified split** (balanced class distribution per split): use `sklearn.model_selection.StratifiedShuffleSplit` on a per-image class label array. For multi-label images, use iterative stratification (`iterstrat` package).

---

## 2. Training YOLOv8 (Ultralytics)

### Installation

```bash
pip install ultralytics
# Optional but recommended:
pip install wandb tensorboard
```

### Model Size Selection

| Model | Parameters | Speed (T4 GPU) | mAP@50-95 (COCO) | Use Case |
|-------|-----------|----------------|-------------------|----------|
| `yolov8n.pt` | 3.2M | ~80 FPS | 37.3 | Edge / mobile |
| `yolov8s.pt` | 11.2M | ~60 FPS | 44.9 | Balanced |
| `yolov8m.pt` | 25.9M | ~40 FPS | 50.2 | Higher accuracy |
| `yolov8l.pt` | 43.7M | ~25 FPS | 52.9 | Accuracy-first |
| `yolov8x.pt` | 68.2M | ~15 FPS | 53.9 | Max accuracy |

### Full Training Script

```python
from ultralytics import YOLO

# Load a pretrained model (transfer learning from COCO weights)
model = YOLO("yolov8s.pt")

# Train
results = model.train(
    # --- Data ---
    data="dataset/data.yaml",
    imgsz=640,              # Input resolution (square)

    # --- Training duration ---
    epochs=100,
    patience=20,            # Early stopping: stop if no improvement for N epochs

    # --- Batch & hardware ---
    batch=16,               # -1 for auto-batch (uses ~60% GPU memory)
    device=0,               # GPU id, or "cpu", or [0,1] for multi-GPU

    # --- Optimizer ---
    optimizer="AdamW",      # SGD | Adam | AdamW | RMSProp
    lr0=0.01,               # Initial learning rate
    lrf=0.01,               # Final lr = lr0 * lrf
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,

    # --- Augmentation ---
    augment=True,
    mosaic=1.0,             # Mosaic probability (highly recommended)
    mixup=0.0,              # Mixup alpha
    copy_paste=0.0,         # Copy-paste augmentation
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,              # Random scale ± factor

    # --- Output ---
    project="runs/detect",
    name="my_detector",
    save_period=10,         # Save checkpoint every N epochs
    exist_ok=False,

    # --- Logging ---
    # Set WANDB_API_KEY env var to enable W&B logging automatically
    # Or use: wandb=True
)

print(f"Best mAP@50-95: {results.results_dict['metrics/mAP50-95(B)']:.4f}")
```

### Transfer Learning Notes

- Starting from `yolov8s.pt` (COCO pretrained) typically converges 3-5x faster than training from scratch.
- For very different domains (e.g., medical, satellite imagery), use a lower `lr0` (e.g., `0.001`) and more `warmup_epochs`.
- Freeze backbone layers for small datasets: add `freeze=10` to freeze the first 10 layers.

### Monitoring Training

**TensorBoard:**
```bash
tensorboard --logdir runs/detect/my_detector
```

**Weights & Biases:**
```bash
pip install wandb
wandb login
# Then just run training — Ultralytics auto-detects wandb
```

### Interpreting Results

After training, `runs/detect/my_detector/results.csv` contains per-epoch metrics:

| Metric | Meaning | Target |
|--------|---------|--------|
| `train/box_loss` | Bounding box regression loss | Decreasing |
| `train/cls_loss` | Classification loss | Decreasing |
| `train/dfl_loss` | Distribution focal loss | Decreasing |
| `metrics/precision(B)` | TP / (TP + FP) | > 0.8 |
| `metrics/recall(B)` | TP / (TP + FN) | > 0.8 |
| `metrics/mAP50(B)` | mAP at IoU=0.5 | > 0.75 for most tasks |
| `metrics/mAP50-95(B)` | mAP at IoU 0.5:0.05:0.95 | > 0.5 for most tasks |

Key file: `runs/detect/my_detector/weights/best.pt` — checkpoint with best validation mAP.

---

## 3. Exporting to ONNX for OpenCV

### Export Command

```python
from ultralytics import YOLO

model = YOLO("runs/detect/my_detector/weights/best.pt")

model.export(
    format="onnx",
    opset=12,           # CRITICAL: OpenCV DNN requires opset <= 12
    simplify=True,      # Uses onnx-simplifier to clean the graph
    dynamic=False,      # Fixed input shape required for OpenCV DNN
    imgsz=640,          # Must match training imgsz
    half=False,         # FP16 not supported in OpenCV DNN CPU
)
# Output: runs/detect/my_detector/weights/best.onnx
```

**Why opset=12?** OpenCV's DNN module does not support newer ONNX opsets. opset 12 covers all YOLOv8 operations while maintaining compatibility.

**Why dynamic=False?** OpenCV DNN requires a static input shape. Dynamic batching is not supported.

### Verifying the ONNX Export

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

# Print input/output info
for inp in session.get_inputs():
    print(f"Input:  {inp.name}  shape={inp.shape}  dtype={inp.type}")
for out in session.get_outputs():
    print(f"Output: {out.name}  shape={out.shape}  dtype={out.type}")

# Test inference
dummy = np.random.randn(1, 3, 640, 640).astype(np.float32)
outputs = session.run(None, {session.get_inputs()[0].name: dummy})
print(f"Output shape: {outputs[0].shape}")
# Expected: (1, 4+num_classes, 8400)  for 640x640 input
```

### Output Node Names and Shapes

For a model with `N` custom classes and 640x640 input:
- **Input node**: `images`, shape `[1, 3, 640, 640]`, dtype float32
- **Output node**: `output0`, shape `[1, 4+N, 8400]`, dtype float32
  - `8400 = 80*80 + 40*40 + 20*20` (multi-scale anchors for 640px input)
  - First 4 rows: `[cx, cy, w, h]` in pixel coordinates (not normalized)
  - Remaining N rows: class confidence scores (already sigmoid-activated in YOLOv8)

---

## 4. Running Custom YOLOv8 ONNX in OpenCV

### Full Inference Pipeline

```python
import cv2
import numpy as np

class YOLOv8OpenCV:
    def __init__(self, model_path, class_names, conf_thresh=0.25, iou_thresh=0.45, imgsz=640):
        self.net = cv2.dnn.readNetFromONNX(model_path)
        self.class_names = class_names
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self.num_classes = len(class_names)

        # Optional: use GPU
        # self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def preprocess(self, image):
        """Letterbox resize + normalize."""
        h, w = image.shape[:2]
        scale = self.imgsz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # Pad to imgsz x imgsz
        canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        pad_top  = (self.imgsz - new_h) // 2
        pad_left = (self.imgsz - new_w) // 2
        canvas[pad_top:pad_top+new_h, pad_left:pad_left+new_w] = resized

        blob = cv2.dnn.blobFromImage(
            canvas,
            scalefactor=1.0 / 255.0,
            size=(self.imgsz, self.imgsz),
            mean=(0, 0, 0),
            swapRB=True,        # BGR -> RGB
            crop=False
        )
        return blob, scale, pad_left, pad_top

    def postprocess(self, output, orig_shape, scale, pad_left, pad_top):
        """
        output shape: [1, 4+num_classes, 8400]
        Returns list of (x1, y1, x2, y2, confidence, class_id)
        """
        preds = output[0]            # [4+num_classes, 8400]
        preds = preds.T              # [8400, 4+num_classes]

        boxes_xywh = preds[:, :4]   # cx, cy, w, h in letterboxed-image pixels
        scores     = preds[:, 4:]   # [8400, num_classes]

        class_ids  = np.argmax(scores, axis=1)
        confidences = scores[np.arange(len(scores)), class_ids]

        # Filter by confidence threshold
        mask = confidences >= self.conf_thresh
        boxes_xywh  = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids   = class_ids[mask]

        if len(boxes_xywh) == 0:
            return []

        # Convert cx,cy,w,h -> x1,y1,x2,y2 (still in letterbox space)
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # Map back to original image coordinates
        x1 = (x1 - pad_left) / scale
        y1 = (y1 - pad_top)  / scale
        x2 = (x2 - pad_left) / scale
        y2 = (y2 - pad_top)  / scale

        # Clip to image bounds
        oh, ow = orig_shape[:2]
        x1 = np.clip(x1, 0, ow)
        y1 = np.clip(y1, 0, oh)
        x2 = np.clip(x2, 0, ow)
        y2 = np.clip(y2, 0, oh)

        # NMS per class
        results = []
        for cls in np.unique(class_ids):
            idx = np.where(class_ids == cls)[0]
            cls_boxes = np.stack([x1[idx], y1[idx], x2[idx]-x1[idx], y2[idx]-y1[idx]], axis=1)
            cls_confs = confidences[idx].tolist()
            keep = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_confs, self.conf_thresh, self.iou_thresh)
            if len(keep) > 0:
                for k in keep.flatten():
                    results.append((
                        int(x1[idx[k]]), int(y1[idx[k]]),
                        int(x2[idx[k]]), int(y2[idx[k]]),
                        float(confidences[idx[k]]),
                        int(cls)
                    ))
        return results

    def detect(self, image):
        blob, scale, pad_left, pad_top = self.preprocess(image)
        self.net.setInput(blob)
        output = self.net.forward()[0]   # [4+num_classes, 8400]
        return self.postprocess(output, image.shape, scale, pad_left, pad_top)

    def draw(self, image, detections):
        for (x1, y1, x2, y2, conf, cls_id) in detections:
            label = f"{self.class_names[cls_id]}: {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return image


# Usage
if __name__ == "__main__":
    detector = YOLOv8OpenCV(
        model_path="best.onnx",
        class_names=["cat", "dog", "person"],
        conf_thresh=0.25,
        iou_thresh=0.45
    )
    img = cv2.imread("test.jpg")
    detections = detector.detect(img)
    result = detector.draw(img.copy(), detections)
    cv2.imshow("Detections", result)
    cv2.waitKey(0)
```

---

## 5. Training Haar Cascade (Classical Custom Detector)

Haar Cascades are fast and run entirely on CPU, making them suitable for embedded systems. However, they work best for rigid, frontal objects (faces, eyes) and require thousands of samples.

### Workflow Overview

```
Positive images  -->  opencv_createsamples  -->  .vec file
Negative images  -->  (directly referenced)
                                     |
                                     v
                           opencv_traincascade
                                     |
                                     v
                               cascade.xml
```

### Step 1 — Prepare Samples

```bash
# Directory structure
haar_training/
├── pos/          # ~1000+ positive images (containing the object)
├── neg/          # ~3000+ negative images (no object)
├── pos.txt       # one line per positive image with annotation
└── neg.txt       # one line per negative image (just the path)

# pos.txt format:
# pos/obj001.jpg  1  x y w h
# (number of objects on the line, then bounding box for each)

# neg.txt format:
# neg/bg001.jpg
# neg/bg002.jpg

# Create the vector file from positives
opencv_createsamples \
  -info pos.txt \
  -num 1000 \
  -w 24 \
  -h 24 \
  -vec positives.vec
```

### Step 2 — Train the Cascade

```bash
mkdir -p cascade_output

opencv_traincascade \
  -data cascade_output/ \
  -vec positives.vec \
  -bg neg.txt \
  -numPos 900 \             # slightly fewer than total positives
  -numNeg 1800 \
  -numStages 20 \           # more stages = higher precision, slower training
  -w 24 \
  -h 24 \
  -featureType HAAR \       # or LBP for faster (less accurate) cascades
  -minHitRate 0.999 \       # per-stage recall requirement
  -maxFalseAlarmRate 0.5 \  # per-stage false alarm tolerance
  -mode ALL                 # use upright + 45-degree features
```

**Key parameters:**
- `numStages`: 15-25 typical. More stages = lower false positive rate, slower.
- `minHitRate`: Per-stage recall floor. 0.995-0.999 is typical.
- `maxFalseAlarmRate`: Per-stage rejection ratio. 0.5 means each stage halves false alarms.
- `w` / `h`: Must match what was used in `opencv_createsamples`.

### Step 3 — Use the Trained Cascade

```python
import cv2

cascade = cv2.CascadeClassifier("cascade_output/cascade.xml")

def detect_with_cascade(image, scale_factor=1.1, min_neighbors=5, min_size=(20, 20)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # improves detection in varying light
    detections = cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,   # pyramid scale step (1.05 for thorough, 1.3 for fast)
        minNeighbors=min_neighbors, # higher = fewer false positives
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image, detections
```

---

## 6. Training HOG + SVM (Custom Detector)

HOG (Histogram of Oriented Gradients) + SVM provides a solid baseline for custom detectors, especially when you have limited compute or need a framework-free solution.

### Full Training Pipeline

```python
import cv2
import numpy as np
from glob import glob

# 1. Configure HOG descriptor
win_size    = (64, 128)   # detection window size
block_size  = (16, 16)
block_stride = (8, 8)
cell_size   = (8, 8)
nbins       = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
# Feature vector length: ((win_size/cell_size - block_size/cell_size + 1) blocks) * cells_per_block * nbins
# For (64,128): 7*15 * 4 * 9 = 3780 features

# 2. Extract features from positive and negative samples
def extract_features(image_paths, label):
    features, labels = [], []
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.resize(img, win_size)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        desc = hog.compute(gray)
        features.append(desc.flatten())
        labels.append(label)
    return features, labels

pos_paths = glob("training_data/positive/*.jpg")
neg_paths = glob("training_data/negative/*.jpg")

pos_feat, pos_lbl = extract_features(pos_paths, label=1)
neg_feat, neg_lbl = extract_features(neg_paths, label=0)

X = np.array(pos_feat + neg_feat, dtype=np.float32)
y = np.array(pos_lbl  + neg_lbl,  dtype=np.int32)

# 3. Train SVM
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)    # RBF kernel handles nonlinear boundaries
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-6))

# Auto-tune C and gamma
svm.trainAuto(cv2.ml.TrainData.create(X, cv2.ml.ROW_SAMPLE, y))

# 4. Save and load
svm.save("hog_svm.xml")
# Later:
loaded_svm = cv2.ml.SVM_load("hog_svm.xml")

# 5. Sliding window detector
def sliding_window(image, step=8, win_size=(64, 128)):
    for y in range(0, image.shape[0] - win_size[1], step):
        for x in range(0, image.shape[1] - win_size[0], step):
            yield x, y, image[y:y+win_size[1], x:x+win_size[0]]

def detect_hog_svm(image, svm, scale_factor=1.25, min_scale=0.5, conf_thresh=0.6):
    detections = []
    h, w = image.shape[:2]
    scale = 1.0

    while min(h * scale, w * scale) >= min(win_size):
        scaled = cv2.resize(image, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

        for sx, sy, window in sliding_window(gray):
            if window.shape[:2] != (win_size[1], win_size[0]):
                continue
            feat = hog.compute(window).flatten().reshape(1, -1).astype(np.float32)
            _, result = svm.predict(feat)
            score = result[0, 0]
            if score == 1:
                # Map coordinates back to original scale
                x1 = int(sx / scale)
                y1 = int(sy / scale)
                x2 = int((sx + win_size[0]) / scale)
                y2 = int((sy + win_size[1]) / scale)
                detections.append([x1, y1, x2 - x1, y2 - y1])

        scale *= (1.0 / scale_factor)
        if scale < min_scale:
            break

    # Apply NMS
    if detections:
        weights = [1.0] * len(detections)
        keep = cv2.dnn.NMSBoxes(detections, weights, score_threshold=0.0, nms_threshold=0.4)
        detections = [detections[k] for k in keep.flatten()]

    return detections
```

---

## 7. Evaluation Metrics

### Core Definitions

Given a set of predictions at a fixed confidence threshold:
- **TP** (True Positive): prediction IoU >= threshold with a ground-truth box of the correct class
- **FP** (False Positive): prediction with no matching ground-truth, or IoU < threshold
- **FN** (False Negative): ground-truth box not matched by any prediction

```
Precision = TP / (TP + FP)    [What fraction of predictions are correct?]
Recall    = TP / (TP + FN)    [What fraction of ground-truths are found?]
F1        = 2 * P * R / (P + R)
```

### IoU (Intersection over Union)

```python
def compute_iou(box1, box2):
    """
    box format: [x1, y1, x2, y2]
    Returns IoU in [0, 1].
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0
```

### Average Precision (AP)

AP is the area under the Precision-Recall curve. PASCAL VOC uses 11-point interpolation; COCO uses all-point interpolation.

```python
import numpy as np

def compute_ap(recalls, precisions):
    """
    Compute AP using all-point interpolation (COCO style).
    recalls, precisions: arrays sorted by recall ascending.
    """
    # Append sentinel values
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Find steps where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_pr_curve(pred_boxes, pred_scores, gt_boxes, iou_thresh=0.5):
    """
    Compute precision-recall curve for a single class.

    pred_boxes:  list of [x1, y1, x2, y2]
    pred_scores: list of confidence scores
    gt_boxes:    list of [x1, y1, x2, y2]
    """
    # Sort by descending confidence
    sorted_idx = np.argsort(pred_scores)[::-1]
    pred_boxes  = [pred_boxes[i]  for i in sorted_idx]
    pred_scores = [pred_scores[i] for i in sorted_idx]

    matched_gt = set()
    tp_list, fp_list = [], []

    for pb in pred_boxes:
        best_iou, best_gt = 0.0, -1
        for gi, gb in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_gt = iou, gi

        if best_iou >= iou_thresh:
            tp_list.append(1)
            fp_list.append(0)
            matched_gt.add(best_gt)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    n_gt   = len(gt_boxes)

    recalls    = tp_cum / n_gt if n_gt > 0 else np.zeros_like(tp_cum)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    return recalls, precisions, compute_ap(recalls, precisions)
```

### mAP@50 and mAP@50-95

```python
def compute_map50(predictions_per_class, gt_per_class):
    """
    predictions_per_class: dict {class_name: (boxes, scores)}
    gt_per_class:          dict {class_name: boxes}
    Returns mAP@50.
    """
    aps = []
    for cls in gt_per_class:
        pred_b, pred_s = predictions_per_class.get(cls, ([], []))
        gt_b = gt_per_class[cls]
        _, _, ap = compute_pr_curve(pred_b, pred_s, gt_b, iou_thresh=0.5)
        aps.append(ap)
    return np.mean(aps)

def compute_map50_95(predictions_per_class, gt_per_class):
    """
    Computes mAP averaged over IoU thresholds 0.5, 0.55, ..., 0.95.
    """
    thresholds = np.arange(0.5, 1.0, 0.05)
    all_maps = []
    for iou_t in thresholds:
        aps = []
        for cls in gt_per_class:
            pred_b, pred_s = predictions_per_class.get(cls, ([], []))
            gt_b = gt_per_class[cls]
            _, _, ap = compute_pr_curve(pred_b, pred_s, gt_b, iou_thresh=iou_t)
            aps.append(ap)
        all_maps.append(np.mean(aps))
    return np.mean(all_maps)
```

### Interpreting mAP Values

| mAP@50-95 | Interpretation |
|-----------|---------------|
| > 0.60 | Strong — production-ready for most tasks |
| 0.40–0.60 | Reasonable — may need more data or tuning |
| 0.25–0.40 | Marginal — significant room for improvement |
| < 0.25 | Poor — revisit data quality or model choice |

> Note: COCO benchmark models reach ~50-55 mAP@50-95 across 80 categories. Single-class custom detectors on clean data regularly exceed 0.70.

---

## Quick Reference: Method Comparison

| Method | Speed | Accuracy | Training Data Needed | OpenCV Load |
|--------|-------|----------|---------------------|-------------|
| YOLOv8n ONNX | Fast (GPU) | High | ~500+ images/class | `readNetFromONNX` |
| Haar Cascade | Very fast | Moderate | 1000+ pos + 3000+ neg | `CascadeClassifier` |
| HOG + SVM | Fast (CPU) | Moderate | 200+ pos + 500+ neg | `ml.SVM_load` |

---

*Guide covers OpenCV 4.x / 5.x, Ultralytics YOLOv8, Python 3.8+.*
