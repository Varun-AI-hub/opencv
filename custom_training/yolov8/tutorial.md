# YOLOv8 Custom Training: Complete End-to-End Tutorial

This tutorial walks you through every stage of training a custom YOLOv8 object detector — from environment setup and dataset creation, through model training, export, and inference with OpenCV DNN. All scripts in this directory are fully runnable.

---

## Stage 0: Environment Setup

```bash
pip install ultralytics opencv-python-headless numpy Pillow onnxruntime
pip install pyyaml matplotlib
pip install roboflow  # optional — for dataset management via Roboflow
```

**Dependency breakdown:**

| Package | Role |
|---|---|
| `ultralytics` | Official YOLOv8 package — training, validation, export |
| `opencv-python-headless` | Image I/O and preprocessing for inference (no GUI); swap for `opencv-python` if you need `imshow` |
| `numpy` | Array math throughout the pipeline |
| `Pillow` | PIL Image support used internally by ultralytics |
| `onnxruntime` | Run ONNX models without PyTorch after export |
| `pyyaml` | Read/write `data.yaml` dataset configs |
| `matplotlib` | Plot loss curves and PR curves |
| `roboflow` | Upload images, annotate online, export in YOLO format |

Verify the install:

```python
import ultralytics
ultralytics.checks()   # prints CUDA availability, versions
```

---

## Stage 1: Synthetic Dataset Generation (for learning the pipeline)

### Why synthetic data?

When learning a new ML pipeline, debugging real data simultaneously adds unnecessary noise. Synthetic data lets you:
- Control class balance perfectly.
- Instantly verify that labels are correct (you generated them).
- Iterate in seconds rather than hours of annotation.
- Prove the pipeline works before spending annotation effort.

The script `01_generate_dataset.py` creates:
- **500 training images** + YOLO labels
- **100 validation images** + YOLO labels
- **3 classes**: circles (0), rectangles (1), triangles (2)
- Images are 128×128 PNG with random backgrounds
- Labels are in YOLO format: `class_id cx cy w h` (all values normalized 0–1)
- A `data.yaml` config file ready for ultralytics

Run it:

```bash
python 01_generate_dataset.py
# Creates: synthetic_dataset/{images,labels}/{train,val}/
```

### YOLO label format explained

Each image has a `.txt` file with the same stem in the matching `labels/` folder. One line per object:

```
class_id  cx  cy  w  h
```

- `class_id` — integer index (0-based, matching `data.yaml` order)
- `cx`, `cy` — bounding-box center, **normalized** by image width/height (0.0–1.0)
- `w`, `h` — bounding-box width and height, **normalized** (0.0–1.0)

Example for a circle at pixel (64, 48) with diameter 20 in a 128×128 image:

```
0  0.5000  0.3750  0.1563  0.1563
```

Empty images (no objects) have an empty `.txt` file.

### data.yaml format

```yaml
path: /absolute/path/to/synthetic_dataset  # root directory
train: images/train                         # relative to path
val:   images/val
nc: 3                                       # number of classes
names: ['circle', 'rectangle', 'triangle'] # class names in order
```

---

## Stage 2: Real Dataset Annotation

### Option A — LabelImg (local, free)

```bash
pip install labelImg
labelImg
```

1. Open Image Dir → select your images folder.
2. Change Save Dir → select matching labels folder.
3. **File → Change Save Format → YOLO** (critical — default is Pascal VOC).
4. Draw boxes with `W` key, select class, save with `Ctrl+S`.
5. Navigate with `A`/`D`.

LabelImg writes one `.txt` per image in YOLO format automatically.

### Option B — CVAT (online, team-friendly)

1. Go to [cvat.ai](https://cvat.ai) and create a project.
2. Upload images, define labels.
3. Annotate with bounding boxes.
4. Export → **YOLO 1.1** format → download zip.

### Option C — Roboflow (recommended for beginners)

```bash
pip install roboflow
```

1. Create a workspace at [roboflow.com](https://roboflow.com).
2. Upload images → annotate in the browser.
3. Apply augmentations (optional at this stage).
4. Export → **YOLOv8** → get Python snippet:

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_KEY")
project = rf.workspace("workspace").project("project")
dataset = project.version(1).download("yolov8")
# dataset.location contains data.yaml and image folders
```

---

## Stage 3: YOLOv8 Model Selection

| Model | Params | GFLOPs | mAP val50-95 | Speed CPU (ms) |
|---|---|---|---|---|
| yolov8n | 3.2M | 8.7 | 37.3 | 80 |
| yolov8s | 11.2M | 28.6 | 44.9 | 128 |
| yolov8m | 25.9M | 78.9 | 50.2 | 234 |
| yolov8l | 43.7M | 165.2 | 52.9 | 375 |
| yolov8x | 68.2M | 257.8 | 53.9 | 479 |

**Rule of thumb — how to choose:**

| Situation | Recommendation |
|---|---|
| Learning / prototyping | `yolov8n` — fastest iteration |
| Edge device (Raspberry Pi, Jetson Nano) | `yolov8n` or `yolov8s` |
| Server CPU inference | `yolov8s` or `yolov8m` |
| GPU inference, accuracy matters | `yolov8m` or `yolov8l` |
| State-of-the-art accuracy, GPU only | `yolov8x` |
| < 500 training images | `yolov8n` — less prone to overfitting |

Always start with `n`. Only move up if validation mAP plateaus and you have sufficient data (roughly 1000+ images per class for larger models).

---

## Stage 4: Training

### Quickstart

```bash
python 02_train_yolov8.py --data synthetic_dataset/data.yaml --model yolov8n --epochs 50
```

### Every training argument explained

| Argument | Default | Effect |
|---|---|---|
| `epochs` | 100 | Total training iterations over the dataset. Use 100–300 for small datasets (<5K images). More epochs = more overfitting risk without early stopping. |
| `imgsz` | 640 | Square input resolution in pixels. Use 416 for speed, 640 for standard accuracy, 1280 for small objects. Larger = slower, more memory. |
| `batch` | 16 | Images per gradient step. `-1` = auto (ultralytics picks the largest that fits in GPU memory). Larger batch = more stable gradients, needs more memory. |
| `patience` | 50 | Early stopping: stop if val mAP does not improve for this many epochs. Set `0` to disable. |
| `lr0` | 0.01 | Initial learning rate. The optimizer starts here after warmup. |
| `lrf` | 0.01 | Final learning rate as a fraction of `lr0` (cosine schedule). Final LR = `lr0 * lrf`. |
| `warmup_epochs` | 3.0 | Linear warmup: LR ramps from 0 to `lr0` over this many epochs. Prevents large gradient updates at the start. |
| `mosaic` | 1.0 | Probability (0–1) of mosaic augmentation (4 images tiled). Greatly improves generalization. Set to 0 for the last 10 epochs automatically. |
| `mixup` | 0.0 | Alpha for mixup augmentation (blends two images). Helps with classification; less effect on detection. |
| `copy_paste` | 0.0 | Segment copy-paste probability (only meaningful for segmentation tasks). |
| `degrees` | 0.0 | Random rotation range ±degrees. |
| `translate` | 0.1 | Random translate ±fraction of image size. |
| `scale` | 0.5 | Random scale ±fraction. |
| `shear` | 0.0 | Random shear ±degrees. |
| `flipud` | 0.0 | Vertical flip probability. Use for aerial/satellite images. |
| `fliplr` | 0.5 | Horizontal flip probability. Disable if left/right matters (e.g., text). |
| `hsv_h` | 0.015 | Hue jitter fraction. |
| `hsv_s` | 0.7 | Saturation jitter fraction. |
| `hsv_v` | 0.4 | Value (brightness) jitter fraction. |
| `pretrained` | True | Load COCO-pretrained weights. **Almost always True** — dramatically reduces epochs needed. |
| `device` | '' (auto) | `'0'` for first GPU, `'0,1'` for multi-GPU, `'cpu'` to force CPU. |
| `workers` | 8 | DataLoader worker threads. Set to 0 on Windows if you get multiprocessing errors. |
| `project` | 'runs/detect' | Output root directory. |
| `name` | 'train' | Subdirectory name under `project`. |
| `exist_ok` | False | Overwrite existing run directory instead of auto-incrementing. |
| `optimizer` | 'auto' | Optimizer: `'SGD'`, `'Adam'`, `'AdamW'`, `'auto'`. |
| `seed` | 0 | Random seed for reproducibility. |
| `amp` | True | Automatic mixed precision (FP16). Speeds up GPU training significantly. |
| `val` | True | Run validation after each epoch. |
| `plots` | True | Save training plots (loss curves, PR curves, confusion matrix). |

### Recommended settings for small custom datasets (<2K images)

```python
model.train(
    data="data.yaml",
    epochs=150,
    imgsz=640,
    batch=16,
    patience=30,
    lr0=0.01,
    lrf=0.01,
    warmup_epochs=5,
    mosaic=1.0,
    flipud=0.0,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    pretrained=True,
    plots=True,
)
```

---

## Stage 5: Understanding Training Output

After training, results are saved to `runs/detect/train/` (or your custom `project/name`):

```
runs/detect/train/
├── weights/
│   ├── best.pt       <- best checkpoint (by val mAP50-95)
│   └── last.pt       <- final epoch checkpoint
├── results.csv       <- per-epoch metrics
├── confusion_matrix.png
├── confusion_matrix_normalized.png
├── PR_curve.png
├── F1_curve.png
├── P_curve.png
├── R_curve.png
├── labels.jpg        <- visualization of training label distribution
├── labels_correlogram.jpg
└── train_batchX.jpg  <- sample augmented training batch
```

### results.csv columns

| Column | Meaning |
|---|---|
| `train/box_loss` | Regression loss (CIoU) on training set |
| `train/cls_loss` | Classification loss on training set |
| `train/dfl_loss` | Distribution Focal Loss (localization) on training set |
| `metrics/precision(B)` | Precision at default confidence threshold on val |
| `metrics/recall(B)` | Recall at default confidence threshold on val |
| `metrics/mAP50(B)` | mAP @ IoU=0.50 on val |
| `metrics/mAP50-95(B)` | mAP averaged over IoU=0.50:0.95 on val |
| `val/box_loss` | Box loss on validation set |
| `val/cls_loss` | Classification loss on validation set |
| `val/dfl_loss` | DFL loss on validation set |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | Learning rates for each parameter group |

### Diagnosing training health

| Pattern | Diagnosis | Fix |
|---|---|---|
| Val loss tracks train loss, both decrease | Healthy training | Keep going |
| Train loss drops, val loss rises after epoch N | Overfitting starts at epoch N | Enable early stopping, add augmentation, reduce model size |
| Both losses plateau high from the start | Underfitting | More epochs, larger model, check labels |
| Val mAP oscillates wildly | LR too high | Reduce `lr0` by 10x |
| Loss goes to NaN | LR way too high or bad data | Check for empty/corrupt images, reduce `lr0` |

---

## Stage 6: ONNX Export and Optimization

### Why export to ONNX?

- Deploy without PyTorch installed.
- Use OpenCV DNN backend for C++ or Python inference.
- Compatible with ONNX Runtime, TensorRT, OpenVINO, CoreML pipelines.
- Smaller deployment artifact.

### Export

```python
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
# Writes: runs/detect/train/weights/best.onnx
```

### Parameter explanations

| Parameter | Value | Reason |
|---|---|---|
| `opset` | 12 | ONNX opset version. OpenCV DNN supports up to opset 13; use 12 for maximum compatibility with older OpenCV builds. |
| `simplify` | True | Runs onnx-simplifier to fold constants, remove redundant reshape ops, and simplify the graph. Reduces model size and speeds up inference. |
| `dynamic` | False | Static input shape `[1, 3, 640, 640]`. Dynamic shapes add complexity; use False unless you need variable batch sizes. |
| `imgsz` | 640 | Bake this resolution into the model graph. Must match the resolution you use at inference time. |

### Verify with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("best.onnx")
dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
outputs = sess.run(None, {sess.get_inputs()[0].name: dummy})
print(outputs[0].shape)  # expect: (1, 4+num_classes, 8400)
```

---

## Stage 7: Inference with OpenCV DNN

The script `03_inference_opencv.py` implements the complete pipeline. Here is a step-by-step explanation.

### 1. Load model

```python
net = cv2.dnn.readNetFromONNX("best.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
```

### 2. Preprocess with blobFromImage

```python
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1/255.0,  # normalize pixel values to [0, 1]
    size=(640, 640),       # resize to model input size
    mean=(0, 0, 0),        # no mean subtraction (YOLO normalizes differently)
    swapRB=True,           # OpenCV loads BGR; YOLO expects RGB — swap channels
    crop=False,            # resize without cropping (letterboxing handled separately)
)
net.setInput(blob)
```

**swapRB=True is critical.** If omitted, the color channels are reversed and the model sees a completely different image — boxes will be wrong or absent.

### 3. Forward pass

```python
outputs = net.forward()   # shape: (1, 4 + num_classes, 8400)
```

### 4. Understanding the output tensor

YOLOv8 uses an anchor-free head with a single output tensor:

```
shape: [1, 4 + num_classes, 8400]
         ^  ^                ^
         |  |                8400 = sum of grid cells across 3 scales
         |  4 box params (cx, cy, w, h) + class probabilities
         batch size
```

The 8400 predictions come from three detection scales:
- 80×80 grid = 6400 cells (small objects)
- 40×40 grid = 1600 cells (medium objects)
- 20×20 grid =  400 cells (large objects)
- Total = 8400

### 5. Decode predictions

```python
# Transpose to [8400, 4 + num_classes] for easier indexing
preds = outputs[0].T   # shape: (8400, 4 + num_classes)

boxes   = preds[:, :4]          # cx, cy, w, h in input-image pixel space
scores  = preds[:, 4:]          # class probabilities, shape (8400, num_classes)

conf    = scores.max(axis=1)    # best class confidence per prediction
class_ids = scores.argmax(axis=1)

# Filter by confidence threshold
mask    = conf > conf_threshold
boxes   = boxes[mask]
conf    = conf[mask]
class_ids = class_ids[mask]
```

### 6. Convert box format and scale to original image

The cx, cy, w, h values are in the **resized input image** coordinate space (e.g., 640×640). Scale back to original:

```python
scale_x = orig_w / input_w   # e.g., 1280 / 640 = 2.0
scale_y = orig_h / input_h

x1 = (boxes[:, 0] - boxes[:, 2] / 2) * scale_x   # left
y1 = (boxes[:, 1] - boxes[:, 3] / 2) * scale_y   # top
x2 = (boxes[:, 0] + boxes[:, 2] / 2) * scale_x   # right
y2 = (boxes[:, 1] + boxes[:, 3] / 2) * scale_y   # bottom
```

### 7. Non-Maximum Suppression (NMS)

```python
indices = cv2.dnn.NMSBoxes(
    bboxes=xyxy_boxes.tolist(),
    scores=conf.tolist(),
    score_threshold=conf_threshold,
    nms_threshold=nms_threshold,   # IoU threshold, typically 0.45
)
```

NMS removes duplicate detections of the same object by keeping only the highest-confidence box when two boxes overlap more than `nms_threshold`.

### 8. Draw results

```python
for i in indices.flatten():
    x1, y1, x2, y2 = xyxy_boxes[i].astype(int)
    label = f"{class_names[class_ids[i]]}: {conf[i]:.2f}"
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
```

---

## Stage 8: Hyperparameter Tuning

### Learning rate

- Default `lr0=0.01` works well with SGD and pretrained COCO weights.
- If loss oscillates: halve `lr0`.
- If loss barely moves after 20 epochs: double `lr0` (rare with pretrained weights).
- `lrf` controls the cosine decay endpoint: `lr_final = lr0 * lrf`. Default `0.01` gives 100× decay.

### Image size (`imgsz`)

| Object size relative to image | Recommended imgsz |
|---|---|
| Large (>30% of image) | 320–416 (faster) |
| Medium (5–30%) | 640 (standard) |
| Small (<5%) | 1280 (necessary for small objects) |

Doubling `imgsz` quadruples the number of grid cells, so small objects are more likely to land in a grid cell. It also quadruples GPU memory usage.

### Batch size and batch norm

Batch normalization statistics (running mean/variance) are computed per batch. Very small batches (batch=4 or less) produce noisy BN statistics and degrade performance. Recommended minimum: batch=8. For fine-tuning with large models, batch=4 with gradient accumulation (`accumulate`) is a reasonable workaround.

### Anchor-free detection (YOLOv8)

YOLOv8 is anchor-free: it predicts the bounding box directly as `(cx, cy, w, h)` offsets using Distribution Focal Loss (DFL) rather than predicting offsets from predefined anchor boxes. This means:
- No anchor tuning required.
- The model generalizes better to unusual aspect ratios.
- Simpler post-processing.

### Confidence and NMS thresholds for deployment

| Use case | Confidence threshold | NMS IoU threshold |
|---|---|---|
| High recall (miss nothing) | 0.1–0.25 | 0.45 |
| Balanced (default) | 0.25 | 0.45 |
| High precision (few false positives) | 0.5–0.7 | 0.5 |
| Counting objects (dense scenes) | 0.25 | 0.3 |

---

## Stage 9: Common Issues and Solutions

| Issue | Symptom | Root Cause | Solution |
|---|---|---|---|
| Overfitting | Train loss low, val loss rising | Too little data or too large a model | Add augmentation (increase `mosaic`, `hsv_s`), use smaller model, collect more data |
| Underfitting | Both losses plateau above baseline | Model too small or insufficient epochs | Use larger model variant, train longer, check labels |
| Class imbalance | mAP good on common class, poor on rare class | Frequency mismatch | Oversample rare classes, use `cls_pw` weight, collect more rare-class images |
| Wrong labels | Loss does not decrease at all; `train_batch.jpg` shows misaligned boxes | Annotation error | Open a few images with `plot=True` (ultralytics), verify `.txt` files exist and are non-empty |
| ONNX output all zeros | Inference returns no boxes or all-zero confidences | Missing `swapRB=True` or wrong normalization | Set `swapRB=True` in `blobFromImage`, ensure `scalefactor=1/255.0` |
| Memory OOM during training | CUDA out of memory | Batch too large for GPU VRAM | Reduce `batch` or use `batch=-1` (auto) |
| `workers` multiprocessing error | Freezes or crashes on Windows | Python multiprocessing limitation | Set `workers=0` |
| mAP stuck at 0 | Validation shows 0 mAP for all epochs | data.yaml path wrong or labels missing | Print `dataset.yaml` resolved paths, check that `labels/` mirrors `images/` structure |
| Model trains but ONNX export fails | onnxruntime version mismatch | Unsupported ops in installed opset | Try `opset=11`, update onnxruntime: `pip install -U onnxruntime` |

---

## Quick Reference: Full Pipeline Commands

```bash
# 1. Install dependencies
pip install ultralytics opencv-python-headless numpy Pillow onnxruntime pyyaml matplotlib

# 2. Generate synthetic dataset
python 01_generate_dataset.py

# 3. Train YOLOv8n for 50 epochs
python 02_train_yolov8.py --data synthetic_dataset/data.yaml --model yolov8n --epochs 50

# 4. Run inference on an image
python 03_inference_opencv.py \
    --model runs/detect/train/weights/best.onnx \
    --source path/to/image.jpg

# 5. Run inference on webcam
python 03_inference_opencv.py \
    --model runs/detect/train/weights/best.onnx \
    --source 0

# 6. Evaluate on validation set
python 04_evaluate.py \
    --model runs/detect/train/weights/best.onnx \
    --data synthetic_dataset/data.yaml
```
