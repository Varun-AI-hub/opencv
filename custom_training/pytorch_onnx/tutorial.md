# PyTorch -> ONNX -> OpenCV: Complete Pipeline Tutorial

## Overview

This tutorial covers the complete workflow for training a custom object detector in PyTorch,
exporting it to ONNX format, and deploying it using OpenCV's DNN module.

Training flow: PyTorch model -> validate -> export ONNX -> verify -> OpenCV DNN inference.

Scripts in this directory:
- `01_synthetic_detection_dataset.py` — generate training data
- `02_pytorch_detector.py` — define and train the model
- `03_export_to_onnx.py` — export checkpoint to ONNX
- `04_opencv_inference.py` — run inference with OpenCV DNN

---

## Part 1: PyTorch Model Architecture

### 1.1 Simple CNN Detector (for learning)

A lightweight classification-then-localization network:

- **Input**: 224x224x3
- **Backbone**: 4 conv blocks (Conv2d + BatchNorm + ReLU + MaxPool2d)
- **Detection head**: two branches
  - Classification: Linear(512, num_classes) + Softmax
  - Regression: Linear(512, 4) for bounding box (cx, cy, w, h)
- **Loss**: CrossEntropyLoss + SmoothL1Loss

**Why separate heads?**
Multi-task learning lets one backbone learn shared visual features while two independent heads
specialise on different tasks. Classification needs features that are invariant to position;
regression needs spatially precise features. Separate heads also let you tune each loss weight
independently (alpha * cls_loss + beta * reg_loss) without one task dominating the other.

**Balancing loss weights**: typical starting point is alpha=1.0, beta=1.0, then monitor both
loss terms — if one is orders of magnitude larger it will dominate gradients and suppress the other.

### 1.2 Anchor-Based Detection Head

Adding an anchor-based detection head to any backbone:

**Grid cells at multiple scales**
Feature maps at strides 8, 16, 32 produce grids of size H/8 x W/8, H/16 x W/16, H/32 x W/32.
Placing anchors at each grid cell gives multi-scale coverage with no extra computation.

**Per-cell predictions**
For each anchor at each grid cell: predict `[objectness (1), class probs (C), box delta (4)]`
giving a tensor of shape `[B, A*(1+C+4), H, W]` where A is anchors-per-cell.

**Anchor assignment**
For each ground-truth box, compute IoU with every anchor. Assign the ground-truth to the
anchor with the highest IoU (and optionally all anchors above an IoU threshold). Unmatched
anchors are background for objectness loss. This separates the detection problem into:
"is something here?" (objectness) and "which class and where exactly?" (cls + reg).

**Loss breakdown**:
- BCE loss on objectness (positive anchors = 1, background = 0)
- Focal loss on classification (positive anchors only)
- Smooth-L1 on box deltas (positive anchors only)

### 1.3 Using torchvision Faster R-CNN

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Load COCO-pretrained model
model = fasterrcnn_resnet50_fpn(weights='COCO_V1')

# Replace head for custom num_classes (including background class)
in_features = model.roi_heads.box_predictor.cls_score.in_features
num_classes = 4  # background + 3 custom classes
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Fine-tuning: freeze backbone, train head first
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = torch.optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.005, momentum=0.9, weight_decay=0.0005
)

# After a few epochs, unfreeze backbone for full fine-tuning
for param in model.backbone.parameters():
    param.requires_grad = True
```

**Custom dataset for torchvision Faster R-CNN**:
The model expects a list of dicts with keys `image` (FloatTensor C,H,W in [0,1]) and
`target` dict containing `boxes` (FloatTensor N,4 in x1,y1,x2,y2) and `labels` (Int64Tensor N).

```python
class CocoDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.imgs[img_id]
        img = Image.open(os.path.join(self.img_dir, img_info['file_name'])).convert('RGB')
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = torch.tensor([[a['bbox'][0], a['bbox'][1],
                               a['bbox'][0]+a['bbox'][2],
                               a['bbox'][1]+a['bbox'][3]] for a in anns], dtype=torch.float32)
        labels = torch.tensor([a['category_id'] for a in anns], dtype=torch.int64)
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([img_id])}
        img = transforms.ToTensor()(img)
        if self.transforms:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)
```

---

## Part 2: Custom Dataset in PyTorch

### Dataset Class

Full implementation pattern for a PyTorch Dataset supporting both COCO JSON and YOLO txt formats:

```python
class DetectionDataset(Dataset):
    def __init__(self, img_dir, ann_path, fmt='coco', transforms=None):
        """
        img_dir:  directory containing image files
        ann_path: COCO JSON file path or directory of YOLO txt files
        fmt:      'coco' or 'yolo'
        """
        self.img_dir = img_dir
        self.transforms = transforms
        if fmt == 'coco':
            self._load_coco(ann_path)
        else:
            self._load_yolo(ann_path)

    def _load_coco(self, ann_file):
        with open(ann_file) as f:
            data = json.load(f)
        self.images = {img['id']: img for img in data['images']}
        self.img_ids = [img['id'] for img in data['images']]
        self.ann_by_img = defaultdict(list)
        for ann in data['annotations']:
            self.ann_by_img[ann['image_id']].append(ann)

    def _load_yolo(self, lbl_dir):
        self.img_ids = []
        self.ann_by_img = {}
        for lbl_file in sorted(Path(lbl_dir).glob('*.txt')):
            img_id = lbl_file.stem
            self.img_ids.append(img_id)
            self.ann_by_img[img_id] = self._parse_yolo(lbl_file)

    def _parse_yolo(self, path):
        boxes = []
        with open(path) as f:
            for line in f:
                cls, cx, cy, w, h = map(float, line.strip().split())
                boxes.append({'category_id': int(cls), 'cx': cx, 'cy': cy, 'w': w, 'h': h})
        return boxes

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        # Load image
        if isinstance(img_id, int):
            fname = self.images[img_id]['file_name']
        else:
            fname = img_id + '.jpg'
        img = cv2.imread(os.path.join(self.img_dir, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # Parse annotations
        anns = self.ann_by_img[img_id]
        boxes, labels = [], []
        for ann in anns:
            if 'bbox' in ann:  # COCO format: x, y, width, height
                x, y, bw, bh = ann['bbox']
                boxes.append([x, y, x + bw, y + bh])
            else:              # YOLO format: cx cy w h (normalised)
                cx, cy, bw, bh = ann['cx']*w, ann['cy']*h, ann['w']*w, ann['h']*h
                boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
            labels.append(ann['category_id'])
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        if self.transforms:
            transformed = self.transforms(image=img, bboxes=boxes.tolist(), labels=labels.tolist())
            img = transformed['image']
            boxes = np.array(transformed['bboxes'], dtype=np.float32)
            labels = np.array(transformed['labels'], dtype=np.int64)
        return img, {'boxes': torch.from_numpy(boxes), 'labels': torch.from_numpy(labels)}

    def __len__(self):
        return len(self.img_ids)
```

**Collate function** for variable-size targets (DataLoader default collate cannot stack dicts
with varying-length tensors):

```python
def detection_collate(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, dim=0)
    return images, targets  # targets remain a list of dicts

loader = DataLoader(dataset, batch_size=8, collate_fn=detection_collate)
```

### Data Augmentation with Albumentations

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.GaussianBlur(blur_limit=(3, 7), p=0.1),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
], bbox_params=A.BboxParams(
    format='pascal_voc',        # expects x1, y1, x2, y2 absolute pixels
    label_fields=['labels'],    # parallel list of class labels
    min_visibility=0.3,         # drop boxes that become mostly occluded
))
```

**Why `bbox_params` is required**: standard image augmentations (flip, rotation, crop) spatially
transform the image. Without `bbox_params`, the boxes stay in their original positions and no
longer match the transformed image. Albumentations uses `bbox_params` to apply the same spatial
transform to all bounding boxes in lockstep with the image, and to filter out boxes that fall
outside the new image boundary.

---

## Part 3: Training Loop

### 3.1 Optimizer and Scheduler

**Adam vs SGD**
- Adam: fast convergence, works well out-of-the-box, higher memory usage (stores m, v per param).
  Good for quick experiments and transformer-based backbones.
- SGD with momentum: slower to converge but often generalises better; industry standard for
  training from scratch on image tasks. Use `momentum=0.9, weight_decay=1e-4`.

**Learning rate schedules**
- `OneCycleLR`: starts low, ramps to max_lr over ~30% of training, then decays.
  Very effective for fine-tuning; prevents early divergence.
- `CosineAnnealingLR`: smooth cosine decay from initial lr to eta_min. Good default.
- `StepLR`: multiply lr by gamma every N epochs. Simple and predictable.

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3,
    steps_per_epoch=len(train_loader),
    epochs=num_epochs,
    pct_start=0.3,
)
```

**Gradient clipping**: prevents exploding gradients especially in early training.

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Linear warm-up**: for the first `warmup_epochs` epochs, scale lr linearly from 0 to the
initial lr. Avoids large gradient steps when weights are randomly initialised.

```python
def warmup_lambda(epoch):
    if epoch < warmup_epochs:
        return float(epoch + 1) / warmup_epochs
    return 1.0
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda)
```

### 3.2 Loss Functions for Detection

**Focal Loss** (for class imbalance — many background anchors, few foreground)

FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where gamma=2 is typical. The modulating factor (1-p_t)^gamma down-weights easy examples
(high confidence, p_t close to 1) so the model focuses on hard, misclassified examples.

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        ce = F.cross_entropy(preds, targets, reduction='none')
        p_t = torch.exp(-ce)
        focal_weight = self.alpha * (1 - p_t) ** self.gamma
        return (focal_weight * ce).mean()
```

**Smooth L1 Loss** (less sensitive to outliers than MSE; used in box regression)

```
L_delta(x) = 0.5 * x^2           if |x| < delta
           = delta*|x| - 0.5*delta^2  otherwise
```

PyTorch: `nn.SmoothL1Loss(beta=1.0)` (beta here equals delta in the formula above).

**IoU-based losses**: GIoU, DIoU, CIoU improve on Smooth-L1 for boxes that do not overlap
(where Smooth-L1 gradient is constant regardless of how far apart the boxes are).
CIoU additionally penalises aspect ratio difference, which speeds box convergence.

### 3.3 Full Training Loop

```python
best_val_loss = float('inf')

for epoch in range(num_epochs):
    # --- Training ---
    model.train()
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        cls_out, box_out = model(images)
        cls_loss = criterion_cls(cls_out, ...)
        box_loss = criterion_box(box_out, ...)
        loss = cls_loss + box_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # if using OneCycleLR (per-step)
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            cls_out, box_out = model(images)
            val_loss += (criterion_cls(cls_out, ...) + criterion_box(box_out, ...)).item()
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({'epoch': epoch, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'val_loss': val_loss},
                   'best.pth')

    if epoch_scheduler:  # StepLR / CosineAnnealingLR (per-epoch)
        epoch_scheduler.step()
```

### 3.4 Mixed Precision Training

Mixed precision (FP16 compute, FP32 master weights) roughly doubles throughput on Ampere+ GPUs
and halves GPU memory usage with no accuracy loss in practice.

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for images, targets in train_loader:
    optimizer.zero_grad()
    with autocast():                        # FP16 forward pass
        cls_out, box_out = model(images)
        loss = cls_loss + box_loss

    scaler.scale(loss).backward()           # scale gradients to avoid FP16 underflow
    scaler.unscale_(optimizer)              # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## Part 4: ONNX Export

### 4.1 What ONNX Is

ONNX (Open Neural Network Exchange) is a standardised computation graph format.
- **Nodes** are operators (Conv, BatchNorm, ReLU, Gemm, ...) with over 150 standard ops.
- **Edges** are named tensors carrying data between nodes.
- **Opset version** governs which operators are available. OpenCV DNN supports up to opset 13
  (as of OpenCV 4.x); opset 12 is the safest choice for broad runtime compatibility.

The ONNX graph is portable: export once, run on ONNX Runtime, OpenCV DNN, TensorRT, CoreML, etc.

### 4.2 Export with torch.onnx.export

```python
import torch
import torch.onnx

model.eval()  # IMPORTANT: set eval mode before export (affects BN, Dropout)

dummy_input = torch.randn(1, 3, 224, 224)  # must match training input shape

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=12,           # OpenCV DNN supports up to 13; 12 is safest
    input_names=["input"],
    output_names=["boxes", "scores", "labels"],
    dynamic_axes={              # omit for static shapes (required by some runtimes)
        "input": {0: "batch_size"},
    },
    export_params=True,         # include trained weights in the file
    do_constant_folding=True,   # fold constant subexpressions at export time
    verbose=False,
)
print("Exported model.onnx")
```

**Key flags explained**:
- `do_constant_folding=True` pre-computes subgraphs with no data-dependent outputs (e.g. shape
  computations), reducing runtime node count.
- `dynamic_axes` lets you mark specific dimensions as variable-length. Use only if you actually
  need variable batch/spatial sizes — dynamic shapes can complicate some runtimes.
- Call `model.eval()` first: BatchNorm and Dropout behave differently in train vs eval mode and
  the export captures the current mode.

### 4.3 ONNX Model Simplification

```bash
pip install onnxsim
python -m onnxsim model.onnx model_simplified.onnx
```

`onnxsim` (onnx-simplifier) performs:
- Constant node folding beyond what PyTorch's exporter does
- Removal of identity operators and no-op reshapes
- Shape inference across the graph
- Removal of unused initializers

The simplified model has fewer nodes, is faster to load, and is more likely to be compatible with
deployment runtimes like OpenCV DNN.

### 4.4 Verifying the Export

Always verify that the ONNX model produces numerically identical outputs to the PyTorch model
before deploying.

```python
import numpy as np
import onnxruntime as ort

# PyTorch reference output
model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    pt_boxes, pt_scores = model(dummy)

# ONNX Runtime output
sess = ort.InferenceSession("model_simplified.onnx",
                            providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
print("ONNX input name:", input_name)
print("ONNX input shape:", sess.get_inputs()[0].shape)
for out in sess.get_outputs():
    print("ONNX output:", out.name, out.shape)

ort_boxes, ort_scores = sess.run(None, {input_name: dummy.numpy()})

np.testing.assert_allclose(pt_boxes.numpy(), ort_boxes, rtol=1e-4, atol=1e-5)
np.testing.assert_allclose(pt_scores.numpy(), ort_scores, rtol=1e-4, atol=1e-5)
print("Outputs match. Export verified.")
```

### 4.5 OpenCV DNN Compatibility Checklist

| Issue | Cause | Fix |
|---|---|---|
| Unsupported op error | opset > 13 or custom op | Use opset_version=12; rewrite custom ops as standard ONNX ops |
| Wrong output shape | Dynamic axes cause shape ambiguity | Remove dynamic_axes={} for static-shape export |
| Colour mismatch | OpenCV reads BGR; model trained on RGB | Set swapRB=True in blobFromImage |
| Normalisation mismatch | Different mean/std at inference | Match blobFromImage mean/scalefactor to training Normalize |
| NaN outputs | FP16 overflow in export | Keep export in FP32; cast inside model if needed |
| Slow inference | Using DNN_BACKEND_DEFAULT | Set DNN_BACKEND_OPENCV + DNN_TARGET_CPU, or CUDA if available |
| Model fails to load | Non-standard node attributes | Simplify with onnxsim; visualise with Netron to identify problematic nodes |

**Netron** is a free graph visualiser for ONNX, TensorFlow, and other formats:
`pip install netron && netron model.onnx`
Use it to inspect node names, shapes, and operator types before debugging inference.

---

## Part 5: OpenCV DNN Inference

### 5.1 Loading and Preprocessing

```python
import cv2
import numpy as np

# Load model
net = cv2.dnn.readNetFromONNX("model_simplified.onnx")

# Backend / target selection
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
# For NVIDIA GPU (requires OpenCV built with CUDA):
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Preprocessing — must exactly match training pipeline
# Training used: Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
# blobFromImage applies: output = (pixel/scalefactor - mean) / std
# To match: scalefactor = 1/255, then subtract mean (in 0-1 scale -> pass as 0-255 values)
mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
std = (0.229, 0.224, 0.225)

img = cv2.imread("test.jpg")
blob = cv2.dnn.blobFromImage(
    img,
    scalefactor=1.0 / 255.0,
    size=(224, 224),
    mean=mean,
    swapRB=True,    # OpenCV is BGR; model expects RGB
    crop=False,
)

# Note: blobFromImage does NOT divide by std. For std normalisation, post-divide:
for c, s in enumerate(std):
    blob[0, c] /= s

net.setInput(blob)
```

### 5.2 Post-Processing Custom Output

The exact parsing depends on your model's output format. General pattern:

```python
# Run inference
output_names = ['boxes', 'scores']  # names used during export
outputs = net.forward(output_names)
boxes_out, scores_out = outputs     # shapes depend on architecture

# boxes_out: [1, N, 4] (cx, cy, w, h) normalised 0-1
# scores_out: [1, N, num_classes]

boxes_out = boxes_out[0]    # [N, 4]
scores_out = scores_out[0]  # [N, num_classes]

IMG_W, IMG_H = img.shape[1], img.shape[0]
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

class_ids, confidences, bboxes = [], [], []

for i in range(len(scores_out)):
    scores = scores_out[i]
    class_id = np.argmax(scores)
    confidence = scores[class_id]
    if confidence < CONF_THRESHOLD:
        continue
    cx, cy, w, h = boxes_out[i]
    # Convert from normalised cx,cy,w,h to pixel x1,y1,x2,y2
    x1 = int((cx - w / 2) * IMG_W)
    y1 = int((cy - h / 2) * IMG_H)
    bw = int(w * IMG_W)
    bh = int(h * IMG_H)
    class_ids.append(class_id)
    confidences.append(float(confidence))
    bboxes.append([x1, y1, bw, bh])  # NMSBoxes expects x,y,w,h

# Non-maximum suppression
indices = cv2.dnn.NMSBoxes(bboxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

CLASS_NAMES = ['circle', 'rectangle', 'triangle']
for i in indices.flatten():
    x, y, w, h = bboxes[i]
    label = f"{CLASS_NAMES[class_ids[i]]}: {confidences[i]:.2f}"
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Detections", img)
cv2.waitKey(0)
```

**Tips for debugging post-processing**:
1. Use Netron to confirm output node names and tensor shapes.
2. Verify output with ONNX Runtime first (same Python, no OpenCV) to rule out preprocessing issues.
3. Print raw output statistics (min, max, mean) to check for NaN/Inf or unexpected ranges.
4. If boxes appear in wrong positions, check whether coordinates are (cx,cy,w,h) or (x1,y1,x2,y2)
   and whether they are normalised (0-1) or absolute pixels.
