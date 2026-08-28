# Custom Training: HOG+SVM and Haar Cascade

This tutorial covers training classical computer-vision detectors from scratch.
It is structured in two parts: **Part A** covers the HOG+SVM pipeline,
and **Part B** covers Haar cascade training.

All Python scripts in this directory are fully runnable. Install dependencies with:

```bash
pip install opencv-python numpy scikit-learn matplotlib
```

---

## Part A: HOG+SVM Custom Detector

### 1. HOG Feature Theory

Histogram of Oriented Gradients was introduced by Dalal & Triggs at CVPR 2005.
The key insight is that local gradient orientations are a robust description of
object appearance that tolerates small geometric and photometric changes.

#### Feature extraction pipeline

```
Input patch (e.g. 64×128 for pedestrians)
         │
         ▼
[Optional] Gamma normalization:  I' = sqrt(I)
         │
         ▼
Compute gradients:
    Gx = I * [-1, 0, 1]   (horizontal Sobel or 1-D diff)
    Gy = Gx^T             (vertical)
         │
         ▼
Per-pixel magnitude and orientation:
    m(x,y) = sqrt(Gx² + Gy²)
    θ(x,y) = arctan(Gy / Gx)   [0°-180° unsigned]
         │
         ▼
Divide into 8×8 pixel cells.
Build a 9-bin histogram of gradient orientations, weighted by magnitude.
         │
         ▼
Group 2×2 adjacent cells into a block (16×16 px).
Concatenate the four cell histograms → 36-dim vector per block.
L2-normalise the block: v / sqrt(||v||² + ε²)
         │
         ▼
Stride block across the window (block stride = 8 px).
Concatenate all block descriptors → final feature vector.
```

#### Feature vector length formula

For a window of size `(W, H)`:

```
n_blocks_x = (W - block_width)  / block_stride_x  + 1
n_blocks_y = (H - block_height) / block_stride_y  + 1
feature_dim = n_blocks_x × n_blocks_y × cells_per_block_x × cells_per_block_y × n_bins
```

Example — 64×128 pedestrian window:

```
n_blocks_x = (64  - 16) / 8 + 1 = 7
n_blocks_y = (128 - 16) / 8 + 1 = 15
feature_dim = 7 × 15 × 2 × 2 × 9 = 3780
```

#### Manual HOG computation in Python

```python
import cv2
import numpy as np

def hog_manual(img_gray, cell_size=8, n_bins=9):
    """Visualise the HOG pipeline step-by-step (non-optimised)."""
    # Step 1 – gradient
    gx = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    angle = angle % 180          # unsigned

    h, w = img_gray.shape
    n_cells_y, n_cells_x = h // cell_size, w // cell_size
    bin_edges = np.linspace(0, 180, n_bins + 1)
    cell_hists = np.zeros((n_cells_y, n_cells_x, n_bins))

    # Step 2 – cell histograms (soft binning via linear interpolation)
    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            cell_mag   = mag  [cy*cell_size:(cy+1)*cell_size,
                                cx*cell_size:(cx+1)*cell_size].ravel()
            cell_angle = angle[cy*cell_size:(cy+1)*cell_size,
                                cx*cell_size:(cx+1)*cell_size].ravel()
            hist = np.zeros(n_bins)
            for m, a in zip(cell_mag, cell_angle):
                bin_idx = np.searchsorted(bin_edges[1:-1], a)
                hist[bin_idx % n_bins] += m
            cell_hists[cy, cx] = hist

    # Step 3 – block normalisation (2×2 cells, stride 1 cell)
    eps = 1e-5
    blocks = []
    for by in range(n_cells_y - 1):
        for bx in range(n_cells_x - 1):
            block = cell_hists[by:by+2, bx:bx+2].ravel()
            block = block / np.sqrt(np.sum(block**2) + eps**2)
            blocks.append(block)

    return np.concatenate(blocks)
```

#### cv2.HOGDescriptor setup

```python
import cv2

hog = cv2.HOGDescriptor(
    _winSize    = (64, 64),   # detection window — must match training
    _blockSize  = (16, 16),
    _blockStride= (8, 8),
    _cellSize   = (8, 8),
    _nbins      = 9,
)
# Feature vector length:
print(hog.getDescriptorSize())   # → 1764 for 64×64 window
```

Key parameters:

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `winSize` | (64, 64) or (64, 128) | Detection window; must equal training patch size |
| `blockSize` | (16, 16) | Larger → more spatial pooling |
| `blockStride` | (8, 8) | Smaller → more descriptors, higher overlap |
| `cellSize` | (8, 8) | Smaller → finer spatial resolution |
| `nbins` | 9 | 0°–180° split into 9 bins of 20° each |

---

### 2. SVM Theory

A Support Vector Machine finds the hyperplane

```
  w^T x + b = 0
```

that maximises the margin `2 / ||w||` between the two classes.

**Primal problem (C-SVC):**

```
  minimise  ½ ||w||²  +  C Σ ξᵢ
  subject to  yᵢ(w^T xᵢ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0
```

`C` is the regularisation parameter — large `C` penalises misclassification
more and yields a smaller-margin solution.

**Kernel trick** — replace the dot product with a kernel function to learn
non-linear boundaries without explicitly mapping to high-dimensional space:

| Kernel | Formula |
|--------|---------|
| LINEAR | K(x,z) = x^T z |
| POLY   | K(x,z) = (γ x^T z + r)^d |
| RBF    | K(x,z) = exp(−γ ‖x−z‖²) |
| SIGMOID| K(x,z) = tanh(γ x^T z + r) |
| CHI2   | K(x,z) = exp(−γ Σ (xᵢ−zᵢ)²/(xᵢ+zᵢ)) |
| INTER  | K(x,z) = Σ min(xᵢ, zᵢ) |

**OpenCV SVM types:**

| Type | Use case |
|------|----------|
| `C_SVC` | Standard classification (most common) |
| `NU_SVC` | Classification with ν parameter instead of C |
| `ONE_CLASS` | Novelty / outlier detection |
| `EPS_SVR` | ε-insensitive regression |
| `NU_SVR` | ν-SVR regression |

For HOG+SVM object detection always use `C_SVC` with a `LINEAR` or `RBF` kernel.
A linear SVM is faster to evaluate and generalises well when the HOG feature
space is already highly discriminative.

---

### 3. Full Training Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  1. Collect positive samples                            │
│     • Crop/resize to winSize, e.g. 64×64               │
│     • 500–5000 samples recommended                      │
├─────────────────────────────────────────────────────────┤
│  2. Collect negative samples                            │
│     • Background images (no target object)             │
│     • Typically 2–3× more than positives               │
├─────────────────────────────────────────────────────────┤
│  3. Extract HOG features                               │
│     pos_feats  [N_pos × D]   label +1                   │
│     neg_feats  [N_neg × D]   label −1                   │
├─────────────────────────────────────────────────────────┤
│  4. Train initial SVM on pos_feats + neg_feats          │
├─────────────────────────────────────────────────────────┤
│  5. Hard Negative Mining  ←──────────────┐             │
│     a. Run sliding-window detector on    │             │
│        full negative images              │             │
│     b. Collect all false-positive crops  │             │
│     c. Append to negative training set   │             │
│     d. Retrain SVM                       │             │
│     e. Repeat 2–3 iterations  ───────────┘             │
├─────────────────────────────────────────────────────────┤
│  6. Evaluate on held-out validation set                 │
│     Precision, Recall, F1 @ IoU ≥ 0.5                  │
├─────────────────────────────────────────────────────────┤
│  7. Set decision threshold                              │
│     svm.predict(feat) returns score; choose threshold  │
│     that gives acceptable precision/recall trade-off   │
└─────────────────────────────────────────────────────────┘
```

Scripts in this directory implement the entire pipeline:

1. `01_generate_hog_dataset.py` — generate synthetic dataset
2. `02_train_hog_svm.py` — train with hard negative mining
3. `03_detect_hog_svm.py` — run detector on image or webcam
4. `04_evaluate_detector.py` — compute PR curve and metrics

---

### 4. Sliding Window Detection

At inference time, the HOG+SVM detector scans an image at multiple scales
using a sliding window:

```
image
 │
 ▼
Build scale pyramid: img_scale = resize(img, scale_factor^k)
for k = 0, 1, 2, ... while min_dim >= winSize
 │
 ├─ For each (x, y) position in img_scale with stride (8, 8):
 │     crop = img_scale[y:y+winH, x:x+winW]
 │     feat = hog.compute(crop)
 │     score = w^T feat + b          ← SVM decision function
 │     if score > threshold:
 │         detections.append((x/scale, y/scale, score))
 │
 ▼
Apply NMS (Non-Maximum Suppression)
 │
 ▼
Final detections
```

**`detect()` vs `detectMultiScale()`:**

| Method | Description |
|--------|-------------|
| `hog.detect(img)` | Single scale, full dense sliding window. Returns pixel locations. |
| `hog.detectMultiScale(img)` | Builds scale pyramid internally. Returns bounding boxes. |

`detectMultiScale` is the typical inference API. You can also call `detect()` at
each pyramid level manually for finer control over padding and stride.

```python
import cv2

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

boxes, weights = hog.detectMultiScale(
    img,
    winStride=(8, 8),      # sliding stride (x, y)
    padding=(16, 16),       # padding around image border
    scale=1.05,             # pyramid scale factor (closer to 1 = more scales)
    finalThreshold=2,       # NMS group threshold
)
```

---

### 5. Precision-Recall and Threshold Selection

A detection is a **true positive (TP)** when:

```
IoU(predicted_box, ground_truth_box) > 0.5
```

where `IoU = area(intersection) / area(union)`.

**To plot a precision-recall curve:**

1. Run the detector on all validation images, keeping the raw SVM score.
2. Sort all detections by descending score.
3. Iterate through the sorted list. For each detection:
   - If IoU > 0.5 with an unmatched ground-truth box → TP
   - Otherwise → FP
4. Compute cumulative precision and recall at each rank.
5. Plot precision (y) vs recall (x).

**Choosing a threshold:**

- High threshold → fewer but more reliable detections (high precision, low recall)
- Low threshold → more detections but more false positives (high recall, low precision)
- F1 score = 2·P·R/(P+R) helps find a balanced operating point.

```python
# Example: pick threshold that maximises F1
best_f1, best_thresh = 0, 0
for thresh in np.linspace(-2, 2, 200):
    pred = (scores > thresh).astype(int)
    tp = np.sum((pred == 1) & (labels == 1))
    fp = np.sum((pred == 1) & (labels == 0))
    fn = np.sum((pred == 0) & (labels == 1))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    if f1 > best_f1:
        best_f1, best_thresh = f1, thresh
```

---

## Part B: Haar Cascade Training

### 1. How Haar Cascade Training Works

Haar cascades use **Haar-like features** — rectangular patterns that compute
differences between sums of pixels in adjacent regions — calculated in O(1)
using an **integral image**.

```
Integral image I(x,y) = Σ_{x'≤x, y'≤y} img(x', y')

Sum of any rectangle = I(BR) − I(BL) − I(TR) + I(TL)   (4 lookups)
```

Example Haar-like features:

```
 ┌──┬──┐   ┌──┬──┬──┐   ┌──────┐
 │  │▓▓│   │  │▓▓│  │   │▓▓▓▓▓▓│
 │  │▓▓│   │  │▓▓│  │   ├──────┤
 └──┴──┘   └──┴──┴──┘   │      │
 edge      line          └──────┘
```

**AdaBoost** selects the most discriminative features:

1. Assign equal weight to each training sample.
2. For each candidate feature, find the threshold that minimises weighted error.
3. Select the best feature → weak classifier.
4. Increase weight of misclassified samples, decrease weight of correctly classified.
5. Repeat for `T` rounds. Combine weak classifiers into a strong classifier:
   ```
   H(x) = sign( Σ αₜ hₜ(x) )
   ```

**Cascade structure** (Viola-Jones):

```
Image patch
    │
    ▼
Stage 1 (few features, low threshold)
    │ reject → discard quickly (most patches are background)
    │ pass ↓
Stage 2 (more features)
    │ reject → discard
    │ pass ↓
   ...
    │
Stage N (many features, high threshold)
    │ pass → DETECTION
```

Each stage has a low false-negative rate (≥99.9% hit rate) and a moderate
false-positive rate (≤50%). Cascading ~20 stages gives an overall FPR of
`0.5^20 ≈ 10^−6` while maintaining a cumulative hit rate of `0.999^20 ≈ 98%`.

---

### 2. Data Requirements

| Requirement | Recommendation |
|-------------|---------------|
| Positive samples | 1,000–5,000 annotated bounding boxes |
| Negative samples | 3,000–5,000 background images (≥ 2× positives) |
| Positive image size | Match training window (e.g. 24×24 for face) |
| Negative image size | Larger than window (random crops are taken) |
| Annotation format | `img.jpg 1 x y w h` per line (opencv_createsamples format) |

You can create synthetic positives from a single sample image using
`opencv_createsamples` with random perspective and rotation.

---

### 3. Tool Pipeline (opencv_traincascade)

These tools ship with the full OpenCV build (not the pip package).

```bash
# Step 1 — create positive .vec file from annotated images
# annotations.txt format: "img.jpg N x1 y1 w1 h1 [x2 y2 w2 h2 ...]"
opencv_createsamples \
    -info  annotations.txt \   # input annotation file
    -bg    negatives.txt   \   # background images list (optional, for augmentation)
    -vec   positives.vec   \   # output binary vector file
    -w 24 -h 24            \   # sample output size (must match training window)
    -num   1000                # number of samples to generate

# Step 2 — train cascade
opencv_traincascade \
    -data       cascade/          \   # output directory (created automatically)
    -vec        positives.vec     \   # positive samples
    -bg         negatives.txt     \   # list of background image paths
    -numPos     1000              \   # positives per stage (< total in .vec)
    -numNeg     500               \   # negatives per stage
    -numStages  20                \   # cascade depth
    -w          24 -h 24          \   # detection window (must match createsamples)
    -minHitRate       0.999       \   # per-stage minimum recall
    -maxFalseAlarmRate 0.5        \   # per-stage maximum FPR
    -precalcValBufSize  1024      \   # feature value cache (MB)
    -precalcIdxBufSize  1024          # feature index cache (MB)
```

#### Flag reference

| Flag | Meaning |
|------|---------|
| `-numStages` | More stages → better rejection, longer training |
| `-minHitRate` | Per-stage recall (0.999 typical). Lower → faster training, worse recall |
| `-maxFalseAlarmRate` | Per-stage FPR (0.5 typical). Lower → each stage must be stronger |
| `-w`, `-h` | Training window size. Must match `opencv_createsamples` |
| `-numPos` | Positives used per stage. Set to ~90% of .vec file count |
| `-numNeg` | Negatives per stage. 0.5–2× numPos is typical |
| `-precalcValBufSize` | RAM for precalculated feature values (increase for speed) |

Typical training time for a 20-stage cascade on 1000 positives: 1–4 hours on a
modern CPU, depending on window size and hardware.

#### Using the trained cascade

```python
import cv2

cascade = cv2.CascadeClassifier("cascade/cascade.xml")
img = cv2.imread("test.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

detections = cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,     # pyramid scale (1.05–1.3 typical)
    minNeighbors=3,      # group threshold — higher = fewer but more reliable
    minSize=(24, 24),
)
for (x, y, w, h) in detections:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

---

### 4. Python Simulation of Training Logic

Because `opencv_traincascade` requires a compiled binary not available via pip,
`02_train_hog_svm.py` provides a complete HOG+SVM pipeline that demonstrates
the same core concepts:

- Hard negative mining (the same technique used inside cascade training)
- Per-iteration evaluation of precision and recall
- Saving a reusable model file

Conceptually, what cascade training achieves vs HOG+SVM:

| Aspect | HOG+SVM | Haar Cascade |
|--------|---------|-------------|
| Features | Dense HOG histogram | Sparse Haar-like |
| Classifier | Single SVM decision | Cascade of AdaBoost stages |
| Speed | Moderate | Very fast (early rejection) |
| Accuracy | High (with HNM) | Good for frontal faces/rigid objects |
| Training | Minutes–hours | Hours–days |
| Best for | Arbitrary objects | Well-aligned frontal objects |

---

## Quick-Start Workflow

```bash
# 1. Generate synthetic data (circles on noise backgrounds)
python 01_generate_hog_dataset.py

# 2. Train HOG+SVM detector with hard negative mining
python 02_train_hog_svm.py --pos positives/ --neg negatives/ --output detector.xml

# 3. Run detection on a test image
python 03_detect_hog_svm.py --model detector.xml --image test_image.jpg

# 4. Evaluate on a test set with annotations
python 04_evaluate_detector.py --model detector.xml \
    --test-dir test_images/ --annotations annotations.csv
```

---

## References

- Dalal, N. & Triggs, B. (2005). *Histograms of Oriented Gradients for Human Detection*. CVPR.
- Viola, P. & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.
- OpenCV HOGDescriptor docs: https://docs.opencv.org/4.x/d5/d33/structcv_1_1HOGDescriptor.html
- OpenCV CascadeClassifier docs: https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html
- OpenCV train cascade tutorial: https://docs.opencv.org/4.x/dc/d88/tutorial_traincascade.html
