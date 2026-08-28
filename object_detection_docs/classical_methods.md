# Classical Object Detection Methods in OpenCV

This document covers the foundational classical techniques for object detection available
in OpenCV — before deep learning dominated the field. These methods remain practical for
embedded systems, real-time pipelines with limited compute, and educational purposes.

---

## Table of Contents

1. [Template Matching](#1-template-matching)
2. [Haar Cascade Classifiers](#2-haar-cascade-classifiers)
3. [HOG + SVM](#3-hog--svm-histogram-of-oriented-gradients)
4. [Background Subtraction](#4-background-subtraction)
5. [Contour-Based Detection](#5-contour-based-detection)
6. [Non-Maximum Suppression (NMS)](#6-non-maximum-suppression-nms)

---

## 1. Template Matching

### How It Works

Template matching slides a small **template image** (the object to find) over the
**source image** pixel by pixel, computing a similarity (or dissimilarity) score at every
position. The result is a response map of the same spatial dimensions as the source image.

```
Source image (W x H):            Template (w x h):
+---------------------------+     +-------+
|                           |     |  obj  |
|   ...slide template...   |     +-------+
|                           |
+---------------------------+
         |
         v  matchTemplate
+---------------------------+
|  response map (float32)   |   ← peak = best match location
+---------------------------+
```

### cv2.matchTemplate — The Six Methods

| Method              | Formula                                          | Best value | Notes                          |
|---------------------|--------------------------------------------------|------------|--------------------------------|
| `TM_SQDIFF`         | sum( (T(x,y) - I(x,y))^2 )                      | minimum    | Sensitive to intensity shifts  |
| `TM_SQDIFF_NORMED`  | TM_SQDIFF / norm(T) * norm(I)                    | minimum 0  | Normalised, range [0, 1]       |
| `TM_CCORR`          | sum( T(x,y) * I(x,y) )                           | maximum    | Cross-correlation              |
| `TM_CCORR_NORMED`   | TM_CCORR / (norm(T) * norm(I))                   | maximum 1  | Cosine similarity              |
| `TM_CCOEFF`         | sum( T'(x,y) * I'(x,y) ) (mean-subtracted)      | maximum    | More robust than raw CCORR     |
| `TM_CCOEFF_NORMED`  | TM_CCOEFF / (norm(T') * norm(I'))                | maximum 1  | **Recommended general use**    |

### Key Parameters

| Parameter          | Type           | Description                                          |
|--------------------|----------------|------------------------------------------------------|
| `image`            | ndarray        | Grayscale or BGR source image                        |
| `templ`            | ndarray        | Template; must be <= source size                     |
| `method`           | int (enum)     | One of the six methods above                         |
| `result`           | ndarray output | Response map; shape = (H-h+1, W-w+1)                |
| `mask` (optional)  | ndarray        | Mask applied to template (TM_SQDIFF, TM_CCORR_NORMED)|

### Full Code Example — Logo Detection with NMS

```python
import cv2
import numpy as np

def template_match_multi(source_gray, template_gray, threshold=0.8):
    """Return list of (x, y, w, h, score) above threshold, after NMS."""
    result = cv2.matchTemplate(source_gray, template_gray,
                               cv2.TM_CCOEFF_NORMED)
    h, w = template_gray.shape[:2]

    # Collect all locations above threshold
    locs = np.where(result >= threshold)
    boxes, scores = [], []
    for y, x in zip(*locs):
        boxes.append([x, y, x + w, y + h])          # x1,y1,x2,y2
        scores.append(float(result[y, x]))

    if not boxes:
        return []

    # Apply NMS
    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(
        [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes],
        scores_np.tolist(), threshold, 0.4
    )

    detections = []
    for i in indices.flatten():
        x1, y1, x2, y2 = boxes[i]
        detections.append((x1, y1, x2 - x1, y2 - y1, scores[i]))
    return detections


# --- Demo with synthetic data ---
source = np.ones((300, 400, 3), dtype=np.uint8) * 200
cv2.rectangle(source, (50, 60), (110, 120), (30, 80, 200), -1)   # logo instance 1
cv2.rectangle(source, (200, 150), (260, 210), (30, 80, 200), -1) # logo instance 2
template = source[60:120, 50:110].copy()

src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

detections = template_match_multi(src_gray, tpl_gray, threshold=0.95)
for (x, y, w, h, score) in detections:
    cv2.rectangle(source, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(source, f"{score:.2f}", (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

cv2.imwrite("template_result.png", source)
print(f"Found {len(detections)} matches")
```

### Limitations

- **Scale sensitivity**: Template must match the object's scale exactly. Workaround: build
  a scale pyramid and run matching at each level.
- **Rotation sensitivity**: Template must be upright. Workaround: rotate the template
  through angles and keep the best response.
- **Occlusion**: Partial matches drop the score sharply.
- **Texture**: Works poorly on uniform-color regions.
- **Speed**: O(W * H * w * h) — use `cv2.matchTemplate` with FFT backend (automatic in
  recent OpenCV) for large images.

---

## 2. Haar Cascade Classifiers

### History: Viola-Jones (2001)

Paul Viola and Michael Jones introduced a real-time face detection framework in 2001 that
revolutionised computer vision. The key insights:

1. **Integral images** for O(1) rectangular-sum computation.
2. **AdaBoost** to select a small set of discriminative features from a huge pool.
3. **Cascade of classifiers** (attentional cascade) for fast rejection of non-faces.

### Integral Images

```
Original image I:        Integral image II:
+--+--+--+             +--+--+--+
| 1| 2| 3|             | 1| 3| 6|
+--+--+--+    ----->   | 5| 9|15|
| 4| 2| 3|             |12|18|27|
+--+--+--+
```

`II(x,y) = sum of all pixels above and to the left of (x,y)`

Any rectangular sum is computed in exactly 4 array lookups regardless of region size.

### Feature Types

```
Edge feature:     Line feature:    Four-rectangle:
+---+---+         +--++--++--+     +--+--+
| + | - |         |+ || - ||+ |    |+ |- |
+---+---+         +--++--++--+     +--+--+
                                   |- |+ |
                                   +--+--+
```

Each feature = weighted sum of white - dark rectangles.

### Cascade Structure

```
Window ──> Stage 1 ──FAIL──> Discard (negative)
              |PASS
              v
           Stage 2 ──FAIL──> Discard
              |PASS
              v
           Stage N ──PASS──> Detection!
```

Early stages contain very few features (2-5) and reject ~50% of negatives very quickly.
Later stages are more complex but rarely reached.

### Pretrained Cascades in OpenCV

| File                                    | Detects               |
|-----------------------------------------|-----------------------|
| `haarcascade_frontalface_default.xml`   | Frontal faces         |
| `haarcascade_frontalface_alt.xml`       | Frontal faces (alt)   |
| `haarcascade_frontalface_alt2.xml`      | Frontal faces (alt2)  |
| `haarcascade_profileface.xml`           | Profile faces         |
| `haarcascade_eye.xml`                   | Eyes                  |
| `haarcascade_eye_tree_eyeglasses.xml`   | Eyes with glasses     |
| `haarcascade_smile.xml`                 | Smiles / mouths       |
| `haarcascade_fullbody.xml`              | Full body             |
| `haarcascade_upperbody.xml`             | Upper body            |
| `haarcascade_lowerbody.xml`             | Lower body            |
| `haarcascade_russian_plate_number.xml`  | Russian licence plate |
| `haarcascade_frontalcatface.xml`        | Cat faces             |

Cascade XML files are located at:
`<opencv_data>/haarcascades/` — typically `/usr/share/opencv4/haarcascades/`
or `cv2.data.haarcascades` in Python.

### detectMultiScale Parameters

| Parameter       | Default | Description                                                               |
|-----------------|---------|---------------------------------------------------------------------------|
| `image`         | —       | Grayscale input image                                                     |
| `scaleFactor`   | 1.1     | How much the image size is reduced at each scale (1.05 = finer, slower)  |
| `minNeighbors`  | 3       | Minimum neighbours a candidate rectangle must retain (higher = fewer FP) |
| `flags`         | 0       | Legacy flags; use 0                                                       |
| `minSize`       | (0,0)   | Minimum object size (e.g. (30,30) to skip tiny detections)               |
| `maxSize`       | —       | Maximum object size                                                       |

**Tuning guide:**
- Too many false positives → increase `minNeighbors` (try 5-8) or increase `minSize`
- Missing detections → decrease `scaleFactor` (e.g. 1.05) or decrease `minNeighbors`
- Slow → increase `scaleFactor` (1.3) and increase `minSize`

### Full Code Example — Face + Eye Detection

```python
import cv2
import numpy as np

# Load cascades (adjust path if needed)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_faces_and_eyes(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)   # improve contrast

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    output = image_bgr.copy()
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(output, (fx, fy), (fx+fw, fy+fh), (255, 0, 0), 2)
        # Search for eyes only within the face ROI (upper half)
        roi_gray  = gray[fy:fy+fh//2, fx:fx+fw]
        roi_color = output[fy:fy+fh//2, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(15,15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return output, len(faces)


# Webcam real-time detection
def webcam_face_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result, n = detect_faces_and_eyes(frame)
        cv2.putText(result, f"Faces: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Face Detection", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# To run: webcam_face_detection()
```

### Performance Notes

- Frontal face detection: ~30-60 fps on modern CPU at 640x480
- Works best on well-lit, frontal, unoccluded faces
- Fails on profile views, partial occlusion, and extreme lighting

---

## 3. HOG + SVM (Histogram of Oriented Gradients)

### HOG Feature Extraction

Dalal and Triggs (2005) proposed HOG for pedestrian detection. The pipeline:

```
Image
  |
  v
Compute gradients (Sobel/[-1,0,1])
  |
  v
Divide into CELLS (e.g. 8x8 pixels)
  |
  v
Build orientation histogram per cell (9 bins, 0-180 degrees)
  |
  v
Group cells into BLOCKS (e.g. 2x2 cells)
  |
  v
L2 normalise each block (contrast normalisation)
  |
  v
Concatenate all block descriptors -> HOG feature vector
```

For the default 64x128 pedestrian detector:
- 8x8 px cells -> 8 cols, 16 rows of cells
- 2x2 cell blocks with 1-cell stride -> 7x15 = 105 blocks
- 9 orientation bins * 4 cells/block = 36 values/block
- **Total: 105 * 36 = 3780-dimensional vector**

### cv2.HOGDescriptor Parameters

| Parameter         | Default for people | Description                               |
|-------------------|--------------------|-------------------------------------------|
| `winSize`         | (64, 128)          | Detection window size                     |
| `blockSize`       | (16, 16)           | Block size in pixels                      |
| `blockStride`     | (8, 8)             | Block stride (overlap = blockSize/2)      |
| `cellSize`        | (8, 8)             | Cell size in pixels                       |
| `nbins`           | 9                  | Number of orientation bins                |
| `derivAperture`   | 1                  | Sobel kernel size                         |
| `winSigma`        | -1 (auto)          | Gaussian window for block smoothing       |
| `histogramNormType` | 0 (L2Hys)        | Block normalisation method                |
| `L2HysThreshold`  | 0.2                | Clipping threshold for L2-Hys            |
| `gammaCorrection` | True               | Whether to apply sqrt gamma correction    |
| `nlevels`         | 64                 | Max number of detection window scales     |

### detectMultiScale Parameters (HOG)

| Parameter       | Description                                              |
|-----------------|----------------------------------------------------------|
| `img`           | Input image (grayscale or BGR)                           |
| `winStride`     | Step size for sliding window (e.g. (8,8))               |
| `padding`       | Padding around image (e.g. (16,16))                     |
| `scale`         | Scale factor between pyramid levels (default 1.05)      |
| `finalThreshold`| SVM decision threshold; lower -> more detections        |

### Full Code Example — Pedestrian Detection

```python
import cv2
import numpy as np

def detect_pedestrians(image_bgr, scale=1.05, win_stride=(8, 8),
                        padding=(16, 16), final_threshold=2.0):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Resize for speed if image is large
    max_dim = 800
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        factor = max_dim / max(h, w)
        image_bgr = cv2.resize(image_bgr, (int(w*factor), int(h*factor)))

    rects, weights = hog.detectMultiScale(
        image_bgr,
        winStride=win_stride,
        padding=padding,
        scale=scale,
        finalThreshold=final_threshold
    )

    output = image_bgr.copy()
    if len(rects):
        # Convert to x1,y1,x2,y2 for NMS
        boxes_xywh = [[x, y, w, h] for (x, y, w, h) in rects]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, weights.flatten().tolist(),
                                   0.0, 0.65)
        for i in indices.flatten():
            x, y, w, h = rects[i]
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)

    return output, len(rects)


# --- Custom HOG+SVM training sketch ---
def train_hog_svm_example():
    """
    Illustrates how to train a custom HOG+SVM binary classifier.
    Replace positives/negatives with real cropped image patches.
    """
    hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)

    # Simulate 10 positive patches (64x128 px) and 10 negatives
    rng = np.random.default_rng(42)
    positives = [rng.integers(0, 255, (128, 64, 3), dtype=np.uint8)
                 for _ in range(10)]
    negatives = [rng.integers(0, 255, (128, 64, 3), dtype=np.uint8)
                 for _ in range(10)]

    features, labels = [], []
    for img in positives:
        feat = hog.compute(img).flatten()
        features.append(feat)
        labels.append(1)
    for img in negatives:
        feat = hog.compute(img).flatten()
        features.append(feat)
        labels.append(-1)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6))
    svm.train(X, cv2.ml.ROW_SAMPLE, y)

    # Extract support vectors to use with HOGDescriptor
    sv = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    # Detector vector = -sv.T (for binary HOG detector convention)
    detector = np.zeros(sv.shape[1] + 1, dtype=np.float32)
    detector[:-1] = -sv[0]
    detector[-1] = float(rho)

    hog.setSVMDetector(detector)
    print("Custom HOG+SVM trained. Descriptor length:", len(detector))
    return hog
```

### Performance Notes

- HOG+SVM achieves ~90% detection rate at ~10^-4 FPPW on INRIA pedestrian dataset
- Typical throughput: 2-5 fps on CPU at 640x480 (single scale)
- Use `winStride=(16,16)` and larger `scale` (1.1-1.2) for faster but coarser detection
- Hard-negative mining significantly improves SVM accuracy

---

## 4. Background Subtraction

### Use Case

Background subtraction separates moving foreground objects from a static (or slowly
changing) background in video. Typical applications: traffic monitoring, security cameras,
people counting.

```
Frame t:          Background model:     Foreground mask:
+----------+      +----------+          +----------+
|  [car]   |  -   |          |    =     | [  FG  ] |
|  road    |      |  road    |          | 000000   |
+----------+      +----------+          +----------+
```

### MOG2 — Gaussian Mixture Model

`cv2.createBackgroundSubtractorMOG2` models each pixel as a mixture of K Gaussian
distributions (default K=5). It automatically adapts to gradual illumination changes and
marks shadows separately.

| Parameter        | Default | Description                                               |
|------------------|---------|-----------------------------------------------------------|
| `history`        | 500     | Number of frames for background model learning            |
| `varThreshold`   | 16      | Mahalanobis distance^2 threshold for foreground/BG        |
| `detectShadows`  | True    | If True, shadows are marked grey (127) in the mask        |

### KNN — K-Nearest Neighbours

`cv2.createBackgroundSubtractorKNN` models the background with the K nearest neighbours
from recent pixel samples. Better for scenes with few foreground objects.

| Parameter        | Default | Description                                               |
|------------------|---------|-----------------------------------------------------------|
| `history`        | 500     | Number of frames in background history                    |
| `dist2Threshold` | 400     | Squared distance threshold for classification             |
| `detectShadows`  | True    | Shadow detection toggle                                   |

### Full Code Example — Vehicle Detection in Video

```python
import cv2
import numpy as np

def detect_moving_objects(video_path_or_int=0,
                          method='MOG2',
                          min_area=500):
    """
    Detect moving objects using background subtraction.
    video_path_or_int: file path or camera index
    method: 'MOG2' or 'KNN'
    min_area: minimum contour area in pixels
    """
    cap = cv2.VideoCapture(video_path_or_int)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path_or_int}")

    if method == 'MOG2':
        bgsub = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True)
    else:
        bgsub = cv2.createBackgroundSubtractorKNN(
            history=500, dist2Threshold=400, detectShadows=True)

    # Morphological kernels
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fg_mask = bgsub.apply(frame)

        # Remove shadows (grey pixels = 127)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_open)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        output = frame.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(output, f"area={int(area)}", (x, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Foreground", fg_mask)
        cv2.imshow("Detection",  output)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# --- Synthetic demo (no video file needed) ---
def bgrsub_synthetic_demo():
    bgsub = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=20)
    rng = np.random.default_rng(0)

    # "Train" on 50 background-only frames
    for _ in range(50):
        bg = (rng.integers(90, 110, (200, 300), dtype=np.uint8))
        bgsub.apply(bg)

    # Now add a moving "vehicle" (white rectangle)
    frame = (rng.integers(90, 110, (200, 300), dtype=np.uint8))
    cv2.rectangle(frame, (80, 70), (140, 120), 255, -1)  # vehicle

    mask = bgsub.apply(frame)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    print(f"Foreground pixels detected: {cv2.countNonZero(mask)}")
    return mask
```

### Morphological Post-Processing

```
Noisy fg_mask:      After OPEN:         After CLOSE:
.....X..X.          .....X....          .....XXXX.
....XXX...    -->   ....XXX...    -->   ....XXXXX.
...XXXXX..          ...XXXXX..          ...XXXXXX.
..X.XX.X..          ...XXX....          ...XXXXX..
```

- **Opening** (erosion then dilation): removes small noise blobs
- **Closing** (dilation then erosion): fills holes inside objects

---

## 5. Contour-Based Detection

### Overview

After thresholding or background subtraction, `cv2.findContours` extracts the boundaries
of connected white regions. Combined with shape descriptors, this allows robust filtering.

### Key Functions

```python
# Find contours
contours, hierarchy = cv2.findContours(
    binary_image,           # uint8, values 0 or 255
    cv2.RETR_EXTERNAL,      # retrieval mode
    cv2.CHAIN_APPROX_SIMPLE # compression method
)

# Bounding box
x, y, w, h = cv2.boundingRect(contour)

# Minimum-area rotated rectangle
rect = cv2.minAreaRect(contour)   # -> (center, (w,h), angle)
box  = cv2.boxPoints(rect).astype(int)

# Connected components (faster alternative)
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
```

### Retrieval Modes

| Mode              | Description                                           |
|-------------------|-------------------------------------------------------|
| `RETR_EXTERNAL`   | Only outermost contours                               |
| `RETR_LIST`       | All contours, no hierarchy                            |
| `RETR_CCOMP`      | Two-level hierarchy (external + holes)                |
| `RETR_TREE`       | Full hierarchy tree                                   |

### Compression Methods

| Method               | Description                                        |
|----------------------|----------------------------------------------------|
| `CHAIN_APPROX_NONE`  | All boundary points stored                         |
| `CHAIN_APPROX_SIMPLE`| Only endpoints of horizontal/vertical/diagonal runs|
| `CHAIN_APPROX_TC89_*`| Teh-Chin chain approximation                       |

### Shape Descriptors for Filtering

| Descriptor    | Formula                              | Good for                        |
|---------------|--------------------------------------|---------------------------------|
| Area          | `cv2.contourArea(cnt)`               | Size filter                     |
| Perimeter     | `cv2.arcLength(cnt, True)`           | Shape complexity                |
| Aspect ratio  | `float(w) / h`                       | Square vs elongated             |
| Extent        | `area / (w * h)`                     | Fills bounding box?             |
| Solidity      | `area / hull_area`                   | Convex vs concave               |
| Circularity   | `4*pi*area / perimeter^2`            | Circle detection                |

### Full Code Example — Multi-Object Detection with Filtering

```python
import cv2
import numpy as np

def detect_objects_contour(binary_img,
                           min_area=100,
                           max_area=50000,
                           min_aspect=0.3,
                           max_aspect=3.5,
                           min_solidity=0.6):
    """
    Detect objects in a binary image by contour analysis.
    Returns list of (x, y, w, h) bounding boxes after shape filtering.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        if not (min_aspect < aspect < max_aspect):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue

        detections.append((x, y, w, h))

    return detections


# --- Synthetic demo ---
binary = np.zeros((300, 400), dtype=np.uint8)
cv2.circle(binary, (100, 100), 40, 255, -1)         # circle object
cv2.rectangle(binary, (200, 60), (280, 140), 255, -1) # rectangle object
cv2.circle(binary, (50, 250), 5, 255, -1)            # tiny noise blob

boxes = detect_objects_contour(binary)
vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
for (x, y, w, h) in boxes:
    cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
print(f"Objects detected (after filtering): {len(boxes)}")

# connectedComponentsWithStats alternative
n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
print(f"Connected components (incl. background): {n}")
for i in range(1, n):   # skip background (label 0)
    x, y, w, h, area = stats[i]
    print(f"  Component {i}: bbox=({x},{y},{w},{h}), area={area}")
```

---

## 6. Non-Maximum Suppression (NMS)

### The Problem

Sliding-window and multi-scale detectors typically produce multiple overlapping boxes
around the same object:

```
+---------+
| box 1   +-------+
|    | box 2      |
|    |    +---+   |
+----+    |box|   |
     +----+ 3 +---+
          +---+
```

NMS keeps the highest-scoring box and removes boxes that overlap with it beyond an IoU
threshold.

### IoU — Intersection over Union

```
Box A:  (xa1,ya1,xa2,ya2)       Box B:  (xb1,yb1,xb2,yb2)
         +-------+
         |   A   |
         |  +----+----+
         +--+----+    |
            | I  | B  |
            +----+----+

IoU = area(Intersection) / area(Union)
    = area(I) / (area(A) + area(B) - area(I))
```

- IoU = 0: no overlap
- IoU = 1: perfect overlap (identical boxes)
- Typical NMS threshold: 0.4-0.5

### cv2.dnn.NMSBoxes

```python
indices = cv2.dnn.NMSBoxes(
    bboxes,          # list of [x, y, w, h]  (not x2,y2)
    scores,          # list of confidence scores (floats)
    score_threshold, # minimum confidence to consider a box
    nms_threshold,   # IoU threshold for suppression
    eta=1.0,         # coefficient for adaptive threshold (rarely used)
    top_k=0          # keep at most top_k boxes; 0 = no limit
)
# Returns: numpy array of kept indices (or empty tuple if nothing kept)
```

### Full Code Example — NMS from Scratch + cv2.dnn.NMSBoxes

```python
import cv2
import numpy as np


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms_manual(boxes_xyxy, scores, iou_threshold=0.5):
    """
    Pure-Python NMS. Returns indices of kept boxes.
    boxes_xyxy: list of [x1, y1, x2, y2]
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order
                 if compute_iou(boxes_xyxy[best], boxes_xyxy[i]) < iou_threshold]
    return kept


# ---- Demo ----
boxes_xyxy = [
    [100, 100, 200, 200],   # score 0.9  <- should keep
    [110, 105, 210, 205],   # score 0.75 <- overlaps heavily -> suppress
    [300, 100, 400, 200],   # score 0.85 <- should keep (separate region)
    [305, 102, 405, 202],   # score 0.6  <- overlaps -> suppress
]
scores = [0.9, 0.75, 0.85, 0.6]

# Manual NMS
kept_manual = nms_manual(boxes_xyxy, scores, iou_threshold=0.45)
print(f"Manual NMS kept indices: {kept_manual}")

# cv2.dnn.NMSBoxes  (expects [x,y,w,h] format)
boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_xyxy]
kept_cv2 = cv2.dnn.NMSBoxes(boxes_xywh, scores,
                             score_threshold=0.5,
                             nms_threshold=0.45)
print(f"cv2.dnn.NMSBoxes kept indices: {kept_cv2.flatten().tolist()}")

# Visualise
vis = np.ones((350, 500, 3), dtype=np.uint8) * 240
colors_all  = (180, 180, 180)
colors_kept = (0, 200, 0)

for i, (b, s) in enumerate(zip(boxes_xyxy, scores)):
    cv2.rectangle(vis, (b[0],b[1]), (b[2],b[3]), colors_all, 1)
for i in kept_manual:
    b = boxes_xyxy[i]
    cv2.rectangle(vis, (b[0],b[1]), (b[2],b[3]), colors_kept, 3)
    cv2.putText(vis, f"{scores[i]:.2f}", (b[0], b[1]-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors_kept, 2)

cv2.imwrite("nms_result.png", vis)
print("IoU between box 0 and box 1:",
      round(compute_iou(boxes_xyxy[0], boxes_xyxy[1]), 3))
```

### NMS Variants

| Variant         | Description                                              |
|-----------------|----------------------------------------------------------|
| Standard NMS    | Hard suppression above IoU threshold                     |
| Soft-NMS        | Decay scores of overlapping boxes rather than remove     |
| DIoU-NMS        | Uses distance-IoU for better box regression              |
| Batched NMS     | Applies NMS per class independently                      |

`cv2.dnn.NMSBoxes` implements standard NMS with optional adaptive threshold (`eta < 1`).

---

## Summary Comparison

| Method               | Speed      | Accuracy  | Scale inv. | Rotation inv. | Best use case                         |
|----------------------|------------|-----------|------------|---------------|---------------------------------------|
| Template Matching    | Fast       | High*     | No         | No            | Rigid templates, controlled scenes    |
| Haar Cascades        | Very fast  | Moderate  | Yes        | Partial       | Frontal face detection                |
| HOG + SVM            | Moderate   | Good      | Yes        | No            | Pedestrians, fixed-aspect objects     |
| Background Sub.      | Fast       | Scene dep.| N/A        | N/A           | Moving object detection in video      |
| Contour-based        | Very fast  | Shape dep.| Yes        | Partial       | Binary/segmented scenes               |

*High accuracy when object is rigid and at known scale/orientation.

---

## References

- Viola, P., & Jones, M. (2001). Rapid object detection using a boosted cascade of simple features. CVPR.
- Dalal, N., & Triggs, B. (2005). Histograms of oriented gradients for human detection. CVPR.
- KaewTraKulPong, P., & Bowden, R. (2002). An improved adaptive background mixture model for real-time tracking. AVBS.
- Zivkovic, Z. (2004). Improved adaptive Gaussian mixture model for background subtraction. ICPR.
- OpenCV documentation: https://docs.opencv.org/4.x/
