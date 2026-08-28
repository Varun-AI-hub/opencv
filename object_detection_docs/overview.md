# Object Detection with OpenCV — Complete Guide

## Overview

This documentation covers every major object detection approach available in or commonly used with OpenCV, from traditional image-processing pipelines through to state-of-the-art deep learning models.  It is intended for computer vision practitioners who need to choose the right tool for a given scenario, understand the underlying algorithms, and integrate detectors into production pipelines.

The documentation is organised into four themes:

- **Classical Methods** — Template Matching, Haar Cascades, HOG+SVM, Background Subtraction, Contour Detection
- **DNN-Based Methods** — YOLO (v3/v4/v8), MobileNet-SSD, Faster R-CNN via `cv2.dnn`
- **Custom Training** — dataset preparation, YOLOv8 fine-tuning, ONNX export, HOG+SVM custom training
- **Advanced Topics** — multi-scale pyramids, tracking integration, edge deployment, quantisation

---

## Method Selection Guide

| Scenario | Recommended Method | Why |
|---|---|---|
| No GPU, simple rigid shapes | Template Matching | Fast, zero training, no dependencies |
| Face/eye/body detection, CPU only | Haar Cascade | Pretrained XMLs included with OpenCV, real-time on CPU |
| Pedestrians / generic rigid objects, no GPU | HOG + SVM | Robust pretrained pedestrian detector; linear SVM fast at inference |
| Moving objects in video, no labelled data | Background Subtraction (MOG2) | No training needed; handles illumination change |
| General objects (80 COCO classes), GPU available | YOLOv8 ONNX via `cv2.dnn` | Best accuracy/speed trade-off; opset-12 ONNX runs in OpenCV |
| Edge device / low-power board | MobileNet-SSD | Small model (20 MB), fast on CPU/OpenCL |
| Custom classes, training data available | Train YOLOv8 → export ONNX | Best end-to-end pipeline; Ultralytics tooling handles aug + metrics |
| Highest accuracy, GPU required | Faster R-CNN (ResNet-50 FPN) | Two-stage pipeline maximises mAP at cost of speed |
| Industrial inspection, template drift | Template Matching + SSIM | Deterministic, interpretable confidence score |

---

## Contents

- [Classical Methods](classical_methods.md)
- [DNN-Based Methods](dnn_methods.md)
- [Custom Training](custom_training.md)
- [Slides (LaTeX Beamer)](slides.tex)

---

## Quick Start Code

The snippet below loads the OpenCV built-in frontal-face Haar Cascade and runs detection on a synthetically generated test image.  No external files are required.

```python
import cv2
import numpy as np

# -------------------------------------------------------------------
# 1. Load the built-in frontal-face Haar Cascade
# -------------------------------------------------------------------
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError(f'Could not load cascade from {cascade_path}')

# -------------------------------------------------------------------
# 2. Create a synthetic test image (white background + grey oval "face")
# -------------------------------------------------------------------
img = np.ones((480, 640, 3), dtype=np.uint8) * 220   # light grey background
# Draw a rough ellipse to approximate a face region
cv2.ellipse(img, (320, 240), (80, 110), 0, 0, 360, (180, 160, 140), -1)
# Eyes
cv2.circle(img, (290, 210), 15, (60, 60, 60), -1)
cv2.circle(img, (350, 210), 15, (60, 60, 60), -1)
# Mouth
cv2.ellipse(img, (320, 270), (30, 15), 0, 0, 180, (100, 80, 80), 2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------------------------
# 3. Detect faces
# -------------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=3,
    minSize=(30, 30),
)

# -------------------------------------------------------------------
# 4. Draw bounding boxes
# -------------------------------------------------------------------
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 200, 0), 2)
    cv2.putText(img, 'face', (x, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)

print(f'Detected {len(faces)} face(s)')
cv2.imwrite('quickstart_result.jpg', img)
# cv2.imshow('Quick Start', img); cv2.waitKey(0)   # uncomment for interactive display
```

---

## Performance Summary

Benchmarks measured on a single image (batch = 1).
GPU: NVIDIA RTX 3080 (CUDA 11.8).  CPU: Intel Core i7-11800H.

| Model | mAP COCO val2017 | FPS (GPU) | FPS (CPU) | Model Size | Input Resolution |
|---|---|---|---|---|---|
| Haar Cascade (face) | N/A | 30+ | 15+ | < 1 MB | 24 × 24 (window) |
| HOG + SVM (pedestrian) | N/A | 5 | 2 | < 1 MB | 64 × 128 (window) |
| MobileNet-SSD v2 | 22.1 | 90 | 10 | 20 MB | 300 × 300 |
| YOLOv3 | 55.3 | 65 | 2 | 237 MB | 416 × 416 |
| YOLOv4 | 65.7 | 62 | 2 | 245 MB | 416 × 416 |
| YOLOv8n | 37.3 | 120 | 8 | 6 MB | 640 × 640 |
| YOLOv8s | 44.9 | 100 | 4 | 22 MB | 640 × 640 |
| YOLOv8m | 50.2 | 75 | 2 | 52 MB | 640 × 640 |
| YOLOv8x | 53.9 | 55 | 1 | 131 MB | 640 × 640 |
| Faster R-CNN (ResNet-50 FPN) | 37.8 | 22 | 0.5 | 160 MB | variable |
| EfficientDet-D0 | 33.8 | 58 | 3 | 15 MB | 512 × 512 |

> **Note**: FPS values are approximate and vary significantly with hardware configuration, image resolution, and OpenCV build options (CUDA, OpenCL).  mAP figures are from the respective original papers or official Ultralytics benchmarks on COCO val2017.

---

## References

1. P. Viola and M. Jones, "Rapid Object Detection using a Boosted Cascade of Simple Features," *IEEE CVPR*, 2001.
2. N. Dalal and B. Triggs, "Histograms of Oriented Gradients for Human Detection," *IEEE CVPR*, 2005.
3. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," *IEEE CVPR*, 2016.
4. W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg, "SSD: Single Shot MultiBox Detector," *ECCV*, 2016.
5. S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," *NeurIPS*, 2015.
6. G. Jocher, A. Chaurasia, and J. Qiu, "Ultralytics YOLOv8," GitHub, 2023. https://github.com/ultralytics/ultralytics
7. OpenCV DNN Module Documentation: https://docs.opencv.org/master/d6/d0f/group__dnn.html
