# Feature: ONNX Model Metadata API (Issue #23277)

## ELI5 (Explain Like I'm 5)

Imagine a book with a **sticky note on the cover** saying:

```
Author: Alice
Labels: cat, dog, bird
Preprocessing: normalize to [0, 1]
```

Before this fix, OpenCV could read the book content (the neural network weights)
but **threw away the sticky note**.

**The fix** adds a way to read the sticky note:

```python
net.getMetaData('author')  # returns 'Alice'
net.getMetaData('classes') # returns 'cat,dog,bird'
```

## What was missing

ONNX model files can store arbitrary `metadata_props` — key/value pairs
set by the model author (framework version, training dataset, class names, etc.).

OpenCV parsed the ONNX file but never exposed these properties to users.

## ASCII diagram

```
model.onnx file
+------------------------------------------+
|  Graph (weights, layers, ops)            |
|                                          |
|  metadata_props:                         |
|    key="author"     value="OpenCV Team"  |
|    key="classes"    value="cat,dog,bird" |
|    key="version"    value="1.3"          |
+------------------------------------------+
            |
            | NEW API
            v
  net.getMetaData("author")  --> "OpenCV Team"
  net.getMetaData("classes") --> "cat,dog,bird"
```

## Files changed

- `modules/dnn/include/opencv2/dnn/dnn.hpp`  (new API declaration)
- `modules/dnn/src/onnx/onnx_importer.cpp`   (parse metadata_props)
- `modules/dnn/src/net.cpp`                  (store & expose metadata)
