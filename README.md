## OpenCV: Open Source Computer Vision Library


### Resources

* Homepage: <https://opencv.org>
  * Courses: <https://opencv.org/courses>
* Docs: <https://docs.opencv.org/5.x/>
* Q&A forum: <https://forum.opencv.org>
  * previous forum (read only): <http://answers.opencv.org>
* Issue tracking: <https://github.com/opencv/opencv/issues>
* Additional OpenCV functionality: <https://github.com/opencv/opencv_contrib>
* Donate to OpenCV: <https://opencv.org/support/>


### Contributing

Please read the [contribution guidelines](https://github.com/opencv/opencv/wiki/How_to_contribute) before starting work on a pull request.

#### Summary of the guidelines:

* One pull request per issue;
* Choose the right base branch;
* Include tests and documentation;
* Clean up "oops" commits before submitting;
* Follow the [coding style guide](https://github.com/opencv/opencv/wiki/Coding_Style_Guide).

### Additional Resources

* [Submit your OpenCV-based project](https://form.jotform.com/233105358823151) for inclusion in Community Friday on opencv.org
* [Subscribe to the OpenCV YouTube Channel](https://youtube.com/@opencvofficial) featuring OpenCV Live, an hour-long streaming show
* [Follow OpenCV on LinkedIn](https://linkedin.com/company/opencv/) for daily posts showing the state-of-the-art in computer vision & AI
* [Apply to be an OpenCV Volunteer](https://form.jotform.com/232745316792159) to help organize events and online campaigns as well as amplify them
* [Follow OpenCV on Mastodon](https://mastodon.social/@opencv) in the Fediverse
* [Follow OpenCV on Twitter](https://twitter.com/opencvlive)
* [OpenCV.ai](https://opencv.ai): Computer Vision and AI development services from the OpenCV team.

---

## Object Detection Documentation (Varun-AI-hub Fork)

This fork includes a comprehensive object detection documentation and tutorial series covering classical and deep learning methods in OpenCV.

### Documentation Branches

| Branch | Description |
|---|---|
| `docs/object-detection-classical` | Template matching, Haar cascades, HOG+SVM, background subtraction, NMS |
| `docs/object-detection-dnn` | YOLO, SSD, Faster R-CNN, EfficientDet via OpenCV DNN module |
| `docs/object-detection-training` | Custom training overview, YOLOv8 ONNX export, evaluation metrics |
| `docs/object-detection-slides` | 40-frame LaTeX Beamer presentation + master overview |
| `docs/custom-training-yolov8` | End-to-end YOLOv8: dataset generation, training, ONNX export, OpenCV inference |
| `docs/custom-training-classical` | HOG+SVM custom training with hard negative mining, Haar cascade workflow |
| `docs/custom-training-pytorch-onnx` | PyTorch custom detector → ONNX → OpenCV DNN complete pipeline |

### Key Files

- `object_detection_docs/classical_methods.md` — Classical methods reference
- `object_detection_docs/dnn_methods.md` — Deep learning methods reference
- `object_detection_docs/custom_training.md` — Custom training guide
- `object_detection_docs/overview.md` — Master index and method selection guide
- `object_detection_docs/slides.tex` — LaTeX Beamer slides (compile with pdflatex)
- `custom_training/yolov8/tutorial.md` — YOLOv8 step-by-step tutorial
- `custom_training/classical/tutorial.md` — HOG+SVM step-by-step tutorial
- `custom_training/pytorch_onnx/tutorial.md` — PyTorch → ONNX tutorial

### Quick Start

```python
# Load and run any pretrained OpenCV detector in 5 lines
import cv2
net = cv2.dnn.readNetFromONNX("model.onnx")
blob = cv2.dnn.blobFromImage(img, 1/255.0, (640, 640), swapRB=True)
net.setInput(blob)
outputs = net.forward(net.getUnconnectedOutLayersNames())
```

### Bug Fixes Included

This fork also contains fixes for 19 OpenCV issues across modules: features, imgproc, dnn, ptcloud, stereo, geometry, videoio, and js.
See branches prefixed with `fix/` for individual issue fixes with tests and documentation.
