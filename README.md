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

This fork includes a comprehensive object detection documentation and tutorial series. See these branches:

| Branch | Description |
|---|---|
| `docs/object-detection-classical` | Template matching, Haar cascades, HOG+SVM, background subtraction |
| `docs/object-detection-dnn` | YOLO, SSD, Faster R-CNN via OpenCV DNN |
| `docs/object-detection-training` | Custom training, YOLOv8 ONNX export, evaluation |
| `docs/object-detection-slides` | LaTeX Beamer slides (40 frames) |
| `docs/custom-training-yolov8` | YOLOv8 end-to-end: dataset → train → ONNX → OpenCV |
| `docs/custom-training-classical` | HOG+SVM with hard negative mining |
| `docs/custom-training-pytorch-onnx` | PyTorch → ONNX → OpenCV DNN pipeline |

### This Branch

This branch contains a bug fix for an OpenCV issue. See the `dumbdown/` folder for a plain-English explanation and `algorithm_math_explained/` for the full mathematical documentation.