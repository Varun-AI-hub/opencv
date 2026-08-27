Blob Detection with SimpleBlobDetector {#tutorial_blob_detection}
======================================

@tableofcontents

@prev_tutorial{tutorial_akaze_tracking}
@next_tutorial{tutorial_homography}

|    |    |
| -: | :- |
| Compatibility | OpenCV >= 3.0 |

Goal
----

In this tutorial you will learn how to:

-   Use @ref cv::SimpleBlobDetector to find blobs in a grayscale image.
-   Configure the detection parameters (area, circularity, convexity, inertia, color) to
    target specific blob shapes.
-   Draw detected blobs with @ref cv::drawKeypoints.

What is a Blob?
---------------

A **blob** is a group of connected pixels that are brighter (or darker) than their surroundings
and whose shape satisfies a set of user-defined criteria. Common examples are:

-   Circular markers / fiducials on a calibration board.
-   Cell nuclei in a microscopy image.
-   Spots on a printed circuit board.

SimpleBlobDetector works by binarizing the source image at multiple threshold levels, finding
connected components at each level, and merging the results. The final keypoint center is the
average of the component centers across all thresholds, and the keypoint size is twice the
estimated blob radius (so that you can draw it with @ref cv::DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS).

Algorithm Overview
------------------

1.  Binarize the image at thresholds
    `minThreshold`, `minThreshold + thresholdStep`, `minThreshold + 2*thresholdStep`, …
    up to (but not including) `maxThreshold`.
2.  Extract connected components from each binary image and compute their centroids.
3.  Merge centroids that are closer than `minDistBetweenBlobs` pixels across the threshold
    stack. A merged group must appear in at least `minRepeatability` binary images to be kept.
4.  Apply the enabled filters (color, area, circularity, inertia, convexity) to the merged
    candidates.
5.  Return surviving blobs as @ref cv::KeyPoint objects. The `pt` field holds the center and
    the `size` field holds twice the estimated radius.

Detection Parameters
--------------------

All parameters are grouped in the @ref cv::SimpleBlobDetector::Params structure. The key ones are:

### Thresholding

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `minThreshold` | 50 | Lowest binarization threshold (inclusive). |
| `maxThreshold` | 220 | Upper binarization threshold (exclusive). |
| `thresholdStep` | 10 | Step between consecutive thresholds. |
| `minRepeatability` | 2 | How many threshold levels a blob must appear in to be retained. |
| `minDistBetweenBlobs` | 10 | Minimum pixel distance between distinct blob centers. |

### Filter by Color

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `filterByColor` | true | Enable color filtering. |
| `blobColor` | 0 | Target intensity: **0** for dark blobs, **255** for light blobs. |

### Filter by Area

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `filterByArea` | true | Enable area filtering. |
| `minArea` | 25 | Minimum blob area in pixels² (inclusive). |
| `maxArea` | 5000 | Maximum blob area in pixels² (exclusive). |

### Filter by Circularity

Circularity = \f$\frac{4\pi \cdot \text{Area}}{\text{perimeter}^2}\f$. A perfect circle
gives 1.0; a thin elongated shape gives a value close to 0.

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `filterByCircularity` | false | Enable circularity filtering. |
| `minCircularity` | 0.8 | Minimum circularity (inclusive). |
| `maxCircularity` | FLT_MAX | Maximum circularity (exclusive). |

### Filter by Inertia Ratio

The inertia ratio is the square root of the ratio of the minimum to maximum second-order
moment of the contour. It equals 1 for a circle and approaches 0 for a line.

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `filterByInertia` | true | Enable inertia ratio filtering. |
| `minInertiaRatio` | 0.1 | Minimum inertia ratio (inclusive). |
| `maxInertiaRatio` | FLT_MAX | Maximum inertia ratio (exclusive). |

### Filter by Convexity

Convexity = blob area / convex hull area. A perfectly convex blob gives 1.0; a star-shaped
or concave blob gives a smaller value.

| Parameter | Default | Description |
| :-------- | :-----: | :---------- |
| `filterByConvexity` | true | Enable convexity filtering. |
| `minConvexity` | 0.95 | Minimum convexity (inclusive). |
| `maxConvexity` | FLT_MAX | Maximum convexity (exclusive). |

Code Examples
-------------

### C++

@code{.cpp}
#include <opencv2/features.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

int main()
{
    // Load a grayscale image
    cv::Mat gray = cv::imread("blobs.png", cv::IMREAD_GRAYSCALE);

    // --- Configure detector ---
    cv::SimpleBlobDetector::Params params;

    // Detect dark blobs (blobColor = 0) on a light background
    params.filterByColor = true;
    params.blobColor     = 0;

    // Keep blobs with area between 100 and 5000 px²
    params.filterByArea  = true;
    params.minArea       = 100.0f;
    params.maxArea       = 5000.0f;

    // Require fairly circular blobs
    params.filterByCircularity = true;
    params.minCircularity      = 0.7f;

    // Disable the remaining filters
    params.filterByInertia   = false;
    params.filterByConvexity = false;

    // --- Create detector and run ---
    cv::Ptr<cv::SimpleBlobDetector> detector =
        cv::SimpleBlobDetector::create(params);

    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray, keypoints);

    // --- Visualize ---
    cv::Mat result;
    cv::drawKeypoints(gray, keypoints, result,
                      cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::imshow("Blobs", result);
    cv::waitKey(0);
    return 0;
}
@endcode

### Python

@code{.py}
import cv2

# Load a grayscale image
gray = cv2.imread("blobs.png", cv2.IMREAD_GRAYSCALE)

# --- Configure detector ---
params = cv2.SimpleBlobDetector_Params()

# Detect dark blobs on a light background
params.filterByColor = True
params.blobColor     = 0        # 0 = dark, 255 = light

# Keep blobs with area between 100 and 5000 px²
params.filterByArea = True
params.minArea      = 100
params.maxArea      = 5000

# Require fairly circular blobs
params.filterByCircularity = True
params.minCircularity      = 0.7

# Disable the remaining filters
params.filterByInertia   = False
params.filterByConvexity = False

# --- Create detector and run ---
detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)

# --- Visualize ---
result = cv2.drawKeypoints(
    gray, keypoints, None,
    (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Blobs", result)
cv2.waitKey(0)
@endcode

Saving and Loading Parameters
------------------------------

Parameters can be serialized to/from an OpenCV FileStorage (XML/YAML/JSON), which is useful
for tuning offline and loading at run-time:

@code{.cpp}
// Save
cv::FileStorage fs("blob_params.yml", cv::FileStorage::WRITE);
params.write(fs);
fs.release();

// Load
cv::FileStorage fs2("blob_params.yml", cv::FileStorage::READ);
cv::SimpleBlobDetector::Params loaded;
loaded.read(fs2.root());
fs2.release();

auto detector = cv::SimpleBlobDetector::create(loaded);
@endcode

Retrieving Blob Contours
-------------------------

If you need the actual contour of each detected blob (e.g., for precise shape analysis),
enable contour collection before calling detect():

@code{.cpp}
params.collectContours = true;
auto detector = cv::SimpleBlobDetector::create(params);
detector->detect(gray, keypoints);

// One contour per detected blob, in the same order as keypoints
const auto& contours = detector->getBlobContours();
for (size_t i = 0; i < contours.size(); i++) {
    cv::drawContours(result, contours,
                     static_cast<int>(i), cv::Scalar(0, 255, 0), 2);
}
@endcode

Tips and Common Pitfalls
--------------------------

-   **Dark vs. light blobs.** The default settings detect *dark* blobs (`blobColor = 0`).
    To detect light blobs, set `blobColor = 255` and adjust `minThreshold`/`maxThreshold`
    so that the blobs are captured as foreground regions.

-   **Thresholding range.** If `maxThreshold - minThreshold <= thresholdStep` there is only
    one binary image, `minRepeatability` is effectively 1, and `minDistBetweenBlobs` is
    ignored. Widen the range or decrease the step to use those parameters.

-   **Area units.** `minArea` and `maxArea` are in *pixels squared*, not pixels.

-   **Circularity near 1.** Due to discrete pixel boundaries, even round blobs rarely
    achieve circularity above ~0.9. Start with `minCircularity = 0.7` and tune from there.

-   **Performance.** Reduce the number of threshold levels (increase `thresholdStep` or
    narrow the `[minThreshold, maxThreshold)` range) to speed up detection on large images.
