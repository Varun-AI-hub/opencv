// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {
TEST(Features2d_BlobDetector, bug_6667)
{
    cv::Mat image = cv::Mat(cv::Size(100, 100), CV_8UC1, cv::Scalar(255, 255, 255));
    cv::circle(image, Point(50, 50), 20, cv::Scalar(0), -1);
    SimpleBlobDetector::Params params;
    params.minThreshold = 250;
    params.maxThreshold = 260;
    params.minRepeatability = 1;  // https://github.com/opencv/opencv/issues/6667
    std::vector<KeyPoint> keypoints;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);
    ASSERT_NE((int) keypoints.size(), 0);
}

TEST(Features2d_BlobDetector, withContours)
{
    cv::Mat image = cv::Mat(cv::Size(100, 100), CV_8UC1, cv::Scalar(255, 255, 255));
    cv::circle(image, Point(50, 50), 20, cv::Scalar(0), -1);
    SimpleBlobDetector::Params params;
    params.minThreshold = 250;
    params.maxThreshold = 260;
    params.minRepeatability = 1; // https://github.com/opencv/opencv/issues/6667
    params.collectContours = true;
    std::vector<KeyPoint> keypoints;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);
    ASSERT_NE((int)keypoints.size(), 0);

    ASSERT_GT((int)detector->getBlobContours().size(), 0);
    std::vector<Point> contour = detector->getBlobContours()[0];
    ASSERT_TRUE(std::any_of(contour.begin(), contour.end(),
                            [](Point p)
                            {
                                return abs(p.x - 30) < 2 && abs(p.y - 50) < 2;
                            }));
}
// Regression test for https://github.com/opencv/opencv/issues/24388
// SimpleBlobDetector should report the same blob center whether the blob is dark
// on a bright background (blobColor=0) or bright on a dark background (blobColor=255,
// the "inverted" image scenario).
TEST(Features2d_BlobDetector, issue_24388_inverted_center)
{
    const int IMG_SIZE = 100;
    const Point blobCenter(60, 40);
    const int blobRadius = 15;
    const double centerTolerance = 1.5; // pixels

    // --- normal image: dark blob on white background, blobColor=0 ---
    cv::Mat normalImage(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(255));
    cv::circle(normalImage, blobCenter, blobRadius, cv::Scalar(0), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 5000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> kpNormal;
    Ptr<SimpleBlobDetector> detNormal = SimpleBlobDetector::create(params);
    detNormal->detect(normalImage, kpNormal);
    ASSERT_EQ((int)kpNormal.size(), 1) << "Normal image: expected exactly one blob";
    EXPECT_NEAR(kpNormal[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Normal image (blobColor=0): X center is wrong";
    EXPECT_NEAR(kpNormal[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Normal image (blobColor=0): Y center is wrong";

    // --- inverted image: bright blob on dark background, blobColor=255 ---
    cv::Mat invertedImage(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(0));
    cv::circle(invertedImage, blobCenter, blobRadius, cv::Scalar(255), -1);

    params.blobColor = 255;
    std::vector<KeyPoint> kpInverted;
    Ptr<SimpleBlobDetector> detInverted = SimpleBlobDetector::create(params);
    detInverted->detect(invertedImage, kpInverted);
    ASSERT_EQ((int)kpInverted.size(), 1) << "Inverted image: expected exactly one blob";
    EXPECT_NEAR(kpInverted[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Inverted image (blobColor=255): X center is wrong (issue #24388)";
    EXPECT_NEAR(kpInverted[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Inverted image (blobColor=255): Y center is wrong (issue #24388)";

    // Both detections should agree with each other.
    EXPECT_NEAR(kpNormal[0].pt.x, kpInverted[0].pt.x, centerTolerance)
        << "Centers for normal vs inverted image disagree on X";
    EXPECT_NEAR(kpNormal[0].pt.y, kpInverted[0].pt.y, centerTolerance)
        << "Centers for normal vs inverted image disagree on Y";
}
// issue_24388: multiple blobs in an inverted image should all be detected at correct positions
TEST(Features2d_BlobDetector, issue_24388_multiple_blobs_inverted)
{
    const int IMG_SIZE = 150;
    const double centerTolerance = 2.0;

    // Three distinct circle positions
    const Point centers[3] = { Point(30, 30), Point(80, 40), Point(60, 110) };
    const int blobRadius = 12;

    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 3; ++i)
        cv::circle(image, centers[i], blobRadius, cv::Scalar(255), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 255;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 5000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);

    ASSERT_EQ((int)keypoints.size(), 3) << "Expected exactly 3 blobs in inverted image";

    // For each ground-truth center, find the closest detected keypoint and check distance
    for (int i = 0; i < 3; ++i)
    {
        double minDist = std::numeric_limits<double>::max();
        for (const auto& kp : keypoints)
        {
            double dx = kp.pt.x - centers[i].x;
            double dy = kp.pt.y - centers[i].y;
            double dist = std::sqrt(dx * dx + dy * dy);
            if (dist < minDist) minDist = dist;
        }
        EXPECT_LE(minDist, centerTolerance)
            << "Blob " << i << " at (" << centers[i].x << "," << centers[i].y
            << ") not detected within tolerance";
    }
}

// issue_24388: blob exactly at the image center should be detected symmetrically
TEST(Features2d_BlobDetector, issue_24388_center_at_image_center)
{
    const int IMG_SIZE = 100;
    const Point blobCenter(50, 50);
    const int blobRadius = 15;
    const double centerTolerance = 1.0;

    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(0));
    cv::circle(image, blobCenter, blobRadius, cv::Scalar(255), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 255;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 5000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);

    ASSERT_EQ((int)keypoints.size(), 1) << "Expected exactly one blob at image center";
    EXPECT_NEAR(keypoints[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Center blob X is wrong (issue #24388)";
    EXPECT_NEAR(keypoints[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Center blob Y is wrong (issue #24388)";
}

// issue_24388: small blob (radius=5) should be detected at the correct position
TEST(Features2d_BlobDetector, issue_24388_small_blob)
{
    const int IMG_SIZE = 100;
    const Point blobCenter(30, 30);
    const int blobRadius = 5;
    const double centerTolerance = 2.0;

    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(0));
    cv::circle(image, blobCenter, blobRadius, cv::Scalar(255), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 255;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 5000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);

    ASSERT_EQ((int)keypoints.size(), 1) << "Expected exactly one small blob";
    EXPECT_NEAR(keypoints[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Small blob X center is wrong (issue #24388)";
    EXPECT_NEAR(keypoints[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Small blob Y center is wrong (issue #24388)";
}

// issue_24388: large blob (radius=25) should be detected at the correct position
TEST(Features2d_BlobDetector, issue_24388_large_blob)
{
    const int IMG_SIZE = 120;
    const Point blobCenter(50, 50);
    const int blobRadius = 25;
    const double centerTolerance = 1.5;

    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(0));
    cv::circle(image, blobCenter, blobRadius, cv::Scalar(255), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 255;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 10000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);

    ASSERT_EQ((int)keypoints.size(), 1) << "Expected exactly one large blob";
    EXPECT_NEAR(keypoints[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Large blob X center is wrong (issue #24388)";
    EXPECT_NEAR(keypoints[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Large blob Y center is wrong (issue #24388)";
}

// Regression guard: blobColor=0 (dark blob on bright background) must still work correctly
// after the THRESH_BINARY_INV change introduced to fix issue #24388.
TEST(Features2d_BlobDetector, issue_24388_blobcolor0_unchanged)
{
    const int IMG_SIZE = 100;
    const Point blobCenter(60, 40);
    const int blobRadius = 15;
    const double centerTolerance = 1.5;

    // Dark blob on white background
    cv::Mat image(IMG_SIZE, IMG_SIZE, CV_8UC1, cv::Scalar(255));
    cv::circle(image, blobCenter, blobRadius, cv::Scalar(0), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 220;
    params.thresholdStep = 10;
    params.minRepeatability = 1;
    params.filterByColor = true;
    params.blobColor = 0;
    params.filterByArea = true;
    params.minArea = 100;
    params.maxArea = 5000;
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;

    std::vector<KeyPoint> keypoints;
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);

    ASSERT_EQ((int)keypoints.size(), 1)
        << "Regression (blobColor=0): expected exactly one dark blob after fix";
    EXPECT_NEAR(keypoints[0].pt.x, (float)blobCenter.x, centerTolerance)
        << "Regression (blobColor=0): X center is wrong after THRESH_BINARY_INV change";
    EXPECT_NEAR(keypoints[0].pt.y, (float)blobCenter.y, centerTolerance)
        << "Regression (blobColor=0): Y center is wrong after THRESH_BINARY_INV change";
}

}} // namespace
