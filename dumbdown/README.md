# ELI5: Fix for stereoRectify() ROI Off-by-One (Issue #7240)

## What is stereo rectification?

When a stereo camera (like two eyes) captures two pictures, the same real-world point
appears at slightly different spots in each image. Rectification "lines up" both images
so that matching points always appear on the SAME horizontal line. This makes finding
matches much easier — you only have to search left-right instead of everywhere.

## What does `alpha=1` mean?

The `alpha` parameter controls how much of each image to keep after rectification:

- `alpha=0` → crop to only the region both cameras share (smaller image, no black edges)
- `alpha=1` → keep ALL valid pixels from both cameras (larger image, might have black corners)

The returned `validPixROI` rectangle tells you exactly which pixels are "good" (not black,
not distorted) when `alpha=1`.

## The Bug (before the fix)

```
Sensor pixels: [0 .. 639]
                |___________________________|
                floor(0.1) = 0              floor(639.9) = 639   <-- correct

BUT THE CODE DID:
                floor(left edge)  and  floor(right edge)
                floor(0.1) = 0    and  floor(639.9) = 639  <-- right edge got floored!

This makes the ROI width = 639 - 0 = 639 instead of 640.
One pixel lost on the right!
```

The fix: use `ceil()` on the right/bottom edges instead of `floor()`.

## ASCII Diagram

```
BEFORE FIX (alpha=1):

  Full sensor pixels
  +------------------------------------------+
  |                                          |
  | +--------------------------------------+ |  <-- Computed ROI
  | |  floor both ends = too narrow by 1  | |      (1 px gap each side)
  | |                                      | |
  | +--------------------------------------+ |
  |                                          |
  +------------------------------------------+
    ^                                      ^
    gap (1 px)                          gap (1 px)

AFTER FIX (alpha=1):

  Full sensor pixels
  +------------------------------------------+
  |                                          |
  +------------------------------------------+  <-- Fixed ROI
  |  floor left, ceil right = exact fit      |      (no gap)
  +------------------------------------------+
```

## What was changed?

In `modules/calib3d/src/calibration.cpp`, the ROI computation changed from:

```cpp
// BEFORE: floor on both sides
roi.x      = cvFloor(inner.x);
roi.width  = cvFloor(inner.x + inner.width)  - roi.x;
roi.y      = cvFloor(inner.y);
roi.height = cvFloor(inner.y + inner.height) - roi.y;
```

```cpp
// AFTER: floor on start, ceil on end
roi.x      = cvFloor(inner.x);
roi.width  = cvCeil(inner.x + inner.width)   - roi.x;
roi.y      = cvFloor(inner.y);
roi.height = cvCeil(inner.y + inner.height)  - roi.y;
```

## Real-world impact

If you were using `validPixROI` to mask or crop your stereo images, you were silently
throwing away 1-2 pixel columns/rows around the edges. For a 640x480 camera this is
~0.2% of the image area — small but enough to cause issues in tight calibration workflows.
