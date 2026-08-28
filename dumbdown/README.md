# ELI5: Fix for findTransformECC() Scale Underestimation (Issue #29774)

## What is findTransformECC?

`findTransformECC` (Enhanced Correlation Coefficient) aligns two images of the same scene.
Given a "template" image and a "source" image that may be shifted, rotated, or scaled,
it finds the exact transformation matrix that maps one to the other.

It's used in video stabilization, panorama stitching, and medical image registration.

## The Bug

When estimating **scale** (how much one image is zoomed compared to the other), the result
was systematically off by about **-0.028%** — always a tiny bit too small.

This sounds tiny, but for precision alignment applications, a systematic bias is worse
than random noise because it never cancels out.

## Why did it happen?

The algorithm works by computing **image gradients** (how pixel values change across the
image). These gradients feed into the "Jacobian" — the matrix that drives the optimization.

Before computing gradients, the code **blurs** the template image with a Gaussian filter.
The border handling mode for this blur was `BORDER_REFLECT_101` (mirror padding):

```
Real image:      [ 10  20  30  40  50 ]
With mirror pad: [ 30  20 | 10  20  30  40  50 | 40  30 ]
                           ^                   ^
                        left edge           right edge
                        (reflected)         (reflected)
```

The mirrored edge pixels create **fake gradients** at the image borders.

For **translation**, the left-side fake gradient and right-side fake gradient are equal and
opposite — they cancel out. No bias.

For **scale**, the gradient contribution depends on DISTANCE FROM CENTER. Left and right
edges are symmetric in distance but their fake values push scale estimates in the SAME
direction (both make the image look "smaller"). They don't cancel — they ADD UP, causing
the systematic -0.028% underestimate.

## The Fix

Change border padding from `BORDER_REFLECT_101` to `BORDER_CONSTANT` (zero/black padding):

```
Real image:      [ 10  20  30  40  50 ]
With zero pad:   [  0   0 | 10  20  30  40  50 |  0   0 ]
                           ^                   ^
                        left edge           right edge
                        (zeros)             (zeros)
```

Zero-padded edges have zero gradient — they contribute NOTHING to the Jacobian.
No fake pixels, no systematic bias.

## ASCII Diagram

```
BORDER_REFLECT_101 (BEFORE - buggy):

Template:  | 10  20  30  40  50 |
           ^                   ^
Reflected: 20  10 | ... actual ... | 40  30

Gradient at left border uses REFLECTED value:
  grad_left = pixel[1] - reflected[-1] = 20 - 10 = +10  <-- FAKE

Gradient at right border uses REFLECTED value:
  grad_right = reflected[N] - pixel[N-1] = 40 - 50 = -10  <-- FAKE

For scale Jacobian (weights by x position):
  left contribution:  -x * grad_left  (x is negative, near left edge)
  right contribution: +x * grad_right (x is positive, near right edge)
  
  Both push scale LOWER. They add, not cancel.
  Result: scale estimate is too small by ~0.028%

BORDER_CONSTANT (AFTER - fixed):

Template:  | 10  20  30  40  50 |
           ^                   ^
Zero pad:   0   0 | ... actual ... |  0   0

Gradient at left border: 20 - 0 = real edge gradient
Gradient at right border: 0 - 40 = real edge gradient

Zero padding adds no FAKE structure.
No systematic bias in scale estimate.
```

## What was changed?

In `modules/video/src/ecc.cpp`, the `GaussianBlur` call for the template image changed:

```cpp
// BEFORE:
GaussianBlur(templateImage, blurred, Size(0,0), sigma,
             sigma, BORDER_REFLECT_101);

// AFTER:
GaussianBlur(templateImage, blurred, Size(0,0), sigma,
             sigma, BORDER_CONSTANT);
```

## Real-world impact

Any application using `MOTION_EUCLIDEAN` or `MOTION_HOMOGRAPHY` warp modes that include
scale in findTransformECC would get a systematic -0.028% scale error per alignment step.
In iterative pipelines (video stabilization, template tracking), this small error can
accumulate over time, causing slowly drifting zoom artifacts.
