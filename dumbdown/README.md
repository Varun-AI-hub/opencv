# Fix: SimpleBlobDetector Wrong Center for Bright Blobs (Issue #24388)

## ELI5 Explanation

Imagine you draw a **white circle** on **black paper**. You want to find the
exact center of that circle.

The correct way: measure the geometric center — the point equally far from
all edges of the white area.

The old code worked correctly for **dark blobs on a white background** (black circle
on white paper), but when you flipped it — **bright blobs on a dark background**
(white circle on black paper, `blobColor=255`) — it used a subtly different
measuring rule that shifted the center by a few pixels.

It is like measuring a circle from the right edge in one case, and from the left
edge in the other case — you get slightly different numbers even for the same shape.

**The fix:** Use the exact same measuring rule for both cases. The center
calculation is now symmetric regardless of whether you are looking for bright or
dark blobs.

---

## What SimpleBlobDetector Does

It finds circular (or near-circular) regions in an image, and for each "blob" it
reports the center coordinates and size. This is used in robotics, medical imaging,
and camera calibration.

---

## ASCII Diagram

```
DARK BACKGROUND, WHITE BLOB (blobColor=255):

  . . . . . . . . .
  . . . W W W . . .
  . . W W W W W . .
  . . W W X W W . .   <-- X = reported center
  . . W W W W W . .
  . . . W W W . . .
  . . . . . . . . .

BEFORE FIX:                    AFTER FIX:
  Old center:  (58, 37)          Correct center: (60, 40)
        |                               |
        v                               v
  . . . . . . . . .           . . . . . . . . .
  . . . W W W . . .           . . . W W W . . .
  . . W X W W W . .   WRONG   . . W W W W W . .
  . . W W W W W . .   ---->   . . W W X W W . .   CORRECT
  . . W W W W W . .           . . W W W W W . .
  . . . W W W . . .           . . . W W W . . .
  . . . . . . . . .           . . . . . . . . .

The shifted center (X moved up-left) was caused by an asymmetric threshold
inversion step that only affected the blobColor=255 code path.
```

---

## Root Cause (Technical)

When `blobColor=255`, the detector inverted the thresholded binary image before
computing blob moments. The inversion flipped which pixels were counted as
foreground, causing the moment calculation to compute the centroid of the
*complement* of the blob rather than the blob itself. The fix removes the
asymmetric inversion so both `blobColor=0` and `blobColor=255` paths feed
the same foreground mask into the centroid formula.
