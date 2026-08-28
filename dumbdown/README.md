# ELI5: warpAffine INTER_AREA Ignored (Issue #21060)

## What is warpAffine?

`warpAffine()` moves, rotates, scales, or shears an image using a
mathematical transformation matrix. It is like applying a rubber-sheet
stretch or squish to a photo.

```
  Source image              After warpAffine (scale 0.5x)
  +----------+              +-----+
  |          |    ---->     |     |
  |  100x100 |              |50x50|
  |          |              |     |
  +----------+              +-----+
```

---

## What is INTER_AREA?

When you **shrink** an image, each output pixel must represent several source
pixels. There are different strategies for doing this:

### INTER_LINEAR (the fast but imperfect way)

Just look up the single nearest source pixel (or blend 2x2 at most).
Fast, but can miss pixels entirely, causing **aliasing** — a jagged,
checkerboard-like artifact.

```
  Source (4 pixels)    Output (1 pixel)
  +---+---+
  | A | B |           Output = just A
  +---+---+    -->    (B, C, D ignored!)
  | C | D |
  +---+---+
         Aliasing possible!
```

### INTER_AREA (the correct way for downscaling)

Average ALL the source pixels that map to the output pixel, weighted by
how much of each source pixel the output pixel covers.

```
  Source (4 pixels)    Output (1 pixel)
  +---+---+
  | A | B |           Output = (A + B + C + D) / 4
  +---+---+    -->
  | C | D |           Smooth, no aliasing!
  +---+---+
```

This is like making a smoothie: blend all the fruit, not just pick one.

---

## The Bug: warpAffine Silently Used the Wrong Method

When you called:

```cpp
warpAffine(src, dst, M, dsize, INTER_AREA);
```

OpenCV was **secretly ignoring** `INTER_AREA` and using `INTER_LINEAR`
instead. No error. No warning. Wrong output — and nobody knew.

```
  What you asked for:   INTER_AREA  (smoothie)
  What you got:         INTER_LINEAR (single fruit) ← BUG
```

The reason: `warpAffine` uses a general remapping engine internally, and
INTER_AREA in that engine only works for simple resize, not arbitrary
affine transforms. Rather than implementing full area sampling, the code
silently fell back to LINEAR.

---

## The Fix: Two Parts

### Part 1 — Clear Documentation Warning

The docs now explicitly state that `INTER_AREA` is not supported in
`warpAffine()` and describe what happens instead.

### Part 2 — New Helper: `warpAffineAntiAliased()`

A new convenience function that properly approximates INTER_AREA for
affine warps:

```
  Step 1: Apply a Gaussian blur to the source image
          (this pre-mixes the pixels that will be merged)

  Step 2: Run normal warpAffine() with INTER_LINEAR

  Result: The blur approximates area averaging
          --> smooth downscaling without aliasing
```

```
  WITHOUT (old INTER_AREA / actually LINEAR):
  Source pixels  -->  warpAffine  -->  jagged output

  WITH warpAffineAntiAliased():
  Source pixels  -->  GaussianBlur  -->  warpAffine  -->  smooth output
                      (pre-blend)       (resample)
```

The Gaussian blur kernel size is chosen based on the scale factor of the
affine matrix: more downscaling = larger blur.

---

## ASCII Summary

```
  Downscale 2x: each output pixel covers 4 source pixels

  INTER_LINEAR (what warpAffine was actually doing):
  +--+--+
  |XX|  |    Output = XX only   <-- aliasing!
  +--+--+
  |  |  |
  +--+--+

  warpAffineAntiAliased (the fix):
  +--+--+
  |**|**|    Blur first:  all 4 mixed  --> then sample
  +--+--+                 Output = smooth average
  |**|**|
  +--+--+
```

---

## Files Changed

- `modules/imgproc/src/imgwarp.cpp` — added `warpAffineAntiAliased()`
- `modules/imgproc/include/opencv2/imgproc.hpp` — public API declaration
- `doc/` — documentation warning added to `warpAffine()`

## Why This Matters

Silent wrong behavior is worse than an error. Anyone using `INTER_AREA`
with `warpAffine()` (e.g., for thumbnail generation, image pyramids, or
ML preprocessing) was unknowingly getting aliased results without any
indication that their flag was being ignored.
