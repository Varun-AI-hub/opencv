# Fix: cv::dilate() Crash with CV_Bool Kernel (Issue #29798)

## ELI5 Explanation

Think of **dilation** like using a rubber stamp to spread ink on a page.

The **stamp shape** (called a kernel or structuring element) tells the software
*where* the ink should spread. For example, a plus-shaped stamp spreads ink
in a cross pattern.

The software accepted stamps made of regular paper (type `CV_8U`, values 0 or 255).
But if you handed it a stamp made of special **boolean paper** (type `CV_Bool`,
values `true` or `false`) — even though it looks and works *exactly the same way* —
the software would **crash**.

**The fix:** If someone gives you a boolean-paper stamp, silently convert it to
regular-paper first, then proceed normally. Both stamps do the same job.

---

## What Dilation Does

Dilation expands bright regions in an image. For every pixel in the output,
it looks at all neighbors defined by the kernel shape. If *any* of those neighbors
is bright (non-zero), the output pixel is bright too.

```
Input:     Kernel (3x3 cross):    Output (dilated):
. . . . .   . X .                 . X . . .
. . X . .   X X X                 . X X X .
. . . . .   . X .                 . X . . .
                                  . . . . .
```

---

## ASCII Diagram

```
BEFORE FIX:
                    +------------------+
  Source image ---> |                  |
                    |  cv::dilate()    | <--- CV_Bool kernel --> CRASH! [X]
                    |                  |
                    +------------------+

AFTER FIX:
                    +------------------+
  Source image ---> |                  |
                    |  cv::dilate()    | <--- CV_Bool kernel
                    |  (auto-converts  |       |
                    |   CV_Bool to     |       v
                    |   CV_8U first)   |  [converted to CV_8U]
                    +------------------+       |
                           |                   v
                           +---------> Correct dilated output [OK]

CV_Bool kernel:    CV_8U equivalent:
[ T  F  T ]        [ 255   0  255 ]
[ F  T  F ]   ==>  [   0 255    0 ]
[ T  F  T ]        [ 255   0  255 ]
   (same shape, same effect, just different type label)
```

---

## Root Cause (Technical)

Inside `cv::dilate()`, the kernel type was checked against a whitelist of supported
types. `CV_Bool` was not in the whitelist, so the function threw an error or
crashed before even starting. The fix adds a conversion step: if the kernel type
is `CV_Bool`, cast it to `CV_8U` (where `true` becomes 255, `false` becomes 0)
before handing it to the core algorithm.
