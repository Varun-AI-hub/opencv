# Fix: connectedComponents() Phantom Background Group (Issue #17902)

## ELI5 Explanation

Imagine you are in a room with 5 people, and everyone is wearing a **red shirt**.
Your job is to count how many different shirt-color groups exist.

The answer is **1 group** — everyone is wearing red.

But the old software was also counting an imaginary group of people in **invisible shirts** —
a group that does not exist at all. So it reported **2 groups** instead of 1.

**The fix:** Stop counting the invisible shirt group. If the entire image is one color,
there is only 1 group, and that is all we should report.

---

## What connectedComponents() Does

It scans an image and finds all connected blobs of the same color, then labels each blob
with a number. The background (black, value=0) traditionally gets label 0, and the
first real group gets label 1.

The bug: on an all-white image, the code assigned label 0 to "background" (which didn't
exist), then label 1 to the white region — and counted 2 labels. The correct answer is 1.

---

## ASCII Diagram

```
INPUT IMAGE (all white):         CORRECT OUTPUT:
+-------+                        +-------+
| W W W |    connectedComponents  | 1 1 1 |    --> count = 1
| W W W |  -------------------->  | 1 1 1 |
| W W W |                        | 1 1 1 |
+-------+                        +-------+

BUGGY OUTPUT (before fix):
+-------+
| 1 1 1 |    --> BUG: count = 2
| 1 1 1 |         (phantom "group 0" background reported even though it doesn't exist)
| 1 1 1 |
+-------+

Groups counted:
  Before fix:  [phantom background] + [white region] = 2  <-- WRONG
  After fix:   [white region]                         = 1  <-- CORRECT
```

---

## Root Cause (Technical)

The function iterated over labels and included label 0 (reserved for background) in the
count even when no background pixels existed. The fix ensures the returned count only
includes labels that are actually present in the image.
