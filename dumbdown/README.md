# ELI5: Subdiv2D Delaunay Triangulation Bug (Issue #16763)

## What is Delaunay Triangulation?

Imagine you have a bunch of dots on a piece of paper. Delaunay triangulation
connects those dots with triangles — but with a special rule: **no dot is
allowed to sit inside the circle that perfectly wraps around any triangle**.

That "perfect circle" is called a **circumcircle** — the one unique circle
that passes through all three corners of a triangle.

```
    circumcircle
      .......
    .         .
   .   A-----B  .
   .   |    /   .
   .   |   /    .
   .   |  /     .
   .   | /      .
   .   |/       .
    .  D      .
      .......
  (circle through A, B, D)
```

---

## The Bug: What Went Wrong?

Consider **4 points forming a perfect square**:

```
  A---B
  |   |
  D---C
```

You can split this square into 2 triangles in exactly **two ways**:

```
  Option 1        Option 2
  A---B            A---B
  |\ |             | /|
  | \|             |/ |
  D---C            D---C
  (AC diagonal)   (BD diagonal)
```

Here is the key insight: for a perfect square, all **4 corners lie exactly on
the same circle**. Both splits are mathematically equally valid — neither is
"more correct" than the other.

The algorithm decides which diagonal to keep by asking:
> "Is point D inside the circumcircle of triangle ABC?"

In math, the answer is a **determinant**. For a perfect square:
- The answer is **exactly zero** (D is exactly ON the circle, not inside it).
- Zero means "don't flip — the current triangulation is fine."

---

## The Problem: Floating-Point Rounding

Computers cannot store every decimal perfectly. Pi is not exactly
3.14159... in a computer — it gets rounded. Same thing happens with square
coordinates.

When the algorithm computes that determinant, instead of getting clean zero,
it gets a tiny number like:

```
  Expected:   0.000000000000000
  Got:       +0.000000000000002   <-- slightly positive  (flip!)
    or:      -0.000000000000001   <-- slightly negative  (don't flip)
```

Depending on which way the rounding went, the algorithm would:
- Sometimes flip the diagonal (giving 2 triangles — correct!)
- Sometimes NOT flip (giving only 1 triangle, or a broken mesh — WRONG!)

The result was **non-deterministic** and input-dependent: the same 4 points
could produce different (wrong) results on different machines or compilers.

---

## The Fix: Add a Tiny Tolerance

```
  Instead of:   if determinant > 0  → flip
  Now:          if determinant > epsilon  → flip
                (where epsilon ≈ machine rounding noise)
```

If the determinant is within a tiny tolerance band around zero, the algorithm
says "close enough to zero — don't flip, accept what we have."

```
  ← flip | don't flip | flip →
  --------[----0----]---------
           -ε     +ε
           tolerance band
```

This makes the algorithm **stable**: degenerate cases (all on one circle) no
longer cause random wrong answers.

---

## ASCII Summary

```
4 points forming a square:    Both triangulations are valid:
  A---B                         A---B    A---B
  |   |           -->           |  /|    |\  |
  D---C                         | / |    | \ |
                                 |/  |    |  \|
                                 D---C    D---C
                                (either is correct)

All 4 lie on ONE circle --> determinant = 0 in theory
Floating-point gives +/-epsilon --> old code: random flip decision

THE FIX:
  |det| < epsilon  -->  treat as 0  -->  stable result
```

---

## Files Changed

- `modules/imgproc/src/subdivision2d.cpp` — the `isRightOf` / in-circle test
  now includes an epsilon tolerance instead of a strict `> 0` comparison.

## Why This Matters

Subdiv2D is used for 3D face alignment, optical flow initialization, and
anywhere you need to mesh a point cloud. A random "1 triangle for 4 points"
bug could silently corrupt downstream geometry code.
