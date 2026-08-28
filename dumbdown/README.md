# Fix: Fisheye Camera Math Produces NaN (Issue #23961)

## ELI5 Explanation

Fisheye lenses take extremely wide-angle photos by bending light like — you
guessed it — a fish eye. The image looks curved and distorted. To undo this
bending in software (called **undistortion**), OpenCV uses a mathematical trick
called **Newton's method**.

Newton's method works like this: start with a guess, then keep improving the
guess step by step until it is close enough to the right answer. Each improvement
step divides by the "slope" of the curve at the current guess.

**The bug:** For certain lens distortion settings, the slope at the starting point
is **exactly zero** — like a perfectly flat hilltop. Dividing by zero gives
infinity. Then doing more math on infinity gives **NaN** (Not a Number) — a
special value that means "this number is broken". NaN then contaminates every
downstream calculation.

**The fix:** Before doing the division, check if the slope is near zero.
If it is, skip the step and keep the current guess as-is. This is safe
because a near-zero slope means the improvement would be wildly off anyway.

---

## What Fisheye Undistortion Does

A fisheye lens maps a 3D angle theta (angle from the optical axis) to a 2D
image radius r. The undistortion step inverts this mapping: given a pixel's
image radius r, find the original angle theta. Newton's method solves this
iteratively.

The distortion model: `r = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9`

The derivative (slope) is: `dr/dtheta = 1 + 3*k1*theta^2 + 5*k2*theta^4 + ...`

When `k1 = -1/3` and `theta = 1`, the slope evaluates to exactly `0`.

---

## ASCII Diagram

```
NEWTON'S METHOD ITERATION:

  f(theta) curve:

    |      /
    |     / <-- normal slope: step is small & safe
    |    .
    |   /
    |--/---------- theta axis
       theta_0

  -------------------------------------------------------

  BUG: flat tangent (slope = 0):

    |
    |....._____.....  <-- slope = 0 (flat) at theta=1
    |       ^
    |       |
    |    theta_0 = 1.0 (with k1 = -1/3)

  Newton step: theta_new = theta - f(theta) / f'(theta)
                         = theta - f(theta) / 0
                         = theta - (+/- infinity)
                         = +/- infinity

  Next iteration: math on infinity --> NaN

  NaN spreads to:  undistorted pixel coords --> output image --> garbage

  -------------------------------------------------------

  FIX: slope safety check

    if |f'(theta)| < epsilon:
        skip this Newton step
        keep theta unchanged
    else:
        theta = theta - f(theta) / f'(theta)

  Result: iteration stalls for one step, then converges safely.
  No infinity. No NaN.

SLOPE CONDITION that triggers the bug:
  k1 = -1/3, theta = 1.0:
  dr/dtheta = 1 + 3 * (-1/3) * 1^2 = 1 - 1 = 0  <-- EXACTLY ZERO
```

---

## Root Cause (Technical)

In `cv::fisheye::undistortPoints()`, Newton's method iterates:

```
theta_new = theta - f(theta) / f_prime(theta)
```

where `f(theta) = r - (theta + k1*theta^3 + ...)` and
`f_prime(theta) = 1 + 3*k1*theta^2 + 5*k2*theta^4 + ...`.

When `f_prime` evaluates to exactly or near zero, division produces `Inf`,
and subsequent operations produce `NaN`. The fix adds:

```cpp
if (std::abs(f_prime) < epsilon)
    continue;   // skip this iteration step
```

This prevents the runaway division and lets the iteration recover or converge
to the nearest valid solution.
