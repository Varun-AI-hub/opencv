# ELI5: Fix for solvePnP() Behind-Camera Pose (Issue #8813)

## What is solvePnP?

`solvePnP` answers the question: "Given that I can see these 2D points in my camera image,
and I know where the corresponding 3D points are in the real world — where is my camera
and which way is it pointing?"

The output is a **rotation** (which way the camera faces) and a **translation** (where the
camera is in 3D space).

## The Two-Solution Problem

Mathematically, for some sets of points the equations have **two valid solutions**:
1. The real solution: object is IN FRONT of the camera (positive depth, z > 0)
2. A "mirror flip": object is mathematically BEHIND the camera (negative depth, z < 0)

The second solution is physically impossible — your camera can't see things behind it —
but the math produces it anyway because flipping the rotation by 180° and negating the
translation also satisfies the projection equations.

## The Bug (before the fix)

The algorithm didn't check which solution it returned. Sometimes it picked the impossible
"behind-camera" solution, giving you a perfectly valid-looking pose that was physically
nonsensical.

```
Camera is pointing --->

WRONG answer:          CORRECT answer:

[Object]  [Camera]-->  [Camera]-->  [Object]
    ^                                   ^
    |                                   |
    Behind camera!                   In front (z > 0)
    z < 0  BAD                       z > 0  GOOD
```

## The Fix

After computing the pose, check the **depth (z-coordinate)** of every input 3D point
when projected through the solution. If ALL of them come out with z < 0 (behind camera),
flip the solution:

```python
# Pseudocode of the check
for each 3D point P:
    depth = (R * P + t).z
    if depth < 0:
        count_behind += 1

if count_behind == total_points:
    # Flip: negate rotation axis and translation
    rvec = -rvec
    tvec = -tvec
```

## ASCII Diagram

```
solvePnP returns TWO mathematically valid solutions:

Solution A (WRONG sometimes chosen):        Solution B (CORRECT):

    z-axis of camera                            z-axis of camera
         |                                           |
         v (pointing right)                          v (pointing right)
         
  [Obj] [CAM]-->                           [CAM]--> [Obj]
  
  Object depth = -2.5m                     Object depth = +2.5m
  (BEHIND camera, impossible)              (IN FRONT, physically real)

Fix: after solving, check all point depths.
     If all negative -> swap to other solution.
```

## What was changed?

In `modules/calib3d/src/solvepnp.cpp`, after the core PnP solve, a depth-sign check
was added. If the computed translation places all known points behind the camera plane
(z < 0 in camera coordinates), the rotation vector is negated and translation is negated
to select the physically valid solution.

## Real-world impact

Applications using solvePnP for AR overlay, robot arm calibration, or camera pose
estimation could silently get a pose that was geometrically correct but physically
impossible — placing virtual objects on the wrong side of the camera or causing robot
commands to flip 180°.
