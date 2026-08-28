# Fix: RGB Odometry Assertion Crash (Issue #23103)

## ELI5 (Explain Like I'm 5)

**Visual odometry** is like a robot figuring out how it moved by looking at
two photos taken a second apart — "I must have moved 10 cm to the right!"

There are **3 modes**:
- **RGB mode**: use only color photos
- **DEPTH mode**: use only depth sensor
- **RGB+DEPTH mode**: use both

**The bug**: even in color-only mode, the code demanded a depth image and
crashed if none was given.

It's like a **car GPS refusing to start** unless you also plug in a compass,
even if you didn't ask for compass navigation.

## What broke

```
OpenCV Error: Assertion failed (!depth.empty())
```

Calling `odometry.compute(frame1, frame2)` in RGB-only mode without providing
a depth image triggered an assertion that should never fire in that mode.

## ASCII diagram

```
Odometry modes:

  [RGB mode]        [DEPTH mode]      [RGB+DEPTH mode]
  color only        depth only        both required
      |                 |                   |
      v                 v                   v
  BEFORE FIX:      works fine          works fine
  assert(!depth    
   .empty())  
  --> CRASH! X    

  AFTER FIX:       works fine          works fine
  depth check
  skipped  ✓
```

## The technical fix

The assertion `CV_Assert(!depth.empty())` fired unconditionally.
The fix wraps it so it only runs when the odometry type includes `DEPTH`:

```cpp
// Before:
CV_Assert(!depth.empty());

// After:
if (type == OdometryType::DEPTH || type == OdometryType::RGB_DEPTH)
    CV_Assert(!depth.empty());
```

## Files changed

- `modules/rgbd/src/odometry.cpp`
