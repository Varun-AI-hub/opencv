# Fix: HashTSDF fetchPointsNormals Too Many Points (Issue #22896)

## ELI5 (Explain Like I'm 5)

Imagine scanning a **3D sculpture** with a depth camera. The scan builds a
3D map where each tiny cube (voxel) stores a signed number:

- **Positive** (+) = outside the sculpture
- **Zero** (0) = exactly on the surface
- **Negative** (-) = inside the sculpture

To get the sculpture's **surface**, you should only pick cubes where the
number **changes from positive to negative** (the zero-crossing boundary).

**The bug**: the code was picking ALL cubes that had any number — so you
got a thick blob of points instead of a clean surface outline.

**The fix**: only pick cubes at the positive-to-negative boundary.

## ASCII cross-section diagram

```
Cross-section through a sphere (voxel values):

  [ +0.8 ][ +0.4 ][ 0.0 ][ -0.4 ][ -0.8 ]
                         ^
                         | zero-crossing = surface

  BEFORE FIX (all voxels included):
  [XXXXXX][XXXXXX][XXXXX][XXXXXX][XXXXXX]   <-- thick blob

  AFTER FIX (only boundary):
  [      ][      ][ *** ][      ][      ]   <-- clean surface
```

## The technical fix

`fetchPointsNormals()` iterated over all filled voxels. The fix adds a
zero-crossing check: only emit a point when the current voxel has a positive
TSDF value and at least one neighbor has a negative TSDF value.

## Files changed

- `modules/3d/src/hash_tsdf.cpp`
