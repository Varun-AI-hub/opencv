# Fix: DNN Layer Fusion Deletes Needed Intermediate Results (Issue #17944)

## ELI5 (Explain Like I'm 5)

When running a neural network, OpenCV combines multiple operations into one
for speed — this is called **fusion**. For example:

```
Conv → BatchNorm → ReLU
```
gets fused into a single pass instead of three.

**The bug**: if another part of the network also needs the BatchNorm output,
fusion would erase it — like a **chef combining prep steps** and accidentally
throwing away an ingredient that was needed for a different dish.

**The fix**: before fusing, check "does anyone else need this intermediate
result?" If yes, don't fuse those layers.

## What broke

Network branches that used an intermediate layer's output (e.g. BatchNorm)
as input to **two different places** in the network would get garbage or
empty output on one branch after fusion.

## ASCII diagram

```
Network graph:

       [Conv]
         |
       [BN]  <--- also used by branch B
         |
       [ReLU]
         |
      Output A       Output B (from BN)

BEFORE FIX:
  Conv+BN+ReLU fused --> Output B is EMPTY / garbage  X

AFTER FIX:
  BN has multiple consumers --> fusion blocked
  Conv+BN kept separate --> Output B preserved  ✓
```

## The technical fix

The fusion logic now counts how many downstream layers consume each
intermediate output. Fusion is only allowed when the intermediate result
has exactly **one consumer** — ensuring no branch loses its input.

## Files changed

- `modules/dnn/src/net.cpp` (fusion eligibility check)
