# Fix: DNN ONNX Bool Tensor Support (Issue #19366)

## ELI5 (Explain Like I'm 5)

Neural networks sometimes use **true/false flags** — like gates that say "use this value" or "skip it".

When loading a model file (ONNX format), OpenCV knew how to read numbers
(floats, ints) but **didn't know how to read true/false values**.

It's like a translator who knows Spanish and French but panics when given
a sentence in Italian.

**The fix** teaches OpenCV to also read boolean (true/false) data.

## What broke

```
Error: Unsupported data type: BOOL
```

ONNX models that contained boolean tensors (output of comparison operators
like `Greater`, `Less`, `Equal`) would crash on load.

## ASCII diagram

```
ONNX Model Pipeline:

  [Input Tensor]         [Greater Than op]       [BOOL Tensor]
   (float data)    -->    (compares values)  -->  (true/false)
                                                       |
                                               BEFORE: CRASH  X
                                               AFTER:  passes through
                                                       |
                                                  [Where op]
                                                  (select values)
                                                       |
                                               [Output Tensor]
                                                (float values)
```

## The technical fix

- Added `BOOL` case to the `getMatType()` data-type switch in the ONNX parser
- Boolean tensors are stored as `CV_8U` (unsigned 8-bit), mapping `false→0`, `true→1`
- Initializer tensors with bool data are now correctly read from protobuf bytes

## Files changed

- `modules/dnn/src/onnx/onnx_importer.cpp`
