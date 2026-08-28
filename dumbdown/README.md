# Fix: VideoCapture.get(CAP_PROP_FRAME_COUNT) Misleading Return Value (Issue #29722)

## ELI5 Explanation

Imagine a library catalog that says a book has **200 pages**.

You walk over, open the book — and every single page is **blank**. The book is
unreadable. But the catalog still says 200 pages, because it read the page count
from the cover without ever opening the book and checking the contents.

OpenCV's `VideoCapture` does the same thing. It reads the frame count from the
video file's **header** (the cover) without trying to actually decode a single frame.
So even if the video is corrupted, encoded in an unsupported codec, or otherwise
unreadable, `get(CAP_PROP_FRAME_COUNT)` can still return a big positive number
like 1200.

A programmer might write a loop that runs 1200 times expecting to get 1200 frames —
but `read()` fails every single iteration and they get nothing.

**The fix:** Add a clear warning in the documentation so every programmer knows:
*always check that `read()` returns true. Never trust the frame count alone.*

---

## What CAP_PROP_FRAME_COUNT Does

This property is supposed to return the total number of frames in a video file.
It is useful for building progress bars, pre-allocating arrays, or validating files.

The key caveat: it reads metadata from the file container, not from actual decoded
content. The value may be wrong or misleading if the file is damaged or the codec
is unsupported.

---

## ASCII Diagram

```
MISLEADING SCENARIO:

Step 1:  cap.open("video.mp4")
         |
         +---> Returns: true  (file opened OK, header read)

Step 2:  cap.get(CAP_PROP_FRAME_COUNT)
         |
         +---> Returns: 1200  (read from file header/metadata)
                               ^
                               |
                        [does NOT try to decode any frames!]

Step 3:  cap.read(frame)
         |
         +---> Returns: false  (FAILS -- can't actually decode!)
                                ^
                                |
                         [corrupt codec / unsupported format]

PROGRAMMER'S LOOP (broken assumption):

  for i in range(1200):          <-- loops 1200 times
      ret, frame = cap.read()    <-- fails EVERY time
      process(frame)             <-- never runs

  Result: 0 frames processed, no error thrown. Silent failure.

THE FIX (documentation warning):

  !! WARNING !!
  CAP_PROP_FRAME_COUNT reads from metadata.
  ALWAYS check ret from read() before using frame.
  Do NOT assume frame_count valid decodes exist.

  Safe pattern:
  while True:
      ret, frame = cap.read()
      if not ret:
          break               <-- stop when decode actually fails
      process(frame)
```

---

## Root Cause (Technical)

`CAP_PROP_FRAME_COUNT` is fetched from container-level metadata (e.g., the `moov`
atom in MP4, or the AVI header). This value is stored independently from whether
the codec can actually decode the stream. The fix is documentation-only: it adds
an explicit caveat to the API docs stating that the returned count is metadata-only
and must not be relied upon as a decode guarantee.
