"""
classical_methods_code.py
=========================
Comprehensive runnable demonstration of classical object detection methods in OpenCV.

Sections:
  1. Template Matching with NMS
  2. Haar Cascade Face & Eye Detection
  3. HOG + SVM Pedestrian Detection
  4. Background Subtraction (MOG2 and KNN)
  5. Contour-Based Detection with Shape Filtering
  6. IoU and NMS Utilities

All sections use synthetic/generated images so the script runs without any
external image or video files.  Where real media paths are referenced, they are
clearly marked and can be swapped in.

Requirements: opencv-python (cv2), numpy
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def show_or_save(title: str, image: np.ndarray, save: bool = True) -> None:
    """Either display or save an image.  Set save=False to use imshow."""
    if save:
        path = f"{title.replace(' ', '_')}.png"
        cv2.imwrite(path, image)
        print(f"  [saved] {path}")
    else:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# SECTION 1: Template Matching with NMS
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """
    Compute Intersection-over-Union for two boxes in [x1, y1, x2, y2] format.

    Args:
        box1, box2: sequences of four numbers (x1, y1, x2, y2)

    Returns:
        float in [0, 1]
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter   = inter_w * inter_h
    area1   = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2   = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union   = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def nms_manual(boxes_xyxy, scores, iou_threshold=0.45):
    """
    Pure-Python greedy NMS.

    Args:
        boxes_xyxy:     list of [x1, y1, x2, y2]
        scores:         list of confidence floats (same length)
        iou_threshold:  suppress if IoU > this value

    Returns:
        list of kept indices sorted by descending score
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept  = []
    while order:
        best = order.pop(0)
        kept.append(best)
        order = [i for i in order
                 if compute_iou(boxes_xyxy[best], boxes_xyxy[i]) < iou_threshold]
    return kept


def section1_template_matching():
    print("\n=== SECTION 1: Template Matching with NMS ===")

    # ------------------------------------------------------------------
    # Build a synthetic source image with two instances of a "logo"
    # ------------------------------------------------------------------
    source = np.full((280, 400, 3), 210, dtype=np.uint8)  # light grey bg

    # Draw a simple blue circle + rectangle "logo"
    def draw_logo(img, cx, cy):
        cv2.circle(img, (cx, cy), 20, (30, 80, 200), -1)
        cv2.rectangle(img, (cx - 25, cy + 20), (cx + 25, cy + 35), (30, 80, 200), -1)

    draw_logo(source, 80,  80)
    draw_logo(source, 280, 160)

    # Add Gaussian noise so matching is non-trivial
    noise = np.random.default_rng(7).integers(-15, 16, source.shape, dtype=np.int16)
    source = np.clip(source.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Crop template from a clean copy
    template_clean = np.full((55, 50, 3), 210, dtype=np.uint8)
    draw_logo(template_clean, 25, 30)

    src_gray = cv2.cvtColor(source,         cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(template_clean, cv2.COLOR_BGR2GRAY)

    th, tw = tpl_gray.shape[:2]

    # ------------------------------------------------------------------
    # Run matchTemplate (TM_CCOEFF_NORMED is recommended)
    # ------------------------------------------------------------------
    result = cv2.matchTemplate(src_gray, tpl_gray, cv2.TM_CCOEFF_NORMED)

    threshold = 0.55
    locs = np.where(result >= threshold)
    boxes_xyxy, scores = [], []
    for y, x in zip(*locs):
        boxes_xyxy.append([int(x), int(y), int(x + tw), int(y + th)])
        scores.append(float(result[y, x]))

    print(f"  Raw detections above {threshold}: {len(boxes_xyxy)}")

    # ------------------------------------------------------------------
    # Apply NMS to remove overlapping candidates
    # ------------------------------------------------------------------
    kept = nms_manual(boxes_xyxy, scores, iou_threshold=0.3)
    print(f"  After NMS: {len(kept)} detections")

    output = source.copy()
    for i in kept:
        x1, y1, x2, y2 = boxes_xyxy[i]
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(output, f"{scores[i]:.2f}", (x1, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

    show_or_save("section1_template_matching", output)

    # ------------------------------------------------------------------
    # Also show: all 6 matchTemplate methods on a simple 1-match image
    # ------------------------------------------------------------------
    methods = [
        ("TM_SQDIFF",        cv2.TM_SQDIFF),
        ("TM_SQDIFF_NORMED", cv2.TM_SQDIFF_NORMED),
        ("TM_CCORR",         cv2.TM_CCORR),
        ("TM_CCORR_NORMED",  cv2.TM_CCORR_NORMED),
        ("TM_CCOEFF",        cv2.TM_CCOEFF),
        ("TM_CCOEFF_NORMED", cv2.TM_CCOEFF_NORMED),
    ]
    simple_src = np.full((100, 200, 3), 200, dtype=np.uint8)
    cv2.rectangle(simple_src, (60, 30), (90, 70), (0, 120, 255), -1)
    simple_tmpl = simple_src[30:70, 60:90].copy()
    ss_gray = cv2.cvtColor(simple_src,  cv2.COLOR_BGR2GRAY)
    st_gray = cv2.cvtColor(simple_tmpl, cv2.COLOR_BGR2GRAY)

    print("  Method comparison (single instance, best match location):")
    for name, method in methods:
        res = cv2.matchTemplate(ss_gray, st_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
            best_loc, best_val = min_loc, min_val
        else:
            best_loc, best_val = max_loc, max_val
        print(f"    {name:<22} best_loc={best_loc}  val={best_val:.4f}")


# ---------------------------------------------------------------------------
# SECTION 2: Haar Cascade Face & Eye Detection
# ---------------------------------------------------------------------------

def section2_haar_cascades():
    print("\n=== SECTION 2: Haar Cascade Face & Eye Detection ===")

    # ------------------------------------------------------------------
    # Create a synthetic "face-like" test image using ovals and circles
    # (In real use, replace this with an actual photo)
    # ------------------------------------------------------------------
    img = np.full((300, 300, 3), 220, dtype=np.uint8)
    # Face oval
    cv2.ellipse(img, (150, 140), (80, 100), 0, 0, 360, (180, 150, 130), -1)
    # Eyes
    cv2.circle(img, (115, 110), 15, (230, 200, 180), -1)
    cv2.circle(img, (185, 110), 15, (230, 200, 180), -1)
    cv2.circle(img, (115, 112), 7,  (50,  30,  20),  -1)
    cv2.circle(img, (185, 112), 7,  (50,  30,  20),  -1)
    # Nose
    cv2.ellipse(img, (150, 145), (10, 15), 0, 0, 360, (160, 120, 110), -1)
    # Mouth
    cv2.ellipse(img, (150, 185), (30, 12), 0, 0, 180, (140, 80, 80), 3)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    # ------------------------------------------------------------------
    # Load cascades
    # ------------------------------------------------------------------
    face_xml = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    eye_xml  = cv2.data.haarcascades + 'haarcascade_eye.xml'

    face_cascade = cv2.CascadeClassifier(face_xml)
    eye_cascade  = cv2.CascadeClassifier(eye_xml)

    if face_cascade.empty():
        print(f"  WARNING: could not load {face_xml}. Skipping Haar demo.")
        return

    # ------------------------------------------------------------------
    # detectMultiScale — face
    # ------------------------------------------------------------------
    faces = face_cascade.detectMultiScale(
        gray_eq,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(40, 40),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    output = img.copy()
    print(f"  Faces detected: {len(faces)}")
    for (fx, fy, fw, fh) in faces:
        cv2.rectangle(output, (fx, fy), (fx+fw, fy+fh), (255, 50, 50), 2)

        # Search eyes in upper-half of face ROI
        roi_gray  = gray_eq[fy:fy + fh//2, fx:fx + fw]
        roi_color = output[fy:fy + fh//2, fx:fx + fw]

        if not eye_cascade.empty():
            eyes = eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
            print(f"  Eyes detected in face ROI: {len(eyes)}")
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (50, 255, 50), 2)

    # ------------------------------------------------------------------
    # Parameter sensitivity table (printed)
    # ------------------------------------------------------------------
    print("\n  scaleFactor / minNeighbors sensitivity scan:")
    print(f"  {'scaleFactor':>12} | {'minNeighbors':>13} | {'detections':>10}")
    print(f"  {'-'*12}-+-{'-'*13}-+-{'-'*10}")
    for sf in [1.05, 1.1, 1.3]:
        for mn in [1, 3, 6]:
            det = face_cascade.detectMultiScale(gray_eq, scaleFactor=sf,
                                                minNeighbors=mn, minSize=(20, 20))
            n = len(det) if isinstance(det, np.ndarray) else 0
            print(f"  {sf:>12.2f} | {mn:>13} | {n:>10}")

    show_or_save("section2_haar_faces", output)

    # ------------------------------------------------------------------
    # Webcam snippet (not run automatically — uncomment to use)
    # ------------------------------------------------------------------
    # def webcam_detection():
    #     cap = cv2.VideoCapture(0)
    #     while True:
    #         ret, frame = cap.read()
    #         if not ret: break
    #         g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         g = cv2.equalizeHist(g)
    #         faces = face_cascade.detectMultiScale(g, 1.1, 5, minSize=(30,30))
    #         for (x, y, w, h) in faces:
    #             cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    #         cv2.imshow("Webcam", frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'): break
    #     cap.release(); cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# SECTION 3: HOG + SVM Pedestrian Detection
# ---------------------------------------------------------------------------

def section3_hog_svm():
    print("\n=== SECTION 3: HOG + SVM Pedestrian Detection ===")

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # ------------------------------------------------------------------
    # Synthetic "street scene": dark background, upright rectangles
    # representing pedestrian silhouettes
    # ------------------------------------------------------------------
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    scene[:] = (60, 55, 50)  # dark road-like bg

    def draw_person(img, cx, top, height):
        """Draw a rough person silhouette at (cx, top) with given height."""
        w = height // 3
        # Body
        cv2.rectangle(img,
                      (cx - w//2, top + height//4),
                      (cx + w//2, top + height),
                      (200, 190, 180), -1)
        # Head
        cv2.circle(img, (cx, top + height//8), height//8, (210, 190, 170), -1)
        # Legs
        cv2.line(img, (cx, top + height),
                 (cx - w//4, top + height + height//4), (180, 170, 160), 4)
        cv2.line(img, (cx, top + height),
                 (cx + w//4, top + height + height//4), (180, 170, 160), 4)

    draw_person(scene, 160, 180, 200)
    draw_person(scene, 400, 220, 160)

    # Blur to make silhouettes less artificial
    scene = cv2.GaussianBlur(scene, (5, 5), 1)

    # ------------------------------------------------------------------
    # Detect
    # ------------------------------------------------------------------
    rects, weights = hog.detectMultiScale(
        scene,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05,
        finalThreshold=2.0
    )

    print(f"  Raw HOG detections: {len(rects)}")

    output = scene.copy()
    if len(rects):
        boxes_xywh = [[int(x), int(y), int(w), int(h)] for (x, y, w, h) in rects]
        wts = [float(v) for v in weights.flatten()]
        indices = cv2.dnn.NMSBoxes(boxes_xywh, wts,
                                   score_threshold=0.0,
                                   nms_threshold=0.65)
        n_after = 0
        if len(indices):
            for i in indices.flatten():
                x, y, w, h = rects[i]
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 80, 255), 2)
                cv2.putText(output, f"{wts[i]:.2f}", (x, y-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
                n_after += 1
        print(f"  After NMS: {n_after} pedestrians")
    else:
        print("  No detections on synthetic image "
              "(expected — silhouettes lack real HOG gradients)")
        # Draw ground-truth boxes for illustration
        for cx, top, height in [(160, 180, 200), (400, 220, 160)]:
            w = height // 3
            cv2.rectangle(output, (cx - w//2, top), (cx + w//2, top + height),
                          (0, 200, 200), 2)
            cv2.putText(output, "GT", (cx - w//2, top - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    show_or_save("section3_hog_pedestrian", output)

    # ------------------------------------------------------------------
    # HOG feature extraction demo
    # ------------------------------------------------------------------
    patch = cv2.resize(scene[170:390, 130:200], (64, 128))
    descriptor = hog.compute(patch)
    print(f"  HOG descriptor length for 64x128 patch: {len(descriptor)}")
    print(f"  Descriptor stats — min:{descriptor.min():.4f}  "
          f"max:{descriptor.max():.4f}  mean:{descriptor.mean():.4f}")

    # ------------------------------------------------------------------
    # Custom HOG+SVM training (minimal synthetic example)
    # ------------------------------------------------------------------
    print("\n  Custom HOG+SVM training (synthetic data):")
    custom_hog = cv2.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
    rng = np.random.default_rng(42)

    features, labels_list = [], []
    for _ in range(20):
        # Positive: patches with a bright vertical stripe (person-like)
        pos = rng.integers(30, 80, (128, 64), dtype=np.uint8)
        pos[:, 24:40] = rng.integers(160, 220, (128, 16), dtype=np.uint8)
        pos_bgr = cv2.cvtColor(pos, cv2.COLOR_GRAY2BGR)
        features.append(custom_hog.compute(pos_bgr).flatten())
        labels_list.append(1)
    for _ in range(20):
        # Negative: random noise
        neg = rng.integers(30, 220, (128, 64, 3), dtype=np.uint8)
        features.append(custom_hog.compute(neg).flatten())
        labels_list.append(-1)

    X = np.array(features, dtype=np.float32)
    y = np.array(labels_list, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(1.0)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS,
                         500, 1e-6))
    svm.train(X, cv2.ml.ROW_SAMPLE, y)

    # Derive HOG detector vector from SVM support vectors
    sv  = svm.getSupportVectors()
    rho, _, _ = svm.getDecisionFunction(0)
    detector = np.zeros(sv.shape[1] + 1, dtype=np.float64)
    detector[:-1] = -sv[0]
    detector[-1]  = float(rho)
    print(f"  SVM trained. Detector vector length: {len(detector)}")
    print(f"  Support vectors: {sv.shape[0]}")


# ---------------------------------------------------------------------------
# SECTION 4: Background Subtraction (MOG2 and KNN)
# ---------------------------------------------------------------------------

def section4_background_subtraction():
    print("\n=== SECTION 4: Background Subtraction ===")

    rng = np.random.default_rng(0)
    H, W = 200, 320

    def make_background_frame(seed_offset=0):
        """Simulate a static road background with slight noise."""
        bg = np.full((H, W), 110, dtype=np.uint8)
        noise = rng.integers(-8, 9, (H, W), dtype=np.int16)
        return np.clip(bg.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    for method_name, bgsub in [
        ("MOG2", cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=25, detectShadows=True)),
        ("KNN",  cv2.createBackgroundSubtractorKNN(
            history=50, dist2Threshold=400, detectShadows=True)),
    ]:
        # Phase 1: feed background-only frames to train the model
        for _ in range(60):
            bgsub.apply(make_background_frame())

        # Phase 2: frame with a "vehicle" (bright rectangle)
        frame = make_background_frame()
        cv2.rectangle(frame, (80, 80), (160, 130), 220, -1)   # vehicle body
        cv2.rectangle(frame, (90, 70), (150,  85), 200, -1)   # roof

        fg_mask = bgsub.apply(frame)

        # Remove shadow pixels (grey = 127) — keep only definite FG (255)
        _, fg_binary = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_clean = cv2.morphologyEx(fg_binary, cv2.MORPH_OPEN,  kernel)
        fg_clean = cv2.morphologyEx(fg_clean,  cv2.MORPH_CLOSE, kernel)

        # Detect contours on cleaned mask
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        vehicles = [cv2.boundingRect(c) for c in contours
                    if cv2.contourArea(c) > 200]

        result_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in vehicles:
            cv2.rectangle(result_bgr, (x, y), (x+w, y+h), (0, 255, 80), 2)

        print(f"  [{method_name}]  raw FG pixels: {cv2.countNonZero(fg_binary)}"
              f"  vehicles detected: {len(vehicles)}")

        # Stack frame / raw mask / cleaned / detections side by side
        frame_3ch  = cv2.cvtColor(frame,      cv2.COLOR_GRAY2BGR)
        raw_3ch    = cv2.cvtColor(fg_binary,  cv2.COLOR_GRAY2BGR)
        clean_3ch  = cv2.cvtColor(fg_clean,   cv2.COLOR_GRAY2BGR)
        composite  = np.hstack([frame_3ch, raw_3ch, clean_3ch, result_bgr])
        show_or_save(f"section4_bgsub_{method_name}", composite)


# ---------------------------------------------------------------------------
# SECTION 5: Contour-Based Detection with Shape Filtering
# ---------------------------------------------------------------------------

def section5_contour_detection():
    print("\n=== SECTION 5: Contour-Based Detection with Shape Filtering ===")

    # ------------------------------------------------------------------
    # Create a binary image with objects of various shapes
    # ------------------------------------------------------------------
    binary = np.zeros((400, 500), dtype=np.uint8)

    # Large filled circle — should pass all filters
    cv2.circle(binary, (100, 100), 55, 255, -1)

    # Wide rectangle — should pass (aspect ~2.5)
    cv2.rectangle(binary, (200, 60), (380, 140), 255, -1)

    # Narrow tall rectangle — aspect ratio filter may exclude
    cv2.rectangle(binary, (430, 60), (470, 200), 255, -1)

    # Crescent / concave shape (low solidity) — should be excluded by solidity
    cv2.circle(binary, (100, 300), 55, 255, -1)
    cv2.circle(binary, (115, 280), 45, 0,   -1)  # bite out

    # Tiny noise blob — should be excluded by min_area
    cv2.circle(binary, (280, 320), 5, 255, -1)

    # L-shaped object (moderate solidity ~0.5) — borderline
    cv2.rectangle(binary, (350, 250), (450, 310), 255, -1)
    cv2.rectangle(binary, (350, 310), (390, 370), 255, -1)

    # ------------------------------------------------------------------
    # Helper: compute shape descriptors for a contour
    # ------------------------------------------------------------------
    def describe_contour(cnt):
        area       = cv2.contourArea(cnt)
        perim      = cv2.arcLength(cnt, True)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect     = float(w) / h if h > 0 else 0
        extent     = area / (w * h) if w * h > 0 else 0
        hull       = cv2.convexHull(cnt)
        hull_area  = cv2.contourArea(hull)
        solidity   = area / hull_area if hull_area > 0 else 0
        circularity = (4 * np.pi * area / perim**2) if perim > 0 else 0
        return dict(area=area, aspect=round(aspect, 2),
                    extent=round(extent, 2), solidity=round(solidity, 2),
                    circularity=round(circularity, 2),
                    bbox=(x, y, w, h))

    # ------------------------------------------------------------------
    # Detect and filter contours
    # ------------------------------------------------------------------
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    print(f"  Total contours found: {len(contours)}")
    print(f"\n  {'ID':>3} | {'area':>7} | {'aspect':>7} | {'solidity':>8} | "
          f"{'circ':>6} | {'pass?':>6}")
    print(f"  {'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*6}-+-{'-'*6}")

    min_area, max_area = 500, 30000
    min_aspect, max_aspect = 0.25, 4.0
    min_solidity = 0.65

    kept_contours = []
    for idx, cnt in enumerate(contours):
        d      = describe_contour(cnt)
        passes = (min_area < d['area'] < max_area
                  and min_aspect < d['aspect'] < max_aspect
                  and d['solidity'] >= min_solidity)
        print(f"  {idx:>3} | {d['area']:>7.0f} | {d['aspect']:>7.2f} | "
              f"{d['solidity']:>8.2f} | {d['circularity']:>6.2f} | "
              f"{'YES' if passes else 'no':>6}")
        if passes:
            kept_contours.append(cnt)

    print(f"\n  Detections after filtering: {len(kept_contours)}")

    # Visualise
    vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(vis, contours, -1, (100, 100, 100), 1)      # all: grey
    cv2.drawContours(vis, kept_contours, -1, (0, 255, 0), 2)     # kept: green
    for cnt in kept_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 255), 1)

    show_or_save("section5_contours", vis)

    # ------------------------------------------------------------------
    # connectedComponentsWithStats (faster alternative for blobs)
    # ------------------------------------------------------------------
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8)
    print(f"\n  connectedComponentsWithStats: {n - 1} components (excl. bg)")
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        print(f"    comp {i}: bbox=({x},{y},{w},{h}) area={area} "
              f"centroid=({cx:.1f},{cy:.1f})")


# ---------------------------------------------------------------------------
# SECTION 6: IoU Utilities and NMS Comparison
# ---------------------------------------------------------------------------

def section6_iou_and_nms():
    print("\n=== SECTION 6: IoU Utilities and NMS Comparison ===")

    # ------------------------------------------------------------------
    # IoU sanity checks
    # ------------------------------------------------------------------
    tests = [
        ([0, 0, 100, 100], [0, 0, 100, 100], 1.0,  "identical"),
        ([0, 0, 100, 100], [200, 200, 300, 300], 0.0, "no overlap"),
        ([0, 0, 100, 100], [50, 50, 150, 150], None, "50% overlap"),
        ([10, 10,  60,  60], [30, 30, 80, 80], None, "partial overlap"),
    ]
    print("  IoU sanity checks:")
    for b1, b2, expected, desc in tests:
        iou = compute_iou(b1, b2)
        ok  = (expected is None) or (abs(iou - expected) < 1e-6)
        print(f"    {desc:<20} IoU={iou:.4f}  {'OK' if ok else 'FAIL'}")

    # ------------------------------------------------------------------
    # Multi-box NMS demo
    # ------------------------------------------------------------------
    boxes_xyxy = [
        [50,  50, 180, 180],   # score 0.95 — keep
        [55,  55, 185, 185],   # score 0.80 — suppress (high overlap with above)
        [60,  60, 190, 190],   # score 0.72 — suppress
        [300,  50, 430, 180],  # score 0.88 — keep (separate region)
        [305,  52, 435, 182],  # score 0.65 — suppress
        [150, 250, 280, 380],  # score 0.55 — keep (another region)
    ]
    scores = [0.95, 0.80, 0.72, 0.88, 0.65, 0.55]

    kept_manual = nms_manual(boxes_xyxy, scores, iou_threshold=0.45)

    boxes_xywh = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes_xyxy]
    kept_cv2_raw = cv2.dnn.NMSBoxes(
        boxes_xywh, scores, score_threshold=0.4, nms_threshold=0.45)
    kept_cv2 = (kept_cv2_raw.flatten().tolist()
                if len(kept_cv2_raw) else [])

    print(f"\n  Input boxes: {len(boxes_xyxy)}")
    print(f"  Manual NMS kept:       {sorted(kept_manual)}")
    print(f"  cv2.dnn.NMSBoxes kept: {sorted(kept_cv2)}")

    # Visualise
    vis = np.full((450, 480, 3), 230, dtype=np.uint8)
    for i, (b, s) in enumerate(zip(boxes_xyxy, scores)):
        # All boxes in grey
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (170, 170, 170), 1)

    for i in kept_manual:
        b = boxes_xyxy[i]
        cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), (0, 180, 0), 3)
        cv2.putText(vis, f"keep {scores[i]:.2f}", (b[0], b[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 150, 0), 1)

    show_or_save("section6_nms", vis)

    # ------------------------------------------------------------------
    # IoU grid: show pairwise IoU for all input boxes
    # ------------------------------------------------------------------
    n = len(boxes_xyxy)
    print(f"\n  Pairwise IoU matrix ({n}x{n}):")
    header = "     " + "".join(f"  B{i}" for i in range(n))
    print(f"  {header}")
    for i in range(n):
        row = f"  B{i} "
        for j in range(n):
            iou = compute_iou(boxes_xyxy[i], boxes_xyxy[j])
            row += f" {iou:4.2f}"
        print(row)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Classical Object Detection Methods — OpenCV Demo")
    print("=" * 60)

    section1_template_matching()
    section2_haar_cascades()
    section3_hog_svm()
    section4_background_subtraction()
    section5_contour_detection()
    section6_iou_and_nms()

    print("\nAll sections complete.  Output images saved as PNG files.")


if __name__ == "__main__":
    main()
