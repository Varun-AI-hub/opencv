"""
02_train_hog_svm.py
-------------------
Train a custom HOG+SVM detector with hard negative mining.

Usage
-----
    python 02_train_hog_svm.py \
        --pos positives/ \
        --neg negatives/ \
        --output detector.xml \
        [--win-size 64] \
        [--hnm-rounds 2] \
        [--test-split 0.2]

Outputs
-------
    <output>.xml     — OpenCV SVM model (cv2.ml.SVM.save)
    <output>.pkl     — HOG descriptor params + SVM support vectors as numpy
                       arrays (used by 03_detect_hog_svm.py)
    training_log.txt — Per-round accuracy / precision / recall

The script implements the complete pipeline:
  1. Load images and extract HOG features
  2. Train/validation split (stratified)
  3. Train initial SVM (RBF kernel, grid-search for C and gamma)
  4. Hard Negative Mining loop
  5. Evaluate final model
  6. Save model and HOG config
"""

import argparse
import os
import pickle
import sys
import time

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# HOG descriptor configuration
# ---------------------------------------------------------------------------

HOG_WIN_SIZE = (64, 64)       # (width, height) — must match data generation
HOG_BLOCK_SIZE = (16, 16)
HOG_BLOCK_STRIDE = (8, 8)
HOG_CELL_SIZE = (8, 8)
HOG_N_BINS = 9

# Sliding-window params used during hard negative mining
HNM_WIN_STRIDE = (8, 8)
HNM_SCALE = 1.05
HNM_PADDING = (0, 0)
# SVM decision threshold for collecting hard negatives (low to catch more FP)
HNM_THRESHOLD = 0.0


def build_hog() -> cv2.HOGDescriptor:
    """Return a configured HOGDescriptor."""
    return cv2.HOGDescriptor(
        HOG_WIN_SIZE,
        HOG_BLOCK_SIZE,
        HOG_BLOCK_STRIDE,
        HOG_CELL_SIZE,
        HOG_N_BINS,
    )


def feature_dim(hog: cv2.HOGDescriptor) -> int:
    return int(hog.getDescriptorSize())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_images_from_dir(directory: str) -> list[np.ndarray]:
    """Load all PNG/JPG images from directory as grayscale numpy arrays."""
    imgs = []
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        path = os.path.join(directory, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  [warn] could not read {path}")
            continue
        imgs.append(img)
    return imgs


def extract_hog_features(images: list[np.ndarray], hog: cv2.HOGDescriptor,
                          win_size: tuple[int, int]) -> np.ndarray:
    """Extract HOG feature vector for each image.
    Images are resized to win_size if necessary.
    Returns float32 array of shape (N, D).
    """
    D = feature_dim(hog)
    feats = np.zeros((len(images), D), dtype=np.float32)
    for i, img in enumerate(images):
        if img.shape[:2] != (win_size[1], win_size[0]):
            img = cv2.resize(img, win_size)
        feat = hog.compute(img, winStride=(win_size[0], win_size[1]),
                           padding=(0, 0))
        feats[i] = feat.ravel()
    return feats


# ---------------------------------------------------------------------------
# SVM helpers
# ---------------------------------------------------------------------------

def build_svm() -> cv2.ml.SVM:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10.0)
    svm.setGamma(1e-4)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                         5000, 1e-6))
    return svm


def train_svm(X: np.ndarray, y: np.ndarray,
              auto_train: bool = False) -> cv2.ml.SVM:
    """Train SVM. If auto_train=True, uses OpenCV's built-in grid search."""
    svm = build_svm()
    print(f"  Training SVM on {X.shape[0]} samples, "
          f"dim={X.shape[1]}, pos={int((y==1).sum())}, neg={int((y==-1).sum())}")
    t0 = time.time()
    td = cv2.ml.TrainData_create(
        X, cv2.ml.ROW_SAMPLE, y.reshape(-1, 1).astype(np.int32)
    )
    if auto_train:
        # Grid search over C and gamma — slower but potentially better
        svm.trainAuto(td, kFold=3)
    else:
        svm.train(td)
    elapsed = time.time() - t0
    print(f"  Training done in {elapsed:.1f}s  "
          f"(support vectors: {svm.getSupportVectors().shape[0]})")
    return svm


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate(svm: cv2.ml.SVM, X: np.ndarray,
             y_true: np.ndarray) -> dict[str, float]:
    """Return accuracy, precision, recall, F1 on a feature matrix."""
    _, y_pred_mat = svm.predict(X)
    y_pred = y_pred_mat.ravel().astype(np.int32)
    y_true_i = y_true.ravel().astype(np.int32)

    tp = int(np.sum((y_pred == 1) & (y_true_i == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true_i == -1)))
    fn = int(np.sum((y_pred == -1) & (y_true_i == 1)))
    tn = int(np.sum((y_pred == -1) & (y_true_i == -1)))

    accuracy  = (tp + tn) / len(y_true_i)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {"accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def print_metrics(metrics: dict, prefix: str = ""):
    print(f"  {prefix}accuracy={metrics['accuracy']:.4f}  "
          f"precision={metrics['precision']:.4f}  "
          f"recall={metrics['recall']:.4f}  "
          f"F1={metrics['f1']:.4f}  "
          f"(TP={metrics['tp']} FP={metrics['fp']} "
          f"FN={metrics['fn']} TN={metrics['tn']})")


# ---------------------------------------------------------------------------
# Hard Negative Mining
# ---------------------------------------------------------------------------

def svm_decision_scores(svm: cv2.ml.SVM, X: np.ndarray) -> np.ndarray:
    """Return raw SVM decision values (signed distance to hyperplane)."""
    _, scores = svm.predict(X, flags=cv2.ml.StatModel_RAW_OUTPUT)
    return scores.ravel()


def hard_negative_mining(svm: cv2.ml.SVM,
                         hog: cv2.HOGDescriptor,
                         neg_images: list[np.ndarray],
                         win_size: tuple[int, int],
                         threshold: float = HNM_THRESHOLD,
                         max_per_image: int = 10) -> np.ndarray:
    """
    Run a sliding-window detector on negative images (which should not
    trigger the detector at all).  Collect false-positive crops as hard
    negative features.

    Returns feature matrix of hard negatives (may be empty).
    """
    W, H = win_size
    D = feature_dim(hog)
    hard_negs: list[np.ndarray] = []

    for img in neg_images:
        if img.shape[0] < H or img.shape[1] < W:
            # Image too small — just skip
            continue

        # Ensure image is large enough for at least one window
        h_img, w_img = img.shape[:2]
        collected = 0

        # Build simple scale pyramid manually
        scale = 1.0
        while True:
            new_w = int(w_img / scale)
            new_h = int(h_img / scale)
            if new_w < W or new_h < H:
                break

            img_scaled = cv2.resize(img, (new_w, new_h))

            # Slide window
            for y in range(0, new_h - H + 1, HNM_WIN_STRIDE[1]):
                for x in range(0, new_w - W + 1, HNM_WIN_STRIDE[0]):
                    crop = img_scaled[y:y + H, x:x + W]
                    feat = hog.compute(
                        crop,
                        winStride=(W, H),
                        padding=(0, 0),
                    ).ravel().reshape(1, -1).astype(np.float32)

                    score = svm_decision_scores(svm, feat)[0]
                    if score > threshold:
                        hard_negs.append(feat[0])
                        collected += 1
                        if collected >= max_per_image:
                            break
                if collected >= max_per_image:
                    break

            scale *= HNM_SCALE

    if not hard_negs:
        return np.empty((0, D), dtype=np.float32)

    return np.vstack(hard_negs).astype(np.float32)


# ---------------------------------------------------------------------------
# Model serialisation
# ---------------------------------------------------------------------------

def svm_to_detector_array(svm: cv2.ml.SVM) -> np.ndarray:
    """
    Convert a LINEAR or RBF SVM to the 1-D detector array expected by
    cv2.HOGDescriptor.setSVMDetector().

    For a linear SVM: the detector = [w_0, w_1, ..., w_{D-1}, bias].
    For non-linear: this is an approximation — use the raw SVM model for
    accurate predictions. The HOGDescriptor fast-path only supports linear.
    """
    sv = svm.getSupportVectors()          # shape (n_sv, D)
    rho, _, _ = svm.getDecisionFunction(0)
    alpha_mat, sv_idx_mat = svm.getSupportVectors(), None

    # Retrieve dual coefficients (alpha * y)
    alphas = np.zeros((sv.shape[0], 1), dtype=np.float64)
    for i in range(sv.shape[0]):
        # getDecisionFunction returns all alpha for class 0
        # For a 2-class C-SVC the dual coefficients are stored per SV
        pass

    # Simplified: use decision function projection for linear kernel
    # w = Σ α_i * y_i * x_i
    # For non-linear SVMs we compute a linear approximation by projecting SVs
    # through the kernel map. Here we just export the support vectors directly.
    result = np.zeros((sv.shape[1] + 1,), dtype=np.float64)
    result[:-1] = sv.mean(axis=0)        # crude approximation
    result[-1]  = -rho
    return result.astype(np.float32)


def save_model(svm: cv2.ml.SVM, hog: cv2.HOGDescriptor,
               xml_path: str, pkl_path: str):
    """Save SVM as XML and HOG+SVM config as pickle."""
    svm.save(xml_path)
    print(f"  SVM model saved → {xml_path}")

    hog_config = {
        "win_size":     HOG_WIN_SIZE,
        "block_size":   HOG_BLOCK_SIZE,
        "block_stride": HOG_BLOCK_STRIDE,
        "cell_size":    HOG_CELL_SIZE,
        "n_bins":       HOG_N_BINS,
        "support_vectors": svm.getSupportVectors(),
        "rho": svm.getDecisionFunction(0)[0],
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(hog_config, f)
    print(f"  HOG config saved  → {pkl_path}")


# ---------------------------------------------------------------------------
# Train/validation split (stratified)
# ---------------------------------------------------------------------------

def stratified_split(X: np.ndarray, y: np.ndarray,
                     test_frac: float = 0.2,
                     seed: int = 42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == -1)[0]

    def split_indices(idx):
        n_test = max(1, int(len(idx) * test_frac))
        perm = rng.permutation(len(idx))
        return idx[perm[n_test:]], idx[perm[:n_test]]

    pos_train, pos_val = split_indices(pos_idx)
    neg_train, neg_val = split_indices(neg_idx)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx   = np.concatenate([pos_val,   neg_val])

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HOG+SVM detector with hard negative mining",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pos",        default="positives", help="Positive images directory")
    parser.add_argument("--neg",        default="negatives", help="Negative images directory")
    parser.add_argument("--output",     default="detector.xml", help="Output SVM model path (.xml)")
    parser.add_argument("--win-size",   type=int, default=64,
                        help="Detection window size (square)")
    parser.add_argument("--hnm-rounds", type=int, default=2,
                        help="Number of hard negative mining rounds")
    parser.add_argument("--test-split", type=float, default=0.2,
                        help="Fraction of data held out for validation")
    parser.add_argument("--auto-train", action="store_true",
                        help="Use OpenCV grid search for SVM hyperparameters (slow)")
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    global HOG_WIN_SIZE
    HOG_WIN_SIZE = (args.win_size, args.win_size)

    pkl_path = args.output.replace(".xml", ".pkl")
    log_path = "training_log.txt"
    log_lines: list[str] = []

    # ---- 1. Load images ---------------------------------------------------
    print("Loading images ...")
    pos_imgs = load_images_from_dir(args.pos)
    neg_imgs = load_images_from_dir(args.neg)

    if not pos_imgs:
        sys.exit(f"ERROR: no images found in {args.pos}")
    if not neg_imgs:
        sys.exit(f"ERROR: no images found in {args.neg}")

    print(f"  Positives: {len(pos_imgs)}  Negatives: {len(neg_imgs)}")
    log_lines.append(f"Positives: {len(pos_imgs)}  Negatives: {len(neg_imgs)}")

    # ---- 2. Extract HOG features ------------------------------------------
    hog = build_hog()
    win_size = HOG_WIN_SIZE
    print(f"\nExtracting HOG features (win={win_size}, "
          f"dim={feature_dim(hog)}) ...")

    t0 = time.time()
    X_pos = extract_hog_features(pos_imgs, hog, win_size)
    X_neg = extract_hog_features(neg_imgs, hog, win_size)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s  "
          f"(pos shape={X_pos.shape}, neg shape={X_neg.shape})")

    y_pos = np.ones(len(X_pos),  dtype=np.int32)
    y_neg = -np.ones(len(X_neg), dtype=np.int32)

    X_all = np.vstack([X_pos, X_neg])
    y_all = np.concatenate([y_pos, y_neg])

    # ---- 3. Train/val split -----------------------------------------------
    X_train, y_train, X_val, y_val = stratified_split(
        X_all, y_all, test_frac=args.test_split, seed=args.seed
    )
    print(f"\nTrain/val split: {len(X_train)} train, {len(X_val)} val")

    # ---- 4. Initial SVM training ------------------------------------------
    print("\n--- Round 0: Initial training ---")
    svm = train_svm(X_train, y_train, auto_train=args.auto_train)
    m = evaluate(svm, X_val, y_val)
    print_metrics(m, "Val ")
    log_lines.append(f"Round 0: {m}")

    # ---- 5. Hard Negative Mining -------------------------------------------
    X_train_cur = X_train.copy()
    y_train_cur = y_train.copy()

    for hnm_round in range(1, args.hnm_rounds + 1):
        print(f"\n--- Round {hnm_round}: Hard Negative Mining ---")
        hard_negs = hard_negative_mining(
            svm, hog, neg_imgs, win_size,
            threshold=HNM_THRESHOLD, max_per_image=10
        )
        n_hard = hard_negs.shape[0]
        print(f"  Collected {n_hard} hard negative samples")
        log_lines.append(f"Round {hnm_round}: hard negatives collected = {n_hard}")

        if n_hard == 0:
            print("  No new hard negatives — stopping early.")
            break

        # Append hard negatives to training set
        y_hard = -np.ones(n_hard, dtype=np.int32)
        X_train_cur = np.vstack([X_train_cur, hard_negs])
        y_train_cur = np.concatenate([y_train_cur, y_hard])

        svm = train_svm(X_train_cur, y_train_cur, auto_train=args.auto_train)
        m = evaluate(svm, X_val, y_val)
        print_metrics(m, "Val ")
        log_lines.append(f"Round {hnm_round} val: {m}")

    # ---- 6. Final evaluation -----------------------------------------------
    print("\n--- Final Evaluation ---")
    m_final = evaluate(svm, X_val, y_val)
    print_metrics(m_final, "Final val ")

    # Also compute train metrics (to check overfitting)
    m_train = evaluate(svm, X_train_cur, y_train_cur)
    print_metrics(m_train, "Train      ")
    log_lines.append(f"Final val: {m_final}")
    log_lines.append(f"Train:     {m_train}")

    # ---- 7. Save model -----------------------------------------------------
    print(f"\nSaving model ...")
    save_model(svm, hog, args.output, pkl_path)

    # Write training log
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"  Training log → {log_path}")

    print("\nDone. Next steps:")
    print(f"  Detect:   python 03_detect_hog_svm.py --model {args.output} "
          "--image test_image.jpg")
    print(f"  Evaluate: python 04_evaluate_detector.py --model {args.output} "
          "--test-dir test_images/ --annotations annotations.csv")


if __name__ == "__main__":
    main()
