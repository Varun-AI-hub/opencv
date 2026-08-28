"""
01_generate_hog_dataset.py
--------------------------
Generate a synthetic dataset for HOG+SVM training.

Positive class: 64×64 images containing a bright circle on a dark/noisy
background, simulating a round object (ball, coin, wheel, etc.).

Negative class: 64×64 images of random noise/gradient textures that
contain no circular structure.

Usage
-----
    python 01_generate_hog_dataset.py [--pos-dir positives/] \
                                       [--neg-dir negatives/] \
                                       [--n-pos 500] \
                                       [--n-neg 1500] \
                                       [--seed 42]

Output
------
    positives/<index>.png   — positive samples
    negatives/<index>.png   — negative samples
    dataset_stats.txt       — summary statistics
"""

import argparse
import os
import random
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

def make_positive(img_size: int = 64, rng: np.random.Generator = None) -> np.ndarray:
    """Return an (img_size × img_size) grayscale image with a circle."""
    if rng is None:
        rng = np.random.default_rng()

    img = rng.integers(10, 50, size=(img_size, img_size), dtype=np.uint8)

    # Random circle parameters
    min_r = img_size // 6
    max_r = img_size // 2 - 4
    radius = int(rng.integers(min_r, max_r + 1))
    cx = int(rng.integers(radius + 2, img_size - radius - 2))
    cy = int(rng.integers(radius + 2, img_size - radius - 2))
    intensity = int(rng.integers(160, 256))

    cv2.circle(img, (cx, cy), radius, intensity, thickness=-1)

    # Optional: add a faint edge ring for realism
    edge_intensity = max(0, intensity - int(rng.integers(20, 60)))
    cv2.circle(img, (cx, cy), radius, edge_intensity, thickness=2)

    # Optional: Gaussian blur to smooth hard edges (simulate camera PSF)
    blur_amount = int(rng.choice([0, 1, 3]))
    if blur_amount > 0:
        img = cv2.GaussianBlur(img, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)

    # Add low-level noise
    noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def make_negative(img_size: int = 64, rng: np.random.Generator = None) -> np.ndarray:
    """Return an (img_size × img_size) grayscale image with no circular structure."""
    if rng is None:
        rng = np.random.default_rng()

    choice = rng.integers(0, 4)

    if choice == 0:
        # Pure Gaussian noise
        img = rng.integers(30, 220, size=(img_size, img_size), dtype=np.uint8)
        noise = rng.normal(0, 25, size=(img_size, img_size))
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    elif choice == 1:
        # Horizontal gradient
        gradient = np.linspace(0, 255, img_size, dtype=np.float32)
        img = np.tile(gradient, (img_size, 1)).astype(np.uint8)
        # Rotate randomly
        angle = float(rng.integers(0, 4)) * 90.0
        M = cv2.getRotationMatrix2D((img_size / 2, img_size / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (img_size, img_size))
        noise = rng.integers(-20, 21, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    elif choice == 2:
        # Diagonal stripe pattern
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        stripe_width = int(rng.integers(4, 16))
        for row in range(img_size):
            for col in range(img_size):
                if ((row + col) // stripe_width) % 2 == 0:
                    img[row, col] = int(rng.integers(100, 200))
                else:
                    img[row, col] = int(rng.integers(30, 80))
        img = cv2.GaussianBlur(img, (3, 3), 0)
        noise = rng.integers(-10, 11, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    else:
        # Random rectangles / blobs (no circles)
        img = rng.integers(40, 100, size=(img_size, img_size), dtype=np.uint8)
        n_rects = int(rng.integers(2, 6))
        for _ in range(n_rects):
            x1 = int(rng.integers(0, img_size - 8))
            y1 = int(rng.integers(0, img_size - 8))
            x2 = int(rng.integers(x1 + 4, min(x1 + img_size // 2, img_size)))
            y2 = int(rng.integers(y1 + 4, min(y1 + img_size // 2, img_size)))
            color = int(rng.integers(120, 230))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
        noise = rng.integers(-15, 16, size=img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic HOG+SVM training data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pos-dir", default="positives", help="Output dir for positive samples")
    parser.add_argument("--neg-dir", default="negatives", help="Output dir for negative samples")
    parser.add_argument("--n-pos", type=int, default=500, help="Number of positive images")
    parser.add_argument("--n-neg", type=int, default=1500, help="Number of negative images")
    parser.add_argument("--img-size", type=int, default=64, help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def compute_stats(directory: str, label: str) -> dict:
    """Compute mean/std pixel statistics over all images in a directory."""
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    means, stds = [], []
    for fname in files:
        img = cv2.imread(os.path.join(directory, fname), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            means.append(float(img.mean()))
            stds.append(float(img.std()))
    return {
        "label": label,
        "count": len(files),
        "mean_pixel": float(np.mean(means)) if means else 0.0,
        "std_pixel": float(np.mean(stds)) if stds else 0.0,
    }


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    os.makedirs(args.pos_dir, exist_ok=True)
    os.makedirs(args.neg_dir, exist_ok=True)

    print(f"Generating {args.n_pos} positive samples → {args.pos_dir}/")
    for i in range(args.n_pos):
        img = make_positive(args.img_size, rng)
        path = os.path.join(args.pos_dir, f"{i:05d}.png")
        cv2.imwrite(path, img)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_pos} positives written")

    print(f"\nGenerating {args.n_neg} negative samples → {args.neg_dir}/")
    for i in range(args.n_neg):
        img = make_negative(args.img_size, rng)
        path = os.path.join(args.neg_dir, f"{i:05d}.png")
        cv2.imwrite(path, img)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{args.n_neg} negatives written")

    # ---- Statistics --------------------------------------------------------
    pos_stats = compute_stats(args.pos_dir, "positive")
    neg_stats = compute_stats(args.neg_dir, "negative")

    lines = [
        "=== Dataset Statistics ===",
        f"Image size      : {args.img_size}×{args.img_size}",
        f"Random seed     : {args.seed}",
        "",
        f"Class           : {pos_stats['label']}",
        f"  Count         : {pos_stats['count']}",
        f"  Mean pixel    : {pos_stats['mean_pixel']:.2f}",
        f"  Std  pixel    : {pos_stats['std_pixel']:.2f}",
        "",
        f"Class           : {neg_stats['label']}",
        f"  Count         : {neg_stats['count']}",
        f"  Mean pixel    : {neg_stats['mean_pixel']:.2f}",
        f"  Std  pixel    : {neg_stats['std_pixel']:.2f}",
        "",
        f"Class balance   : 1 : {args.n_neg / args.n_pos:.1f} (pos : neg)",
    ]
    stats_text = "\n".join(lines)
    print("\n" + stats_text)

    stats_path = "dataset_stats.txt"
    with open(stats_path, "w") as f:
        f.write(stats_text + "\n")
    print(f"\nStatistics saved to {stats_path}")

    print("\nDone. Next step:")
    print("  python 02_train_hog_svm.py "
          f"--pos {args.pos_dir}/ --neg {args.neg_dir}/ --output detector.xml")


if __name__ == "__main__":
    main()
