"""
01_generate_dataset.py — Generate a synthetic YOLO dataset for pipeline testing.

Creates 500 training + 100 validation images (128x128 PNG) with three classes:
  0: circle
  1: rectangle
  2: triangle

Labels are written in YOLO format:  class_id  cx  cy  w  h  (all normalized 0-1)
A data.yaml config file is written to the dataset root.

Usage:
    python 01_generate_dataset.py [--output synthetic_dataset] [--seed 42]

Requirements:
    pip install opencv-python-headless numpy pyyaml
"""

import argparse
import math
import os
import random

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CLASS_NAMES = ["circle", "rectangle", "triangle"]
IMG_SIZE = 128          # square image side length in pixels
MIN_SHAPE_SIZE = 16     # minimum bounding-box side (pixels)
MAX_SHAPE_SIZE = 48     # maximum bounding-box side (pixels)
MAX_OBJECTS = 3         # maximum objects per image


# ---------------------------------------------------------------------------
# Drawing helpers — each returns (img, bbox_xyxy)
# bbox_xyxy = (x1, y1, x2, y2) in pixel coordinates (integers)
# ---------------------------------------------------------------------------

def draw_circle(img, color):
    """Draw a filled circle at a random position. Returns updated img and bbox."""
    h, w = img.shape[:2]
    radius = random.randint(MIN_SHAPE_SIZE // 2, MAX_SHAPE_SIZE // 2)
    cx = random.randint(radius, w - radius - 1)
    cy = random.randint(radius, h - radius - 1)
    cv2.circle(img, (cx, cy), radius, color, -1)
    x1, y1 = cx - radius, cy - radius
    x2, y2 = cx + radius, cy + radius
    return img, (x1, y1, x2, y2)


def draw_rectangle(img, color):
    """Draw a filled rectangle at a random position. Returns updated img and bbox."""
    h, w = img.shape[:2]
    bw = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    bh = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    x1 = random.randint(0, w - bw - 1)
    y1 = random.randint(0, h - bh - 1)
    x2, y2 = x1 + bw, y1 + bh
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return img, (x1, y1, x2, y2)


def draw_triangle(img, color):
    """Draw a filled triangle at a random position. Returns updated img and bbox."""
    h, w = img.shape[:2]
    size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
    # Equilateral triangle pointing upward
    cx = random.randint(size, w - size - 1)
    cy = random.randint(size, h - size - 1)
    half = size // 2
    height = int(size * math.sqrt(3) / 2)
    pts = np.array([
        [cx,            cy - height // 2],
        [cx - half,     cy + height // 2],
        [cx + half,     cy + height // 2],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], color)
    x1 = max(0, int(pts[:, 0].min()))
    y1 = max(0, int(pts[:, 1].min()))
    x2 = min(w - 1, int(pts[:, 0].max()))
    y2 = min(h - 1, int(pts[:, 1].max()))
    return img, (x1, y1, x2, y2)


DRAW_FUNCS = [draw_circle, draw_rectangle, draw_triangle]


# ---------------------------------------------------------------------------
# Image + label generation
# ---------------------------------------------------------------------------

def random_color():
    """Return a random BGR color tuple."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def make_background(size):
    """Return a random background image: either Gaussian noise or a solid color."""
    if random.random() < 0.5:
        # Gaussian noise background
        bg = np.random.randint(0, 80, (size, size, 3), dtype=np.uint8)
    else:
        # Solid low-intensity color
        color = [random.randint(0, 80) for _ in range(3)]
        bg = np.full((size, size, 3), color, dtype=np.uint8)
    return bg


def pixel_to_yolo(bbox_xyxy, img_w, img_h):
    """Convert pixel (x1,y1,x2,y2) to YOLO (cx,cy,w,h) normalized."""
    x1, y1, x2, y2 = bbox_xyxy
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1) / float(img_w)
    bh = (y2 - y1) / float(img_h)
    # Clamp to [0, 1]
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)
    return cx, cy, bw, bh


def generate_image(img_size=IMG_SIZE, max_objects=MAX_OBJECTS):
    """
    Generate one synthetic image with random shapes.
    Returns:
        img: numpy uint8 BGR image
        labels: list of (class_id, cx, cy, w, h) — all floats normalized 0-1
    """
    img = make_background(img_size)
    labels = []
    n_objects = random.randint(1, max_objects)
    for _ in range(n_objects):
        class_id = random.randint(0, len(CLASS_NAMES) - 1)
        color = random_color()
        draw_fn = DRAW_FUNCS[class_id]
        img, bbox = draw_fn(img, color)
        cx, cy, bw, bh = pixel_to_yolo(bbox, img_size, img_size)
        # Sanity check: skip degenerate boxes
        if bw > 0.01 and bh > 0.01:
            labels.append((class_id, cx, cy, bw, bh))
    return img, labels


def write_label_file(label_path, labels):
    """Write a YOLO-format label file."""
    with open(label_path, "w") as f:
        for class_id, cx, cy, bw, bh in labels:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def generate_split(images_dir, labels_dir, n_images, start_idx=0):
    """
    Generate `n_images` images + label files into the given directories.
    Returns class_counts dict.
    """
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    class_counts = {name: 0 for name in CLASS_NAMES}
    for i in range(n_images):
        img_idx = start_idx + i
        img, labels = generate_image()
        img_filename = f"img_{img_idx:05d}.png"
        lbl_filename = f"img_{img_idx:05d}.txt"
        cv2.imwrite(os.path.join(images_dir, img_filename), img)
        write_label_file(os.path.join(labels_dir, lbl_filename), labels)
        for class_id, *_ in labels:
            class_counts[CLASS_NAMES[class_id]] += 1
    return class_counts


def write_data_yaml(output_dir, num_train, num_val):
    """Write data.yaml for ultralytics training."""
    abs_output = os.path.abspath(output_dir)
    data = {
        "path": abs_output,
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    yaml_path = os.path.join(output_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f"  data.yaml written to: {yaml_path}")
    print(f"  Absolute dataset path: {abs_output}")
    return yaml_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic YOLO dataset with circles, rectangles, triangles."
    )
    parser.add_argument(
        "--output", default="synthetic_dataset",
        help="Output directory for the dataset (default: synthetic_dataset)"
    )
    parser.add_argument(
        "--train", type=int, default=500,
        help="Number of training images (default: 500)"
    )
    parser.add_argument(
        "--val", type=int, default=100,
        help="Number of validation images (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--imgsize", type=int, default=IMG_SIZE,
        help=f"Image side length in pixels (default: {IMG_SIZE})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("Synthetic YOLO Dataset Generator")
    print("=" * 60)
    print(f"Output directory : {args.output}")
    print(f"Image size       : {args.imgsize}x{args.imgsize}")
    print(f"Training images  : {args.train}")
    print(f"Validation images: {args.val}")
    print(f"Classes          : {CLASS_NAMES}")
    print(f"Random seed      : {args.seed}")
    print()

    # Directory layout:
    #   synthetic_dataset/
    #     images/train/   images/val/
    #     labels/train/   labels/val/
    #     data.yaml
    base = args.output
    train_img_dir = os.path.join(base, "images", "train")
    train_lbl_dir = os.path.join(base, "labels", "train")
    val_img_dir   = os.path.join(base, "images", "val")
    val_lbl_dir   = os.path.join(base, "labels", "val")

    # Override global IMG_SIZE if user changed it
    global IMG_SIZE
    IMG_SIZE = args.imgsize

    print("Generating training set...")
    train_counts = generate_split(train_img_dir, train_lbl_dir, args.train, start_idx=0)
    print(f"  Saved {args.train} images to {train_img_dir}")
    print(f"  Object counts: {train_counts}")

    print("Generating validation set...")
    val_counts = generate_split(val_img_dir, val_lbl_dir, args.val, start_idx=args.train)
    print(f"  Saved {args.val} images to {val_img_dir}")
    print(f"  Object counts: {val_counts}")

    print("Writing data.yaml...")
    yaml_path = write_data_yaml(base, args.train, args.val)

    print()
    print("=" * 60)
    print("Dataset generation complete!")
    print(f"Total training objects  : {sum(train_counts.values())}")
    print(f"Total validation objects: {sum(val_counts.values())}")
    print()
    print("Next step — train YOLOv8:")
    print(f"  python 02_train_yolov8.py --data {yaml_path} --model yolov8n --epochs 50")
    print("=" * 60)


if __name__ == "__main__":
    main()
