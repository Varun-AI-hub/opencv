"""
01_synthetic_detection_dataset.py
==================================
Generate a synthetic object-detection dataset for PyTorch training experiments.

Output
------
  output_dir/
    images/      224x224 PNG images
    labels/      YOLO txt labels (one file per image)
    annotations.json   COCO-format annotation file

Classes
-------
  0  circle
  1  rectangle
  2  triangle

Usage
-----
  python 01_synthetic_detection_dataset.py --output ./data --num_images 1000
"""

import argparse
import json
import math
import os
import random
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def random_color():
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_circle(img, cx, cy, r, color):
    cv2.circle(img, (cx, cy), r, color, -1)
    x1, y1 = cx - r, cy - r
    x2, y2 = cx + r, cy + r
    return x1, y1, x2, y2


def draw_rectangle(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    return x1, y1, x2, y2


def draw_triangle(img, cx, cy, r, color):
    pts = []
    for angle_deg in [90, 210, 330]:
        angle = math.radians(angle_deg)
        px = int(cx + r * math.cos(angle))
        py = int(cy - r * math.sin(angle))
        pts.append([px, py])
    pts = np.array([pts], dtype=np.int32)
    cv2.fillPoly(img, pts, color)
    xs = [p[0] for p in pts[0]]
    ys = [p[1] for p in pts[0]]
    return min(xs), min(ys), max(xs), max(ys)


# ---------------------------------------------------------------------------
# Single image generator
# ---------------------------------------------------------------------------

CLASS_NAMES = ['circle', 'rectangle', 'triangle']
IMG_SIZE = 224


def generate_image(num_objects_range=(1, 4)):
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    # Random background colour (dark)
    bg = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
    img[:] = bg

    num_objects = random.randint(*num_objects_range)
    annotations = []  # list of (class_id, x1, y1, x2, y2)

    for _ in range(num_objects):
        cls = random.randint(0, 2)
        color = random_color()
        margin = 20
        r = random.randint(15, 45)

        cx = random.randint(margin + r, IMG_SIZE - margin - r)
        cy = random.randint(margin + r, IMG_SIZE - margin - r)

        if cls == 0:  # circle
            x1, y1, x2, y2 = draw_circle(img, cx, cy, r, color)
        elif cls == 1:  # rectangle
            half_w = random.randint(15, 50)
            half_h = random.randint(15, 50)
            rx1 = max(0, cx - half_w)
            ry1 = max(0, cy - half_h)
            rx2 = min(IMG_SIZE - 1, cx + half_w)
            ry2 = min(IMG_SIZE - 1, cy + half_h)
            x1, y1, x2, y2 = draw_rectangle(img, rx1, ry1, rx2, ry2, color)
        else:  # triangle
            x1, y1, x2, y2 = draw_triangle(img, cx, cy, r, color)

        # Clamp to image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(IMG_SIZE - 1, int(x2))
        y2 = min(IMG_SIZE - 1, int(y2))

        if x2 > x1 and y2 > y1:
            annotations.append((cls, x1, y1, x2, y2))

    return img, annotations


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(output_dir, num_images=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    img_dir = os.path.join(output_dir, 'images')
    lbl_dir = os.path.join(output_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # COCO scaffold
    coco = {
        'info': {'description': 'Synthetic shape detection dataset', 'version': '1.0'},
        'categories': [
            {'id': 0, 'name': 'circle', 'supercategory': 'shape'},
            {'id': 1, 'name': 'rectangle', 'supercategory': 'shape'},
            {'id': 2, 'name': 'triangle', 'supercategory': 'shape'},
        ],
        'images': [],
        'annotations': [],
    }

    ann_id = 0
    issues = 0

    for i in range(num_images):
        fname = f'{i:05d}.png'
        fpath = os.path.join(img_dir, fname)

        img, anns = generate_image()
        cv2.imwrite(fpath, img)

        # COCO image record
        coco['images'].append({
            'id': i,
            'file_name': fname,
            'width': IMG_SIZE,
            'height': IMG_SIZE,
        })

        # YOLO label file
        yolo_lines = []
        for cls, x1, y1, x2, y2 in anns:
            bw = x2 - x1
            bh = y2 - y1
            # Validate
            if bw <= 0 or bh <= 0:
                issues += 1
                continue
            if x1 < 0 or y1 < 0 or x2 >= IMG_SIZE or y2 >= IMG_SIZE:
                issues += 1
                continue

            # COCO bbox: x, y, width, height (absolute)
            coco['annotations'].append({
                'id': ann_id,
                'image_id': i,
                'category_id': cls,
                'bbox': [x1, y1, bw, bh],
                'area': bw * bh,
                'iscrowd': 0,
            })

            # YOLO: class cx cy w h (normalised)
            cx_n = (x1 + bw / 2) / IMG_SIZE
            cy_n = (y1 + bh / 2) / IMG_SIZE
            w_n = bw / IMG_SIZE
            h_n = bh / IMG_SIZE
            yolo_lines.append(f'{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}')

            ann_id += 1

        lbl_path = os.path.join(lbl_dir, f'{i:05d}.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

    # Save COCO JSON
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(coco, f, indent=2)

    return num_images, ann_id, issues


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_dataset(output_dir):
    ann_path = os.path.join(output_dir, 'annotations.json')
    with open(ann_path) as f:
        coco = json.load(f)

    img_dir = os.path.join(output_dir, 'images')
    errors = []

    for ann in coco['annotations']:
        img_id = ann['image_id']
        img_info = next(im for im in coco['images'] if im['id'] == img_id)
        W, H = img_info['width'], img_info['height']
        x, y, bw, bh = ann['bbox']
        if bw <= 0 or bh <= 0:
            errors.append(f"ann {ann['id']}: zero-size box")
        if x < 0 or y < 0 or (x + bw) > W or (y + bh) > H:
            errors.append(f"ann {ann['id']}: box out of bounds {ann['bbox']}")

    img_count = len(coco['images'])
    for img_info in coco['images']:
        fpath = os.path.join(img_dir, img_info['file_name'])
        if not os.path.isfile(fpath):
            errors.append(f"missing image: {img_info['file_name']}")

    return img_count, len(coco['annotations']), errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic detection dataset')
    parser.add_argument('--output', default='./data', help='Output directory')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate (default: 1000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    print(f"Generating {args.num_images} images in '{args.output}' ...")
    n_imgs, n_anns, issues = generate_dataset(args.output, args.num_images, args.seed)
    print(f"  Generated {n_imgs} images, {n_anns} annotations, {issues} skipped (bad boxes).")

    print("Verifying dataset ...")
    img_count, ann_count, errors = verify_dataset(args.output)
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors[:20]:
            print(f"    {e}")
        sys.exit(1)
    else:
        print(f"  OK: {img_count} images, {ann_count} annotations, no issues found.")
        print(f"\nDataset ready:")
        print(f"  Images:      {os.path.join(args.output, 'images')}/")
        print(f"  YOLO labels: {os.path.join(args.output, 'labels')}/")
        print(f"  COCO JSON:   {os.path.join(args.output, 'annotations.json')}")


if __name__ == '__main__':
    main()
