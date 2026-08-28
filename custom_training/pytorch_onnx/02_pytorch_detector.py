"""
02_pytorch_detector.py
=======================
Train a simple custom object detector in PyTorch.

Architecture
------------
  SimpleDetector
    Backbone: 4 x (Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d)
    Shared FC: Flatten -> Linear(512*7*7, 512) -> ReLU -> Dropout
    Classification head: Linear(512, num_classes)
    Regression head:     Linear(512, 4)  [normalised cx, cy, w, h of primary object]

This is a "dominant object" detector — it predicts one box per image.
For multi-object detection on the same dataset see Part 1.2 of tutorial.md.

Usage
-----
  # Generate data first:
  python 01_synthetic_detection_dataset.py --output ./data --num_images 1000

  # Train:
  python 02_pytorch_detector.py --data ./data --epochs 30 --lr 1e-3 --batch 32 --output ./runs

  # Resume from checkpoint:
  python 02_pytorch_detector.py --data ./data --epochs 50 --resume ./runs/best.pth
"""

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class SimpleDetector(nn.Module):
    """
    Lightweight CNN backbone + dual detection head.

    Input:  (B, 3, 224, 224)
    Output: cls_logits (B, num_classes), box_pred (B, 4) normalised (cx,cy,w,h)
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: 224 -> 112 -> 56 -> 28 -> 14
        self.backbone = nn.Sequential(
            ConvBlock(3,   32),   # -> (B, 32,  112, 112)
            ConvBlock(32,  64),   # -> (B, 64,   56,  56)
            ConvBlock(64,  128),  # -> (B, 128,  28,  28)
            ConvBlock(128, 256),  # -> (B, 256,  14,  14)
        )

        # Global average pool -> (B, 256, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Shared representation
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Classification head
        self.cls_head = nn.Linear(512, num_classes)

        # Regression head (outputs cx, cy, w, h all in [0, 1])
        self.reg_head = nn.Sequential(
            nn.Linear(512, 4),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.gap(feat)
        feat = self.shared(feat)
        cls_logits = self.cls_head(feat)
        box_pred = self.reg_head(feat)
        return cls_logits, box_pred


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal loss for multi-class classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_weight = self.alpha * (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def detection_loss(cls_logits, box_pred, cls_targets, box_targets,
                   cls_weight=1.0, reg_weight=1.0, use_focal=True):
    """Combined classification + regression loss."""
    if use_focal:
        cls_loss = FocalLoss()(cls_logits, cls_targets)
    else:
        cls_loss = F.cross_entropy(cls_logits, cls_targets)

    # Only compute regression loss where we have valid boxes (all-zero box = no object)
    valid = (box_targets.sum(dim=1) > 0).float().unsqueeze(1)
    reg_loss = F.smooth_l1_loss(box_pred * valid, box_targets * valid, beta=1.0)

    total = cls_weight * cls_loss + reg_weight * reg_loss
    return total, cls_loss.item(), reg_loss.item()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DetectionDataset(Dataset):
    """
    Loads images and COCO JSON annotations.

    Targets are the dominant (largest-area) object per image.
    Box format returned: normalised (cx, cy, w, h) in [0, 1].
    """

    IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, img_dir, ann_file, img_size=224, augment=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.augment = augment

        with open(ann_file) as f:
            data = json.load(f)

        self.images = {im['id']: im for im in data['images']}
        self.img_ids = [im['id'] for im in data['images']]
        self.ann_by_img = defaultdict(list)
        for ann in data['annotations']:
            self.ann_by_img[ann['image_id']].append(ann)

    # --- augmentation helpers ---

    def _augment(self, img):
        # Horizontal flip
        if random.random() < 0.5:
            img = cv2.flip(img, 1)
        # Brightness/contrast jitter
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)  # contrast
            beta = random.uniform(-20, 20)    # brightness
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)
        return img

    # --- core ---

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        info = self.images[img_id]
        fpath = os.path.join(self.img_dir, info['file_name'])

        img = cv2.imread(fpath)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {fpath}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]

        if self.augment:
            img = self._augment(img)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Normalise
        img = img.astype(np.float32) / 255.0
        img = (img - self.IMG_MEAN) / self.IMG_STD
        img = torch.from_numpy(img.transpose(2, 0, 1))  # (3, H, W)

        # Pick dominant (largest area) object
        anns = self.ann_by_img.get(img_id, [])
        if not anns:
            # No annotation: class 0, zero box
            cls_target = 0
            box_target = torch.zeros(4, dtype=torch.float32)
        else:
            ann = max(anns, key=lambda a: a['bbox'][2] * a['bbox'][3])
            cls_target = ann['category_id']
            x, y, bw, bh = ann['bbox']
            cx = (x + bw / 2) / W
            cy = (y + bh / 2) / H
            wn = bw / W
            hn = bh / H
            box_target = torch.tensor([cx, cy, wn, hn], dtype=torch.float32)

        return img, torch.tensor(cls_target, dtype=torch.long), box_target

    def __len__(self):
        return len(self.img_ids)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, device, scaler=None):
    model.train()
    total_loss = total_cls = total_reg = 0.0

    for imgs, cls_tgt, box_tgt in loader:
        imgs = imgs.to(device)
        cls_tgt = cls_tgt.to(device)
        box_tgt = box_tgt.to(device)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast('cuda'):
                cls_logits, box_pred = model(imgs)
                loss, c, r = detection_loss(cls_logits, box_pred, cls_tgt, box_tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            cls_logits, box_pred = model(imgs)
            loss, c, r = detection_loss(cls_logits, box_pred, cls_tgt, box_tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        total_cls += c
        total_reg += r

    n = len(loader)
    return total_loss / n, total_cls / n, total_reg / n


def evaluate(model, loader, device):
    model.eval()
    total_loss = total_cls = total_reg = 0.0
    correct = 0
    count = 0

    with torch.no_grad():
        for imgs, cls_tgt, box_tgt in loader:
            imgs = imgs.to(device)
            cls_tgt = cls_tgt.to(device)
            box_tgt = box_tgt.to(device)
            cls_logits, box_pred = model(imgs)
            loss, c, r = detection_loss(cls_logits, box_pred, cls_tgt, box_tgt)
            total_loss += loss.item()
            total_cls += c
            total_reg += r
            correct += (cls_logits.argmax(1) == cls_tgt).sum().item()
            count += len(imgs)

    n = len(loader)
    return total_loss / n, total_cls / n, total_reg / n, correct / max(count, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train SimpleDetector')
    parser.add_argument('--data', default='./data',
                        help='Dataset directory (must contain images/ and annotations.json)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Fraction of data used for validation')
    parser.add_argument('--output', default='./runs',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--resume', default='',
                        help='Path to checkpoint to resume from')
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision training (requires CUDA)')
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Dataset
    ann_file = os.path.join(args.data, 'annotations.json')
    img_dir = os.path.join(args.data, 'images')
    full_dataset = DetectionDataset(img_dir, ann_file, augment=False)

    n_val = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    # Enable augmentation on train split
    train_ds.dataset.augment = False  # keep simple for split safety; use Albumentations for real use

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    print(f"Train: {n_train}  Val: {n_val}")

    # Model
    model = SimpleDetector(num_classes=args.num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3,
    )

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # Training loop
    log_path = os.path.join(args.output, 'train_log.csv')
    with open(log_path, 'w') as log_f:
        log_f.write('epoch,train_loss,train_cls,train_reg,val_loss,val_cls,val_reg,val_acc\n')

    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_cls, tr_reg = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, scaler)
        vl_loss, vl_cls, vl_reg, vl_acc = evaluate(model, val_loader, device)

        lr_now = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:3d}/{args.epochs} | "
            f"train loss={tr_loss:.4f} (cls={tr_cls:.4f} reg={tr_reg:.4f}) | "
            f"val loss={vl_loss:.4f} (cls={vl_cls:.4f} reg={vl_reg:.4f}) | "
            f"val acc={vl_acc:.3f} | lr={lr_now:.6f}"
        )

        with open(log_path, 'a') as log_f:
            log_f.write(f"{epoch+1},{tr_loss:.6f},{tr_cls:.6f},{tr_reg:.6f},"
                        f"{vl_loss:.6f},{vl_cls:.6f},{vl_reg:.6f},{vl_acc:.6f}\n")

        # Save best checkpoint
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            ckpt_path = os.path.join(args.output, 'best.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': vl_loss,
                'val_acc': vl_acc,
                'num_classes': args.num_classes,
            }, ckpt_path)
            print(f"  -> Saved best checkpoint: {ckpt_path}")

        # Save latest checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            ckpt_path = os.path.join(args.output, f'epoch_{epoch+1:03d}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': vl_loss,
                'num_classes': args.num_classes,
            }, ckpt_path)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best checkpoint: {os.path.join(args.output, 'best.pth')}")
    print(f"Log: {log_path}")


if __name__ == '__main__':
    main()
