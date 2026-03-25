import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/CamVid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 10
LR = 5e-5

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES = ["background", "road", "sidewalk", "vehicle", "pedestrian"]

# -----------------------------
# CamVid mapping (same as before)
# -----------------------------
CLASS_COLORS = {
    (128, 128, 128): 0,
    (128, 0, 0): 1,
    (192, 192, 128): 2,
    (128, 64, 128): 3,
    (0, 0, 192): 4,
    (128, 128, 0): 5,
    (192, 128, 128): 6,
    (64, 64, 128): 7,
    (64, 0, 128): 8,
    (64, 64, 0): 9,
    (0, 128, 192): 10,
    (0, 0, 0): 11
}

REDUCED_MAP = {
    0: 0, 1: 0, 2: 0,
    3: 1,
    4: 2,
    5: 0, 6: 0, 7: 0,
    8: 3,
    9: 4,
    10: 0,
    11: 0
}

def rgb_to_mask(mask):
    mask = np.array(mask)
    orig_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for rgb, idx in CLASS_COLORS.items():
        matches = np.all(mask == np.array(rgb), axis=-1)
        orig_mask[matches] = idx

    reduced_mask = np.zeros_like(orig_mask)
    for orig_cls, reduced_cls in REDUCED_MAP.items():
        reduced_mask[orig_mask == orig_cls] = reduced_cls

    return reduced_mask

# -----------------------------
# Dataset
# -----------------------------
class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_dir, self.img_files[idx])).convert("RGB").resize(IMG_SIZE)
        mask = Image.open(os.path.join(self.mask_dir, self.mask_files[idx])).convert("RGB").resize(IMG_SIZE, Image.NEAREST)

        mask = rgb_to_mask(mask)

        return np.array(img), mask

# -----------------------------
# Load model
# -----------------------------
processor = SegformerImageProcessor(do_resize=False)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# -----------------------------
# Data loaders
# -----------------------------
train_loader = DataLoader(
    CamVidDataset(os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "train_labels")),
    batch_size=BATCH_SIZE, shuffle=True
)

val_loader = DataLoader(
    CamVidDataset(os.path.join(DATA_DIR, "val"), os.path.join(DATA_DIR, "val_labels")),
    batch_size=BATCH_SIZE
)

# -----------------------------
# IoU
# -----------------------------
def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        inter = np.logical_and(pred == cls, target == cls).sum()
        union = np.logical_or(pred == cls, target == cls).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(inter / union)
    return ious

# -----------------------------
# Training
# -----------------------------
best_miou = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        inputs = processor(images=list(imgs), return_tensors="pt").to(DEVICE)
        labels = torch.tensor(masks).long().to(DEVICE)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    all_ious = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
            inputs = processor(images=list(imgs), return_tensors="pt").to(DEVICE)

            outputs = model(**inputs)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            for p, t in zip(preds, masks):
                all_ious.append(compute_iou(p, t, NUM_CLASSES))

    all_ious = np.array(all_ious)
    class_iou = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(class_iou)

    print(f"\nEpoch {epoch+1} mIoU: {miou:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"IoU {name}: {class_iou[i]:.4f}")

    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_segformer.pth"))
        print("Saved best SegFormer model")

print("Training complete. Best mIoU:", best_miou)
