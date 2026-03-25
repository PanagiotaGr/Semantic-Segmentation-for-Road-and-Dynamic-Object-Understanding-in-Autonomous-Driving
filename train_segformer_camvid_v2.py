import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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
# CamVid original color mapping
# -----------------------------
CLASS_COLORS = {
    (128, 128, 128): 0,   # Sky
    (128, 0, 0): 1,       # Building
    (192, 192, 128): 2,   # Pole
    (128, 64, 128): 3,    # Road
    (0, 0, 192): 4,       # Pavement
    (128, 128, 0): 5,     # Tree
    (192, 128, 128): 6,   # SignSymbol
    (64, 64, 128): 7,     # Fence
    (64, 0, 128): 8,      # Car
    (64, 64, 0): 9,       # Pedestrian
    (0, 128, 192): 10,    # Bicyclist
    (0, 0, 0): 11         # Void
}

# -----------------------------
# Reduced mapping
# 0 background, 1 road, 2 sidewalk, 3 vehicle, 4 pedestrian
# -----------------------------
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
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_size = img_size

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("RGB").resize(self.img_size, Image.NEAREST)

        image = np.array(image, dtype=np.uint8)
        mask = rgb_to_mask(mask)

        return image, torch.tensor(mask, dtype=torch.long)

# -----------------------------
# IoU computation
# -----------------------------
def compute_iou_per_class(pred, target, num_classes):
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
# DataLoaders
# -----------------------------
train_loader = DataLoader(
    CamVidDataset(
        os.path.join(DATA_DIR, "train"),
        os.path.join(DATA_DIR, "train_labels"),
        IMG_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    CamVidDataset(
        os.path.join(DATA_DIR, "val"),
        os.path.join(DATA_DIR, "val_labels"),
        IMG_SIZE
    ),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# -----------------------------
# SegFormer processor + model
# -----------------------------
processor = SegformerImageProcessor(do_resize=False)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

best_miou = -1.0

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        # imgs comes as batch of numpy arrays
        img_list = [img.numpy() if torch.is_tensor(img) else img for img in imgs]
        inputs = processor(images=img_list, return_tensors="pt").to(DEVICE)

        labels = masks.long().to(DEVICE)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    all_ious = []
    val_loss = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
            img_list = [img.numpy() if torch.is_tensor(img) else img for img in imgs]
            inputs = processor(images=img_list, return_tensors="pt").to(DEVICE)
            labels = masks.long().to(DEVICE)

            outputs = model(**inputs, labels=labels)
            val_loss += outputs.loss.item()

            logits = outputs.logits

            # Upsample to ground-truth size
            logits = F.interpolate(
                logits,
                size=labels.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            for p, t in zip(preds, targets):
                all_ious.append(compute_iou_per_class(p, t, NUM_CLASSES))

    all_ious = np.array(all_ious, dtype=np.float32)
    class_iou = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(class_iou)

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val mIoU:   {miou:.4f}")

    for i, name in enumerate(CLASS_NAMES):
        print(f"IoU {name}: {class_iou[i]:.4f}")

    if miou > best_miou:
        best_miou = miou
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_segformer_camvid_v2.pth"))
        print("Saved best SegFormer model.")

print(f"\nTraining complete. Best SegFormer mIoU: {best_miou:.4f}")
