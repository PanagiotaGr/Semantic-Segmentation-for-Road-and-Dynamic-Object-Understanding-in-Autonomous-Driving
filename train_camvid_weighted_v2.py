import os
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/CamVid"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

CSV_PATH = "weighted_history.csv"

CLASS_NAMES = ["background", "road", "sidewalk", "vehicle", "pedestrian"]

# -----------------------------
# Original CamVid colors
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

class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_size = img_size
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("RGB").resize(self.img_size, Image.NEAREST)

        image = self.transform(image)
        mask = torch.tensor(rgb_to_mask(mask), dtype=torch.long)

        return image, mask

def compute_iou_per_class(preds, targets, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(np.nan)
        else:
            ious.append(intersection / union)
    return ious

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

# -----------------------------
# Data
# -----------------------------
train_loader = DataLoader(
    CamVidDataset(os.path.join(DATA_DIR, "train"), os.path.join(DATA_DIR, "train_labels")),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    CamVidDataset(os.path.join(DATA_DIR, "val"), os.path.join(DATA_DIR, "val_labels")),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

# -----------------------------
# Model
# -----------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

# Πιο ήπια weights για να μη χαλάει πολύ το global mIoU
class_weights = torch.tensor([1.0, 1.0, 1.1, 1.6, 2.8], dtype=torch.float32).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_miou = -1.0
best_ped_iou = -1.0
best_epoch_miou = -1
best_epoch_ped = -1

# -----------------------------
# CSV logging
# -----------------------------
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "train_loss", "val_loss", "miou",
        "iou_background", "iou_road", "iou_sidewalk", "iou_vehicle", "iou_pedestrian"
    ])

# -----------------------------
# Train loop
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Train"):
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    all_ious = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            targets = masks.cpu().numpy()

            for p, t in zip(preds, targets):
                all_ious.append(compute_iou_per_class(p, t, NUM_CLASSES))

    all_ious = np.array(all_ious, dtype=np.float32)
    class_iou = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(class_iou)
    pedestrian_iou = class_iou[4]

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val mIoU:   {miou:.4f}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"IoU {name}: {class_iou[i]:.4f}")

    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch + 1,
            f"{avg_train_loss:.4f}",
            f"{avg_val_loss:.4f}",
            f"{miou:.4f}",
            f"{class_iou[0]:.4f}",
            f"{class_iou[1]:.4f}",
            f"{class_iou[2]:.4f}",
            f"{class_iou[3]:.4f}",
            f"{class_iou[4]:.4f}"
        ])

    if miou > best_miou:
        best_miou = miou
        best_epoch_miou = epoch + 1
        save_checkpoint(model, os.path.join(MODEL_DIR, "best_unet_camvid_weighted_miou.pth"))
        print("Saved best mIoU model.")

    if pedestrian_iou > best_ped_iou:
        best_ped_iou = pedestrian_iou
        best_epoch_ped = epoch + 1
        save_checkpoint(model, os.path.join(MODEL_DIR, "best_unet_camvid_weighted_pedestrian.pth"))
        print("Saved best pedestrian model.")

print("\nTraining complete.")
print(f"Best mIoU: {best_miou:.4f} at epoch {best_epoch_miou}")
print(f"Best pedestrian IoU: {best_ped_iou:.4f} at epoch {best_epoch_ped}")
print(f"History saved to: {CSV_PATH}")
