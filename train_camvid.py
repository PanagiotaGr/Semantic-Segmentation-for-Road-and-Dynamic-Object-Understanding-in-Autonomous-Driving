import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import jaccard_score

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
NUM_CLASSES = 12
IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 10
LR = 1e-3

# -----------------------------
# CamVid class mapping
# Μπορείς να το αλλάξεις ανάλογα με το label set σου
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
    (0, 0, 0): 11         # Void / Background
}

def rgb_to_mask(mask):
    mask = np.array(mask)
    class_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for rgb, idx in CLASS_COLORS.items():
        matches = np.all(mask == np.array(rgb), axis=-1)
        class_mask[matches] = idx
    return class_mask

class CamVidDataset(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(256, 256)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.img_size = img_size
        self.img_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB").resize(self.img_size)
        mask = Image.open(mask_path).convert("RGB").resize(self.img_size, Image.NEAREST)

        image = self.img_transform(image)
        mask = torch.tensor(rgb_to_mask(mask), dtype=torch.long)

        return image, mask

def mean_iou(preds, targets, num_classes):
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious) if len(ious) > 0 else 0.0

train_dataset = CamVidDataset(
    os.path.join(DATA_DIR, "train"),
    os.path.join(DATA_DIR, "train_labels"),
    IMG_SIZE
)

val_dataset = CamVidDataset(
    os.path.join(DATA_DIR, "val"),
    os.path.join(DATA_DIR, "val_labels"),
    IMG_SIZE
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_miou = 0.0

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
    miou_scores = []

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - Val"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            miou_scores.append(mean_iou(preds, masks, NUM_CLASSES))

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_miou = np.mean(miou_scores)

    print(f"\nEpoch {epoch+1}:")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {avg_val_loss:.4f}")
    print(f"Val mIoU:   {avg_miou:.4f}")

    if avg_miou > best_miou:
        best_miou = avg_miou
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/best_unet_camvid.pth")
        print("Saved best model.")

print(f"Training complete. Best mIoU: {best_miou:.4f}")
