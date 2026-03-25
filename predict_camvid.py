import os
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

# -----------------------------
# Config
# -----------------------------
DATA_DIR = "data/CamVid"
MODEL_PATH = "models/best_unet_camvid.pth"
OUTPUT_DIR = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 12
IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Class colors (same as training)
# -----------------------------
CLASS_COLORS = {
    0: (128, 128, 128),   # Sky
    1: (128, 0, 0),       # Building
    2: (192, 192, 128),   # Pole
    3: (128, 64, 128),    # Road
    4: (0, 0, 192),       # Pavement
    5: (128, 128, 0),     # Tree
    6: (192, 128, 128),   # SignSymbol
    7: (64, 64, 128),     # Fence
    8: (64, 0, 128),      # Car
    9: (64, 64, 0),       # Pedestrian
    10: (0, 128, 192),    # Bicyclist
    11: (0, 0, 0)         # Void / Background
}

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb

# -----------------------------
# Model
# -----------------------------
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.Compose([
    T.ToTensor(),
])

# -----------------------------
# Test images
# -----------------------------
test_img_dir = os.path.join(DATA_DIR, "test")
test_mask_dir = os.path.join(DATA_DIR, "test_labels")

test_images = sorted(os.listdir(test_img_dir))
test_masks = sorted(os.listdir(test_mask_dir))

# Πάρε 5 τυχαίες εικόνες
samples = random.sample(list(zip(test_images, test_masks)), 5)

for idx, (img_name, mask_name) in enumerate(samples):
    img_path = os.path.join(test_img_dir, img_name)
    mask_path = os.path.join(test_mask_dir, mask_name)

    image = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    gt_mask = Image.open(mask_path).convert("RGB").resize(IMG_SIZE, Image.NEAREST)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    pred_rgb = mask_to_rgb(pred_mask)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(gt_mask)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    axes[2].imshow(pred_rgb)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"prediction_{idx+1}.png")
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")
