import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

DATA_DIR = "data/CamVid"
MODEL_PATH = "models/best_unet_camvid_weighted.pth"
OUTPUT_DIR = "outputs_full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    10: 0, 11: 0
}

REDUCED_COLORS = {
    0: (0, 0, 0),
    1: (128, 64, 128),
    2: (0, 0, 255),
    3: (255, 0, 0),
    4: (0, 255, 0)
}

def rgb_to_reduced_mask(mask):
    mask = np.array(mask)
    orig_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)

    for rgb, idx in CLASS_COLORS.items():
        matches = np.all(mask == np.array(rgb), axis=-1)
        orig_mask[matches] = idx

    reduced_mask = np.zeros_like(orig_mask)
    for orig_cls, reduced_cls in REDUCED_MAP.items():
        reduced_mask[orig_mask == orig_cls] = reduced_cls

    return reduced_mask

def mask_to_rgb(mask):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls, color in REDUCED_COLORS.items():
        rgb[mask == cls] = color
    return rgb

def overlay(image, mask, alpha=0.4):
    return (image * (1 - alpha) + mask * alpha).astype(np.uint8)

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = T.ToTensor()

test_img_dir = os.path.join(DATA_DIR, "test")
test_images = sorted(os.listdir(test_img_dir))

for img_name in tqdm(test_images):
    img_path = os.path.join(test_img_dir, img_name)

    image = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    image_np = np.array(image)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    pred_rgb = mask_to_rgb(pred_mask)
    overlay_img = overlay(image_np, pred_rgb)

    save_path = os.path.join(OUTPUT_DIR, f"{img_name}")
    Image.fromarray(overlay_img).save(save_path)

print("All predictions saved!")
