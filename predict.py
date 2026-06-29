"""Unified prediction script for trained segmentation checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.model_factory import SUPPORTED_MODELS, create_segmentation_model
from utils.augmentations import get_val_augmentations
from utils.visualization import colorize_mask, overlay_mask


CLASS_NAMES = ["Background", "Road", "Sidewalk", "Vehicle", "Pedestrian"]


def load_image(path: Path, image_size: int):
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    image_array = np.asarray(image)
    transform = get_val_augmentations(image_size)
    tensor = transform(image=image_array, mask=np.zeros((image_size, image_size), dtype=np.uint8))["image"]
    return image_array, tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate segmentation predictions from a trained checkpoint.")
    parser.add_argument("--image-dir", default="data/CamVid/test", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", default="unet", choices=SUPPORTED_MODELS)
    parser.add_argument("--encoder", default="resnet34")
    parser.add_argument("--image-size", default=256, type=int)
    parser.add_argument("--output-dir", default=Path("results/predictions"), type=Path)
    parser.add_argument("--save-overlays", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = args.output_dir / "masks"
    color_dir = args.output_dir / "color_masks"
    overlay_dir = args.output_dir / "overlays"
    mask_dir.mkdir(exist_ok=True)
    color_dir.mkdir(exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(exist_ok=True)

    model = create_segmentation_model(args.model, num_classes=len(CLASS_NAMES), encoder_name=args.encoder, encoder_weights=None).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    image_files = sorted([p for p in args.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not image_files:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    with torch.no_grad():
        for image_path in tqdm(image_files, desc="Predict"):
            image_array, tensor = load_image(image_path, args.image_size)
            logits = model(tensor.unsqueeze(0).to(device))
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            color = colorize_mask(pred)
            Image.fromarray(pred).save(mask_dir / f"{image_path.stem}.png")
            Image.fromarray(color).save(color_dir / f"{image_path.stem}.png")
            if args.save_overlays:
                Image.fromarray(overlay_mask(image_array, pred)).save(overlay_dir / f"{image_path.stem}.png")

    print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    main()
