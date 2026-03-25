import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Models comparison (mIoU)
# -------------------------
models = [
    "U-Net (12)",
    "U-Net (5)",
    "DeepLabV3+",
    "U-Net + Weighted",
    "SegFormer"
]

miou = [0.4797, 0.7240, 0.7132, 0.6823, 0.6135]

plt.figure()
plt.bar(models, miou)
plt.title("mIoU Comparison Across Models")
plt.ylabel("mIoU")
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig("miou_comparison_full.png", dpi=200)
plt.show()


# -------------------------
# Class-wise IoU comparison
# -------------------------
classes = ["background", "road", "sidewalk", "vehicle", "pedestrian"]

unet = [0.9646, 0.9591, 0.8424, 0.5944, 0.2596]
deeplab = [0.9547, 0.9594, 0.8072, 0.6357, 0.2092]
weighted = [0.9421, 0.9542, 0.8151, 0.4824, 0.2176]
segformer = [0.9410, 0.9283, 0.7566, 0.4416, 0.0000]

x = np.arange(len(classes))
width = 0.2

plt.figure()
plt.bar(x - 1.5*width, unet, width, label="U-Net")
plt.bar(x - 0.5*width, deeplab, width, label="DeepLabV3+")
plt.bar(x + 0.5*width, weighted, width, label="Weighted U-Net")
plt.bar(x + 1.5*width, segformer, width, label="SegFormer")

plt.xticks(x, classes)
plt.ylabel("IoU")
plt.title("Class-wise IoU Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("class_iou_full.png", dpi=200)
plt.show()


# -------------------------
# Pedestrian comparison
# -------------------------
ped_models = ["U-Net", "DeepLabV3+", "Weighted", "SegFormer"]
ped_scores = [0.2596, 0.2092, 0.2176, 0.0000]

plt.figure()
plt.bar(ped_models, ped_scores)
plt.title("Pedestrian IoU Comparison")
plt.ylabel("IoU")
plt.tight_layout()
plt.savefig("pedestrian_iou_full.png", dpi=200)
plt.show()


print("All plots saved successfully!")
