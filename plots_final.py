import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Models comparison
# -------------------------
models = ["U-Net (12)", "U-Net (5)", "DeepLabV3+", "U-Net + Weighted"]
miou = [0.4797, 0.7240, 0.7132, 0.6823]

plt.figure()
plt.bar(models, miou)
plt.title("mIoU Comparison Across Models")
plt.ylabel("mIoU")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("miou_comparison.png", dpi=200)
plt.show()


# -------------------------
# Class-wise IoU
# -------------------------
classes = ["background", "road", "sidewalk", "vehicle", "pedestrian"]

unet = [0.9646, 0.9591, 0.8424, 0.5944, 0.2596]
deeplab = [0.9547, 0.9594, 0.8072, 0.6357, 0.2092]
weighted = [0.9421, 0.9542, 0.8151, 0.4824, 0.2176]

x = np.arange(len(classes))
width = 0.25

plt.figure()
plt.bar(x - width, unet, width, label="U-Net")
plt.bar(x, deeplab, width, label="DeepLabV3+")
plt.bar(x + width, weighted, width, label="Weighted U-Net")

plt.xticks(x, classes)
plt.ylabel("IoU")
plt.title("Class-wise IoU Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("class_iou_comparison.png", dpi=200)
plt.show()


# -------------------------
# Optional: Pedestrian focus
# -------------------------
pedestrian_scores = [0.2596, 0.2092, 0.2176]
ped_models = ["U-Net", "DeepLabV3+", "Weighted"]

plt.figure()
plt.bar(ped_models, pedestrian_scores)
plt.title("Pedestrian IoU Comparison")
plt.ylabel("IoU")
plt.tight_layout()
plt.savefig("pedestrian_iou.png", dpi=200)
plt.show()

print("Plots saved!")
