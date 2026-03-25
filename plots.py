import matplotlib.pyplot as plt
import numpy as np

models = ["U-Net (12)", "U-Net (5)", "DeepLabV3+", "U-Net + Weighted"]
miou = [0.4797, 0.7240, 0.7132, 0.74]  # βάλε το δικό σου weighted

plt.figure()
plt.bar(models, miou)
plt.title("mIoU Comparison")
plt.ylabel("mIoU")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("miou_plot.png")
plt.show()


classes = ["background", "road", "sidewalk", "vehicle", "pedestrian"]

unet = [0.9646, 0.9591, 0.8424, 0.5944, 0.2596]
deeplab = [0.9547, 0.9594, 0.8072, 0.6357, 0.2092]

x = np.arange(len(classes))
width = 0.35

plt.figure()
plt.bar(x - width/2, unet, width, label="U-Net")
plt.bar(x + width/2, deeplab, width, label="DeepLabV3+")

plt.xticks(x, classes)
plt.title("IoU per Class")
plt.legend()
plt.tight_layout()
plt.savefig("class_iou_plot.png")
plt.show()
