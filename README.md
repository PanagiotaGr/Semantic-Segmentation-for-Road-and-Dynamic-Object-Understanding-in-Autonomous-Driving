# Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving

## Overview

This project presents a comprehensive study of semantic segmentation techniques for urban scene understanding, with applications in autonomous driving. The work investigates the performance of convolutional neural networks (CNNs) and transformer-based architectures under a reduced, safety-critical label space.

The primary objective is to evaluate how model architecture, label formulation, and loss design influence segmentation performance, particularly for dynamic and safety-critical classes such as pedestrians and vehicles.

---

## Research Contributions

* Implementation of a full semantic segmentation pipeline
* Comparison between CNN-based and transformer-based architectures
* Introduction of a reduced 5-class representation for safety-critical perception
* Evaluation of class imbalance handling via weighted loss
* Extensive quantitative and qualitative analysis

---

## Dataset

The experiments are conducted on the **CamVid dataset**, a widely used benchmark for urban scene understanding.

### Reduced Label Space

To focus on autonomous driving requirements, the original 12 classes were mapped into 5 key categories:

* **Background**
* **Road**
* **Sidewalk**
* **Vehicle**
* **Pedestrian**

This simplification improves interpretability and performance in safety-critical tasks.

---

## Models

The following architectures were evaluated:

### CNN-based

* **U-Net (ResNet34 encoder)**
* **DeepLabV3+**

### Transformer-based

* **SegFormer (MiT-B0 backbone)**

---

## Experimental Setup

* Image resolution: 256×256
* Batch size: 4
* Optimizer: Adam / AdamW
* Learning rate: 1e-3 (CNN), 5e-5 (Transformer)
* Loss functions:

  * Cross-Entropy Loss
  * Weighted Cross-Entropy (class imbalance handling)

---

## Evaluation Metrics

* **Mean Intersection over Union (mIoU)**
* **Class-wise IoU**
* Qualitative segmentation outputs

---

## Results

### Overall Performance

| Model                 |       mIoU |
| --------------------- | ---------: |
| U-Net (12 classes)    |     0.4797 |
| U-Net (5 classes)     | **0.7240** |
| DeepLabV3+            |     0.7132 |
| U-Net + Weighted Loss |     0.6823 |
| SegFormer             |     0.6135 |

---

### Class-wise IoU (Best U-Net Model)

| Class      |    IoU |
| ---------- | -----: |
| Background | 0.9646 |
| Road       | 0.9591 |
| Sidewalk   | 0.8424 |
| Vehicle    | 0.5944 |
| Pedestrian | 0.2596 |

---

## Key Findings

* Reducing the label space from 12 to 5 classes significantly improved performance.
* U-Net achieved the best overall mIoU, demonstrating strong performance in structured urban scenes.
* DeepLabV3+ improved vehicle segmentation but did not surpass U-Net globally.
* Weighted loss did not improve overall performance and introduced trade-offs.
* SegFormer (transformer-based) did not outperform CNN models in this setup and failed to capture the pedestrian class effectively.

---

## Qualitative Results

### Segmentation Examples

![Example1](sample_outputs/reduced_prediction_1.png)
![Example2](sample_outputs/weighted_overlay_1.png)

### Performance Plots

![mIoU](plots/miou_comparison_full.png)
![Class IoU](plots/class_iou_full.png)

---

## Full Outputs

All segmentation outputs for the test dataset are available in the `results/` directory.
This enables full reproducibility and qualitative evaluation of model behavior across diverse urban scenarios.

---

## Discussion

Large and spatially continuous classes such as road and sidewalk are segmented with high accuracy due to their structural consistency. In contrast, dynamic and small-scale objects such as pedestrians remain challenging due to class imbalance and limited pixel representation.

The results indicate that architectural complexity alone does not guarantee improved performance. Instead, task formulation and data representation play a critical role in achieving robust segmentation.

---

## Conclusion

This work demonstrates that semantic segmentation can effectively model urban driving environments when combined with appropriate label design and model selection.

The reduced-class U-Net configuration provides the best trade-off between accuracy and interpretability. Transformer-based approaches, while promising, require further optimization and larger-scale data to fully realize their potential in this domain.

---

## Future Work

* Transformer optimization (SegFormer tuning, Mask2Former)
* Advanced loss functions (Focal Loss, Dice Loss)
* Temporal consistency for video segmentation
* Multi-task learning (segmentation + lane detection)
* Real-time inference optimization

---

## Repository Structure

```
semantic_seg_project/
├── train_*.py
├── predict_*.py
├── results/
├── plots/
├── sample_outputs/
├── README.md
```

---

## Author

Panagiota Grosdouli

---

## License

This project is for academic and research purposes.
