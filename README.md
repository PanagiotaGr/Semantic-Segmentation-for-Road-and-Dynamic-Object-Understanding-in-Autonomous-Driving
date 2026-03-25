# Semantic Segmentation for Autonomous Driving

This project implements semantic segmentation models for urban scene understanding using the CamVid dataset.

## Models
- U-Net (ResNet34)
- DeepLabV3+

## Experiments
- 12-class segmentation
- Reduced 5-class segmentation
- Weighted loss for class imbalance

## Results
- Best mIoU (U-Net 5 classes): 0.7240
- DeepLabV3+: 0.7132

## Classes
- Road
- Sidewalk
- Vehicle
- Pedestrian
- Background

## Metrics
- mIoU
- Class-wise IoU

## Sample Results
(Add your output images here)

## How to Run

```bash
python train_camvid_reduced.py
python train_deeplab_camvid_reduced.py
python train_camvid_weighted.py
