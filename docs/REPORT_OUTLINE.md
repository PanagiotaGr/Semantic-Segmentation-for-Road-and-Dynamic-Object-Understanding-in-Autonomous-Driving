# Report Outline

## Title

Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving

## Abstract

This project investigates semantic segmentation for urban road-scene understanding using the CamVid dataset. The study compares CNN-based and transformer-based segmentation architectures under both original and reduced label-space formulations. The best result is achieved by a U-Net model with a ResNet34 encoder under a reduced 5-class safety-critical label space.

## 1. Introduction

Autonomous-driving systems require reliable perception of road structures and dynamic objects. Semantic segmentation provides pixel-level scene understanding and supports downstream tasks such as path planning, obstacle avoidance, and situational awareness.

## 2. Dataset

The CamVid dataset is used for urban-scene semantic segmentation. The project studies a reduced 5-class formulation that maps the original semantic classes into Background, Road, Sidewalk, Vehicle, and Pedestrian.

## 3. Methodology

The study compares U-Net, DeepLabV3+, and SegFormer. The models are trained and evaluated using mean Intersection over Union and class-wise IoU.

## 4. Experiments

The experiments examine three factors:

* label-space reduction,
* architecture selection,
* class-imbalance handling with weighted loss.

## 5. Results

The reduced-class U-Net achieves the strongest global performance with mIoU 0.7240. Road and sidewalk segmentation are highly accurate, while pedestrian segmentation remains challenging.

## 6. Discussion

The results suggest that label-space design and dataset characteristics strongly influence segmentation quality. CNN-based architectures perform better than the tested transformer configuration in this specific setting.

## 7. Limitations

The main limitations are reduced image resolution, class imbalance, limited transformer tuning, and the absence of temporal video modeling.

## 8. Conclusion

The project demonstrates that a reduced safety-critical label space combined with a strong CNN encoder-decoder model can provide effective road-scene segmentation for autonomous-driving perception.

## 9. Future Work

Future improvements include stronger loss functions, higher-resolution training, temporal consistency, real-time benchmarking, and extension to larger autonomous-driving datasets.
