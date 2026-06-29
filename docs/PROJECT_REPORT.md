# Project Report

## Semantic Segmentation for Road and Dynamic Object Understanding in Autonomous Driving

### Author

Panagiota Grosdouli

## Abstract

This project investigates semantic segmentation for urban road-scene understanding using the CamVid dataset. It compares CNN-based and transformer-based architectures and studies the effect of reducing the original semantic label space into a 5-class safety-critical formulation. The strongest reported result is achieved by U-Net with a ResNet34 encoder under the reduced label space, with an mIoU of 0.7240.

## 1. Introduction

Autonomous-driving systems require reliable perception of road structures and dynamic objects. Semantic segmentation supports this requirement by assigning a semantic label to every image pixel. This allows the system to distinguish road, sidewalk, vehicles, pedestrians, and background regions.

## 2. Dataset

The project uses the CamVid dataset. The original label space is reduced into five categories: Background, Road, Sidewalk, Vehicle, and Pedestrian. This reduction focuses the task on safety-critical perception.

## 3. Models

The evaluated models are U-Net with ResNet34 encoder, DeepLabV3+, and SegFormer with MiT-B0 backbone. These models allow comparison between CNN encoder-decoder architectures, atrous convolution-based segmentation, and transformer-based segmentation.

## 4. Results

| Model | Label setup | Loss | mIoU |
| --- | --- | --- | ---: |
| U-Net | 12 classes | Cross-Entropy | 0.4797 |
| U-Net | 5 classes | Cross-Entropy | 0.7240 |
| DeepLabV3+ | 5 classes | Cross-Entropy | 0.7132 |
| U-Net | 5 classes | Weighted Cross-Entropy | 0.6823 |
| SegFormer | 5 classes | Cross-Entropy | 0.6135 |

## 5. Discussion

The results show that label-space formulation has a strong effect on segmentation performance. The reduced 5-class setup improves global mIoU and makes the task more relevant to autonomous-driving perception.

Pedestrian segmentation remains the most challenging class because pedestrians are small, visually variable, and under-represented compared with road and background pixels.

## 6. Limitations

The main limitations are reduced image resolution, limited transformer tuning, low pedestrian IoU, lack of temporal modelling, and no real-time inference benchmark.

## 7. Future Work

Future work should include higher-resolution training, Focal Loss, Dice Loss, multi-seed evaluation, confusion matrices, FPS benchmarking, and experiments on larger urban-scene datasets.

## 8. Conclusion

The project demonstrates that a reduced safety-critical label space combined with a strong CNN encoder-decoder model can provide effective semantic segmentation for autonomous-driving road-scene understanding.
