# Dataset and Label Mapping

## Dataset

The experiments are based on the CamVid dataset, a road-scene semantic segmentation dataset containing urban driving images and pixel-level annotations.

CamVid is suitable for this project because it contains road, sidewalk, vehicle, pedestrian, building, sky, vegetation, and other urban-scene categories that are relevant to autonomous-driving perception.

## Motivation for Label Reduction

The original semantic label space is useful for complete scene parsing. However, autonomous-driving perception often prioritizes safety-critical categories. For this reason, the project uses a reduced 5-class label space.

The reduced label space makes the problem more focused and improves interpretability. It also allows clearer analysis of road layout and dynamic objects.

## Reduced 5-Class Formulation

| Class index | Reduced class | Autonomous-driving meaning |
| ---: | --- | --- |
| 0 | Background | Non-critical or merged scene elements |
| 1 | Road | Drivable area |
| 2 | Sidewalk | Pedestrian-side area |
| 3 | Vehicle | Dynamic or potentially dynamic road agents |
| 4 | Pedestrian | Vulnerable road users |

## Why This Mapping Matters

The reduced mapping changes the learning objective. Instead of requiring the model to separate every visual category, it asks the model to focus on scene elements that are most relevant for driving decisions.

This can improve global mIoU because visually similar or less critical classes are merged. However, rare classes such as pedestrians can remain difficult because they occupy fewer pixels and have higher visual variation.

## Recommended Dataset Directory

A clean local dataset structure is:

```text
data/
└── CamVid/
    ├── train/
    ├── train_labels/
    ├── val/
    ├── val_labels/
    ├── test/
    └── test_labels/
```

The dataset itself is not stored in the repository because image datasets are usually large and may have separate licensing requirements.

## Preprocessing Notes

Recommended preprocessing steps:

* resize images and masks to 256 x 256,
* normalize RGB images using ImageNet statistics when using pretrained encoders,
* use nearest-neighbor interpolation for segmentation masks,
* ensure that label IDs remain integer class indices,
* apply the same train, validation, and test split across all models.

## Scientific Considerations

When reporting results, the label mapping must be documented clearly because mIoU values are not directly comparable across different label spaces. A 5-class segmentation problem is easier and more task-focused than a 12-class segmentation problem.
