import torch

from src.research.temporal_fusion import TemporalFeatureFusion, TemporalFusionSegmentationHead


def test_temporal_feature_fusion_mean_shape():
    features = torch.randn(2, 3, 16, 8, 8)
    fusion = TemporalFeatureFusion(mode="mean")
    output = fusion(features)
    assert output.shape == (2, 16, 8, 8)


def test_temporal_feature_fusion_concat_shape():
    features = torch.randn(2, 3, 16, 8, 8)
    fusion = TemporalFeatureFusion(mode="concat")
    output = fusion(features)
    assert output.shape == (2, 48, 8, 8)


def test_temporal_fusion_segmentation_head_shape():
    features = torch.randn(2, 16, 8, 8)
    head = TemporalFusionSegmentationHead(input_channels=16, num_classes=5)
    logits = head(features)
    assert logits.shape == (2, 5, 8, 8)
