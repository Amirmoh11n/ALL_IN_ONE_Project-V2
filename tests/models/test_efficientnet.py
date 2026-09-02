"""Tests for src/models/efficientnet.py (EfficientNetB3Classifier).

pretrained=False is used throughout so tests don't require network access to
download ImageNet weights; architecture/shape/freeze behavior is identical
either way.
"""

import torch

from src.data.classes import TumorClasses
from src.models.efficientnet import EfficientNetB3Classifier


def test_output_shape_matches_num_classes():
    model = EfficientNetB3Classifier(num_classes=TumorClasses.num_classes(), pretrained=False)
    model.eval()

    batch = torch.randn(2, 3, 300, 300)
    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, TumorClasses.num_classes())


def test_final_layer_replaced_with_correct_num_classes():
    model = EfficientNetB3Classifier(num_classes=4, pretrained=False)
    final_linear = model.backbone.classifier[1]
    assert isinstance(final_linear, torch.nn.Linear)
    assert final_linear.out_features == 4


def test_builtin_dropout_is_preserved():
    model = EfficientNetB3Classifier(num_classes=4, pretrained=False)
    dropout_layer = model.backbone.classifier[0]
    assert isinstance(dropout_layer, torch.nn.Dropout)
    assert dropout_layer.p == 0.3  # torchvision's EfficientNet-B3 default, left untouched


def test_full_fine_tune_keeps_backbone_trainable():
    model = EfficientNetB3Classifier(num_classes=4, pretrained=False, freeze_backbone=False)
    backbone_params_trainable = all(p.requires_grad for p in model.backbone.features.parameters())
    assert backbone_params_trainable is True


def test_freeze_backbone_freezes_features_but_not_head():
    model = EfficientNetB3Classifier(num_classes=4, pretrained=False, freeze_backbone=True)

    backbone_params_frozen = all(not p.requires_grad for p in model.backbone.features.parameters())
    head_trainable = model.backbone.classifier[1].weight.requires_grad

    assert backbone_params_frozen is True
    assert head_trainable is True


def test_forward_pass_produces_finite_logits():
    model = EfficientNetB3Classifier(num_classes=4, pretrained=False)
    model.eval()
    batch = torch.randn(3, 3, 300, 300)
    with torch.no_grad():
        logits = model(batch)
    assert torch.isfinite(logits).all()
