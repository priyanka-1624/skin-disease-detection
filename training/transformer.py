from __future__ import annotations

from dataclasses import dataclass

import timm
import torch
from torch import nn

SUPPORTED_MODELS = {"vit_base_patch16_224", "swin_tiny_patch4_window7_224"}


@dataclass(slots=True)
class BackboneSpec:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = True

    def validate(self) -> None:
        if self.model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Backbone must be one of {sorted(SUPPORTED_MODELS)}")


def create_classifier_backbone(num_classes: int, spec: BackboneSpec) -> nn.Module:
    spec.validate()
    return timm.create_model(spec.model_name, pretrained=spec.pretrained, num_classes=num_classes)


def create_feature_extractor(spec: BackboneSpec) -> nn.Module:
    spec.validate()
    return timm.create_model(spec.model_name, pretrained=False, num_classes=0, global_pool="avg")


def load_backbone_for_features(extractor: nn.Module, classifier_state: dict[str, torch.Tensor]) -> None:
    filtered = {
        key: value
        for key, value in classifier_state.items()
        if not key.startswith("head") and not key.startswith("classifier")
    }
    missing, unexpected = extractor.load_state_dict(filtered, strict=False)
    tolerated_prefixes = ("head", "classifier", "norm", "fc_norm")
    critical_unexpected = [key for key in unexpected if not key.startswith(tolerated_prefixes)]
    critical_missing = [key for key in missing if not key.startswith(tolerated_prefixes)]
    if critical_unexpected:
        raise RuntimeError(f"Unexpected state keys: {critical_unexpected}")
    if critical_missing:
        raise RuntimeError(f"Missing critical keys: {critical_missing}")
