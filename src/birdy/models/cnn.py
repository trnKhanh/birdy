"""Pretrained deep learning baseline for the BirdCLEF 2026 competition."""

from typing import override

import timm
import torch
from torch import nn


class TIMMModel(nn.Module):
    """Baseline for BirdCLEF 2026 using EfficientNet."""

    def __init__(
        self,
        model_name: str,
        in_channels: int,
        num_classes: int,
        *,
        pretrained: bool = True,
    ) -> None:
        """Initialize EfficientNet baseline.

        Args:
            model_name: name of the TIMM model (see [collections](https://huggingface.co/timm/collections))
            in_channels: number of input channels (e.g. 3 for RGB images)
            num_classes: number of classes for classification
            pretrained: whether to load the pretrained
        """
        super().__init__()

        self._model: nn.Module = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            num_classes=num_classes,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
