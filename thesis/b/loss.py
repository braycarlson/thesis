from __future__ import annotations

import torch

from thesis.coordinates import (
    CoordinatesConverter,
    TensorStrategy
)
from torch import nn
from torchvision.ops import box_iou


class MultiBoxLoss(nn.Module):
    """The MultiBox Loss function."""

    def __init__(self, default: torch.Tensor):
        """Initialize the MultiBox Loss function.

        Args:
            default: The default boxes.

        """

        super().__init__()

        self.background = 0
        self.classes = 2
        self.object = 1

        strategy = TensorStrategy(128, 128)
        self.converter = CoordinatesConverter(strategy)

        self.alpha = 1.0
        self.ratio = 1.0
        self.threshold = 0.50

        self.default = default
        self.amount = self.default.size(0)
        self.xy = self.converter.cxcywh_to_xyxy(default)
        self.entropy = nn.CrossEntropyLoss(reduction='none')
        self.smooth = nn.SmoothL1Loss()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        boxes: list[list[int, int, int, int]],
        labels: list[int]
    ) -> torch.Tensor:
        """Compute the Multi-Box Loss.

        Args:
            x: The predicted coordinates as offsets to default boxes.
            y: The predicted class scores for the default boxes.
            boxes: The ground-truth bounding boxes for each image.
            labels: The ground-truth labels for each image.

        Returns:
            The weighted loss as a scalar.

        """

        length = len(boxes)

        localization = torch.zeros_like(x, dtype=torch.float)

        size = [length, self.amount]
        classification = torch.zeros(size, dtype=torch.long).to('cuda')

        for i in range(length):
            amount = boxes[i].size(0)
            iou = box_iou(boxes[i], self.xy)
            overlap, digit = iou.max(dim=0)

            _, default = iou.max(dim=1)
            digit[default] = torch.arange(amount).cuda()
            overlap[default] = 1.0

            coordinates = self.converter.xyxy_to_cxcywh(boxes[i][digit])
            localization[i] = self.converter.cxcywh_to_gcxgcy(coordinates, self.default)

            classification[i] = labels[i][digit]
            classification[i][overlap < self.threshold] = self.background

        positives = (classification == self.object)
        total = positives.sum()

        smooth = self.smooth(
            x[positives],
            localization[positives]
        )

        entropy = self.entropy(
            y.view(-1, self.classes),
            classification.view(-1)
        )

        entropy = entropy.view(-1, self.amount)

        positive = entropy[positives]

        negative, _ = entropy[~positives].sort(dim=0, descending=True)
        keep = int(total * self.ratio)

        negative = negative[:keep]

        aggregate = positive.sum() + negative.sum()
        entropy = (self.alpha * aggregate) / total
        return smooth + entropy
