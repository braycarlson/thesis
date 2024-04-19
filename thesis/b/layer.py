from __future__ import annotations

import torch

from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from default import DefaultConfiguration


class Base(nn.Module):
    """Base component for feature extraction."""

    def __init__(self, channel: list[int]):
        """Initialize the base component.

        Args:
            channel: A list of filters.

        """

        super().__init__()

        self.module = nn.ModuleList()
        self.identifier = []

        for i in range(len(channel) - 1):
            self.module.extend([
                nn.Conv2d(
                    in_channels=channel[i],
                    out_channels=channel[i+1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=1
                ),
                nn.InstanceNorm2d(channel[i+1]),
                nn.LeakyReLU(0.01),
                nn.Conv2d(
                    in_channels=channel[i+1],
                    out_channels=channel[i+1],
                    kernel_size=3,
                    padding=1,
                    stride=1,
                    groups=1
                ),
                nn.InstanceNorm2d(channel[i+1]),
                nn.LeakyReLU(0.01)
            ])

            self.module.append(nn.MaxPool2d(kernel_size=2))
            self.identifier.append(len(self.module) - 2)

        # Capture the most relevant and high-level features
        self.identifier = self.identifier[-2:]

        # Remove the pooling channel
        self.module = self.module[:-1]

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """A forward pass of the base component.

        Args:
            x: An input tensor.

        Returns:
            A tuple containing the output tensor and intermediate features.

        """

        features = []

        length = len(self.module)

        for i in range(length):
            x = self.module[i](x)

            if i in self.identifier:
                features.append(x)

        return x, features


class Auxiliary(nn.Module):
    """Auxiliary component for feature extraction."""

    def __init__(self, channel: list[int]):
        """Initialize the auxiliary component.

        Args:
            channel: A list of filters.

        """

        super().__init__()

        self.module = nn.ModuleList()
        self.identifier = []

        for i in range(len(channel) - 1):
            self.module.extend([
                nn.Conv2d(
                    in_channels=channel[i],
                    out_channels=channel[i] // 2,
                    kernel_size=1,
                    padding=0
                ),
                nn.InstanceNorm2d(channel[i] // 2),
                nn.LeakyReLU(0.01),
                nn.Conv2d(
                    in_channels=channel[i] // 2,
                    out_channels=channel[i+1],
                    kernel_size=3,
                    padding=1,
                    stride=2
                ),
                nn.InstanceNorm2d(channel[i+1]),
                nn.LeakyReLU(0.01)
            ])

            self.identifier.append(len(self.module) - 1)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """A forward pass of the auxiliary component.

        Args:
            x: An input tensor.

        Returns:
            A list containing intermediate features.

        """

        features = []

        length = len(self.module)

        for i in range(length):
            x = self.module[i](x)

            if i in self.identifier:
                features.append(x)

        return features


class Prediction(nn.Module):
    """Prediction component for localization and classification."""

    def __init__(self, configuration: DefaultConfiguration):
        """Initialize the prediction component.

        Args:
            configuration: The default box configuration instance.

        """

        super().__init__()

        self.configuration = configuration.generate()

        self.channel = self.configuration.get('channel')
        self.classes = 2
        self.pixel = self.configuration.get('pixel')

        self.localization = nn.ModuleList()
        self.classification = nn.ModuleList()

        length = len(self.channel)

        for i in range(length):
            self.localization.append(
                nn.Sequential(
                    nn.InstanceNorm2d(self.channel[i]),
                    nn.Conv2d(
                        self.channel[i],
                        self.pixel[i] * 4,
                        kernel_size=3,
                        padding=1
                    )
                )
            )

            self.classification.append(
                nn.Sequential(
                    nn.InstanceNorm2d(self.channel[i]),
                    nn.Conv2d(
                        self.channel[i],
                        self.pixel[i] * self.classes,
                        kernel_size=3,
                        padding=1
                    )
                )
            )

    def _postprocess(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """The postprocessing step of the tensor.

        Args:
            x: An input tensor.
            k: The size parameter.

        Returns:
            The postprocessed tensor.

        """

        order = [0, 2, 3, 1]

        x = x.permute(order).contiguous()
        return x.view(x.size(0), -1, k)

    def forward(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """A forward pass of the prediction component.

        Args:
            x: A list containing intermediate features.

        Returns:
            A tuple containing localization and classification logits.

        """

        localization = []
        classification = []

        length = len(x)

        for i in range(length):
            tensor = self._postprocess(self.localization[i](x[i]), 4)
            localization.append(tensor)

            tensor = self._postprocess(self.classification[i](x[i]), self.classes)
            classification.append(tensor)

        return (
            torch.cat(localization, dim=1),
            torch.cat(classification, dim=1)
        )
