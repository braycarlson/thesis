from __future__ import annotations

import torch

from typing import TYPE_CHECKING
from typing_extensions import Any

if TYPE_CHECKING:
    from layer import Base, Auxiliary


class DefaultConfiguration:
    """The configuration for creating default boxes"""

    def __init__(
        self,
        base: Base,
        auxiliary: Auxiliary,
        ratio: list[list[int]],
        minimum: float = 0.1,
        maximum: float = 1.0,
        size: tuple[int, int, int, int] = (1, 1, 128, 128)
    ):
        """Initialize the configuration.

        Args:
            base: The base component.
            auxiliary: The auxiliary component.
            ratio: The aspect ratios.
            minimum: The minimum scale.
            maximum: The maximum scale.
            size: The size of the input.

        """

        self.base = base
        self.auxiliary = auxiliary
        self.minimum = minimum
        self.maximum = maximum
        self.ratio = ratio
        self.size = size

    def generate(self) -> dict[str, Any]:
        """Generate the default box configuration.

        Returns:
            A dictionary containing configuration parameters.

        """

        zeros = torch.zeros(self.size)
        _, base = self.base(zeros)
        output = base[-1].size()

        zeros = torch.zeros(output)
        auxiliary = self.auxiliary(zeros)

        features = base + auxiliary

        channel = [
            feature.size(1)
            for feature in features
        ]

        size = [
            feature.size(-1)
            for feature in features
        ]

        amount = len(channel)

        scale = torch.linspace(
            self.minimum,
            self.maximum,
            steps=amount
        )

        ratio = [
            torch.tensor(ratio, dtype=torch.float32)
            for ratio in self.ratio[:amount]
        ]

        factor = [0.90, 1.0, 1.10]

        pixel = [
            len(ratios) + len(factor)
            for ratios in ratio
        ]

        return {
            'amount': amount,
            'channel': channel,
            'factor': factor,
            'pixel': pixel,
            'ratio': ratio,
            'scale': scale,
            'size': size
        }


class DefaultGenerator:
    """The generator for creating default boxes"""

    def __init__(self, configuration: DefaultConfiguration):
        """Initialize the configuration.

        Args:
            configuration: The configuration instance.

        """

        self.configuration = configuration.generate()

    def generate(self) -> torch.Tensor:
        """Generate default boxes based on the configuration.

        Returns:
            A tensor containing default boxes.

        """

        configuration = self.configuration
        amount = configuration['amount']
        factors = configuration['factor']
        ratios = configuration['ratio']
        scales = configuration['scale']
        sizes = configuration['size']

        defaults = []

        for i in range(amount):
            size = sizes[i]
            scale = scales[i]

            ranges = (
                torch
                .arange(size, dtype=torch.float32)
                .cuda() / size
            )

            for j in ranges:
                cy = j + 0.5 / size

                for k in ranges:
                    cx = k + 0.5 / size

                    for ratio in ratios[i]:
                        root = torch.sqrt(ratio).cuda()
                        width = scale * root
                        height = scale / root

                        cx = torch.clamp(cx, 0, 1)
                        cy = torch.clamp(cy, 0, 1)
                        width = torch.clamp(width, 0, 1)
                        height = torch.clamp(height, 0, 1)

                        coordinates = [cx, cy, width, height]

                        default = (
                            torch
                            .tensor(coordinates, dtype=torch.float32)
                            .cuda()
                        )

                        defaults.append(default)

                        if ratio == 1:
                            for factor in factors:
                                square = (
                                    scales[i+1]
                                    if i+1 < len(scales)
                                    else 1.0
                                )

                                square = torch.sqrt(scale * square) * factor
                                square = torch.clamp(square, 0, 1)

                                coordinates = [cx, cy, square, square]

                                default = (
                                    torch
                                    .tensor(coordinates, dtype=torch.float32)
                                    .cuda()
                                )

                                defaults.append(default)

        return torch.stack(defaults)
