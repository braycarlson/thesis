from __future__ import annotations

import numpy as np
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import functional, v2


class GradientTransform:
    """A fading gradient transform to use on a training dataset."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a gradient fade effect to an input image.

        Args:
            tensor: An input image.

        Returns:
            The transformed image.

        """

        rng = np.random.default_rng()

        direction = rng.choice(['horizontal', 'vertical'])
        fraction = rng.uniform(0.25, 0.75)

        uint8 = tensor.dtype == torch.uint8

        if uint8:
            tensor = tensor.float() / 255

        channel, height, width = tensor.shape

        if direction == 'horizontal':
            length = int(width * fraction)
            gradient = torch.linspace(1, 0, length) ** 2

            full = torch.cat([torch.ones(width - length), gradient], dim=0)
            full = full.view(1, 1, width).expand(channel, height, width)
        else:
            length = int(height * fraction)
            gradient = torch.linspace(1, 0, length) ** 2

            full = torch.cat([torch.ones(height - length), gradient], dim=0)
            full = full.view(1, height, 1).expand(channel, height, width)

        tensor = tensor * full

        if uint8:
            tensor = (tensor * 255).to(torch.uint8)

        return tensor


class RandomNoise(torch.nn.Module):
    """A random noise transform to use on a training dataset."""

    def __init__(self):
        """Initialize the random noise transform."""

        super().__init__()

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random noise to an input image.

        Args:
            tensor: The input image and corresponding bounding boxes.

        Returns:
            The transformed image and corresponding bounding boxes.

        """

        image, boxes = tensor

        choice = ['gaussian', 'saltpepper', 'speckle']

        rng = np.random.default_rng()
        noise = rng.choice(choice)

        size = image.size()

        if noise == 'gaussian':
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gaussian = torch.randn(size) * sigma + mean
            image = image + gaussian

        if noise == 'saltpepper':
            ratio = 0.5
            amount = 0.04

            total = image.numel()

            salt = np.ceil(amount * total * ratio)
            salt = int(salt)

            complement = (1. - ratio)
            pepper = np.ceil(amount * total * complement)
            pepper = int(pepper)

            indices = torch.randperm(total)[:salt]
            image = image.clone().view(-1)
            image[indices] = 1

            indices = torch.randperm(total)[:pepper]
            image[indices] = 0

            image = image.view(size)

        if noise == 'speckle':
            noise = torch.randn(size)
            image = image + image * noise

        image = torch.clamp(image, 0, 1)

        return image, boxes


class RandomStroke(torch.nn.Module):
    """A random stroke transform to use on a training dataset."""

    def __init__(self, count: int = 5, width: int = 3):
        """Initialize the random stroke transform.

        Args:
            count: The amount of strokes per image.
            width: The thickness of each stroke.

        """

        super().__init__()

        self.count = count
        self.width = width

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random strokes to an input image.

        Args:
            tensor: The input image and corresponding bounding boxes.

        Returns:
            The transformed image and corresponding bounding boxes.

        """

        image, boxes = tensor
        image = functional.to_pil_image(image, mode='F')
        draw = ImageDraw.Draw(image)

        count = random.randint(1, self.count)

        for _ in range(count):
            start_x = random.randint(0, image.width)
            start_y = random.randint(0, image.height)
            end_x = random.randint(0, image.width)
            end_y = random.randint(0, image.height)

            width = random.randint(1, self.width)
            fill = random.randint(0, 255)

            draw.line(
                (start_x, start_y, end_x, end_y),
                fill=fill,
                width=width
            )

        return functional.to_tensor(image), boxes


class ShiftTransform:
    """A shift transform to use on a training dataset."""

    def __init__(self, maximum: int = 10):
        """Initialize the shift transform.

        Args:
            maximum: The maximum shift distance.

        """

        self.maximum = maximum

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a random shift to an input image.

        Args:
            tensor: An input image.

        Returns:
            The transformed image.

        """

        rng = np.random.default_rng()

        dx = rng.integers(-self.maximum, self.maximum)
        dy = rng.integers(-self.maximum, self.maximum)

        return v2.functional.affine(
            tensor,
            angle=0,
            translate=(dx, dy),
            scale=1,
            shear=0,
            interpolation=v2.InterpolationMode.NEAREST,
            fill=None,
            center=None
        )


class StandardizationTransform:
    """A standardization transform to use on a training dataset."""

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply standardization on an input image.

        Args:
            tensor: An input image.

        Returns:
            The transformed image.

        """

        mean = tensor.mean()
        std = tensor.std()

        if std < 1e-6:
            std = 1e-6

        standardized = (tensor - mean) / std
        return standardized + 0.5
