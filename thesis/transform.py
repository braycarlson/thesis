from __future__ import annotations

import numpy as np
import random
import torch

from PIL import ImageDraw
from torchvision.transforms import functional, v2


class GradientTransform:
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a gradient fade effect.

        Args:
            tensor: An input image tensor.

        Returns:
            The transformed image tensor.

        """

        rng = np.random.default_rng()

        direction = rng.choice(['horizontal', 'vertical'])
        fade_fraction = rng.uniform(0.25, 0.75)

        was_uint8 = tensor.dtype == torch.uint8

        if was_uint8:
            tensor = tensor.float() / 255

        channels, height, width = tensor.shape

        if direction == 'horizontal':
            fade_length = int(width * fade_fraction)
            gradient = torch.linspace(1, 0, fade_length) ** 2
            full = torch.cat([torch.ones(width - fade_length), gradient], dim=0)
            full = full.view(1, 1, width).expand(channels, height, width)
        else:
            fade_length = int(height * fade_fraction)
            gradient = torch.linspace(1, 0, fade_length) ** 2
            full = torch.cat([torch.ones(height - fade_length), gradient], dim=0)
            full = full.view(1, height, 1).expand(channels, height, width)

        tensor *= full

        if was_uint8:
            tensor = (tensor * 255).to(torch.uint8)

        return tensor


class RandomNoise(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data: tuple) -> tuple:
        """Apply random noise.

        Args:
            data: A tuple containing the input image tensor and bounding boxes.

        Returns:
            The transformed image tensor.

        """

        tensor, boxes = data

        choice = ['gaussian', 'sp', 'speckle']
        noise_type = np.random.choice(choice)

        if noise_type == 'gaussian':
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gaussian = torch.randn(tensor.size()) * sigma + mean
            noisy_image = tensor + gaussian

        if noise_type == 'sp':
            s_vs_p = 0.5
            amount = 0.04
            num_salt = int(np.ceil(amount * tensor.numel() * s_vs_p))
            num_pepper = int(np.ceil(amount * tensor.numel() * (1. - s_vs_p)))

            indices = torch.randperm(tensor.numel())[:num_salt]
            noisy_image = tensor.clone().view(-1)
            noisy_image[indices] = 1

            indices = torch.randperm(tensor.numel())[:num_pepper]
            noisy_image[indices] = 0
            noisy_image = noisy_image.view(tensor.size())

        if noise_type == 'speckle':
            noise = torch.randn(tensor.size())
            noisy_image = tensor + tensor * noise

        noisy_image = torch.clamp(noisy_image, 0, 1)

        return noisy_image, boxes


class RandomStroke(torch.nn.Module):
    def __init__(self, count: int = 5, width: int = 3):
        super().__init__()
        self.count = count
        self.width = width

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random strokes.

        Args:
            tensor: An input image tensor.

        Returns:
            The transformed image tensor.

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
    def __init__(self, maximum: int = 10):
        self.maximum = maximum

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply a random shift.

        Args:
            tensor: An input image tensor.

        Returns:
            The transformed image tensor.

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
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply standardization.

        Args:
            tensor: An input image tensor.

        Returns:
            The transformed image tensor.

        """

        mean = tensor.mean()
        std = tensor.std()

        if std < 1e-6:
            std = 1e-6

        standardized = (tensor - mean) / std
        return standardized + 0.5
