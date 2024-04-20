from __future__ import annotations

import numpy as np
import torch

from PIL import Image
from thesis.constant import CWD
from thesis.dataset import BaseDataset


class ClassificationDataset(BaseDataset):
    """The dataset for the classification model during training."""

    def __init__(
        self,
        *args,
        size: tuple[float, float] = (28, 28),
        **kwargs
    ):
        """Initialize the dataset for training.

        Args:
            size: The size of the images.

        """

        super().__init__(*args, **kwargs)

        self.size = size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get an item from the dataset.

        Args:
            index: The index of the item.

        Returns:
            The image and its label.

        """

        file = self.annotation.loc[index, 'file']
        label = self.annotation.loc[index, 'label']

        label = int(label)
        label = torch.tensor(label)

        path = CWD.joinpath(file)
        image = Image.open(path).convert('L')
        image = np.asarray(image).astype('float32')

        image = self.strategy.apply(image)

        return image, label

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'classification'
