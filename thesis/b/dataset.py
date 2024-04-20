from __future__ import annotations

import numpy as np
import torch

from PIL import Image
from thesis.constant import CWD
from thesis.dataset import BaseDataset


class BaseLocalizationDataset(BaseDataset):
    """The base dataset for a localization model during training."""

    def __init__(
        self,
        *args,
        size: tuple[float, float] = (128, 128),
        **kwargs
    ):
        """The base class for a localization dataset.

        Args:
            size: A tuple representing the size of the images.

        """

        super().__init__(*args, **kwargs)

        self.coordinates = 4
        self.size = size


class SingleShotDetectorDataset(BaseLocalizationDataset):
    """The dataset for the CASSD model during training."""

    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.width = self.height = 128

    def __getitem__(self, index: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        """Get an item from the dataset.

        Args:
            index: The index of the item.

        Returns:
            The image, bounding box, and label.

        """

        file = self.annotation.loc[index, 'file']
        boxes = self.annotation.loc[index, 'xyxy']
        labels = self.annotation.loc[index, 'label']

        path = CWD.joinpath(file)

        with Image.open(path).convert('L') as image:
            image = np.asarray(image).astype('uint8')

        image, boxes = self.strategy.apply(
            image=image,
            boxes=boxes
        )

        image = (image - image.mean()) / image.std() + 0.5
        boxes = boxes / self.width
        labels = torch.tensor(labels, dtype=torch.uint8)

        labels = torch.where(labels == 10, 0, 1)

        return image, boxes, labels

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'cassd'
