from __future__ import annotations

import lightning.pytorch as lp
import torch

from torch.utils.data import DataLoader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from thesis.b.transformation import (
        AlbumentationStrategy,
        TorchvisionStrategy
    )
    from torch.utils.data import Dataset


def collate_fn(
    batch: list[torch.Tensor, torch.Tensor, torch.Tensor]
) -> tuple(torch.Tensor, torch.Tensor, torch.Tensor):
    """A collate function for data loading.

    Args:
        batch: A list of tuples containing images, boxes, and labels.

    Returns:
        A tuple containing images, boxes, and labels.

    """

    images = []
    boxes = []
    labels = []

    for item in batch:
        images.append(item[0])
        boxes.append(item[1])
        labels.append(item[2])

    images = torch.stack(images)
    return (images, boxes, labels)


class DataModule(lp.LightningDataModule):
    """A datamodule for localization."""

    def __init__(
        self,
        annotation: dict[str, pd.DataFrame] | None = None,
        batch_size: int = 64,
        dataset: Dataset = None,
        strategy: AlbumentationStrategy | TorchvisionStrategy | None = None,
    ):
        super().__init__()

        self.annotation = annotation
        self.batch_size = batch_size
        self.dataset = dataset
        self.strategy = strategy

    def setup(self, stage: str | None = None) -> None:
        """

        Args:
            stage: The name of the stage to setup for.

        """

        strategy = self.strategy

        # Training
        annotation = self.annotation.get('training')

        if annotation is not None:
            train_dataset = self.dataset(
                annotation=annotation,
                strategy=strategy
            )

            self.train_dataset = train_dataset

        # Testing
        annotation = self.annotation.get('testing')

        if annotation is not None:
            test_dataset = self.dataset(
                annotation=annotation,
                strategy=strategy
            )

            self.test_dataset = test_dataset

        # Validation
        annotation = self.annotation.get('validation')

        if annotation is not None:
            validation_dataset = self.dataset(
                annotation=annotation,
                strategy=strategy
            )

            self.validation_dataset = validation_dataset

    def train_dataloader(self) -> DataLoader:
        """Create a DataLoader for the training data.

        Returns:
            The training dataloader.

        """

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=True
        )

    def test_dataloader(self) -> DataLoader:
        """Create a DataLoader for the test data.

        Returns:
            The test dataloader.

        """

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
            prefetch_factor=2,
            shuffle=False
        )

    def val_dataloader(self) -> DataLoader:
        """Create a DataLoader for the validation data.

        Returns:
            The validation dataloader.

        """

        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            persistent_workers=True,
            shuffle=False
        )
