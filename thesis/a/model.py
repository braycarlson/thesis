from __future__ import annotations

import lightning.pytorch as lp
import torch

from pathlib import Path
from thesis.visualize import ClassificationStrategy, Visualizer
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class ClassificationModel(lp.LightningModule):
    """The classification model"""

    def __init__(self, lr: float = 0.001):
        """Initialize the model.

        Args:
            lr: The learning rate.

        """

        super().__init__()

        self.lr = lr
        self.size = (28, 28)

        self.training_visualized = False
        self.validation_visualized = False

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),

            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def execute(self, tensor: torch.Tensor) -> torch.Tensor:
        """Execute the model on the provided tensor.

        Args:
            tensor: An input tensor.

        Returns:
            The output tensor.

        """

        self.eval()

        with torch.no_grad():
            return self(tensor)

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str
    ) -> dict[str, torch.Tensor]:
        """Perform a single step during training or validation.

        Args:
            batch: The input batch.
            stage: A stage of the step (training or validation).

        Returns:
            A dictionary containing the loss.

        """

        images, target = batch

        logit = self(images)
        loss = nn.functional.nll_loss(logit, target)
        prediction = torch.argmax(logit, dim=1)
        accuracy = (prediction == target).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_epoch=True,
            prog_bar=True
        )

        self.log(
            f"{stage}_accuracy",
            accuracy,
            on_epoch=True,
            prog_bar=True
        )

        return {'loss': loss}

    def _visualize(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str
    ) -> None:
        """Visualize each prediction from a batch of data.

        Args:
            batch: An input batch.
            stage: The stage of the visualization (training or validation).

        """

        images, target = batch

        logit = self(images)
        prediction = torch.argmax(logit, dim=1)

        epoch = str(self.current_epoch).zfill(3)
        filename = f"{epoch}.png"

        path = Path('examples', stage)
        path.mkdir(exist_ok=True, parents=True)

        path = path.joinpath(filename)

        strategy = ClassificationStrategy(images, target, prediction)
        visualizer = Visualizer(strategy)
        visualizer.save(path)

    def configure_optimizers(self) -> Any:
        """Configure the optimizer for training."""

        lr = self.lr

        return torch.optim.Adam(
            self.parameters(),
            lr=lr
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """A forward pass of the model.

        Args:
            x: An input tensor.

        Returns:
            The output tensor after passing through the model.

        """

        x = self.backbone(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _: int
    ) -> dict[str, torch.Tensor]:
        """Perform a single testing step.

        Args:
            batch: An input batch.
            _: The index of the batch.

        Returns:
            A dictionary containing the loss.

        """

        return self._step(batch, 'test')

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""

        self.training_visualized = False

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _: int
    ) -> dict[str, torch.Tensor]:
        """Perform a single training step.

        Args:
            batch: An input batch.
            _: The index of the batch.

        Returns:
            A dictionary containing the loss.

        """

        name = 'training'

        if not self.training_visualized:
            with torch.no_grad():
                self._visualize(batch, name)

            self.training_visualized = True

        return self._step(batch, name)

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""

        self.validation_visualized = False

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        _: int
    ) -> dict[str, torch.Tensor]:
        """Perform a single validation step.

        Args:
            batch: An input batch.
            _: The index of the batch.

        Returns:
            A dictionary containing the loss.

        """

        name = 'validation'

        if not self.validation_visualized:
            with torch.no_grad():
                self._visualize(batch, name)

            self.validation_visualized = True

        return self._step(batch, name)
