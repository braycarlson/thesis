from __future__ import annotations

import lightning.pytorch as lp
import torch

from datetime import datetime, UTC
from pathlib import Path
from thesis.b.default import (
    DefaultConfiguration,
    DefaultGenerator
)
from thesis.b.layer import (
    Auxiliary,
    Base,
    Prediction
)
from thesis.b.loss import MultiBoxLoss
from thesis.coordinates import (
    CoordinatesConverter,
    TensorStrategy
)
from thesis.visualize import (
    ClassAgnosticSingleShotStrategy,
    Visualizer
)
from torch import nn
from torchvision.ops import nms
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class ClassAgnosticSingleShotDetector(lp.LightningModule):
    """The CASSD model."""

    def __init__(
        self,
        ratio: list[list[int]],
        lr: float = 0.001
    ):
        """Initialize the model.

        Args:
            ratio: The aspect ratios for default boxes.
            lr: The learning rate.

        """

        super().__init__()

        self.batch_size = 64
        self.is_training_visualized = False
        self.is_validation_visualized = False
        self.lr = lr
        self.object = 1
        self.ratio = ratio

        self.base = Base(
            [1, 16, 16, 32, 32, 64]
        )

        self.auxiliary = Auxiliary(
            [64, 64, 64]
        )

        self.configuration = DefaultConfiguration(
            self.base,
            self.auxiliary,
            self.ratio
        )

        self.prediction = Prediction(self.configuration)
        self.generator = DefaultGenerator(self.configuration)
        self.default = self.generator.generate()
        self.criterion = MultiBoxLoss(self.default)

        strategy = TensorStrategy(128, 128)
        self.converter = CoordinatesConverter(strategy)

        self.apply(self._initialize)

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'cassd'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'cassd'

    def _initialize(self, layer: nn.Module) -> None:
        """Initialize the weights and biases of each convolutional layer."""

        if type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def localize(
        self,
        defaults: torch.Tensor,
        labels: torch.Tensor,
        score: float = 0.40,
        overlap: float = 0.45
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Localize digit(s) in the image based on predicted boxes and labels.

        Args:
            defaults: The predicted offsets relative to default bounding boxes.
            labels: The predicted class scores.
            score: The confidence score threshold.
            overlap: The overlap threshold for non-maximum suppression.

        Returns:
            The bounding boxes, labels, and confidence scores.

        """

        batch = defaults.size(0)

        probabilities = nn.functional.softmax(labels, dim=2)

        all_boxes = []
        all_labels = []
        all_scores = []

        for i in range(batch):
            decoded = self.converter.gcxgcy_to_cxcywh(defaults[i], self.default)
            decoded_locs = self.converter.cxcywh_to_xyxy(decoded)

            class_score = probabilities[i][:, self.object]
            indices = (class_score > score).nonzero().squeeze(1)

            if indices.nelement() == 0:
                continue

            class_boxes = decoded_locs[indices]
            class_scores = class_score[indices]

            keep = nms(class_boxes, class_scores, overlap)

            _boxes = class_boxes[keep]

            _labels = torch.full(
                [keep.size(0)],
                self.object,
                dtype=torch.long
            )

            _scores = class_scores[keep]

            all_boxes.append(_boxes)
            all_labels.append(_labels)
            all_scores.append(_scores)

        if not all_boxes:
            return [], [], []

        return (
            all_boxes,
            all_labels,
            all_scores
        )

    def execute(
        self,
        tensor: torch.Tensor
    ) -> tuple[list[float], list[int], list[float]]:
        """Execute the model on a single image or batch of images.

        Args:
            tensor: The input images.

        Returns:
            The bounding boxes, class labels, and scores.

        """

        self.eval()

        with torch.no_grad():
            prediction = self(tensor)
            boxes, labels, scores = prediction

            boxes, labels, scores = self.localize(
                boxes,
                labels,
                score=0.85,
                overlap=0.50
            )

            boxes = [self.converter.scale(box, 'xyxy') for box in boxes]
            labels = [label.detach().cpu().numpy() for label in labels]
            scores = [score.detach().cpu().numpy() for score in scores]

            boxes = [box.cpu().tolist() for box in boxes]

            if not boxes:
                return [(0.0, 0.0, 0.0, 0.0)], [0], [0.0]

            return boxes[0], labels[0], scores[0]

    def forward(
        self,
        x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """A forward pass through the CASSD model.

        Args:
            x: The input images.

        Returns:
            The predicted offsets relative to default bounding boxes,
            predicted class scores, and a collection of feature maps.

        """

        x, features = self.base(x)
        x = self.auxiliary(x)

        features.extend(x)

        localization, classification = self.prediction(features)
        return localization, classification, features

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: str
    ) -> dict[str, torch.Tensor]:
        """Perform a single step during training, validation or testing.

        Args:
            batch: The images, bounding boxes, and labels.
            stage: The name of the stage.

        Returns:
            The stage loss and scalar loss.

        """

        image, boxes, labels = batch
        localization, classification, _ = self(image)

        mb_loss = self.criterion(
            localization,
            classification,
            boxes,
            labels
        )

        loss = mb_loss

        self.log(
            f"{stage}_mb_loss",
            mb_loss,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
            prog_bar=True
        )

        self.log(
            f"{stage}_loss",
            loss,
            batch_size=self.batch_size,
            on_epoch=True,
            on_step=True,
            prog_bar=True
        )

        return {
            f"{stage}_loss": loss,
            'loss': loss
        }

    def configure_optimizers(self) -> Any:
        """Configure the optimizer for training.

        Returns:
            The optimizer, scheduler and loss to monitor

        """

        lr = self.lr

        bias = []
        not_bias = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name:
                    bias.append(param)
                else:
                    not_bias.append(param)

        parameters = [
            {'params': bias, 'lr': lr * 2},
            {'params': not_bias, 'lr': lr}
        ]

        if self.current_epoch < 10:
            optimizer = torch.optim.AdamW(
                parameters,
                amsgrad=True,
                lr=lr,
                weight_decay=5e-4
            )
        else:
            optimizer = torch.optim.SGD(
                parameters,
                lr=lr,
                momentum=0.9,
                nesterov=True
            )

        milestones = list(
            range(20, 200, 10)
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones,
            gamma=0.5,
            verbose=False
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'validation_loss'
        }

    def _process(
        self,
        batch: tuple[torch.Tensor, torch.Tensor]
    ) -> dict[str, Any]:
        """Restructure a batch for training or validation.

        Args:
            batch: The images, bounding boxes, and labels.

        Returns:
            The images, bounding boxes, and labels from target and prediction.

        """

        image, boxes, labels = batch

        target = {
            'boxes': boxes,
            'image': image,
            'labels': labels
        }

        boxes, labels, scores = self(image)

        boxes, labels, scores = self.localize(
            boxes,
            labels,
            score=0.85,
            overlap=0.50
        )

        prediction = {
            'boxes': boxes,
            'labels': labels,
            'scores': scores
        }

        return {
            'prediction': prediction,
            'target': target
        }

    def _visualize(
        self,
        batch: dict[str, Any],
        stage: str
    ) -> None:
        """Visualize each target and prediction from a batch of data.

        Args:
            batch: An input target and prediction batch.
            stage: The name of the stage.

        """

        target = batch.get('target')
        prediction = batch.get('prediction')

        identifier = datetime.now(tz=UTC).strftime('%H%M%S')
        epoch = str(self.current_epoch).zfill(3)
        filename = f"{epoch}_{identifier}.png"

        path = Path('examples', stage)
        path.mkdir(exist_ok=True, parents=True)

        path = path.joinpath(filename)

        strategy = ClassAgnosticSingleShotStrategy(
            target,
            prediction
        )

        visualizer = Visualizer(strategy)
        visualizer.save(path)

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
            The stage loss and scalar loss.

        """

        name = 'test'

        return self._step(batch, name)

    def on_train_epoch_end(self) -> None:
        """Execute at the end of each training epoch."""

        self.is_training_visualized = False

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
            The stage loss and scalar loss.

        """

        name = 'training'
        output = self._process(batch)

        if not self.is_training_visualized:
            self._visualize(output, name)
            self.is_training_visualized = True

        return self._step(batch, name)

    def on_validation_epoch_end(self) -> None:
        """Execute at the end of each validation epoch."""

        self.is_validation_visualized = False

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
            The stage loss and scalar loss.

        """

        name = 'validation'
        output = self._process(batch)

        if not self.is_validation_visualized:
            self._visualize(output, name)
            self.is_validation_visualized = True

        return self._step(batch, name)
