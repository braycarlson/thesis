from __future__ import annotations

import math
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from matplotlib import patches
from thesis.coordinates import (
    CoordinatesConverter,
    ScalarStrategy
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from pathlib import Path
    from torch import Tensor
    from typing_extensions import Any


class VisualizationStrategy(ABC):
    @abstractmethod
    def visualize(self, axes: plt.Axes) -> None:
        """The base class for visualization strategies.

        Args:
            axes: The axes to use for the visualization.

        """

        raise NotImplementedError


class ClassAgnosticSingleShotStrategy(VisualizationStrategy):
    def __init__(
        self,
        target: dict[str, Any],
        prediction: dict[str, Any]
    ):
        strategy = ScalarStrategy(128, 128)
        self.converter = CoordinatesConverter(strategy)
        self.prediction = prediction
        self.target = target

        self.image = target.get('image')

    def visualize(self, axes: plt.Axes) -> None:
        """The visualization of the CASSD model during training.

        Create a visualization for target and prediction bounding boxes.

        Args:
            axes: The axes to use for visualization.

        """

        bt = self.target.get('boxes')
        lt = self.target.get('labels')

        bp = self.prediction.get('boxes')
        lp = self.prediction.get('labels')

        for i, ax in enumerate(axes.flatten()):
            if i >= len(self.image):
                break

            ax.imshow(
                self.image[i].permute(1, 2, 0).cpu().numpy(),
                cmap='gray'
            )

            if i < len(bt):
                boxes = [
                    self.converter.scale(box, 'xyxy')
                    for box in bt[i].cpu().numpy()
                ]

                self._draw(
                    ax,
                    boxes,
                    lt[i].cpu().numpy(),
                    'g'
                )

            if i < len(bp):
                boxes = [
                    self.converter.scale(box, 'xyxy')
                    for box in bp[i].detach().cpu().numpy()
                ]

                self._draw(
                    ax,
                    boxes,
                    lp[i].detach().cpu().numpy(),
                    'r'
                )

            ax.set_axis_off()

    def _draw(
        self,
        ax: plt.Axes,
        boxes: list[npt.NDArray],
        labels: npt.NDArray,
        edgecolor: str
    ) -> None:
        """Draw the bounding boxes on an axes.

        Args:
            ax: The axes to draw bounding boxes on.
            boxes: The bounding boxes.
            labels: The labels corresponding to the bounding boxes.
            edgecolor: The edge color for the bounding boxes.

        """

        for box, label in zip(boxes, labels, strict=False):
            if label == 0:
                continue

            x_min, y_min, x_max, y_max = box

            linewidth = 2
            offset = linewidth / 2

            x_min = max(0 + offset, x_min)
            y_min = max(0 + offset, y_min)
            x_max = min(128 - offset, x_max)
            y_max = min(128 - offset, y_max)

            width, height = x_max - x_min, y_max - y_min

            rect = patches.Rectangle(
                (x_min - offset, y_min - offset),
                width,
                height,
                edgecolor=edgecolor,
                facecolor='none',
                linewidth=2
            )

            ax.add_patch(rect)

            ax.text(
                x_min,
                y_min,
                str(label),
                fontsize=8,
                color='white',
                backgroundcolor=edgecolor
            )


class ClassificationStrategy(VisualizationStrategy):
    def __init__(
        self,
        image: list[Tensor],
        target: Tensor,
        prediction: Tensor
    ):
        self.image = image
        self.prediction = prediction
        self.target = target

    def visualize(self, axes: plt.Axes) -> None:
        """The visualization of the classification model during training.

        Create a visualization for target and prediction classes.

        Args:
            axes: The axes to use for visualization.

        """

        length = len(self.image)

        for i, ax in enumerate(axes):
            if i >= length:
                break

            target = self.target[i].item()
            prediction = self.prediction[i].item()

            title = f"Target: {target}, Prediction: {prediction}"

            image = self.image[i].squeeze().cpu().numpy()
            ax.imshow(image, cmap='gray')
            ax.set_title(title, fontsize=20)
            ax.set_axis_off()


class Visualizer:
    """A visualizer to use during training."""

    def __init__(
        self,
        strategy: VisualizationStrategy | None = None
    ):
        """Initialize the visualizer.

        Args:
            strategy: The type of visualization to use.

        """

        self.strategy = strategy

    def save(self, path: Path | str) -> None:
        """Save the visualization to disk.

        Args:
            path: The location to save the visualization.

        """

        length = len(self.strategy.image)

        amount = length // 4
        amount = min(amount, length // 4)
        columns = math.ceil(math.sqrt(amount))
        rows = math.ceil(amount / columns)

        figsize = (
            int(columns) * 5,
            int(rows) * 5
        )

        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figsize
        )

        axes = axes.flatten()

        self.strategy.visualize(axes)

        for ax in axes[length:]:
            ax.set_visible(False)

        plt.tight_layout()

        plt.savefig(
            path,
            dpi=72,
            format='png'
        )

        plt.close(fig)
