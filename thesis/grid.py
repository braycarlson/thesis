from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from PIL import Image, ImageDraw, ImageFont
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class GridStrategy(ABC):
    """The base class for grid strategies."""

    def __init__(
        self,
        images: list[Image.Image] | None = None,
        labels: list[str] | list[int] | None = None,
        title: str | None = None
    ):
        self._grid = None
        self.images = images
        self.labels = labels
        self.title = title

    @abstractmethod
    def generate(self, maximum: int = 0, zoom: int = 2) -> None:
        """Generate the visualization.

        Args:
            maximum: The maximum number of images per row/column in the grid.
            zoom: The zoom factor for the visualization.

        Returns:
            The grid of images.

        """

        raise NotImplementedError

    @abstractmethod
    def save(self, filename: str, *args, **kwargs) -> None:
        """Save the visualization to disk.

        Args:
            filename: The name of the file to save the visualization as.

        """

        raise NotImplementedError

    @abstractmethod
    def show(self) -> None:
        """Display the visualization."""

        raise NotImplementedError

    def cleanup(self) -> None:
        """Close each opened of handle."""

        for image in self.images:
            image.close()

    def size(self, maximum: int) -> int:
        """Calculate the size of the grid.

        Args:
            maximum: The maximum number of images per row/column in the grid.

        Returns:
            The size of the grid.

        """

        if maximum == 0:
            length = len(self.images)
            sqrt = math.sqrt(length)
            closest = math.ceil(sqrt) ** 2

            sqrt = np.sqrt(closest)
            return int(sqrt)

        return maximum


class MatplotlibGrid(GridStrategy):
    """The Matplotlib grid strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, maximum: int = 0, zoom: int = 2) -> tuple[Figure, Axes]:
        """Generate the visualization using Matplotlib.

        Args:
            maximum: The maximum number of images per row/column in the grid.
            zoom: The zoom factor for the visualization.

        Returns:
            The Matplotlib figure and axes.

        """

        size = self.size(maximum)

        first, *_ = self.images
        i, j, _ = np.shape(first)

        nrows = size
        ncols = size

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(
                zoom * ncols,
                zoom * nrows
            )
        )

        fig.patch.set_facecolor('black')

        for axis in ax.flat:
            axis.set_facecolor('black')
            axis.axis('off')

        for index, image in enumerate(self.images):
            row_index = index // ncols
            col_index = index % ncols

            ax[row_index, col_index].imshow(
                image,
                cmap='gray',
                aspect='equal',
                origin='upper',
                interpolation='nearest'
            )

            if self.labels and index < len(self.labels):
                ax[row_index, col_index].set_title(
                    str(self.labels[index]),
                    color='red',
                    fontsize=14
                )

        if self.title:
            fig.suptitle(
                self.title,
                color='white',
                fontsize=28,
                fontweight='bold'
            )

        plt.tight_layout(pad=1.5)

        fig.subplots_adjust(
            left=0.05,
            right=0.95,
            top=0.90,
            bottom=0.05,
            hspace=0.5,
            wspace=0.5
        )

        return fig, ax

    def save(self, filename: str, *args, **kwargs) -> None:
        """Save the visualization using Matplotlib.

        Args:
            filename: The name of the file to save the visualization to.

        """

        plt.savefig(
            filename,
            *args,
            bbox_inches='tight',
            dpi=300,
            format='png',
            pad_inches=0.5,
            **kwargs
        )

        plt.close()

    def show(self) -> None:
        """Display the visualization using Matplotlib."""

        plt.tight_layout()
        plt.show()


class PillowGrid(GridStrategy):
    """PIL grid strategy."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, maximum: int = 0, zoom: int = 2) -> Image.Image:
        """Generate the visualization using PIL.

        Args:
            maximum: The maximum number of images per row/column in the grid.
            zoom: The zoom factor for the visualization.

        Returns:
            A PIL Image.

        """

        maximum = self.size(maximum)

        first, *_ = self.images
        row, column = first.size

        title = 20
        offset = 20
        space = 20
        padding = 20

        height = row * maximum + title * maximum + (maximum - 1) * padding
        width = maximum * column + (maximum - 1) * padding

        dimension = (height, width, 3)
        canvas = np.zeros(dimension, dtype=np.uint8)

        position = 0
        current = 0
        font = ImageFont.truetype('arial.ttf', 10)
        last = 0

        for index, image in enumerate(self.images):
            image = image.convert('RGB')
            array = np.array(image)
            image_height, image_width, _ = array.shape

            if position + image_width > maximum * column:
                if current == maximum - 1:
                    break

                position = 0
                current = current + 1

            start_x = position + (position // column) * padding
            start_y = row * current + title * (current + 1) + current * padding
            end_x = start_x + image_width
            end_y = start_y + image_height

            canvas[start_y:end_y, start_x:end_x, :] = array

            if self.labels and index < len(self.labels):
                label = str(self.labels[index])
                text_width, _ = font.getsize(label)

                image = Image.fromarray(canvas)
                draw = ImageDraw.Draw(image)

                x = (end_x + start_x - text_width) // 2
                y = start_y - title

                draw.text(
                    (x, y),
                    label,
                    fill='red',
                    font=font
                )

                canvas = np.array(image)

            last = end_y
            position = position + image_width

        canvas = canvas[:last, :, :]

        dimension = (height + 2 * padding + offset + space, width + 2 * padding, 3)
        final = np.zeros(dimension, dtype=np.uint8)

        start_y = padding + offset + space
        start_x = padding

        final[start_y:start_y + height, start_x:start_x + width, :] = canvas
        final = final.astype('uint8')

        image = Image.fromarray(final)

        image = image.resize(
            (image.width * zoom, image.height * zoom),
            Image.NEAREST
        )

        self._grid = image.convert('RGB')

        if self.title:
            draw = ImageDraw.Draw(self._grid)
            font = ImageFont.truetype('arial.ttf', 36)
            text_width, _ = font.getsize(self.title)

            x = (self._grid.width - text_width) // 2
            y = padding * zoom

            draw.text(
                (x, y),
                str(self.title),
                fill='white',
                font=font
            )

        return image

    def save(self, filename: str, *args, **kwargs) -> None:
        """Save the visualization using PIL.

        Args:
            filename: The name of the file to save the grid visualization to.

        """

        self._grid.save(filename, *args, **kwargs)

    def show(self) -> None:
        """Display the visualization using PIL."""
        self._grid.show()


class Grid:
    """A concrete class for the grid strategies."""

    def __init__(self):
        self.strategy = MatplotlibGrid()

    @property
    def images(self) -> list[Image.Image]:
        """A list of images to be displayed on the grid."""

        return self.strategy.images

    @images.setter
    def images(self, images: list[Image.Image]) -> None:
        self.strategy.images = images

    def generate(self, *args, **kwargs) -> Image.Image | tuple[Figure, Axes]:
        """Generate the visualization."""

        return self.strategy.generate(*args, **kwargs)

    def save(self, filename: str, *args, **kwargs) -> None:
        """Save the visualization to disk."""

        self.strategy.save(filename, *args, **kwargs)

    def show(self) -> None:
        """Display the visualization."""

        self.strategy.show()
