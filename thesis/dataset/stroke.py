from __future__ import annotations

import numpy as np
import numpy.typing as npt

from itertools import product
from PIL import Image, ImageDraw
from thesis.constant import STROKE


class StrokeDataset:
    def __init__(self, dimension: int = 28, size: int = 3):
        """Initialize the dataset.

        Args:
            dimension: The dimensions of the canvas.
            size: The size of the stroke.

        """

        self.dimension = dimension
        self.size = size

    def draw(self, matrix: npt.NDArray) -> Image.Image:
        """Draw a stroke pattern on a canvas based on the provided matrix.

        Args:
            matrix: The matrix representing the stroke pattern.

        Returns:
            An image containing the stroke pattern.

        """

        canvas = Image.new(
            'RGBA',
            (self.dimension, self.dimension),
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(canvas)

        offset = (self.dimension - self.size) // 2

        for i in range(self.size):
            for j in range(self.size):
                if matrix[i, j] == 1:
                    draw.point(
                        (offset + j, offset + i),
                        fill=(255, 255, 255, 255)
                    )

        return canvas

    def create(self) -> None:
        """Create stroke pattern based on combinations of stroke matrices."""

        repeat = self.size * self.size

        combinations = product(
            [0, 1],
            repeat=repeat
        )

        size = (self.size, self.size)

        for index, combination in enumerate(combinations, 0):
            if index == 0:
                continue

            matrix = np.array(combination).reshape(size)
            image = self.draw(matrix)

            index = str(index).zfill(3)

            filename = f"{index}.png"
            path = STROKE.joinpath(filename)

            image.save(path)
            image.close()


def main() -> None:
    stroke = StrokeDataset(size=4)
    stroke.create()


if __name__ == '__main__':
    main()
