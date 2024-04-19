from __future__ import annotations

import numpy as np
import numpy.typing as npt

from PIL import Image
from thesis.constant import MNIST, SEGMENT
from thesis.image import to_transparent
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class SegmentDataset:
    def __init__(
        self,
        partition: dict[str, list[str | Path]] | None = None,
    ):
        """
        Initialize the dataset.

        Args:
            partition: The partition of the dataset.

        """

        self.counter = 0
        self.partition = partition

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'segment'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'segment'

    def _is_threshold(self, segment: npt.NDArray) -> bool:
        """Determine if a segment meets the threshold for being considered sufficient.

        Args:
            segment: The segment array.

        Returns:
            True if the segment meets the threshold, False otherwise.

        """

        total = (
            segment.shape[0] *
            segment.shape[1]
        )

        count = np.sum(segment[:, :, :3] == 0) / 3
        return (count / total) > 0.95

    def _recenter(self, segment: npt.NDArray) -> Image:
        """Recenter a segment within a 28x28 background.

        Args:
            segment: The segment array.

        Returns:
            An image with the segment recentered.

        """

        dimension = (28, 28, 4)

        background = np.zeros(
            dimension,
            dtype=np.uint8
        )

        background_x, background_y, background_c = np.shape(background)
        segment_x, segment_y, segment_c = np.shape(segment)

        start_y = (background_x - segment_x) // 2
        start_x = (background_y - segment_y) // 2

        if segment_c == 3:
            background[
                start_y:start_y+segment_x,
                start_x:start_x+segment_y,
                :3
            ] = segment

            background[
                start_y:start_y+segment_x,
                start_x:start_x+segment_y,
                3
            ] = 255

        if segment_c == 4:
            background[
                start_y:start_y+segment_x,
                start_x:start_x+segment_y
            ] = segment

        return Image.fromarray(background, 'RGBA')

    def _segment(self, path: Path | str) -> None:
        """Divide an image into top, bottom, left, and right segments.

        Args:
            path: The path to the image.

        Returns:
            A tuple containing the top, bottom, left, and right segments.

        """

        with Image.open(path) as image:
            image = image.convert('L')
            image = to_transparent(image)

        array = np.array(image)

        top, bottom = np.split(array, 2, axis=0)
        left, right = np.split(array, 2, axis=1)

        return (
            top,
            bottom,
            left,
            right
        )

    def create(self) -> None:
        """Create the dataset."""

        for name, images in self.partition.items():
            length = len(images) - 1

            string = str(length)
            fill = len(string)

            base = SEGMENT.joinpath(name)
            base.mkdir(exist_ok=True, parents=True)

            for image in tqdm(images):
                for segment in self._segment(image):
                    if self._is_threshold(segment):
                        continue

                    filename = str(self.counter).zfill(fill) + '.png'
                    path = base.joinpath(filename)

                    self._recenter(segment).save(path)

                    self.counter = self.counter + 1


def main() -> None:
    testing = [
        file
        for file in MNIST.glob('testing/*/*.png')
        if file.is_file()
    ]

    training = [
        file
        for file in MNIST.glob('training/*/*.png')
        if file.is_file()
    ]

    validation = [
        file
        for file in MNIST.glob('validation/*/*.png')
        if file.is_file()
    ]

    partition = {
        'testing': testing,
        'training': training,
        'validation': validation
    }

    segment = SegmentDataset(partition=partition)
    segment.create()


if __name__ == '__main__':
    main()
