from __future__ import annotations

import pandas as pd
import random

from PIL import Image
from thesis.constant import CWD
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing_extensions import Any


class OverlayDataset:
    def __init__(
        self,
        amount: int = 3,
        destination: Path | str | None = None,
        partition: dict[str, list[str | Path]] | None = None,
        stroke: list[Image.Image] | None = None
    ):
        self.amount = amount
        self.destination = destination
        self.partition = partition
        self.stroke = stroke

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'overlay'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'overlay'

    def _process(self, partition: str, image: str) -> dict[Any, Any]:
        """Process a partition of the dataset.

        Args:
            partition: The partition of the dataset.
            image: The image filename.

        Returns:
            The metadata of the processed partition.

        """

        original = image

        label = image.parent.relative_to(image.parent.parent)
        label = str(label)

        path = self.destination.joinpath(partition, label)
        path.mkdir(exist_ok=True, parents=True)

        file = path.joinpath(image.name)

        with Image.open(image) as image:
            image = image.convert('RGBA')

            stroke = self.paste(image, partition)

            image = image.convert('L')
            image.save(file)

        file = file.relative_to(CWD).as_posix()
        label = int(label)

        return {
            'file': file,
            'original': original,
            'label': label,
            'stroke': stroke,
        }

    def apply(self) -> None:
        """Apply the dataset generation process."""

        for partition in tqdm(self.partition):
            dataset = []

            images = self.partition.get(partition)

            for image in tqdm(images):
                metadata = self._process(partition, image)
                dataset.append(metadata)

            dataframe = pd.DataFrame(dataset)

            pickle = self.destination.joinpath(partition + '.pkl')
            dataframe.to_pickle(pickle)

    def paste(self, image: Image.Image, partition: str) -> list[dict]:
        """Paste stroke(s) onto the image.

        Args:
            image: The image to paste strokes onto.
            partition: The partition of the dataset.

        Returns:
            The metadata for the pasted strokes.

        """

        metadata = []

        images = self.stroke[partition]

        for _ in range(self.amount):
            path = random.choice(images)

            with Image.open(path) as stroke:
                stroke = stroke.convert('L')

                x = random.randint(-12, 15)
                y = random.randint(-12, 15)

                image.paste(stroke, (x, y), stroke)

                coordinates = (
                    x,
                    y,
                    x + stroke.width,
                    y + stroke.height
                )

                information = {
                    'file': path.relative_to(CWD).as_posix(),
                    'coordinates': coordinates
                }

                metadata.append(information)

        return metadata
