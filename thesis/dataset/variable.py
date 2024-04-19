from __future__ import annotations

import math
import numpy as np
import pandas as pd
import random
import uuid

from PIL import Image
from thesis.constant import CWD, MNIST, OVERLAP
from thesis.coordinates import CoordinatesConverter, ScalarStrategy
from thesis.image import to_mask, to_transparent
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class VariableOverlapDataset:
    def __init__(
        self,
        amount: int = 1,
        partition: dict[str, list[str | Path]] | None = None,
        canvas: tuple[int, int] = (128, 128)
    ):
        width, height = canvas

        self.amount = amount
        self.partition = partition
        self.canvas = canvas
        strategy = ScalarStrategy(128, 128)
        self.converter = CoordinatesConverter(strategy)

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'variable'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'variable'

    def _is_overlapping(
        self,
        x: tuple[int, int, int, int],
        y: tuple[int, int, int, int],
        rate: float
    ) -> bool:
        """
        Check if two boxes overlap.

        Args:
            x: The coordinates of the first box.
            y: The coordinates of the second box.
            rate: The overlap rate threshold.

        Returns:
            True if boxes overlap more than the threshold, False otherwise.

        """

        x1, y1, x2, y2 = x
        x3, y3, x4, y4 = y

        overlap_x = max(0, min(x2, x4) - max(x1, x3))
        overlap_y = max(0, min(y2, y4) - max(y1, y3))

        overlap_area = overlap_x * overlap_y
        area_x = (x2 - x1) * (y2 - y1)
        area_y = (x4 - x3) * (y4 - y3)

        total = overlap_area / min(area_x, area_y)
        return total > rate

    def _distance(
        self,
        cluster: list[tuple[int, int]],
        minimum: float = 50,
        maximum: float = 120
    ) -> tuple[int, int]:
        """Calculate the distance of a point from existing clusters.

        Args:
            cluster: A list of existing cluster positions.
            minimum: The minimum distance allowed.
            maximum: The maximum distance allowed.

        Returns:
            A tuple representing the coordinates of the new point.

        """

        width, height = self.canvas
        padding = 10

        if not cluster:
            return (
                random.randint(padding, width - padding),
                random.randint(padding, height - padding)
            )

        best_distance = float('inf')
        best_point = (0, 0)

        for _ in range(100):
            candidate_x = random.randint(padding, width - padding - 1)
            candidate_y = random.randint(padding, height - padding - 1)

            min_distance = min(
                math.hypot(candidate_x - x, candidate_y - y)
                for x, y in cluster
            )

            if minimum <= min_distance <= maximum:
                if min_distance < best_distance:
                    best_distance = min_distance
                    best_point = (candidate_x, candidate_y)

        return best_point

    def _process(self, partition: str) -> dict:
        """Process a partition of the dataset.

        Args:
            partition: The partition of the dataset.

        Returns:
            A dictionary containing metadata of the processed partition.

        """

        background = Image.new('L', self.canvas, 0)
        image_mask = np.zeros(self.canvas, dtype=np.uint8)

        clusters = random.randint(1, self.amount)
        existing = []

        images = self.partition[partition]

        strategy = str(self)

        path = OVERLAP.joinpath(strategy, partition)
        path.mkdir(exist_ok=True, parents=True)

        mask_path = path.joinpath('mask')
        mask_path.mkdir(exist_ok=True, parents=True)

        labels = []
        components = []
        areas = []
        mxyxy = []
        xyxy = []
        xywh = []
        cxcywh = []
        normalize = []

        unique = 1

        for _ in range(clusters):
            cluster_x, cluster_y = self._distance(existing)

            position = (cluster_x, cluster_y)
            existing.append(position)

            n = random.randint(2, 3)
            samples = random.sample(images, n)

            for sample in samples:
                label = sample.parent.relative_to(sample.parent.parent)
                label = int(str(label))

                image = Image.open(sample).convert('RGBA')

                scale = random.uniform(0.75, 1.5)
                scaled_width = int(image.width * scale)
                scaled_height = int(image.height * scale)
                size = (scaled_width, scaled_height)

                image = image.resize(size)
                image = to_transparent(image)
                mask = to_mask(image)

                overlap_rate = random.uniform(0.00, 1.00)

                placed = False
                attempt = 0

                while not placed and attempt < 50:
                    x_offset = random.randint(-scaled_width // 2, scaled_width // 2)
                    y_offset = random.randint(-scaled_height // 2, scaled_height // 2)

                    x = cluster_x + x_offset
                    y = cluster_y + y_offset

                    x = min(max(x, 0), background.width - scaled_width)
                    y = min(max(y, 0), background.height - scaled_height)

                    coordinates = (x, y, x + scaled_width, y + scaled_height)

                    condition = (
                        self._is_overlapping(coordinates, other_xyxy, overlap_rate)
                        for other_xyxy in xyxy
                    )

                    if not any(condition):
                        identifier = uuid.uuid1()
                        identifier = str(identifier) + '.png'

                        background.paste(image, (x, y), image)

                        mask_np = np.array(mask)

                        image_mask[y:y + scaled_height, x:x + scaled_width] = np.where(
                            mask_np > 0,
                            unique,
                            image_mask[y:y + scaled_height, x:x + scaled_width]
                        )

                        unique = unique + 1

                        component = sample.relative_to(CWD).as_posix()
                        mask_file_path  = mask_path.joinpath(identifier)
                        area = scaled_width * scaled_height

                        mask_file_path = mask_file_path .relative_to(CWD).as_posix()

                        labels.append(label)
                        components.append(component)
                        areas.append(area)
                        xyxy.append(coordinates)
                        xywh.append(self.converter.xyxy_to_xywh(coordinates))
                        cxcywh.append(self.converter.xyxy_to_cxcywh(coordinates))
                        normalize.append(self.converter.normalize(coordinates, 'xyxy'))

                        placed = True

                    attempt = attempt + 1

                image.close()

        identifier = uuid.uuid1()
        filename = f"{identifier}.png"

        image_mask = Image.fromarray(image_mask)
        image_mask_path = mask_path.joinpath(filename)
        image_mask.save(image_mask_path)
        image_mask.close()

        background_path = path.joinpath(filename)
        background.save(background_path)
        background.close()

        length = len(labels)

        file = background_path.relative_to(CWD).as_posix()
        mask = image_mask_path.relative_to(CWD).as_posix()

        identifier = [
            i + 1
            for i in range(length)
        ]

        objectness = [1.0] * length

        return {
            'file': file,
            'identifier': identifier,
            'label': labels,
            'objectness': objectness,
            'component': components,
            'area': areas,
            'mask': mask,
            'mxyxy': mxyxy,
            'xyxy': xyxy,
            'xywh': xywh,
            'cxcywh': cxcywh,
            'normalize': normalize,
        }

    def apply(self) -> None:
        """Create the variable overlap dataset."""

        for partition, images in self.partition.items():
            dataset = []

            for _ in tqdm(images):
                metadata = self._process(partition)
                dataset.append(metadata)

            dataframe = pd.DataFrame(dataset)

            filename = f"{partition}.pkl"
            strategy = str(self)

            pickle = OVERLAP.joinpath(strategy, filename)
            dataframe.to_pickle(pickle)


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

    dataset = VariableOverlapDataset(
        amount=3,
        canvas=(128, 128),
        partition=partition
    )

    dataset.apply()


if __name__ == '__main__':
    main()
