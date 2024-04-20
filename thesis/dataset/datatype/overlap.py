from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import pickle
import random

from abc import ABC, abstractmethod
from enum import Enum, auto
from thesis.constant import CWD, OVERLAP
from thesis.coordinates import CoordinatesConverter, ScalarStrategy
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from PIL import Image
    from typing_extensions import Any


class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class OverlapStrategy(ABC):
    def __init__(
        self,
        amount: int = 1,
        default: list[int] | None = (8, 16, 32, 64),
        base: str | Path | None = None,
        overlap_rate: float = 0.10,
        partition: dict[str, list[str | Path]] | None = None,
        canvas: tuple[int, int] = (128, 128)
    ):
        """Initialize the overlap strategy.

        Args:
            amount: The number of clusters to generate.
            default: The default sizes for background boxes.
            base: The base directory for saving images.
            overlap_rate: The overlap rate threshold.
            partition: The partition of the dataset.
            canvas: The dimensions of the canvas.

        """

        width, height = canvas

        self.amount = amount
        self.default = default
        self.base = base
        strategy = ScalarStrategy(width, height)
        self.converter = CoordinatesConverter(strategy)
        self.overlap_rate = overlap_rate
        self.partition = partition
        self.canvas = canvas

        with open('configuration.pkl', 'rb') as file:
            self.configuration = pickle.load(file)

    @abstractmethod
    def paste(
        self,
        image: Image.Image,
        background: Image.Image,
        center_width: int,
        center_height: int
    ) -> tuple[int, int, int, int]:
        """Paste an image onto a background.

        Args:
            image: The image to paste onto the background.
            background: The background image.
            center_width: The center width position for pasting.
            center_height: The center height position for pasting.

        Returns:
            The coordinates of the pasted image.

        """

        raise NotImplementedError

    def _process(self, partition: str, image: str) -> dict[Any, Any]:
        """Process a partition of the dataset.

        Args:
            partition: The partition of the dataset.
            image: The image filename.

        Returns:
            The metadata of the processed partition.

        """

        amount = (
            random.randint(1, 10)
            if self.amount == -1
            else self.amount
        )

        component = (
            [] if self.amount == 0
            else random.sample(self.partition[partition], amount)
        )

        component.insert(0, image)

        length = len(component)

        label = [
            image.parent.relative_to(image.parent.parent).as_posix()
            for image in component
        ]

        label = [
            int(digit)
            for digit in label
        ]

        path = f"{self.base}/{partition}/{label[-1]}"

        metadata = self.paste(path, component)

        file = [
            file.get('file')
            for file in metadata
        ]

        file, *_ = file

        mask = [
            file.get('mask')
            for file in metadata
        ]

        mask, *_ = mask

        area = [
            file.get('area')
            for file in metadata
        ]

        xyxy = [
            file.get('coordinates')
            for file in metadata
        ]

        mxyxy = [
            file.get('mxyxy')
            for file in metadata
        ]

        if partition in ('training', 'validation'):
            boxes = self._create_background_boxes(xyxy)

            areas = [
                (box[2] - box[0]) * (box[3] - box[1])
                for box in boxes
            ]

            xyxy.extend(boxes)
            mxyxy.extend(boxes)
            area.extend(areas)

        xywh = [
            self.converter.xyxy_to_xywh(coordinates)
            for coordinates in xyxy
        ]

        cxcywh = [
            self.converter.xyxy_to_cxcywh(coordinates)
            for coordinates in xyxy
        ]

        normalize = [
            self.converter.normalize(coordinates, 'xyxy')
            for coordinates in xyxy
        ]

        identifier = [
            i + 1
            for i in range(len(xyxy))
        ]

        objectness = [1.0] * length

        if partition in ('training', 'validation'):
            amount = len(xyxy) - len(objectness)
            background = [0.0] * amount
            objectness.extend(background)

            background = [10] * amount
            label.extend(background)

        component = [
            path.relative_to(CWD).as_posix()
            for path in component
        ]

        return {
            'identifier': identifier,
            'file': file,
            'label': label,
            'objectness': objectness,
            'component': component,
            'area': area,
            'mask': mask,
            'mxyxy': mxyxy,
            'xyxy': xyxy,
            'xywh': xywh,
            'cxcywh': cxcywh,
            'normalize': normalize,
        }

    def _create_background_boxes(
        self,
        boxes: list[tuple[int, int, int, int]]
    ) -> list[tuple[int, int, int, int]]:
        """Create the background boxes.

        Args:
            boxes: The bounding boxes.

        Returns:
            The background bounding boxes.

        """

        width, height = self.canvas
        canvas_array = np.zeros((height, width))

        for box in boxes:
            x1, y1, x2, y2 = box
            canvas_array[y1:y2, x1:x2] = 1

        def is_placeable(x: int, y: int, size: int) -> bool:
            if x + size > width or y + size > height:
                return False

            return not np.any(canvas_array[y:y+size, x:x+size])

        def place(x: int, y: int, size: int) -> list[int]:
            canvas_array[y:y+size, x:x+size] = 1
            return [x, y, x + size, y + size]

        def find_unoccupied(start_x: int, start_y: int) -> tuple[int, int]:
            for y in range(start_y, height):
                x_start = start_x if y == start_y else 0

                for x in range(x_start, width):
                    if canvas_array[y, x] == 0:
                        return x, y

            return None, None

        background_boxes = []
        start_x, start_y = 0, 0

        while True:
            start_x, start_y = find_unoccupied(start_x, start_y)

            if start_x is None:
                break

            placed = False

            for box_size in sorted(self.default, reverse=True):
                if is_placeable(start_x, start_y, box_size):
                    box = place(start_x, start_y, box_size)
                    background_boxes.append(box)

                    start_x = start_x + box_size
                    placed = True
                    break

            if not placed:
                start_x = start_x + 1

                if start_x >= width:
                    start_x = 0
                    start_y = start_y + 1

        return background_boxes

    def apply(self) -> None:
        """Apply the overlap strategy to generate the dataset."""

        for partition, images in self.partition.items():
            dataset = []

            for image in tqdm(images):
                metadata = self._process(partition, image)
                dataset.append(metadata)

            dataframe = pd.DataFrame(dataset)

            filename = f"{partition}.pkl"
            strategy = str(self)

            pickle = OVERLAP.joinpath(strategy, self.base, filename)
            dataframe.to_pickle(pickle)

    def mask_to_box(self, masks: npt.NDArray) -> npt.NDArray:
        """Convert image masks to bounding boxes.

        Args:
            masks: The image masks.

        Returns:
            The bounding boxes.

        """

        if masks.size == 0:
            return np.zeros((0, 4))

        n = masks.shape[0]
        boxes = np.zeros((n, 4))

        for index, mask in enumerate(masks):
            positions = np.where(mask != 0)
            y, x = positions[0], positions[1]

            if y.size > 0 and x.size > 0:
                x1, y1, x2, y2 = np.min(x), np.min(y), np.max(x), np.max(y)
                boxes[index] = np.array([x1, y1, x2, y2])

        return boxes
