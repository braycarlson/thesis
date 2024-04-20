from __future__ import annotations

import numpy as np
import random
import uuid

from datatype.overlap import OverlapStrategy
from joblib import delayed, Parallel
from PIL import Image
from thesis.constant import CWD, MNIST, OVERLAP
from thesis.image import to_mask, to_transparent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path
    from typing_extensions import Any


class FlexibleOverlapDataset(OverlapStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'flexible'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'flexible'

    def paste(
        self,
        base: Path | str,
        images: list[Path | str],
    ) -> dict[str, Any]:
        """Paste image(s) onto a base image and generate the associated metadata.

        Args:
            base: A path to the base image.
            images: The path(s) to the images to be pasted.

        Returns:
            The metadata associated with the pasted images.

        """

        metadata = []
        background_width, background_height = self.canvas

        key = f"{self.amount}_{self.overlap_rate}"
        configuration = random.choices(self.configuration[key], k=1)

        path = OVERLAP.joinpath(str(self), base)
        path.mkdir(exist_ok=True, parents=True)

        mask_path = path.joinpath('mask')
        mask_path.mkdir(exist_ok=True, parents=True)

        background = Image.new('L', self.canvas, 0)
        image_mask = np.zeros(self.canvas, dtype=np.uint8)

        background_destination = path.joinpath(images[0].name)
        image_mask_destination = mask_path.joinpath(images[0].stem + '_mask.png')

        coordinates = []
        masks = []

        for positions in configuration:
            iterable = zip(images, positions, strict=True)

            for index, (image_path, position) in enumerate(iterable, 0):
                identifier = uuid.uuid1()
                identifier = str(identifier) + '.png'

                with Image.open(image_path).convert('RGBA') as image:
                    width, height = image.size
                    image = to_transparent(image)
                    mask = to_mask(image)

                x, y = int(position[0]), int(position[1])
                position = (x, y)

                xmax = x + width
                ymax = y + height
                coordinate = (x, y, xmax, ymax)

                background.paste(image, position, image)

                mask_np = np.array(mask)

                image_mask[y:y + mask_np.shape[0], x:x + mask_np.shape[1]] = np.where(
                    mask_np > 0,
                    index + 1,
                    image_mask[y:y + mask_np.shape[0], x:x + mask_np.shape[1]]
                )

                individual_mask = np.zeros_like(image_mask)

                individual_mask[y:y + mask_np.shape[0], x:x + mask_np.shape[1]] = np.where(
                    mask_np > 0, 1, 0
                )

                masks.append(individual_mask)
                coordinates.append(coordinate)

                area = width * height

                information = {
                    'file': background_destination.relative_to(CWD).as_posix(),
                    'mask': image_mask_destination.relative_to(CWD).as_posix(),
                    'area': area
                }
                metadata.append(information)

        for index, box in enumerate(coordinates):
            metadata[index]['coordinates'] = box

        array = np.array(masks)
        mxyxy = self.mask_to_box(array)

        for index, box in enumerate(mxyxy):
            metadata[index]['mxyxy'] = box.tolist()

        background = background.convert('L')
        image_mask = Image.fromarray(image_mask)

        background.save(background_destination)
        image_mask.save(image_mask_destination)

        background.close()
        image_mask.close()

        return metadata


def create(amount: int, rate: float, partition: str) -> None:
    """Create the dataset and apply it to a partition.

    Args:
        amount: The number of digits to be placed.
        rate: The overlap rate.
        partition: The partition to apply the dataset to.

    """

    overlap_rate = round(rate, 2)
    integer = int(overlap_rate * 100)

    base = f"{str(amount).zfill(2)}/{str(integer).zfill(2)}"

    dataset = FlexibleOverlapDataset(
        amount=amount,
        base=base,
        canvas=(128, 128),
        overlap_rate=overlap_rate,
        partition=partition
    )

    dataset.apply()


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

    task = []

    for amount in range(1, 9):
        for rate in np.arange(0.00, 1.01, 0.10):
            task.append(
                delayed(create)
                (amount, rate, partition)
            )

    Parallel(n_jobs=8)(task)


if __name__ == '__main__':
    main()
