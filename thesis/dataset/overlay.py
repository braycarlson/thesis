"""
A script meant to create several "testing" datasets using
1-5 strokes per image. These datasets are to be used in
the "sample.py" script
"""

from __future__ import annotations

import random

from datatype.overlay import OverlayDataset
from PIL import Image
from thesis.constant import (
    MNIST,
    NOISE,
    SEGMENT
)


def main() -> None:
    random.seed(42)

    stroke = {
        file: Image.open(file).convert('L')
        for file in SEGMENT.glob('*.png')
        if file.is_file()
    }

    testing = [
        file
        for file in MNIST.glob('testing/*/*.png')
        if file.is_file()
    ]

    partition = {
        'testing': testing,
    }

    for amount in range(1, 6):
        name = str(amount).zfill(2)

        destination = NOISE.joinpath('segment/testing', name)
        destination.mkdir(exist_ok=True, parents=True)

        dataset = OverlayDataset(
            amount=amount,
            destination=destination,
            partition=partition,
            stroke=stroke
        )

        dataset.apply()

    for path in stroke:
        image = stroke[path]
        image.close()


if __name__ == '__main__':
    main()
