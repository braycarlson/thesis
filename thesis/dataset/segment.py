"""
A script meant to create the "main" noise dataset, which
is to be used for training the model
"""

from __future__ import annotations

import random

from datatype.overlay import OverlayDataset
from thesis.constant import (
    MNIST,
    NOISE,
    SEGMENT
)


def main() -> None:
    random.seed(42)

    testing = [
        file
        for file in SEGMENT.glob('testing/*.png')
        if file.is_file()
    ]

    training = [
        file
        for file in SEGMENT.glob('training/*.png')
        if file.is_file()
    ]

    validation = [
        file
        for file in SEGMENT.glob('validation/*.png')
        if file.is_file()
    ]

    stroke = {
        'testing': testing,
        'training': training,
        'validation': validation
    }

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

    for amount in range(1, 6):
        name = str(amount).zfill(2)

        destination = NOISE.joinpath('segment', name)
        destination.mkdir(exist_ok=True, parents=True)

        dataset = OverlayDataset(
            amount=amount,
            destination=destination,
            partition=partition,
            stroke=stroke
        )

        dataset.apply()


if __name__ == '__main__':
    main()
