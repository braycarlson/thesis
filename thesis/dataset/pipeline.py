from __future__ import annotations

import numpy as np

from flexible import FlexibleOverlapDataset
from joblib import delayed, Parallel
from thesis.constant import MNIST


class PipelineOverlapDataset(FlexibleOverlapDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'pipeline'

    def __str__(self) -> str:
        """The string representation of the class.

        Returns:
            A string representation of the class.

        """

        return 'pipeline'


def create(amount: int, overlap_rate: float, partition: str) -> None:
    """Create the dataset and apply it to a partition.

    Args:
        amount: The amount of overlap.
        overlap_rate: The overlap rate.
        partition: The partition to apply the dataset to.

    """

    overlap_rate = round(overlap_rate, 2)
    integer = int(overlap_rate * 100)

    base = f"{str(amount).zfill(2)}/{str(integer).zfill(2)}"

    dataset = PipelineOverlapDataset(
        amount=amount,
        base=base,
        canvas=(128, 128),
        overlap_rate=overlap_rate,
        partition=partition,
    )

    dataset.apply()


def main() -> None:
    testing = [
        file
        for file in MNIST.glob('testing/*/*.png')
        if file.is_file()
    ]

    partition = {'testing': testing}

    task = []

    for amount in range(1, 11):
        for rate in np.arange(0.00, 1.01, 0.10):
            rate = round(rate, 2)

            task.append(
                delayed(create)
                (amount, rate, partition)
            )

    Parallel(n_jobs=8)(task)


if __name__ == '__main__':
    main()
