from __future__ import annotations

import shutil

from thesis.constant import ORIGINAL
from torchvision import datasets


def main() -> None:
    # Training
    root = ORIGINAL.joinpath('train')
    root.mkdir(exist_ok=True, parents=True)

    dataset = datasets.MNIST(
        download=True,
        root=root,
        train=True,
        transform=None
    )

    training = ORIGINAL.joinpath('training')
    training.mkdir(exist_ok=True, parents=True)

    for index, (image, label) in enumerate(dataset):
        digit = str(label)

        folder = training.joinpath(digit)
        folder.mkdir(exist_ok=True, parents=True)

        path = folder.joinpath(f"{index}.png")
        image.save(path)

    if root.exists():
        shutil.rmtree(root)

    # Testing
    root = ORIGINAL.joinpath('test')
    root.mkdir(exist_ok=True, parents=True)

    dataset = datasets.MNIST(
        download=True,
        root=root,
        train=False,
        transform=None
    )

    testing = ORIGINAL.joinpath('testing')
    testing.mkdir(exist_ok=True, parents=True)

    for index, (image, label) in enumerate(dataset):
        digit = str(label)

        folder = testing.joinpath(digit)
        folder.mkdir(exist_ok=True, parents=True)

        path = folder.joinpath(f"{index}.png")
        image.save(path)

    if root.exists():
        shutil.rmtree(root)


if __name__ == '__main__':
    main()
