from __future__ import annotations

import random
import shutil

from thesis.constant import MNIST, ORIGINAL
from tqdm import tqdm


def main() -> None:
    random.seed(42)

    testing = [
        file
        for file in ORIGINAL.glob('testing/*/*.png')
        if file.is_file()
    ]

    training = [
        file
        for file in ORIGINAL.glob('training/*/*.png')
        if file.is_file()
    ]

    random.shuffle(training)

    index = int(len(training) * 0.80)

    validation = training[index:]
    training = training[:index]

    partition = {
        'testing': testing,
        'training': training,
        'validation': validation
    }

    for name in partition:
        base = MNIST.joinpath(name)
        base.mkdir(exist_ok=True, parents=True)

        images = partition[name]

        for image in tqdm(images):
            label = image.parent.relative_to(ORIGINAL).stem

            path = base.joinpath(label)
            path.mkdir(exist_ok=True, parents=True)

            path = path.joinpath(image.name)
            shutil.copy(image, path)


if __name__ == '__main__':
    main()
