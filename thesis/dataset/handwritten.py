from __future__ import annotations

import numpy as np

from PIL import Image
from thesis.constant import HANDWRITTEN


def main() -> None:
    files = [
        file
        for file in HANDWRITTEN.glob('original/*.png')
        if file.is_file()
    ]

    for file in files:
        image = Image.open(file).convert('L')
        array = np.array(image)
        image = Image.fromarray(~array)

        filename = f"{file.stem}.png"
        path = file.parent.parent.joinpath(filename)
        image.save(path)


if __name__ == '__main__':
    main()
