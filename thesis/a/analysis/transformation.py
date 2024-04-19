from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from PIL import Image
from thesis.constant import CWD, HEIGHT, NOISE, WIDTH
from thesis.transform import GradientTransform, ShiftTransform
from thesis.transformation import TorchvisionStrategy
from torchvision.transforms import v2


warnings.filterwarnings(
    'ignore',
    category=UserWarning
)


def main() -> None:
    for _ in range(10):
        size = (128, 128)

        transform = [
            v2.Resize(size, antialias=True),
            ShiftTransform(25),
            v2.RandomChoice([
                GradientTransform(),
                v2.GaussianBlur(kernel_size=3),
                v2.RandomAutocontrast(p=0.5),
                v2.RandomErasing(
                    p=0.5,
                    scale=(0.025, 0.050),
                    ratio=(0.025, 0.050),
                    value=0
                ),
                v2.RandomPerspective(distortion_scale=0.2, p=0.5),
                v2.RandomResizedCrop(
                    size,
                    antialias=True,
                    scale=(0.8, 1.0),
                    ratio=(0.75, 1.33)
                )
            ]),
        ]

        transformation = v2.Compose(transform)
        transformation = TorchvisionStrategy(transformation)

        name = '01'

        path = NOISE.joinpath('segment', name)

        pickle = path.joinpath('training.pkl')
        testing = pd.read_pickle(pickle)

        path = testing.file.iloc[1]
        path = CWD.joinpath(path)

        image = Image.open(path).convert('L')
        image = np.asarray(image).astype('float32')
        image = transformation.apply(image)

        image = image.squeeze(0)

        fig, ax = plt.subplots()

        ax.imshow(image, cmap='gray')

        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=True)
        plt.close()


if __name__ == '__main__':
    main()
