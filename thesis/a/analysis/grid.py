from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import warnings

from io import BytesIO
from thesis.grid import MatplotlibGrid
from PIL import Image
from thesis.constant import (
    ANALYSIS,
    CWD,
    HEIGHT,
    NOISE,
    WIDTH
)
from thesis.transform import (
    GradientTransform,
    ShiftTransform
)
from thesis.transformation import TorchvisionStrategy
from torchvision.transforms import v2
from tqdm import tqdm


warnings.filterwarnings(
    'ignore',
    category=UserWarning
)

warnings.filterwarnings(
    'ignore',
    category=FutureWarning
)


def main() -> None:
    size = (28, 28)

    random_state = 42

    random.seed(random_state)
    torch.manual_seed(random_state)

    transform = [
        v2.Resize(size, antialias=True),
        v2.RandomChoice([
            GradientTransform(),
            ShiftTransform(9),
            v2.GaussianBlur(kernel_size=3),
            v2.RandomAutocontrast(p=0.5),
            v2.RandomErasing(
                p=0.5,
                ratio=(0.025, 0.050),
                scale=(0.025, 0.050),
                value=0
            ),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            v2.RandomResizedCrop(
                size,
                antialias=True,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            )
        ])
    ]

    transformation = v2.Compose(transform)
    strategy = TorchvisionStrategy(transformation)

    strokes = [
        str(i).zfill(2)
        for i in range(1, 6)
    ]

    total = len(strokes)

    for stroke in tqdm(strokes, total=total):
        pickle = NOISE.joinpath('segment', stroke, 'testing.pkl')
        testing  = pd.read_pickle(pickle)

        testing = (
            testing
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        pickle = NOISE.joinpath('segment', stroke, 'training.pkl')
        training  = pd.read_pickle(pickle)

        training = (
            training
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        pickle = NOISE.joinpath('segment', stroke, 'validation.pkl')
        validation  = pd.read_pickle(pickle)

        validation = (
            validation
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        annotation = {
            'testing': testing,
            'training': training,
            'validation': validation
        }

        for name, partition in annotation.items():
            title = name.title()
            title = f"{title}: [{stroke}]"

            dataset = ANALYSIS.joinpath('dataset/classification')
            dataset.mkdir(exist_ok=True, parents=True)

            files = partition.file.tolist()
            labels = partition.label.tolist()

            images = []

            for file, _ in zip(files, labels, strict=True):
                path = CWD.joinpath(file)

                with Image.open(path).convert('L') as image:
                    image = np.asarray(image).astype('float32') / 255.0

                if name in ('training', 'validation') and 'variable' not in str(path):
                    image = strategy.apply(image=image)

                fig, ax = plt.subplots()

                ax.imshow(image.squeeze(), cmap='gray')

                figure_width, figure_height = fig.get_size_inches() * fig.dpi

                x = (WIDTH - figure_width) // 2
                y = (HEIGHT - figure_height) // 2
                y = y - 50
                plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

                plt.tight_layout()
                plt.axis('off')

                position = ax.get_position()

                border = plt.Rectangle(
                    (position.x0, position.y0),
                    position.width,
                    position.height,
                    transform=fig.transFigure,
                    color='#818181',
                    linewidth=4,
                    fill=False
                )

                fig.add_artist(border)

                buffer = BytesIO()

                plt.savefig(
                    buffer,
                    bbox_inches='tight',
                    edgecolor='black',
                    facecolor='black',
                    format='png'
                )

                buffer.seek(0)

                image = Image.open(buffer)
                images.append(image)

                plt.close()

            filename = f"grid_{name}_{stroke}.png"
            path = dataset.joinpath(filename)

            grid = MatplotlibGrid(images=images, title=title)

            grid.generate()
            grid.save(path)
            grid.cleanup()


if __name__ == '__main__':
    main()
