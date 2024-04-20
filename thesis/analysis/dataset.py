from __future__ import annotations

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import scienceplots
import torch
import warnings

from io import BytesIO
from thesis.grid import MatplotlibGrid
from matplotlib import patches
from PIL import Image
from thesis.constant import (
    ANALYSIS,
    COLOR,
    CWD,
    HEIGHT,
    FLEXIBLE,
    WIDTH
)
from thesis.coordinates import CoordinatesConverter, ScalarStrategy
from thesis.transform import RandomNoise, RandomStroke
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
    plt.style.use('science')

    draw = True
    save = False

    size = (128, 128)

    random_state = 42

    random.seed(random_state)
    torch.manual_seed(random_state)

    transform = [
        v2.RandomApply([
            RandomStroke(count=10, width=1)
        ], p=1.0),
        v2.RandomApply([
            RandomNoise()
        ], p=0.25),
        v2.RandomApply([
            v2.RandomHorizontalFlip(p=0.50),
            v2.RandomVerticalFlip(p=0.50),
        ], p=0.25),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=1)
        ], p=0.50),
        v2.Resize(size, antialias=True),
    ]

    strategy = ScalarStrategy(128, 128)
    converter = CoordinatesConverter(strategy)

    transformation = v2.Compose(transform)
    strategy = TorchvisionStrategy(transformation)

    combination = [
        (str(i).zfill(2), str(j * 10).zfill(2))
        for i in range(1, 9) for j in range(10)
    ]

    total = len(combination)

    for amount, overlap in tqdm(combination, total=total):
        columns = [
            'file',
            'label',
            'xyxy'
        ]

        hdf = FLEXIBLE.joinpath(amount, overlap, 'testing.hdf')
        testing  = pd.read_hdf(hdf)

        testing = (
            testing
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        hdf = FLEXIBLE.joinpath(amount, overlap, 'training.hdf')
        training  = pd.read_hdf(hdf)

        training = (
            training
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        hdf = FLEXIBLE.joinpath(amount, overlap, 'validation.hdf')
        validation  = pd.read_hdf(hdf)

        validation = (
            validation
            .sample(n=16, random_state=random_state)
            .reset_index(drop=True)
        )

        columns.pop(0)

        for column in columns:
            testing[column] = testing[column].apply(json.loads)
            training[column] = training[column].apply(json.loads)
            validation[column] = validation[column].apply(json.loads)

        annotation = {
            'testing': testing,
            'training': training,
            'validation': validation
        }

        for name, partition in annotation.items():
            title = name.title()
            title = f"{title}: [{amount}, {overlap}]"

            dataset = ANALYSIS.joinpath('dataset/cassd')
            dataset.mkdir(exist_ok=True, parents=True)

            files = partition.file.tolist()
            boxes = partition.xyxy.tolist()
            labels = partition.label.tolist()

            images = []

            for file, box, label in zip(files, boxes, labels, strict=True):
                path = CWD.joinpath(file)

                with Image.open(path).convert('L') as image:
                    image = np.asarray(image).astype('uint8')

                indices = [
                    index
                    for index, lbl in enumerate(label)
                    if int(lbl) < 10
                ]

                boxes = [box[index] for index in indices]

                if name in ('training', 'validation') and 'variable' not in str(path):
                    image, boxes = strategy.apply(
                        image=image,
                        boxes=boxes
                    )

                    image = (image - image.mean()) / image.std() + 0.5
                    boxes = boxes / 128
                else:
                    image = (image - image.mean()) / image.std() + 0.5

                fig, ax = plt.subplots()

                ax.imshow(image.squeeze(), cmap='gray')

                if draw:
                    for index, box in enumerate(boxes, 0):
                        if name in ('training', 'validation') and 'variable' not in str(path):
                            box = converter.scale(box, 'xyxy')

                        x_min, y_min, x_max, y_max = box
                        width, height = x_max - x_min, y_max - y_min

                        i = (index + 1) % len(COLOR)
                        edgecolor = COLOR[i]

                        rectangle = patches.Rectangle(
                            (x_min, y_min),
                            width,
                            height,
                            edgecolor=edgecolor,
                            facecolor='none',
                            linewidth=2
                        )

                        ax.add_patch(rectangle)

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

            if draw:
                filename = f"grid_{name}_{amount}_{overlap}_boxes.png"
            else:
                filename = f"grid_{name}_{amount}_{overlap}.png"

            path = dataset.joinpath(filename)

            grid = MatplotlibGrid(images=images, title=title)
            grid.generate()

            if save:
                grid.save(path)
            else:
                grid.show()

            grid.cleanup()


if __name__ == '__main__':
    main()
