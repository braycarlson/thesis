from __future__ import annotations

import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch
import warnings

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
            dataset = ANALYSIS.joinpath('dataset/cassd')
            dataset.mkdir(exist_ok=True, parents=True)

            files = partition.file.tolist()
            boxes = partition.xyxy.tolist()
            labels = partition.label.tolist()

            iterable = zip(files, boxes, labels, strict=True)

            for i, (file, box, label) in enumerate(iterable, 0):
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
                    for j, box in enumerate(boxes, 0):
                        if name in ('training', 'validation') and 'variable' not in str(path):
                            box = converter.scale(box, 'xyxy')

                        x_min, y_min, x_max, y_max = box
                        width, height = x_max - x_min, y_max - y_min

                        k = (j + 1) % len(COLOR)
                        edgecolor = COLOR[k]

                        rectangle = patches.Rectangle(
                            (x_min, y_min),
                            width,
                            height,
                            edgecolor=edgecolor,
                            facecolor='none',
                            linewidth=2
                        )

                        ax.add_patch(rectangle)

                plt.tight_layout()
                plt.axis('off')

                if save:
                    filename = f"{name}__{amount}_{overlap}_{i}.png"
                    location = dataset.joinpath(filename)

                    plt.savefig(
                        location,
                        bbox_inches='tight',
                        dpi=300,
                        format='png',
                        transparent=True
                    )
                else:
                    figure_width, figure_height = fig.get_size_inches() * fig.dpi

                    x = (WIDTH - figure_width) // 2
                    y = (HEIGHT - figure_height) // 2
                    y = y - 50

                    plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

                    plt.show()

                plt.close()


if __name__ == '__main__':
    main()
