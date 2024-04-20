from __future__ import annotations

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
    FLEXIBLE,
    HEIGHT,
    WIDTH
)
from thesis.coordinates import CoordinatesConverter, ScalarStrategy
from thesis.factory import ModelFactory
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

    save = False

    model, transformation = ModelFactory.get_model('cassd')

    strategy = ScalarStrategy(128, 128)
    converter = CoordinatesConverter(strategy)

    random_state = 42

    random.seed(random_state)
    torch.manual_seed(random_state)

    dataset = FLEXIBLE.glob('*/*/testing.pkl')

    dataset = [
        path
        for path in dataset
        if '100' not in str(path)
    ]

    sample = random.sample(list(dataset), 5)

    dataframe  = [
        pd.read_pickle(pickle)
        for pickle in sample
    ]

    testing = pd.concat(dataframe, ignore_index=True)

    testing = (
        testing
        .sample(n=128, random_state=random_state)
        .reset_index(drop=True)
    )

    annotation = {
        'testing': testing,
    }

    for name, partition in annotation.items():
        title = name.title()

        chunk = 16
        split = np.array_split(partition, len(partition) // chunk)

        output = ANALYSIS.joinpath('prediction')
        output.mkdir(exist_ok=True, parents=True)

        total = len(split)

        for cid, chunk in tqdm(enumerate(split), total=total):
            images = []

            for _, row in chunk.iterrows():
                path = row.file

                path = CWD.joinpath(path)

                with Image.open(path).convert('L') as image:
                    image = np.asarray(image).astype('uint8')
                    image = (image - image.mean()) / image.std() + 0.5

                fig, ax = plt.subplots()

                ax.imshow(image.squeeze(), cmap='gray')

                tensor = transformation.apply(image=image).unsqueeze(0)
                tensor = tensor.to(model.device)

                model.eval()

                with torch.no_grad():
                    prediction = model(tensor)
                    localization, classification, scores = prediction

                    boxes, labels, scores = model.localize(
                        localization,
                        classification,
                        score=0.50,
                        overlap=0.75
                    )

                    boxes = boxes[0].detach().cpu()
                    scores = scores[0].detach().cpu()

                for index, box in enumerate(boxes, 0):
                    coordinates = converter.scale(box, 'xyxy')

                    x_min, y_min, x_max, y_max = coordinates
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

            filename = f"grid_prediction_{cid}.png"

            path = output.joinpath(filename)

            grid = MatplotlibGrid(images=images, title=title)

            grid.generate()

            if save:
                grid.save(path)
            else:
                grid.show()

            grid.cleanup()


if __name__ == '__main__':
    main()
