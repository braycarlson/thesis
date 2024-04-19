from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import patches
from PIL import Image
from thesis.constant import (
    CWD,
    COLOR,
    FIGURES,
    HEIGHT,
    VARIABLE,
    WIDTH
)
from thesis.factory import ModelFactory


def main() -> None:
    save = False

    model, transformation = ModelFactory.get_model('cassd')

    pickle = VARIABLE.joinpath('testing.pkl')
    dataframe = pd.read_pickle(pickle)

    random_state = 1

    dataframe = (
        dataframe
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    files = dataframe.file.to_list()

    for i, file in enumerate(files, 0):
        path = CWD.joinpath(file).as_posix()

        with Image.open(path).convert('L') as image:
            image = np.asarray(image).astype('uint8')

        tensor = transformation.apply(image=image).unsqueeze(0)
        tensor = tensor.to(model.device)

        boxes, labels, scores = model.execute(tensor)

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='grey')

        iterable = zip(boxes, labels, scores, strict=True)

        for j, (box, label, score) in enumerate(iterable, 0):
            x_min, y_min, x_max, y_max = box

            linewidth = 2
            offset = linewidth / 2

            x_min = max(0 + offset, x_min)
            y_min = max(0 + offset, y_min)
            x_max = min(128 - offset, x_max)
            y_max = min(128 - offset, y_max)

            width, height = x_max - x_min, y_max - y_min

            k = (j + 1) % len(COLOR)
            backgroundcolor = edgecolor = COLOR[k]

            rectangle = patches.Rectangle(
                (x_min - offset, y_min - offset),
                width,
                height,
                edgecolor=edgecolor,
                facecolor='none',
                linewidth=linewidth
            )

            ax.add_patch(rectangle)

            ax.text(
                x_min,
                y_min,
                f"{label}: {score:.2f}",
                backgroundcolor=backgroundcolor,
                color='white',
                fontsize=8
            )

        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.axis('off')

        if save:
            name = str(i).zfill(3)
            filename = name + '.png'

            path = FIGURES.joinpath(filename)

            plt.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png',
                transparent=True
            )
        else:
            plt.show(block=True)

        plt.close()


if __name__ == '__main__':
    main()
