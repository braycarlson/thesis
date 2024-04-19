from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from PIL import Image
from thesis.constant import (
    CWD,
    COLOR,
    HANDWRITTEN,
    HEIGHT,
    WIDTH
)
from thesis.factory import ModelFactory


def main() -> None:
    model, transformation = ModelFactory.get_model('cassd')

    files = [
        file
        for file in HANDWRITTEN.glob('*.png')
        if file.is_file()
    ]

    for file in files:
        path = CWD.joinpath(file).as_posix()

        with Image.open(path).convert('L') as image:
            image = np.asarray(image).astype('uint8')

        tensor = transformation.apply(image=image).unsqueeze(0)
        tensor = tensor.to(model.device)

        boxes, labels, scores = model.execute(tensor)

        fig, ax = plt.subplots(1)
        ax.imshow(image, cmap='grey')

        iterable = zip(boxes, labels, scores, strict=True)

        for index, (box, label, score) in enumerate(iterable, 0):
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

            ax.text(
                x_min,
                y_min,
                f"{label}: {score:.2f}",
                backgroundcolor='red',
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
        plt.show(block=True)
        plt.close()


if __name__ == '__main__':
    main()
