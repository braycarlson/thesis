from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import patches
from PIL import Image
from thesis.constant import COLOR, HEIGHT, PIPELINE, WIDTH
from thesis.factory import ModelFactory


def main() -> None:
    plt.style.use('science')

    save = False

    model, transformation = ModelFactory.get_model('cassd')

    path = PIPELINE.joinpath('02/60/testing/3/2231.png')

    with Image.open(path).convert('L') as image:
        original = image.resize(
            (1024, 1024),
            resample=Image.Resampling.BICUBIC
        )

        original.save('original.png')

        image = np.asarray(image).astype('uint8')

    tensor = transformation.apply(image=image).unsqueeze(0)
    tensor = tensor.to(model.device)

    boxes, labels, scores = model.execute(tensor)

    figsize = (1024 / 96, 1024 / 96)
    fig, ax = plt.subplots(figsize=figsize, dpi=96)
    ax.imshow(image, cmap='grey')

    iterable = zip(boxes, labels, scores, strict=True)

    for index, (box, label, score) in enumerate(iterable, 0):
        x_min, y_min, x_max, y_max = box
        width, height = x_max - x_min, y_max - y_min

        i = (index + 1) % len(COLOR)
        edgecolor = backgroundcolor = facecolor = COLOR[i]

        rectangle = patches.Rectangle(
            (x_min, y_min),
            width,
            height,
            edgecolor=edgecolor,
            facecolor='none',
            linewidth=2
        )

        ax.add_patch(rectangle)

        text_x, text_y = (
            x_min + 0.025 * width,
            y_min + 0.025 * height
        )

        ax.text(
            text_x, text_y,
            f"{label}: {score:.2f}",
            backgroundcolor=backgroundcolor,
            color='white',
            fontsize=12,
            horizontalalignment='left',
            verticalalignment='top',
            bbox={'facecolor': facecolor, 'edgecolor': 'none'},
            clip_on=True
        )

    plt.tight_layout()
    plt.axis('off')

    if save:
        plt.savefig(
            'prediction.png',
            dpi=96,
            edgecolor='black',
            facecolor='black',
            transparent=False
        )
    else:
        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.show(block=True)

    plt.close()


if __name__ == '__main__':
    main()
