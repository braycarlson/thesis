from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image
from thesis.constant import (
    CWD,
    HEIGHT,
    MNIST,
    NOISE,
    WIDTH
)
from thesis.factory import ModelFactory


def main() -> None:
    plt.style.use('science')

    save = False

    model, transformation = ModelFactory.get_model('classification')

    random_state = 42

    pickle = NOISE.joinpath('segment/01/testing.pkl')
    dataframe = pd.read_pickle(pickle)

    dataframe = (
        dataframe
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    files = dataframe.file.to_list()
    labels = dataframe.label.to_list()

    iterable = zip(files, labels, strict=True)

    for index, (file, _) in enumerate(list(iterable), 0):
        digit = dataframe.loc[index, 'original']
        path = MNIST.joinpath(digit)

        with Image.open(path).convert('L') as image:
            original = image.resize(
                (1024, 1024),
                resample=Image.Resampling.NEAREST
            )

            original.save('classification_digit.png')

        path = CWD.joinpath(file)

        with Image.open(path).convert('L') as image:
            original = image.resize(
                (1024, 1024),
                resample=Image.Resampling.NEAREST
            )

            original.save('classification_original.png')

            image = np.asarray(image).astype('float32') / 255.0

            tensor = (
                torch
                .from_numpy(image)
                .unsqueeze(0)
                .unsqueeze(0)
                .to(model.device)
            )

        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).item()

    figsize = (1024 / 96, 1024 / 96)
    fig, ax = plt.subplots(figsize=figsize, dpi=96)
    ax.imshow(image, cmap='grey')

    ax.text(
        0.50,
        0.95,
        f"Prediction: {prediction}",
        color='black',
        fontsize=24,
        ha='center',
        va='center',
        transform=fig.transFigure
    )

    figure_width, figure_height = fig.get_size_inches() * fig.dpi

    x = (WIDTH - figure_width) // 2
    y = (HEIGHT - figure_height) // 2
    y = y - 50

    plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

    plt.tight_layout()
    plt.axis('off')

    if save:
        plt.savefig(
            'classification_prediction.png',
            bbox_inches='tight',
            dpi=96,
            edgecolor='black',
            facecolor='black',
            pad_inches=0,
            transparent=False
        )
    else:
        plt.show(block=True)

    plt.close()


if __name__ == '__main__':
    main()
