from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import torch

from PIL import Image
from thesis.constant import (
    CWD,
    FIGURES,
    HEIGHT,
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

    for i, (file, label) in enumerate(iterable, 0):
        path = CWD.joinpath(file)

        image = Image.open(path).convert('L')
        array = np.asarray(image).astype('float32') / 255
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(model.device)

        model.eval()

        with torch.no_grad():
            logits = model(tensor)
            prediction = torch.argmax(logits, dim=1).item()

        figsize = (6, 6)

        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(image, cmap='grey')

        plt.title(f"True: {label}, Predicted: {prediction}")

        plt.axis('off')
        plt.tight_layout()

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
            figure_width, figure_height = fig.get_size_inches() * fig.dpi

            x = (WIDTH - figure_width) // 2
            y = (HEIGHT - figure_height) // 2
            y = y - 50

            plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

            plt.show()

        plt.close()


if __name__ == '__main__':
    main()
