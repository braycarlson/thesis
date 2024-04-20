from __future__ import annotations

import math
# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import patches
from PIL import Image
from thesis.constant import (
    ANALYSIS,
    CWD,
    HEIGHT,
    PIPELINE,
    WIDTH
)
from thesis.coordinates import (
    CoordinatesConverter,
    ScalarStrategy
)
from thesis.factory import ModelFactory
from torch import nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class Visualizer:
    def __init__(self, model: nn.Module):
        """Initializes the visualizer with a model.

        Args:
            model: A model from which the visualizer will extract data.

        """

        strategy = ScalarStrategy(128, 128)
        self.converter = CoordinatesConverter(strategy)
        self.model = model
        self.figsize = (18, 6)

    def _extract(self, attribute: str, tensor: torch.Tensor) -> list[torch.Tensor]:
        """Extract features or filters from the model.

        Args:
            attribute: The attribute which to extract features or filters.
            tensor: The input tensor to the model.

        Returns:
            The extracted features or filters.

        """

        output = []

        if attribute == 'prediction':
            for module in getattr(self.model, attribute).classification:
                tensor = module(tensor)
                output.append(tensor)

            for module in getattr(self.model, attribute).localization:
                tensor = module(tensor)
                output.append(tensor)
        else:
            for module in getattr(self.model, attribute).module:
                tensor = module(tensor)
                output.append(tensor)

        return output

    def defaults(self) -> None:
        """Visualize the default boxes from the model."""

        defaults = self.model.default.detach().cpu()

        figsize = (8, 8)
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.invert_yaxis()

        for default in defaults:
            x_min, y_min, x_max, y_max = default
            width, height = x_max - x_min, y_max - y_min

            rectangle = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                edgecolor='r',
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
        plt.show(block=True)
        plt.close()

    def features(self, attribute: str, tensor: torch.Tensor) -> None:
        """Visualize the feature maps from a specific layer.

        Args:
            attribute: The attribute which to extract features.
            tensor: The input tensor to the model.

        Returns:
            The extracted features.

        """

        # mpl.rcParams['text.color'] = '#ffffff'

        outputs = self._extract(attribute, tensor)

        destination = ANALYSIS.joinpath('features')
        destination.mkdir(exist_ok=True, parents=True)

        layer = attribute.title()

        for index, output in enumerate(outputs):
            filename = f"{layer}_{index}".lower() + '.png'
            path = destination.joinpath(filename)

            amount = output.shape[1]
            column = 8
            row = math.ceil(amount / column)

            fig, axes = plt.subplots(
                row,
                column,
                figsize=(column * 2, row * 2)
            )

            for j in range(amount):
                feature_map = output[0, j, :, :].detach().cpu()
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min())

                ax = axes[j // column, j % column]
                ax.imshow(feature_map, cmap='gray')
                ax.axis('off')
                ax.set_title(f'Layer {index} - Feature Map {j}', fontsize=8)

            fig.suptitle(f'Feature Maps for {layer} - Layer {index}', fontsize=14)

            plt.subplots_adjust(hspace=1.0)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.axis('off')

            plt.savefig(
                path,
                dpi=300,
                format='png',
                # edgecolor='black',
                # facecolor='black',
                transparent=False
            )

            plt.close()

        return outputs

    def filters(self) -> None:
        """Visualize the filter(s) of convolutional layer(s) in the model."""

        # mpl.rcParams['text.color'] = '#ffffff'

        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                filters = layer.weight.data.clone()

                # Skip 1x1 filters
                if filters.size(2) == 1 and filters.size(3) == 1:
                    continue

                amount = filters.shape[0]
                column = 8
                row = math.ceil(amount / column)

                destination = ANALYSIS.joinpath('filters')
                destination.mkdir(exist_ok=True, parents=True)

                filename = f"{name}_filters".replace('.', '_') + '.png'
                path = destination.joinpath(filename)

                fig, axes = plt.subplots(
                    row,
                    column,
                    figsize=(column * 2, row * 2),
                    squeeze=False
                )

                axes = axes.flatten()

                for ax in axes:
                    ax.axis('off')

                for i in range(amount):
                    filter_map = filters[i, 0, :, :].detach().cpu()
                    filter_map = (filter_map - filter_map.min()) / (filter_map.max() - filter_map.min())
                    axes[i].imshow(filter_map, cmap='gray')
                    axes[i].axis('on')
                    axes[i].set_title(f'Filter {i}', fontsize=8)

                fig.suptitle(f'Filters for {name}', fontsize=14)

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.axis('off')

                plt.savefig(
                    path,
                    dpi=300,
                    format='png',
                    # edgecolor='black',
                    # facecolor='black',
                    transparent=False
                )

                plt.close()

    def single(self, attribute: str, tensor: torch.Tensor) -> None:
        """Visualize individual feature maps of a specific layer.

        Args:
            attribute: The attribute from which to extract features.
            tensor: The input tensor to the model.

        Returns:
            The extracted features.

        """

        features = self._extract(attribute, tensor)

        destination = ANALYSIS.joinpath('features')
        destination.mkdir(exist_ok=True, parents=True)

        layer = attribute.title()
        m = len(features)
        n = len(str(m))

        for i, output in enumerate(features):
            amount = output.shape[1]

            o = len(str(amount))

            for j in range(amount):
                feature = output[0, j, :, :].detach().cpu()

                feature = (feature - feature.min()) / (feature.max() - feature.min())

                figsize = (2, 2)
                fig, ax = plt.subplots(figsize=figsize)
                ax.imshow(feature, cmap='gray')
                ax.axis('off')

                x = str(i).zfill(n)
                y = str(j).zfill(o)

                filename = f"{layer}_{x}_featuremap_{y}.png".lower()
                path = destination.joinpath(filename)

                plt.savefig(
                    path,
                    bbox_inches='tight',
                    dpi=300,
                    format='png',
                    transparent=True
                )

                plt.close()

        return features


def main() -> None:
    model, transformation = ModelFactory.get_model('cassd')

    pickle = PIPELINE.joinpath('03', '40', 'testing.pkl')
    dataframe = pd.read_pickle(pickle)

    random_state = 42

    dataframe = (
        dataframe
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    files = dataframe.file.to_list()

    visualizer = Visualizer(model=model)

    for file in files:
        path = CWD.joinpath(file).as_posix()

        image = Image.open(path).convert('L')
        image = np.asarray(image).astype('uint8')
        image = (image - image.mean()) / image.std() + 0.5

        destination = ANALYSIS.joinpath('original')
        destination.mkdir(exist_ok=True, parents=True)

        path = destination.joinpath(CWD.joinpath(file).stem + '.png')

        figsize = (6, 6)
        fig, ax = plt.subplots(figsize=figsize)

        ax.imshow(image, cmap='gray')
        ax.axis('off')

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png',
            transparent=True
        )

        plt.close()

        tensor = transformation.apply(image=image).unsqueeze(0)
        tensor = tensor.to(model.device)

        visualizer.defaults()

        layer = visualizer.single('base', tensor)
        layer = layer[-1]
        visualizer.single('auxiliary', layer)

        visualizer.filters()


if __name__ == '__main__':
    main()
