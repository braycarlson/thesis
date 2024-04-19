from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from matplotlib import patches
from PIL import Image
from thesis.factory import ModelFactory
from thesis.constant import (
    ANALYSIS,
    COLOR,
    CWD,
    HEIGHT,
    PIPELINE,
    WIDTH
)
from thesis.coordinates import CoordinatesConverter, ScalarStrategy
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis.a.model import ClassificationModel
    from thesis.transformation import TorchvisionStrategy
    from typing_extension import Any


class Pipeline:
    def __init__(
        self,
        classification: tuple[ClassificationModel, TorchvisionStrategy] | None = None,
        converter: CoordinatesConverter | None = None,
        dataframe: pd.DataFrame = None,
        device: str = 'cuda',
        localization: tuple[Any, TorchvisionStrategy] = (None, None),
        visualize: bool = False,
    ):
        self.classification = classification
        self.converter = converter
        self.dataframe = dataframe
        self.device = device
        self.localization = localization
        self.visualize = visualize

    def _classify(self, segment: npt.NDArray) -> tuple[npt.NDArray, float]:
        """Classify a digit within a bounding box.

        Args:
            segment: The image to be classified.

        Returns:
            A tuple containing the predicted label and confidence score.

        """

        model, transformation = self.classification

        size = (28, 28)

        image = Image.fromarray(segment)
        image.thumbnail(size, Image.Resampling.LANCZOS)

        background = Image.new('L', size, color=0)

        width, height = image.size
        x = (28 - width) // 2
        y = (28 - height) // 2

        background.paste(image, (x, y))

        array = np.asarray(background).astype('float32') / 255
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.device)

        logits = model.execute(tensor)

        image.close()
        background.close()

        return torch.argmax(logits, dim=1).item()

    def _localize(self, image: npt.NDArray) -> list[tuple[int, int, int, int]]:
        """Localize digit(s) within an image.

        Args:
            image: The image to localize digit(s) in.

        Returns:
            A list of bounding boxes representing the localized digit(s).

        """

        model, transformation = self.localization

        tensor = transformation.apply(image=image).unsqueeze(0)
        tensor = tensor.to(self.device)

        return model.execute(tensor)

    def _visualize(self, data: dict[str, Any]) -> None:
        """Visualize the localization and classification step.

        Args:
            data: A dictionary containing data to visualize.

        """

        segmentation = data.get('segment')

        amount = len(segmentation)
        cols = 2
        rows = 1 + amount // 2 + amount % 2

        figsize = (10, 2 * rows)
        fig = plt.figure(figsize=figsize)

        label = data.get('label')
        fig.suptitle(f"Truth: {label}", fontsize=16)

        original = data.get('original')
        array = data.get('array')

        ax = plt.subplot2grid(
            (rows, cols),
            (0, 0),
            colspan=1
        )

        ax.imshow(original, cmap='gray')
        ax.set_title('Original')
        ax.axis('off')

        localize = plt.subplot2grid(
            (rows, cols),
            (0, 1),
            colspan=1
        )

        localize.imshow(array, cmap='gray')
        localize.set_title('Localization')
        localize.axis('off')

        boxes = data.get('boxes')

        for index, box in enumerate(boxes, 0):
            x_min, y_min, x_max, y_max = box

            linewidth = 1
            offset = linewidth / 2

            x_min = max(0 + offset, x_min)
            y_min = max(0 + offset, y_min)
            x_max = min(128 - offset, x_max)
            y_max = min(128 - offset, y_max)

            width, height = x_max - x_min, y_max - y_min

            i = (index + 1) % len(COLOR)
            edgecolor = COLOR[i]

            rectangle = patches.Rectangle(
                (x_min - offset, y_min - offset),
                width,
                height,
                linewidth=linewidth,
                edgecolor=edgecolor,
                facecolor='none'
            )

            localize.add_patch(rectangle)

        for i, segment in enumerate(segmentation):
            prediction = segment.get('prediction')
            image = segment.get('segment')

            col_index = i % 2
            row_index = 1 + i // 2

            segment = plt.subplot2grid(
                (rows, cols),
                (row_index, col_index)
            )

            segment.imshow(image, cmap='gray')
            segment.set_title(f'Prediction: {prediction}')
            segment.axis('off')

        figure_width, figure_height = fig.get_size_inches() * fig.dpi

        x = (WIDTH - figure_width) // 2
        y = (HEIGHT - figure_height) // 2
        y = y - 50

        plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

        plt.tight_layout()
        plt.axis('off')
        plt.tight_layout()
        plt.show(block=True)
        plt.close()

    def run(self) -> None:
        """Run the pipeline."""

        metadata = {
            'file': [],
            'target': [],
            'prediction': [],
            'accuracy': []
        }

        files = self.dataframe.file.to_list()
        labels = self.dataframe.label.to_list()

        correct = 0
        total = 0

        if self.visualize:
            visualization = {
                'original': None,
                'boxes': [],
                'segment': []
            }

        iterable = zip(files, labels, strict=True)
        length = len(files)

        for file, label in tqdm(iterable, total=length):
            potential = len(label)

            path = CWD.joinpath(file).as_posix()
            original = Image.open(path)

            array = np.asarray(original).astype('uint8')

            output = self._localize(array)

            boxes, *_ = output

            if self.visualize:
                visualization['original'] = original
                visualization['array'] = array
                visualization['boxes'] = boxes
                visualization['label'] = sorted(label)

            predicted = []
            matched = 0
            clone = label[:]

            for box in boxes:
                minc, minr, maxc, maxr = box
                segment = array[minr:maxr, minc:maxc]

                prediction = self._classify(segment)
                predicted.append(prediction)

                if prediction in clone:
                    matched = matched + 1
                    clone.remove(prediction)

                if self.visualize:
                    visualization['segment'].append({
                        'segment': segment,
                        'prediction': prediction
                    })

            metadata['file'].append(file)
            metadata['target'].append(sorted(label))
            metadata['prediction'].append(sorted(predicted))
            accuracy = matched / len(label) if label else 0
            metadata['accuracy'].append(accuracy)

            length = len(label)
            correct = correct + (potential - length)
            total = total + potential

            if self.visualize:
                self._visualize(visualization)
                visualization['segment'].clear()

        accuracy = metadata['accuracy']
        overall = sum(accuracy) / len(accuracy)
        metadata['overall'] = f"{round(overall * 100, 2):.2f}"

        return metadata


def main() -> None:
    localization = ModelFactory.get_model('cassd')
    classification = ModelFactory.get_model('classification')

    strategy = ScalarStrategy(128, 128)
    converter = CoordinatesConverter(strategy)

    for i in range(1, 9):
        for j in range(1, 11):
            amount = str(i).zfill(2)
            overlap = str(j * 10).zfill(2)

            pickle = PIPELINE.joinpath(amount, overlap, 'testing.pkl')
            dataframe = pd.read_pickle(pickle)

            pipeline = Pipeline(
                classification=classification,
                converter=converter,
                dataframe=dataframe,
                localization=localization,
                visualize=True
            )

            metadata = pipeline.run()

            filename = f"{amount}_{overlap}.csv"

            amount, overlap = int(amount), int(overlap)

            file = metadata.get('file')
            length = len(file)

            model, _ = pipeline.localization
            model = str(model)

            amount = length * [amount]
            overlap = length * [overlap]
            model = length * [model]

            metadata['amount'] = amount
            metadata['overlap'] = overlap
            metadata['model'] = model

            analysis = ANALYSIS.joinpath('csv', 'pipeline')
            analysis.mkdir(exist_ok=True, parents=True)

            path = analysis.joinpath(filename)

            dataframe = pd.DataFrame(metadata)
            dataframe.to_csv(path)


if __name__ == '__main__':
    main()
