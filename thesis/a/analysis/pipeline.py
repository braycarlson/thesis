from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from PIL import Image
from thesis.constant import (
    ANALYSIS,
    CWD,
    HEIGHT,
    NOISE,
    WIDTH
)
from thesis.factory import ModelFactory
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis.a.model import ClassificationModel
    from typing_extensions import Any


class Pipeline:
    def __init__(
        self,
        classification: ClassificationModel = None,
        dataframe: pd.DataFrame = None,
        device: str = 'cuda',
        visualize: bool = False,
    ):
        self.classification = classification
        self.dataframe = dataframe
        self.device = device
        self.visualize = visualize

    def _classify(self, segment: npt.NDArray) -> int:
        """Classify a segment of an image.

        Args:
            segment: A segment of an image.

        Returns:
            The predicted label for the segment.

        """

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

        self.classification.eval()

        with torch.no_grad():
            logits = self.classification(tensor)

        return torch.argmax(logits, dim=1).item()

    def _visualize(self, data: dict[str, Any]) -> None:
        """Visualize an image and its prediction.

        Args:
            data: The input image and its prediction.

        """

        figsize = (10, 4)
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        array = data.get('image')
        prediction = data.get('prediction')

        ax.imshow(array, cmap='gray')
        ax.set_title(f'Prediction: {prediction}')
        ax.axis('off')

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
        """Run the classification pipeline.

        Returns:
            The metadata containing file, target, prediction, and accuracy.

        """

        metadata = {
            'file': [],
            'target': [],
            'prediction': [],
            'accuracy': []
        }

        if self.visualize:
            visualization = {}

        files = self.dataframe.file.to_list()
        labels = self.dataframe.label.to_list()

        correct = 0
        total = 0

        iterable = zip(files, labels, strict=True)
        length = len(files)

        for file, label in tqdm(iterable, total=length):
            path = CWD.joinpath(file).as_posix()
            original = Image.open(path).convert('L')

            array = np.asarray(original).astype('float32')

            prediction = self._classify(array)

            metadata['file'].append(file)
            metadata['target'].append(label)
            metadata['prediction'].append(prediction)
            image_accuracy = 1 if prediction == label else 0
            metadata['accuracy'].append(image_accuracy)

            correct += image_accuracy
            total += 1

            if self.visualize:
                visualization['image'] = array
                visualization['prediction'] = prediction

            if self.visualize:
                self._visualize(visualization)
                visualization.clear()

        overall_accuracy = correct / total if total > 0 else 0
        print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')

        return metadata


def main() -> None:
    classification, transformation = ModelFactory.get_model('classification')

    amount = '01'

    pickle = NOISE.joinpath('segment', amount, 'testing.pkl')
    dataframe = pd.read_pickle(pickle)

    random_state = 42

    dataframe = (
        dataframe
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    pipeline = Pipeline(
        classification=classification,
        dataframe=dataframe,
        device='cuda',
        visualize=False
    )

    metadata = pipeline.run()

    filename = f"{amount}.csv"

    analysis = ANALYSIS.joinpath('classification')
    analysis.mkdir(exist_ok=True, parents=True)

    path = analysis.joinpath(filename)

    dataframe = pd.DataFrame(metadata)
    dataframe.to_csv(path)


if __name__ == '__main__':
    main()
