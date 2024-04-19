from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from joblib import delayed, Parallel
from PIL import Image
from thesis.constant import ANALYSIS, CWD, NOISE
from thesis.factory import ModelFactory
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis.a.model import ClassificationModel
    from thesis.transformation import TorchvisionStrategy
    from typing_extension import Any


class PerformanceAnalyzer:
    def __init__(
        self,
        classification: tuple[ClassificationModel, TorchvisionStrategy] | None = None,
        n_jobs: int = 10,
        subset: int | None = None
    ):
        self.model, self.transformation = classification
        self.n_jobs = n_jobs
        self.subset = subset

    def _classify(self, segment: npt.NDArray) -> int:
        """Classify a segment of an image.

        Args:
            segment: A segment of an image represented as a numpy array.

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

        coordinates = (x, y)
        background.paste(image, coordinates)

        array = np.asarray(background).astype('float32') / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        tensor = tensor.to(self.model.device)

        self.model.eval()

        with torch.no_grad():
            logits = self.model(tensor)

        return torch.argmax(logits, dim=1).item()

    def _process(self, stroke: str) -> dict[str, Any]:
        """Process an entire dataset for analysis.

        Args:
            stroke: The name of the dataset.

        Returns:
            A dictionary containing analysis metadata.

        """

        pickle = NOISE.joinpath('segment', stroke, 'testing.pkl')
        dataframe = pd.read_pickle(pickle)

        dataframe = (
            dataframe[:self.subset]
            if self.subset is not None
            else dataframe
        )

        files = dataframe.file.to_list()
        labels = dataframe.label.to_list()

        correct = 0
        total = 0

        iterable = zip(files, labels, strict=True)
        length = len(files)

        metadata = {
            'file': [],
            'target': [],
            'prediction': [],
            'accuracy': []
        }

        for file, label in tqdm(iterable, total=length):
            path = CWD.joinpath(file).as_posix()
            original = Image.open(path).convert('L')

            array = np.asarray(original).astype('float32')

            prediction = self._classify(array)

            metadata['file'].append(file)
            metadata['target'].append(label)
            metadata['prediction'].append(prediction)
            accuracy = 1 if prediction == label else 0
            metadata['accuracy'].append(accuracy)

            correct = correct + accuracy
            total = total + 1

        accuracy = metadata['accuracy']
        overall = sum(accuracy) / len(accuracy)

        metadata['overall'] = f"{round(overall * 100, 2):.2f}"
        metadata['stroke'] = int(stroke)
        return metadata

    def all(self) -> None:
        """Perform analysis on all datasets."""

        strokes = [
            str(i).zfill(2)
            for i in range(1, 6)
        ]

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process)
            (stroke)
            for stroke in strokes
        )

        dataframe = pd.DataFrame(result)

        path = ANALYSIS.joinpath('classification.pkl')
        dataframe.to_pickle(path)

        path = ANALYSIS.joinpath('classification.csv')
        dataframe.to_csv(path)


def main() -> None:
    classification = ModelFactory.get_model('classification')

    analyzer = PerformanceAnalyzer(
        classification=classification,
        n_jobs=10
    )

    analyzer.all()


if __name__ == '__main__':
    main()
