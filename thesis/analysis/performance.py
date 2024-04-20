from __future__ import annotations

import pandas as pd

from joblib import delayed, Parallel
from thesis.constant import FLEXIBLE
from thesis.coordinates import (
    CoordinatesConverter,
    ScalarStrategy
)
from thesis.factory import ModelFactory
from thesis.pipeline import Pipeline
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thesis.a.model import ClassificationModel
    from thesis.transformation import TorchvisionStrategy
    from typing_extension import Any


class PerformanceAnalyzer:
    def __init__(
        self,
        converter: CoordinatesConverter | None = None,
        classification: tuple[ClassificationModel, TorchvisionStrategy] | None = None,
        localization: tuple[Any, TorchvisionStrategy] = (None, None),
        n_jobs: int = 10,
        subset: int | None = None
    ):
        """Initialize the performance analyzer.

        Args:
            converter: The coordinates converter to convert boxes to xyxy.
            classification: The classification model and its transformation(s).
            localization: The CASSD model and its transformation(s).
            n_jobs: The number of job(s) to use for parallelization.
            subset: The number for segmenting the dataset.

        """

        self.converter = converter
        self.classification = classification
        self.localization = localization
        self.n_jobs = n_jobs
        self.subset = subset

    def _process(self, amount: str, overlap: str) -> dict[str, Any]:
        """Process a single dataset through the pipeline.

        Args:
            amount: The number of digits.
            overlap: The overlap rate.

        """

        pickle = FLEXIBLE.joinpath(amount, overlap, 'testing.pkl')
        dataframe = pd.read_pickle(pickle)

        dataframe = (
            dataframe[:self.subset]
            if self.subset is not None
            else dataframe
        )

        pipeline = Pipeline(
            classification=self.classification,
            converter=self.converter,
            dataframe=dataframe,
            localization=self.localization,
            visualize=False
        )

        amount = int(amount)

        result = pipeline.run()

        result['amount'] = amount
        result['overlap'] = overlap

        return result

    def all(self) -> None:
        """Run the entire dataset through the pipeline."""

        arguments = [
            (str(i).zfill(2), str(j * 10).zfill(2))
            for i in range(1, 9) for j in range(11)
        ]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process)
            (amount, overlap)
            for amount, overlap in arguments
        )

        dataframe = pd.DataFrame(results)
        dataframe.to_pickle('metadata.pkl')
        dataframe.to_csv('metadata.csv')

    def single(self) -> None:
        """Run a single dataset through the pipeline."""

        arguments = [
            (str(i).zfill(2), str(j * 10).zfill(2))
            for i in range(1, 9) for j in range(11)
        ]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process)
            (amount, overlap)
            for amount, overlap in arguments
        )

        dataframe = pd.DataFrame(results)
        dataframe.to_pickle('metadata.pkl')
        dataframe.to_csv('metadata.csv')

    def manual(self) -> None:
        """Run a specific dataset through the pipeline."""

        arguments = [('01', '30')]

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._process)
            (amount, overlap)
            for amount, overlap in arguments
        )

        dataframe = pd.DataFrame(results)
        dataframe.to_pickle('metadata.pkl')
        dataframe.to_csv('metadata.csv')


def main() -> None:
    classification = ModelFactory.get_model('classification')
    localization = ModelFactory.get_model('cassd')

    strategy = ScalarStrategy(128, 128)
    converter = CoordinatesConverter(strategy)

    analyzer = PerformanceAnalyzer(
        classification=classification,
        converter=converter,
        localization=localization,
        n_jobs=10
    )

    analyzer.all()


if __name__ == '__main__':
    main()
