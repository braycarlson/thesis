from __future__ import annotations

from thesis.a.model import ClassificationModel
from thesis.b.model import ClassAgnosticSingleShotDetector
from thesis.constant import (
    CASSD,
    CLASSIFICATION,
    RATIO,
)
from thesis.transform import StandardizationTransform
from thesis.transformation import TorchvisionStrategy
from torchvision.transforms import v2
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extension import Any


class ModelFactory:
    """Get a model and its transformation based on its name"""

    @staticmethod
    def get_model(name: str) -> Any:
        """Get the preloaded model and its transformation.

        Args:
            name: The name of the model.

        Returns:
            An instance of the model requested.

        """

        width = height = 128
        size = (width, height)

        match name:
            case'cassd':
                logs = [
                    file
                    for file in CASSD.glob('*/*/*/*.ckpt')
                    if file.is_file()
                ]

                checkpoint = logs[-1]

                model = ClassAgnosticSingleShotDetector.load_from_checkpoint(
                    checkpoint,
                    ratio=RATIO
                )

                transform = [
                    v2.Resize(size, antialias=True),
                    StandardizationTransform()
                ]

                transformation = v2.Compose(transform)
                transformation = TorchvisionStrategy(transformation)

                return (model, transformation)

            case 'classification':
                width = height = 28
                size = (width, height)

                logs = [
                    file
                    for file in CLASSIFICATION.glob('*/*/*/*.ckpt')
                    if file.is_file()
                ]

                checkpoint = logs[-1]
                model = ClassificationModel.load_from_checkpoint(checkpoint)

                transform = [v2.Resize(size, antialias=True)]
                transformation = v2.Compose(transform)
                transformation = TorchvisionStrategy(transformation)

                return (model, transformation)

            case _:
                message = 'Unknown method'
                raise ValueError(message)
