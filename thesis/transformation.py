from __future__ import annotations

import torch

from abc import ABC, abstractmethod
from torchvision import tv_tensors
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from albumentations.core.composition import Compose as ACompose
    from torchvision.transforms.v2._container import Compose as TVCompose


class TransformationStrategy(ABC):
    @abstractmethod
    def apply(
        self,
        image: npt.NDArray | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
        mask: npt.NDArray | None = None
    ) -> npt.NDArray | tuple[npt.NDArray, list[tuple[int, int, int, int]]]:
        """Apply transformation(s).

        Args:
            image: An input image array.
            boxes: A list of bounding boxes.
            mask: An input mask array.

        Returns:
            The transformed image array or a tuple containing transformed
            image array and transformed bounding boxes.

        """

        raise NotImplementedError


class AlbumentationStrategy(TransformationStrategy):
    def __init__(self, transformation: ACompose):
        self.transformation = transformation

    def apply(
        self,
        image: npt.NDArray | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
        _mask: npt.NDArray | None = None
    ) -> npt.NDArray | tuple[npt.NDArray, list[tuple[int, int, int, int]]]:
        """Apply Albumentations transformation(s).

        Args:
            image: An input image array.
            boxes: A list of bounding boxes.
            _mask: A mask of the input image.

        Returns:
            The transformed image array or a tuple containing transformed
            image array and transformed bounding boxes.

        """

        transformation = self.transformation(
            image=image,
            bboxes=boxes or [],
            labels=[0] * len(boxes) if boxes else []
        )

        image = transformation.get('image')
        boxes = transformation.get('bboxes')

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        return image, boxes


class TorchvisionStrategy(TransformationStrategy):
    def __init__(self, transformation: TVCompose):
        self.transformation = transformation

    def apply(
        self,
        image: npt.NDArray | None = None,
        boxes: list[tuple[int, int, int, int]] | None = None,
        masks: npt.NDArray | None = None
    ) -> npt.NDArray | tuple[npt.NDArray, list[tuple[int, int, int, int]]]:
        """Apply torchvision  transformation(s).

        Args:
            image: An input image array.
            boxes: A list of bounding boxes.
            _mask: A mask of the input image.

        Returns:
            The transformed image array or a tuple containing transformed
            image array and transformed bounding boxes.

        """

        image = tv_tensors.Image(image, dtype=torch.float32)

        if boxes is None and masks is None:
            return self.transformation(image)

        if boxes is not None and masks is None:
            boxes = tv_tensors.BoundingBoxes(
                boxes,
                canvas_size=image.shape[-2:],
                dtype=torch.float32,
                format='XYXY',
            )

            return self.transformation(image, boxes)

        if masks is not None and boxes is None:
            masks = tv_tensors.Mask(masks, dtype=torch.uint8)
            return self.transformation(image, masks)

        boxes = tv_tensors.BoundingBoxes(
            boxes,
            canvas_size=image.shape[-2:],
            dtype=torch.float32,
            format='XYXY',
        )

        masks = tv_tensors.Mask(masks, dtype=torch.uint8)
        return self.transformation(image, boxes, masks)
