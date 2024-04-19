from __future__ import annotations

import numpy as np
import torch

from abc import abstractmethod, ABC


class CoordinateStrategy(ABC):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @abstractmethod
    def xyxy_to_xywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def xywh_to_xyxy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def xyxy_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def cxcywh_to_xyxy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def xywh_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def cxcywh_to_xywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def cxcywh_to_gcxgcy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        anchor: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def gcxgcy_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        anchor: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def normalize(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        system: str = 'xyxy'
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def rescale(
        self,
        boxes: torch.Tensor,
        original: tuple[int, int],
        target: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        raise NotImplementedError

    @abstractmethod
    def scale(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        system: str = 'xyxy'
    ) -> tuple[int, int, int, int] | torch.Tensor:
        raise NotImplementedError


class TensorStrategy(CoordinateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def xyxy_to_xywh(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        pass

    def xywh_to_xyxy(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        pass

    def xyxy_to_cxcywh(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        c_x = (box[:, 2] + box[:, 0]) / 2
        c_y = (box[:, 3] + box[:, 1]) / 2
        w = box[:, 2] - box[:, 0]
        h = box[:, 3] - box[:, 1]
        return torch.stack([c_x, c_y, w, h], dim=1)

    def cxcywh_to_xyxy(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        x_min = box[:, 0] - box[:, 2] / 2
        y_min = box[:, 1] - box[:, 3] / 2
        x_max = box[:, 0] + box[:, 2] / 2
        y_max = box[:, 1] + box[:, 3] / 2
        return torch.stack([x_min, y_min, x_max, y_max], dim=1)

    def xywh_to_cxcywh(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        pass

    def cxcywh_to_xywh(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        pass

    def cxcywh_to_gcxgcy(
        self,
        cxcy: torch.Tensor,
        anchor: torch.Tensor
    ) -> torch.Tensor:
        offset = (cxcy[:, :2] - anchor[:, :2]) / (anchor[:, 2:] / 10)
        scale = torch.log(cxcy[:, 2:] / anchor[:, 2:]) * 5
        return torch.cat([offset, scale], dim=1)

    def gcxgcy_to_cxcywh(
        self,
        gcxgcy: torch.Tensor,
        anchor: torch.Tensor
    ) -> torch.Tensor:
        center = gcxgcy[:, :2] * anchor[:, 2:] / 10 + anchor[:, :2]
        size = torch.exp(gcxgcy[:, 2:] / 5) * anchor[:, 2:]
        return torch.cat([center, size], dim=1)

    def normalize(
        self,
        box: torch.Tensor
    ) -> torch.Tensor:
        pass

    def rescale(
        self,
        boxes: torch.Tensor,
        original: tuple[int, int],
        target: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        pass

    def scale(
        self,
        boxes: torch.Tensor,
        system: str = 'xyxy'
    ) -> torch.Tensor:
        boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1e6, neginf=-1e6)

        boxes = torch.clamp(boxes, min=0)

        x_min, y_min, x_max, y_max = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        x_max = torch.max(x_min, x_max)
        y_max = torch.max(y_min, y_max)

        x_min = torch.round(x_min * self.width).int()
        y_min = torch.round(y_min * self.height).int()
        x_max = torch.round(x_max * self.width).int()
        y_max = torch.round(y_max * self.height).int()

        boxes = torch.stack(
            (x_min, y_min, x_max, y_max),
            dim=1
        )

        if system == 'cxcywh':
            return self.xyxy_to_cxcywh(boxes)

        return boxes


class ScalarStrategy(CoordinateStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def xyxy_to_xywh(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        return (x_min, y_min, width, height)

    def xywh_to_xyxy(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x_min, y_min, width, height = box
        x_max = x_min + width
        y_max = y_min + height
        return (x_min, y_min, x_max, y_max)

    def xyxy_to_cxcywh(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x_min, y_min, x_max, y_max = box
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        return (cx, cy, width, height)

    def cxcywh_to_xyxy(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        cx, cy, width, height = box
        x_min = cx - width / 2
        y_min = cy - height / 2
        x_max = cx + width / 2
        y_max = cy + height / 2
        return (x_min, y_min, x_max, y_max)

    def xywh_to_cxcywh(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x_min, y_min, width, height = box
        cx = x_min + width / 2
        cy = y_min + height / 2
        return (cx, cy, width, height)

    def cxcywh_to_xywh(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        cx, cy, width, height = box
        x_min = cx - width / 2
        y_min = cy - height / 2
        return (x_min, y_min, width, height)

    def cxcywh_to_gcxgcy(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        pass

    def gcxgcy_to_cxcywh(
        self,
        box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        pass

    def normalize(
        self,
        box: tuple[int, int, int, int],
        system: str = 'xyxy'
    ) -> tuple[float, float, float, float]:
        if system == 'xywh':
            box = self.xywh_to_xyxy(box)

        if system == 'cxcywh':
            box = self.cxcywh_to_xyxy(box)

        x1, y1, x2, y2 = box
        x1 = float(x1) / self.width
        y1 = float(y1) / self.height
        x2 = float(x2) / self.width
        y2 = float(y2) / self.height

        return (x1, y1, x2, y2)

    def rescale(
        self,
        boxes: list[tuple[int, int, int, int]],
        original: tuple[int, int],
        target: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        scaled = []

        ow, oh = original
        tw, th = target

        for box in boxes:
            box = self.scale(box, 'xyxy')
            x_min, y_min, x_max, y_max = box

            x_min = int(x_min * tw / ow)
            y_min = int(y_min * th / oh)
            x_max = int(x_max * tw / ow)
            y_max = int(y_max * th / oh)

            coordinates = (x_min, y_min, x_max, y_max)
            scaled.append(coordinates)

        return scaled

    def scale(
        self,
        box: tuple[int, int, int, int],
        system: str = 'xyxy'
    ) -> tuple[float, float, float, float]:
        x_min, y_min, x_max, y_max = box

        x_min = max(
            0,
            np.nan_to_num(x_min, nan=0.0, posinf=1e6, neginf=-1e6)
        )

        y_min = max(
            0,
            np.nan_to_num(y_min, nan=0.0, posinf=1e6, neginf=-1e6)
        )

        x_max = max(
            x_min,
            np.nan_to_num(x_max, nan=0.0, posinf=1e6, neginf=-1e6)
        )

        y_max = max(
            y_min,
            np.nan_to_num(y_max, nan=0.0, posinf=1e6, neginf=-1e6)
        )

        x_min = round(float(x_min) * self.width)
        y_min = round(float(y_min) * self.height)
        x_max = round(float(x_max) * self.width)
        y_max = round(float(y_max) * self.height)

        x_min, x_max = sorted([x_min, x_max])
        y_min, y_max = sorted([y_min, y_max])

        if system == 'cxcywh':
            box = (x_min, y_min, x_max, y_max)
            return self.xyxy_to_cxcywh(box)

        return (x_min, y_min, x_max, y_max)


class CoordinatesConverter:
    def __init__(self, strategy: CoordinateStrategy):
        self.strategy = strategy

    def xyxy_to_xywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.xyxy_to_xywh(box)

    def xywh_to_xyxy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.xywh_to_xyxy(box)

    def xyxy_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.xyxy_to_cxcywh(box)

    def cxcywh_to_xyxy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.cxcywh_to_xyxy(box)

    def xywh_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.xywh_to_cxcywh(box)

    def cxcywh_to_xywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.cxcywh_to_xywh(box)

    def cxcywh_to_gcxgcy(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        anchor: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.cxcywh_to_gcxgcy(box, anchor)

    def gcxgcy_to_cxcywh(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        anchor: tuple[int, int, int, int] | torch.Tensor
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.gcxgcy_to_cxcywh(box, anchor)

    def normalize(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        system: str = 'xyxy'
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.normalize(box, system)

    def rescale(
        self,
        boxes: list[tuple[int, int, int, int]],
        original: tuple[int, int],
        target: tuple[int, int]
    ) -> list[tuple[int, int, int, int]]:
        return self.strategy.rescale(boxes, original, target)

    @abstractmethod
    def scale(
        self,
        box: tuple[int, int, int, int] | torch.Tensor,
        system: str = 'xyxy'
    ) -> tuple[int, int, int, int] | torch.Tensor:
        return self.strategy.scale(box, system)
