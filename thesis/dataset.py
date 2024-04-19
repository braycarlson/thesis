from __future__ import annotations

from torch.utils.data import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from albumentations.core.composition import Compose as ACompose
    from torchvision.transforms.v2._container import Compose as TVCompose


class BaseDataset(Dataset):
    def __init__(
        self,
        annotation: pd.DataFrame | None = None,
        strategy: ACompose | TVCompose | None = None
    ):
        self.annotation = annotation
        self.strategy = strategy

    def __len__(self) -> int:
        """

        Returns:
            The size of the pandas DataFrame.

        """

        return len(self.annotation)
