from __future__ import annotations

import netron
import torch

from thesis.constant import CLASSIFICATION
from thesis.factory import ModelFactory
from torchinfo import summary


def main() -> None:
    model, _ = ModelFactory.get_model('classification')
    model.eval()

    file = CLASSIFICATION.joinpath('classification.pth').as_posix()
    torch.save(model, file)

    netron.start(file)

    col_names = (
        'input_size',
        'output_size',
        'num_params'
    )

    size = (1, 1, 28, 28)

    summary(
        model,
        size,
        col_names=col_names,
        depth=5,
        verbose=1
    )


if __name__ == '__main__':
    main()
