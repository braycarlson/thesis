from __future__ import annotations

import netron
import torch

from thesis.constant import CASSD
from thesis.factory import ModelFactory
from torchinfo import summary


def main() -> None:
    model, _ = ModelFactory.get_model('cassd')
    model.eval()

    file = CASSD.joinpath('cassd.pth').as_posix()
    torch.save(model, file)

    netron.start(file)

    col_names = (
        'input_size',
        'output_size',
        'num_params'
    )

    size = (1, 1, 128, 128)

    summary(
        model,
        size,
        col_names=col_names,
        depth=5,
        verbose=1
    )


if __name__ == '__main__':
    main()
