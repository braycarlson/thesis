from __future__ import annotations

import json
import lightning.pytorch as lp
import pandas as pd
import torch

from datamodule import DataModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from thesis.b.dataset import SingleShotDetectorDataset
from thesis.b.model import ClassAgnosticSingleShotDetector
from thesis.callback import KeyboardInterruptTrialCallback
from thesis.constant import FLEXIBLE, RATIO
from thesis.transform import RandomNoise, RandomStroke
from thesis.transformation import TorchvisionStrategy
from torchinfo import summary
from torchvision.transforms import v2


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    size = (128, 128)

    transform = [
        v2.RandomApply([
            RandomStroke(count=10, width=1)
        ], p=1.0),
        v2.RandomApply([
            RandomNoise()
        ], p=0.25),
        v2.RandomApply([
            v2.RandomHorizontalFlip(p=0.50),
            v2.RandomVerticalFlip(p=0.50),
        ], p=0.25),
        v2.RandomApply([
            v2.GaussianBlur(kernel_size=1)
        ], p=0.50),
        v2.Resize(size, antialias=True),
    ]

    transformation = v2.Compose(transform)
    strategy = TorchvisionStrategy(transformation)

    dataset = SingleShotDetectorDataset

    name = 'cassd'

    columns = [
        'file',
        'label',
        'xyxy'
    ]

    hdf = FLEXIBLE.joinpath('training.hdf')
    training = pd.read_hdf(hdf)

    hdf = FLEXIBLE.joinpath('validation.hdf')
    validation = pd.read_hdf(hdf)

    hdf = FLEXIBLE.joinpath('testing.hdf')
    testing = pd.read_hdf(hdf)

    columns.pop(0)

    for column in columns:
        training[column] = training[column].apply(json.loads)
        validation[column] = validation[column].apply(json.loads)

    annotation = {
        'training': training,
        'validation': validation,
        'testing': testing,
    }

    datamodule = DataModule(
        annotation=annotation,
        batch_size=64,
        dataset=dataset,
        strategy=strategy
    )

    model = ClassAgnosticSingleShotDetector(ratio=RATIO)

    size = (16, 1, 128, 128)
    summary(model, size, depth=10)

    callbacks = [
        EarlyStopping(
            min_delta=0.001,
            mode='min',
            monitor='validation_loss',
            patience=10,
            verbose=True
        ),
        KeyboardInterruptTrialCallback(),
    ]

    logger = TensorBoardLogger(
        'tensorboard',
        name=name
    )

    trainer = lp.Trainer(
        callbacks=callbacks,
        gradient_clip_val=0.50,
        logger=logger,
        max_epochs=50
    )

    trainer.fit(
        datamodule=datamodule,
        model=model
    )


if __name__ == '__main__':
    main()
