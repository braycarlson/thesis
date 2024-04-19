from __future__ import annotations

import lightning.pytorch as lp
import pandas as pd
import torch

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from thesis.a.dataset import ClassificationDataset
from thesis.a.model import ClassificationModel
from thesis.callback import KeyboardInterruptTrialCallback
from thesis.constant import NOISE
from thesis.datamodule import DataModule
from thesis.transform import GradientTransform, ShiftTransform
from thesis.transformation import TorchvisionStrategy
from torchvision.transforms import v2


def main() -> None:
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    size = (28, 28)

    transform = [
        v2.Resize(size, antialias=True),
        v2.RandomChoice([
            GradientTransform(),
            ShiftTransform(9),
            v2.GaussianBlur(kernel_size=3),
            v2.RandomAutocontrast(p=0.5),
            v2.RandomErasing(
                p=0.5,
                scale=(0.025, 0.050),
                ratio=(0.025, 0.050),
                value=0
            ),
            v2.RandomPerspective(distortion_scale=0.2, p=0.5),
            v2.RandomResizedCrop(
                size,
                antialias=True,
                scale=(0.8, 1.0),
                ratio=(0.75, 1.33)
            )
        ]),
    ]

    transformation = v2.Compose(transform)
    strategy = TorchvisionStrategy(transformation)

    dataset = ClassificationDataset

    name = '02'

    path = NOISE.joinpath('segment', name)

    pickle = path.joinpath('training.pkl')
    training = pd.read_pickle(pickle)

    pickle = path.joinpath('testing.pkl')
    testing = pd.read_pickle(pickle)

    pickle = path.joinpath('validation.pkl')
    validation = pd.read_pickle(pickle)

    annotation = {
        'training': training,
        'testing': testing,
        'validation': validation,
    }

    datamodule = DataModule(
        annotation=annotation,
        batch_size=64,
        dataset=dataset,
        strategy=strategy
    )

    model = ClassificationModel()

    callbacks = [
        EarlyStopping(
            mode='min',
            monitor='validation_loss',
            min_delta=0.01,
            patience=10,
            verbose=True,
        ),
        KeyboardInterruptTrialCallback()
    ]

    logger = TensorBoardLogger(
        'tensorboard',
        name=name
    )

    trainer = lp.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=50
    )

    trainer.fit(
        datamodule=datamodule,
        model=model
    )

    dataloaders = datamodule.val_dataloader()

    trainer.validate(
        dataloaders=dataloaders,
        model=model
    )

    dataloaders = datamodule.test_dataloader()

    trainer.test(
        ckpt_path='best',
        dataloaders=dataloaders,
        model=model
    )


if __name__ == '__main__':
    main()
