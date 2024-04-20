from __future__ import annotations

import matplotlib.pyplot as plt

from pathlib import Path
from screeninfo import get_monitors


def walk(file: Path) -> Path | None:
    """Find the root of the project from the "venv".

    Args:
        file: The starting path to search for the "venv".

    Returns:
        A path to the root directory, if found, None otherwise.

    """

    for parent in [file, *file.parents]:
        if parent.is_dir():
            path = parent.joinpath('venv')

            if path.exists() and path.is_dir():
                return path.parent

    return None


file = Path.cwd()
CWD = walk(file)

DATASET = CWD.joinpath('dataset')
DATASET.mkdir(exist_ok=True, parents=True)

ANALYSIS = CWD.joinpath('thesis/analysis')
ANALYSIS.mkdir(exist_ok=True, parents=True)

ANIMATION = CWD.joinpath('thesis/animation')
ANIMATION.mkdir(exist_ok=True, parents=True)

FIGURES = CWD.joinpath('thesis/figures')
FIGURES.mkdir(exist_ok=True, parents=True)

SEGMENT = DATASET.joinpath('segment')
SEGMENT.mkdir(exist_ok=True, parents=True)

STROKE = DATASET.joinpath('stroke')
STROKE.mkdir(exist_ok=True, parents=True)

MNIST = DATASET.joinpath('mnist')
MNIST.mkdir(exist_ok=True, parents=True)

HANDWRITTEN = DATASET.joinpath('handwritten')
HANDWRITTEN.mkdir(exist_ok=True, parents=True)

NOISE = DATASET.joinpath('noise')
NOISE.mkdir(exist_ok=True, parents=True)

ORIGINAL = DATASET.joinpath('original')
ORIGINAL.mkdir(exist_ok=True, parents=True)

OVERLAP = DATASET.joinpath('overlap')
OVERLAP.mkdir(exist_ok=True, parents=True)

CAPTCHA = OVERLAP.joinpath('captcha')
CAPTCHA.mkdir(exist_ok=True, parents=True)

CARDINAL = OVERLAP.joinpath('cardinal')
CARDINAL.mkdir(exist_ok=True, parents=True)

FLEXIBLE = OVERLAP.joinpath('flexible')
FLEXIBLE.mkdir(exist_ok=True, parents=True)

PIPELINE = OVERLAP.joinpath('pipeline')
PIPELINE.mkdir(exist_ok=True, parents=True)

SEPARATION = OVERLAP.joinpath('separation')
SEPARATION.mkdir(exist_ok=True, parents=True)

VARIABLE = OVERLAP.joinpath('variable')
VARIABLE.mkdir(exist_ok=True, parents=True)

CASSD = CWD.joinpath('thesis/b/tensorboard/')
CLASSIFICATION = CWD.joinpath('thesis/a/tensorboard/')

PALETTE = plt.cm.get_cmap('tab20')
length = len(PALETTE.colors)

color = [
    PALETTE(i)[:3]
    for i in range(length)
]

black = (0, 0, 0)
color.insert(0, black)

COLOR = dict(enumerate(color))

NORMALIZE = {
    k: tuple(int(255 * x) for x in v)
    for k, v in COLOR.items()
}

monitor = get_monitors()[0]
WIDTH, HEIGHT = monitor.width, monitor.height

SIZE = 128

RATIO = [
    [0.3, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 2.0, 3.0, 4.0],
    [0.3, 0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 2.0, 2.5, 3.0],
    [0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 3.0, 4.0],
    [0.6, 0.8, 1.0, 1.1, 1.2, 1.4, 1.6, 3.0, 4.0]
]
