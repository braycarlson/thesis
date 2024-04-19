from __future__ import annotations

import numpy as np

from PIL import Image


def to_transparent(image: Image.Image) -> Image.Image:
    """Convert an image to have a transparent background.

    Args:
        image: The image to be converted.

    Returns:
        An image with a transparent background.

    """

    image = image.convert('RGBA')

    array = np.array(image)

    array[
        (array[:, :, 0] < 50) &
        (array[:, :, 1] < 50) &
        (array[:, :, 2] < 50),
        3
    ] = 0

    return Image.fromarray(array, 'RGBA')
