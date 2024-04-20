from __future__ import annotations

import math
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from itertools import product
from matplotlib import patches
from thesis.constant import HEIGHT, WIDTH


CANVAS = (128, 128)
IMAGE = (28, 28)


def rotate(
    x: float,
    y: float,
    angle: float,
    center: tuple[float, float]
) -> tuple[float, float]:
    """Rotate a point around a center by a given angle.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        angle: The angle of rotation in degrees.
        center: The center of rotation as a tuple (x, y).

    Returns:
        The coordinates of the rotated point.

    """

    rad = math.radians(angle)

    x = x - center[0]
    y = y - center[1]

    x_new = x * math.cos(rad) - y * math.sin(rad)
    y_new = x * math.sin(rad) + y * math.cos(rad)

    x_new = x_new + center[0]
    y_new = y_new + center[1]

    return x_new, y_new


def translate(x: float, y: float, dx: float, dy: float) -> tuple[float, float]:
    """Translate a point by a given offset.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.
        dx: The x-offset.
        dy: The y-offset.

    Returns:
        The translated coordinates of the point.

    """

    return x + dx, y + dy


def adjust(x: float, y: float) -> tuple[float, float]:
    """Adjust the coordinates of a point to fit within the canvas.

    Args:
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.

    Returns:
        The adjusted coordinates of the point.

    """

    x = max(0, min(x, CANVAS[0] - IMAGE[0]))
    y = max(0, min(y, CANVAS[1] - IMAGE[1]))

    if x + IMAGE[0] > CANVAS[0]:
        x = CANVAS[0] - IMAGE[0]

    if y + IMAGE[1] > CANVAS[1]:
        y = CANVAS[1] - IMAGE[1]

    return x, y


def transform(
    positions: list[tuple[float, float]],
    angle: float
) -> list[tuple[float, float]]:
    """Apply rotation and translation to the position(s).

    Args:
        positions: The initial positions as tuples (x, y).
        angle: The angle of rotation in degrees.

    Returns:
        The transformed positions.

    """

    center = (CANVAS[0] / 2, CANVAS[1] / 2)

    rotated = [
        rotate(x, y, angle, center)
        for x, y in positions
    ]

    min_x = min(x for x, _ in rotated)
    max_x = max(x for x, _ in rotated)
    min_y = min(y for _, y in rotated)
    max_y = max(y for _, y in rotated)

    dx = random.randint(-int(min_x), int(CANVAS[0] - max_x - IMAGE[0]))
    dy = random.randint(-int(min_y), int(CANVAS[1] - max_y - IMAGE[1]))

    return [translate(x, y, dx, dy) for x, y in rotated]


def precompute(overlap: float, count: int) -> tuple[tuple[int, int], ...]:
    """Precompute the position(s) depending on overlap rate and count.

    Args:
        overlap: The overlap rate.
        count: The number of positions to precompute.

    Returns:
        The precomputed position(s).

    """

    width, height = IMAGE
    center_x = (CANVAS[0] - width) // 2
    center_y = (CANVAS[1] - height) // 2
    positions = [(center_x, center_y)]

    def _inside(x: int, y: int) -> bool:
        """Check if a point is inside the canvas boundaries.

        Args:
            x: The x-coordinate of the point.
            y: The y-coordinate of the point.

        Returns:
            True if the point is inside the canvas, False otherwise.

        """

        return 0 <= x < CANVAS[0] - width and 0 <= y < CANVAS[1] - height

    def _position(distance: int) -> list[tuple[int, int]]:
        """Generate the position(s) based on a given distance.

        Args:
            distance: The distance from the center.

        Returns:
            The position(s) within the distance from the center.

        """

        layer = []

        for dx, dy in product(range(-distance, distance + 1), repeat=2):
            if abs(dx) + abs(dy) == distance:
                x, y = (
                    center_x + dx * (width - width * overlap),
                    center_y + dy * (height - height * overlap)
                )

                if _inside(x, y):
                    layer.append((int(x), int(y)))

        return layer

    distance = 1

    while len(positions) < count:
        layer = _position(distance)

        if not layer:
            distance = distance + 1
            continue

        random.shuffle(layer)

        for pos in layer:
            if len(positions) == count:
                break

            positions.append(pos)

        distance = distance + 1

    return tuple(positions)


def main() -> None:
    show = False

    configuration = {}

    for amount in range(2, 10):
        for rate in np.arange(0.00, 1.01, 0.10):
            rate = round(rate, 2)
            key = f"{amount - 1}_{rate}"

            configuration[key] = []

            possible = set()

            for angle in range(0, 360, 1):
                positions = precompute(rate, amount)
                positions = transform(positions, angle)

                possible.add(tuple(positions))

                if show:
                    fig, ax = plt.subplots()
                    ax.set_xlim(0, CANVAS[0])
                    ax.set_ylim(0, CANVAS[1])
                    ax.invert_yaxis()

                    for (x, y) in positions:
                        rect = patches.Rectangle(
                            (x, y),
                            IMAGE[0],
                            IMAGE[1],
                            linewidth=1,
                            edgecolor='r',
                            facecolor='none'
                        )

                        ax.add_patch(rect)

                    figure_width, figure_height = fig.get_size_inches() * fig.dpi

                    x = (WIDTH - figure_width) // 2
                    y = (HEIGHT - figure_height) // 2
                    y = y - 50

                    plt.get_current_fig_manager().window.wm_geometry(f"+{int(x)}+{int(y)}")

                    plt.tight_layout()
                    plt.show(block=True)
                    plt.close()

            possible = list(possible)
            configuration[key].extend(possible)

    with open('configuration.pkl', 'wb') as handle:
        pickle.dump(configuration, handle)


if __name__ == '__main__':
    main()
