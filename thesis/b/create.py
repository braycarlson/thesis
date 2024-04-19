from __future__ import annotations

import pandas as pd

from thesis.constant import FLEXIBLE, VARIABLE
from tqdm import tqdm


def main() -> None:
    columns = [
        'file',
        'label',
        'xyxy'
    ]

    partition = {
        'training': 48000,
        'testing': 10000,
        'validation': 12000
    }

    for name, amount in partition.items():
        files = [
            path
            for path in FLEXIBLE.glob(f"*/*/{name}.hdf")
            if path.is_file()
        ]

        n = amount // len(files)

        dataframe = pd.DataFrame()

        for file in tqdm(files):
            hdf = pd.read_hdf(file, columns=columns)

            sample = hdf.sample(n=n)

            dataframe = pd.concat(
                [dataframe, sample],
                ignore_index=True
            )

            del hdf

        hdf = VARIABLE.joinpath(f"{name}.hdf")
        variable = pd.read_hdf(hdf, columns=columns)

        dataframe = pd.concat(
            [dataframe, variable],
            ignore_index=True
        )

        filename = name + '.hdf'
        path = FLEXIBLE.joinpath(filename)

        dataframe.to_hdf(
            path,
            format='table',
            key=name
        )

        del variable
        del dataframe


if __name__ == '__main__':
    main()
