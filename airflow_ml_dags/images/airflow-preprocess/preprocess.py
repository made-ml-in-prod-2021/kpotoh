import os
from typing import Tuple

import click
import pandas as pd


def read_data_and_target(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_data = os.path.join(dir_path, 'data.csv')
    path_to_target = os.path.join(dir_path, 'target.csv')

    data = pd.read_csv(path_to_data)
    target = pd.read_csv(path_to_target)
    return data, target


def modify_data(data: pd.DataFrame, target: pd.DataFrame):
    # do nothing
    return data, target


def write_data(output_dir: str, data: pd.DataFrame, target=None):
    os.makedirs(output_dir, exist_ok=True)

    path_to_data = os.path.join(output_dir, 'data.csv')
    data.to_csv(path_to_data, index=None)
    if target is not None:
        path_to_target = os.path.join(output_dir, 'target.csv')
        target.to_csv(path_to_target, index=None)


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    data, target = read_data_and_target(input_dir)
    pdata, ptarget = modify_data(data, target)
    write_data(output_dir, pdata, ptarget)


if __name__ == '__main__':
    preprocess()
