import os
from typing import Tuple

import pandas as pd
import click
from sklearn.model_selection import train_test_split


def read_data_and_target(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_data = os.path.join(dir_path, 'data.csv')
    path_to_target = os.path.join(dir_path, 'target.csv')

    data = pd.read_csv(path_to_data)
    target = pd.read_csv(path_to_target)
    return data, target


def write_data(
    output_dir: str, data: pd.DataFrame, 
    target=None, data_filename='data.csv'):

    os.makedirs(output_dir, exist_ok=True)
    path_to_data = os.path.join(output_dir, data_filename)
    data.to_csv(path_to_data, index=None)
    if target is not None:
        path_to_target = os.path.join(output_dir, 'target.csv')
        target.to_csv(path_to_target, index=None)


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir: str):
    data, target = read_data_and_target(input_dir)
    data = data.merge(target, left_index=True, right_index=True)
    train, val = train_test_split(data, test_size=0.3)

    write_data(output_dir, train, data_filename='train.csv')
    write_data(output_dir, val, data_filename='val.csv')


if __name__ == '__main__':
    split()
