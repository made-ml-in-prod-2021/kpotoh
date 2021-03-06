import os
import pickle
from typing import Tuple

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def read_data(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_train = os.path.join(dir_path, 'train.csv')
    path_to_val = os.path.join(dir_path, 'val.csv')

    train = pd.read_csv(path_to_train)
    val = pd.read_csv(path_to_val)
    return train, val


def train_model(features: np.ndarray, target: np.ndarray) -> LogisticRegression:
    model = LogisticRegression()
    model.fit(features, target)
    return model


def save_model(model: LogisticRegression, model_dir: str):
    """ будем сохранять модель 2 раза: в директорию с датой и 
    в базовую директорию, из которой модель идет на предикт. 
    
    `model_dir` это /data/models/{{ ds }}
    `base_dir` это /data/models
    """
    os.makedirs(model_dir, exist_ok=True)
    base_dir = os.path.dirname(model_dir)

    for my_dir in [model_dir, base_dir]:
        with open(os.path.join(my_dir, "model.pkl"), "wb") as fout:
            pickle.dump(model, fout)


@click.command("train")
@click.option("--data-dir")
@click.option("--model-dir")
def train(data_dir: str, model_dir: str):
    data, _ = read_data(data_dir)
    X = data.drop('target', axis=1).values
    y = data['target'].values

    model = train_model(X, y)
    save_model(model, model_dir)


if __name__ == "__main__":
    train()
