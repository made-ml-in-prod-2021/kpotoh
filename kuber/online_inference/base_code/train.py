import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def read_data(path: str, target_col='target') -> Tuple[np.ndarray]:
    data_raw = pd.read_csv(path)
    target = data_raw[target_col]
    data = data_raw.drop(target_col, axis=1)
    return data.values, target.values


def train_model(data: np.ndarray, target: np.ndarray):
    model = RandomForestClassifier()
    model.fit(data, target)
    return model


def save_model(path: str, model: RandomForestClassifier):
    with open(path, "wb") as fout:
        pickle.dump(model, fout)


def main():
    data_path = 'heart.csv'
    model_path = 'model.pkl'

    data, target = read_data(data_path)
    model = train_model(data, target)
    save_model(model_path, model)


if __name__ == '__main__':
    main()
