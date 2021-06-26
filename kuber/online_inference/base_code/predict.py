import pickle
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier



def load_object(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


def read_data(path: str, target_col='target') -> Tuple[np.ndarray]:
    data_raw = pd.read_csv(path)
    target = data_raw[target_col]
    data = data_raw.drop(target_col, axis=1)
    return data.values, target.values


def make_predict(
    data: List, features: List[str], model: RandomForestClassifier) -> List:
    data, _ = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    return [OutputStruct(predicted_class=int(dead)) for dead in predicts]


def main():
    model = load_object('model.pkl')
    data = pd.read_csv("heart.csv").drop('target', axis=1)

    request_features = list(data.columns)
    request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[0].tolist()
        ]

    df = pd.DataFrame([request_data], columns=request_features)
    print(df)
    preds = model.predict(df)
    print(preds)
    print(type(preds), type(preds[0]))

if __name__ == '__main__':
    main()
