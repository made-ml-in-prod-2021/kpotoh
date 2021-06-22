import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


def read_data(dir_path: str) -> pd.DataFrame:
    path_to_data = os.path.join(dir_path, 'data.csv')
    data = pd.read_csv(path_to_data)
    return data


def load_model(model_dir: str) -> LogisticRegression:
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def write_prediction(preds: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path_to_preds = os.path.join(output_dir, 'preds.csv')
    preds.to_csv(path_to_preds, index=None)


@click.command("predict")
@click.option("--data-dir")
@click.option("--output-dir")
@click.option("--model-dir")
def predict(data_dir: str, output_dir: str, model_dir: str):
    data = read_data(data_dir)
    model = load_model(model_dir)
    data["predict"] = model.predict(data)
    write_prediction(data, output_dir)


if __name__ == '__main__':
    predict()