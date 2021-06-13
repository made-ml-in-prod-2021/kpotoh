import os
import pickle
import json
from typing import Tuple

import pandas as pd
import click
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


def load_model(model_dir: str):
    with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model


def read_data(dir_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path_to_train = os.path.join(dir_path, 'train.csv')
    path_to_val = os.path.join(dir_path, 'val.csv')

    train = pd.read_csv(path_to_train)
    val = pd.read_csv(path_to_val)
    return train, val


def save_metrics(metrics: dict, metrics_path: str):
    with open(os.path.join(metrics_path, "metrics.json"), "w") as fout:
        json.dump(metrics, fout)


@click.command("validate")
@click.option("--data-dir")
@click.option("--model-dir")
def validate(data_dir: str, model_dir: str):
    train, val = read_data(data_dir)
    train_X, train_y = train.drop('target', axis=1).values, train['target'].values
    val_X, val_y = val.drop('target', axis=1).values, val['target'].values
    
    model = load_model(model_dir)
    tpreds = model.predict(train_X)
    vpreds = model.predict(val_X)
    metrics = {
        "train": {
            "accuracy_score": accuracy_score(train_y, tpreds),
            "f1_score": f1_score(train_y, tpreds),
            "roc_auc_score": roc_auc_score(train_y, tpreds),
        }, 
        "val": {
            "accuracy_score": accuracy_score(val_y, vpreds),
            "f1_score": f1_score(val_y, vpreds),
            "roc_auc_score": roc_auc_score(val_y, vpreds),
        }
    }
    save_metrics(metrics, model_dir)


if __name__ == '__main__':
    validate()
