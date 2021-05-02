import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from ml_project.entities import TrainingParams

SklearnRegressionModel = Union[
    RandomForestClassifier, LogisticRegression, KNeighborsClassifier
]


def choose_model(model_type: str) -> SklearnRegressionModel:
    model_types = {
        'LogisticRegression': LogisticRegression(),
        'RandomForestClassifier': RandomForestClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(),
    }
    model = model_types.get(model_type, default=None)
    if model is None:
        raise NotImplementedError
    return model


def train_model(
    features: np.ndarray,
    target: np.ndarray,
    train_params: TrainingParams
) -> SklearnRegressionModel:
    model = choose_model(train_params.model_type)
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: np.ndarray,
) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: np.ndarray):
    scores = {
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
        "roc_auc_score": roc_auc_score(target, predicts),
    }
    return scores


def save_model(model: SklearnRegressionModel, path: str):
    """ save model to pickle file """
    with open(path, "wb") as fout:
        pickle.dump(model, fout)
