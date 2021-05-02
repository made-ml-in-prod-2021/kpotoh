import json
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


def choose_model(model_type: str, random_state: int) -> SklearnRegressionModel:
    model_types = {
        'LogisticRegression': LogisticRegression(
            C=0.108, fit_intercept=True, max_iter=100, random_state=random_state,
        ),
        'RandomForestClassifier': RandomForestClassifier(
            criterion='entropy', max_features='log2',
            min_samples_leaf=1, n_estimators=55, random_state=random_state,
        ),
        'KNeighborsClassifier': KNeighborsClassifier(
            n_neighbors=5, p=1, random_state=random_state
        ),
    }
    model = model_types.get(model_type, None)
    if model is None:
        raise NotImplementedError
    return model


def train_model(
    features: np.ndarray,
    target: np.ndarray,
    train_params: TrainingParams
) -> SklearnRegressionModel:
    model = choose_model(train_params.model_type, train_params.random_state)
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
    return path


def save_metrics(metrics: dict, path: str):
    """ save metrics to json """
    with open(path, 'w') as fout:
        json.dump(metrics, fout)
