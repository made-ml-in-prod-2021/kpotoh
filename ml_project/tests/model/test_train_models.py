import os
import json
import pickle
from zlib import adler32

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from ml_project.entities import FeatureParams
from ml_project.models import (
    train_model, predict_model, evaluate_model, 
    save_model, save_metrics, load_model,
)


@pytest.fixture
def train_params(pipeline_params):
    return pipeline_params.train_params


@pytest.fixture
def fitted_model(train_data, train_params):
    X, y = train_data
    model = train_model(X, y, train_params)
    return model


def test_training(fitted_model, train_data):
    X, _ = train_data
    # if cannot predict, raise exceproin
    fitted_model.predict(X)


def test_pridiction(fitted_model, train_data):
    X, target = train_data
    preds = fitted_model.predict(X)
    assert target.shape == preds.shape
    assert 0 < accuracy_score(target, preds) < 1


def test_evaluation(fitted_model, train_data):
    X, target = train_data
    preds = fitted_model.predict(X)
    scores = evaluate_model(preds, target)
    assert isinstance(scores, dict)
    for metric, score_value in scores.items():
        assert isinstance(metric, str)
        assert isinstance(score_value, float)
        assert 0 < score_value < 1


def test_model_saving(tmpdir, fitted_model):
    model_fio = tmpdir.join('model.pkl')
    save_model(fitted_model, model_fio)
    assert os.path.exists(model_fio)
    with open(model_fio, 'rb') as fin:
        loaded_model = pickle.load(fin)
    assert adler32(pickle.dumps(loaded_model)) == adler32(pickle.dumps(fitted_model))


def test_metrics_saving(tmpdir, fitted_model, train_data):
    X, target = train_data
    preds = fitted_model.predict(X)
    scores = evaluate_model(preds, target)

    metrics_fio = tmpdir.join('metrics.json')
    save_metrics(scores, metrics_fio)

    assert os.path.exists(metrics_fio)
    with open(metrics_fio, 'r') as fin:
        loaded_metrics = json.load(fin)
    assert loaded_metrics == scores


def test_model_loading(tmpdir, fitted_model):
    model_fio = tmpdir.join('model.pkl')
    save_model(fitted_model, model_fio)
    loaded_model = load_model(model_fio)
    assert adler32(pickle.dumps(loaded_model)) == adler32(pickle.dumps(fitted_model))
 