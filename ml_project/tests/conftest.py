import os

import pytest
import numpy as np
import pandas as pd

from ml_project.entities import TrainingPipelineParams
from ml_project.entities import read_training_pipeline_params
from ml_project.features import FeaturesExtractor, extract_target


@pytest.fixture
def synthetic_dataset():
    """ genereate synthetic dataset independently based on true one """
    cat_features = {
        'sex': 2, 'cp': 4, 'fbs': 2, 'restecg': 3, 
        'exang':2, 'slope':3, 'ca': 5, 'thal': 4,
    }
    num_features = {
        'age': (29, 77), 'trestbps': (94, 200), 'chol': (126, 564), 
        'thalach': (71, 202), 'oldpeak': (0, 6.2),
    }
    target = {'target': 2}

    nrows = 100
    data = dict()
    for fea, nvars in cat_features.items():
        data[fea] = np.random.randint(0, nvars, nrows)
    for fea, (minv, maxv) in num_features.items():
        if fea == 'oldpeak':
            data[fea] = np.random.randint(minv*10, (maxv + 1)*10, nrows) / 10
        else:   
            data[fea] = np.random.randint(minv, maxv + 1, nrows)
    for fea, nvars in target.items():
        data[fea] = np.random.randint(0, nvars, nrows)

    data = pd.DataFrame(data)
    return data


@pytest.fixture
def path_to_synthetic_data(tmpdir, synthetic_dataset):
    dataset_fio = tmpdir.join('dataset.txt')
    synthetic_dataset.to_csv(dataset_fio, index=None)
    return dataset_fio


@pytest.fixture
def pipeline_params() -> TrainingPipelineParams:
    path = './tests/configs/train_config.yml'
    assert os.path.exists(path), 'choose another config path'

    params = read_training_pipeline_params(path)
    return params


@pytest.fixture
def train_data(synthetic_dataset, pipeline_params):
    feature_params = pipeline_params.feature_params
    X = FeaturesExtractor(feature_params).fit_transform(synthetic_dataset)
    y = extract_target(synthetic_dataset, feature_params)
    return X, y
    