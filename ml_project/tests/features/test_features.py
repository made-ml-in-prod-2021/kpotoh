import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml_project.entities import FeatureParams
from ml_project.features import (
    FeaturesExtractor, extract_target
)
from tests.global_fixtures import synthetic_dataset, pipeline_params


@pytest.fixture
def feature_params(pipeline_params):
    fea_params = pipeline_params.feature_params
    return fea_params


@pytest.fixture
def extractor_instance(feature_params):
    extractor = FeaturesExtractor(feature_params)
    return extractor


def test_scaling_in_num_features_building(extractor_instance):
    num_pipeline = extractor_instance.build_numerical_pipeline(use_scaling=False)
    scaled_num_pipeline = extractor_instance.build_numerical_pipeline(use_scaling=True)

    steps1 = scaled_num_pipeline.steps
    assert any([
        isinstance(step_process, StandardScaler) for name, step_process in steps1
    ])
    steps2 = num_pipeline.steps
    assert not all([
        isinstance(step_process, StandardScaler) for name, step_process in steps2
    ])


def test_build_transformer(extractor_instance, synthetic_dataset):
    feature_params = extractor_instance.params
    transformer = extractor_instance.build_transformer(use_scaling=True)

    transformed = transformer.fit_transform(synthetic_dataset)
    expected_n_columns = (
        len(feature_params.numerical_features) + \
            synthetic_dataset[feature_params.categorical_features].nunique().sum()
    )
    assert len(transformed) == len(synthetic_dataset)
    assert transformed.shape[1] == expected_n_columns
    assert transformed.max() < 50, "scaling does'n work"


def test_fitting(extractor_instance, synthetic_dataset):
    extractor_instance.fit(synthetic_dataset)
    num_pipe = extractor_instance._transformer.named_transformers_['numerical_pipeline']
    scaler = num_pipe.named_steps['scaler']
    assert len(scaler.scale_) > 0, "scaler didn't fitted correctly"


def test_transformation(extractor_instance, synthetic_dataset):
    extractor_instance.fit(synthetic_dataset)
    feature_params = extractor_instance.params
    transformed = extractor_instance.transform(synthetic_dataset)
    expected_n_columns = (
        len(feature_params.numerical_features) + \
            synthetic_dataset[feature_params.categorical_features].nunique().sum()
    )
    assert len(transformed) == len(synthetic_dataset)
    assert transformed.shape[1] == expected_n_columns
    assert transformed.max() < 50, "scaling does'n work"


def test_target_extraction(feature_params, synthetic_dataset):
    target = extract_target(synthetic_dataset, feature_params)

    assert isinstance(target, np.ndarray)
    assert target.shape == (len(synthetic_dataset), )
    assert np.all(synthetic_dataset[feature_params.target_col] == target)
