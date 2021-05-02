import os
from typing import List

import pytest

from tests.global_fixtures import (
    path_to_synthetic_data, synthetic_dataset, pipeline_params
)
from ml_project.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)
from ml_project.train_pipeline import train_pipeline


@pytest.fixture
def pipeline_params_for_testing(tmpdir, path_to_synthetic_data, pipeline_params):
    output_model_path = tmpdir.join("model.pkl")
    metric_path = tmpdir.join("metrics.json")

    num_features = pipeline_params.feature_params.numerical_features
    cat_features = pipeline_params.feature_params.categorical_features
    target_col = pipeline_params.feature_params.target_col

    params = TrainingPipelineParams(
        input_data_path=path_to_synthetic_data,
        output_model_path=output_model_path,
        metric_path=metric_path,
        splitting_params=SplittingParams(val_size=0.2, random_state=2339),
        feature_params=FeatureParams(
            numerical_features=num_features,
            categorical_features=cat_features,
            target_col=target_col,
            use_scaling_for_num_features=True,
            features_to_drop=pipeline_params.feature_params.features_to_drop,
        ),
        train_params=TrainingParams(model_type="LogisticRegression"),
    )
    return params


@pytest.mark.parametrize(
    "model_type",
    [
        pytest.param("LogisticRegression", id='logreg'),
        pytest.param("RandomForestClassifier", id='trees'),
        pytest.param("KNeighborsClassifier", id='KNN'),
    ]
)
def test_end2end(pipeline_params_for_testing, model_type):
    """ define model and metrics pathes for testing and run train pipeline 
    with each classifier 
    """
    params = pipeline_params_for_testing
    params.train_params.model_type = model_type
    real_model_path, metrics = train_pipeline(params)

    assert metrics["accuracy_score"] > 0
    assert os.path.exists(real_model_path)
    assert os.path.exists(params.metric_path)
