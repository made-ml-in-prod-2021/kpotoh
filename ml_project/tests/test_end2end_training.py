import os
from typing import List

import pytest
import pandas as pd

from ml_project.entities import (
    TrainingPipelineParams,
    SplittingParams,
    FeatureParams,
    TrainingParams,
)
from ml_project.pipeline import train_pipeline, predict_pipeline


@pytest.fixture
def pipeline_params_for_testing(tmpdir, path_to_synthetic_data, pipeline_params):
    output_model_path = tmpdir.join("model.pkl")
    metric_path = tmpdir.join("metrics.json")

    # change path-parameters to temporary files and synthetic data
    pipeline_params.input_data_path = path_to_synthetic_data
    pipeline_params.output_model_path = output_model_path
    pipeline_params.metric_path = metric_path
    return pipeline_params


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


def test_prediction(tmpdir, pipeline_params_for_testing):
    # train model to create model file
    train_pipeline(pipeline_params_for_testing)

    dataset_path = pipeline_params_for_testing.input_data_path
    output_path = tmpdir.join("preds.csv")
    predict_pipeline(dataset_path, output_path, pipeline_params_for_testing)
    
    assert os.path.exists(output_path)
    readed_preds = pd.read_csv(output_path)
    assert readed_preds.columns == ['target']
    assert readed_preds.shape == (100, 1)
    assert readed_preds['target'].nunique() == 2
