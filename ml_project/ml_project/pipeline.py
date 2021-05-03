import sys
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import logging.config

import yaml
import pandas as pd

from ml_project.entities import TrainingPipelineParams, read_training_pipeline_params
from ml_project.data import read_data, split_to_train_val
from ml_project.features import FeaturesExtractor, extract_target
from ml_project.models import (
    train_model, predict_model, evaluate_model, save_model, save_metrics, load_model
)

DEFAULT_LOGGING_CONFIG_PATH = './configs/logging_config.yml'
DEFAULT_CONFIG_PATH = './configs/train_config_logreg.yml'
DEFAULT_DATASET_PATH = './data/heart.csv'
DEFAULT_OUTPUT_PATH = './data/prediction.csv'
APPLICATION_NAME = 'heart_training'

logger = logging.getLogger(APPLICATION_NAME)


def train_pipeline(params: TrainingPipelineParams):
    logger.info("start training pipeline")
    data = read_data(params.input_data_path)
    logger.info(f"data readed from {params.input_data_path}")
    train_df, val_df = split_to_train_val(data, params.splitting_params)
    logger.debug(f"data splitted; train_df size: {train_df.shape}, val_df size: {val_df.shape}")

    feature_extractor = FeaturesExtractor(params.feature_params)
    X_train = feature_extractor.fit_transform(train_df)
    X_val = feature_extractor.transform(val_df)
    y_train = extract_target(train_df, params.feature_params)
    y_val = extract_target(val_df, params.feature_params)

    logger.info("features and target extracted")
    logger.debug(
        f"X_train size: {X_train.shape}, y_train size: {y_train.shape}, "
        f"X_val size: {X_val.shape}, y_val size: {y_val.shape}"
    )

    model = train_model(X_train, y_train, params.train_params)
    logger.info(f"model {params.train_params.model_type} loaded")
    y_pred = predict_model(model, X_val)
    logger.debug(f"prediction done; y_pred size: {y_pred.shape}")
    metrics = evaluate_model(y_pred, y_val)
    logger.info(f"evaluation done; accuracy: {metrics['accuracy_score']}")

    path_to_model = save_model(model, params.output_model_path)
    logger.info(f"model saved to {path_to_model}")
    save_metrics(metrics, params.metric_path)
    logger.info(f"metrics saved to {params.metric_path}")

    return path_to_model, metrics


def predict_pipeline(
        dataset_path: str,
        output_path: str,
        params: TrainingPipelineParams,
    ):
    try:
        model = load_model(params.output_model_path)
    except:
        raise Exception(
            f"Cannot load model from {params.output_model_path}, "
            f"try to train model with model_type = {params.train_params.model_type}"
        )
    logger.info(f"loaded model from {params.output_model_path} for prediction")
    df = read_data(dataset_path)
    logger.info(f"data readed from {dataset_path}")
    logger.debug(f"data size: {df.shape}")
    X = FeaturesExtractor(params.feature_params).fit_transform(df)
    logger.debug(f"features extracted; X size: {X.shape}")
    preds = predict_model(model, X)
    logger.info(f"prediction done; prediction size: {preds.shape}")
    pd.DataFrame(preds, columns=['target']).to_csv(output_path, index=None)


def setup_parser(parser):
    """functionality to build argument parser"""
    subparsers = parser.add_subparsers(help="choose command")

    train_parser = subparsers.add_parser(
        "train",
        help='train models to predict heart disease',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    train_parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH,
        help="path to config to load training parameters from",
        dest="config_path", required=True,
    )
    train_parser.set_defaults(callback=callback_train)

    predict_parser = subparsers.add_parser(
        "predict",
        help='predict heart disease for given dataset',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    predict_parser.add_argument(
        "-d", "--dataset", default=DEFAULT_DATASET_PATH,
        help="path to dataset with features",
        required=True, dest="dataset_path",
    )
    predict_parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH,
        help="path to config to load training parameters from",
        dest="config_path", required=True,
    )
    predict_parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT_PATH,
        help="path for prediction to write to",
        dest="output_path", required=True,
    )
    predict_parser.set_defaults(callback=callback_predict)


def process_train(config_path):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params)


def process_predict(dataset_path, output_path, config_path):
    params = read_training_pipeline_params(config_path)
    predict_pipeline(dataset_path, output_path, params)


def callback_train(arguments):
    process_train(arguments.config_path)


def callback_predict(arguments):
    process_predict(
        arguments.dataset_path,
        arguments.output_path,
        arguments.config_path,
    )


def _setup_logging():
    handler = logging.StreamHandler(sys.stderr)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)


def setup_logging():
    """setting up the logging from yaml config file"""
    with open(DEFAULT_LOGGING_CONFIG_PATH) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


def main():
    """main code to run script with console interface"""
    parser = ArgumentParser(
        description='Script for training and predicting model to deal with the heart disease',
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    setup_parser(parser)
    arguments = parser.parse_args()
    setup_logging()
    arguments.callback(arguments)


if __name__ == "__main__":
    main()
