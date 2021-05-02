from ml_project.entities import TrainingPipelineParams
from ml_project.data import read_data, split_to_train_val
from ml_project.features import FeaturesExtractor, extract_target
from ml_project.models import (
    train_model, predict_model, evaluate_model, save_model, save_metrics
)

def train_pipeline(params: TrainingPipelineParams):
    data = read_data(params.input_data_path)
    train_df, val_df = split_to_train_val(data, params.splitting_params)
    
    feature_extractor = FeaturesExtractor(params.feature_params)
    X_train = feature_extractor.fit_transform(train_df)
    X_val = feature_extractor.transform(val_df)
    y_train = extract_target(train_df, params.feature_params)
    y_val = extract_target(val_df, params.feature_params)

    model = train_model(X_train, y_train, params.train_params)
    y_pred = predict_model(model, X_val)
    metrics = evaluate_model(y_pred, y_val)

    path_to_model = save_model(model, params.output_model_path)
    save_metrics(metrics, params.metric_path)

    return path_to_model, metrics