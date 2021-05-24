import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from ml_project.entities import read_training_pipeline_params
from ml_project.data import read_data
from ml_project.features import FeaturesExtractor, extract_target

PATH_TO_BEST_MODEL_PARAMS = "./models/best_model_params.csv"
TRAIN_CONFIG_PATH = "./configs/train_config_trees.yml"


def best_model_params(clf, grid_params: dict, X, y):
    """ choose best classifier by grid search cross-validation """
    grid_search_output = GridSearchCV(
        clf, grid_params, scoring='roc_auc', n_jobs=-1, cv=5, verbose=1,
    )
    grid_search_output.fit(X, y)
    return grid_search_output.best_params_, grid_search_output.best_score_


def main():
    warnings.filterwarnings('ignore')
    params = read_training_pipeline_params(TRAIN_CONFIG_PATH)

    # load data
    data = read_data(params.input_data_path)
    X = FeaturesExtractor(params.feature_params).fit_transform(data)
    y = extract_target(data, params.feature_params)

    # determine models and its parameters
    logreg = LogisticRegression()
    logreg_grid_params = {
        "fit_intercept": [True, False],
        "max_iter": [100, 500, 1000],
        "C": np.logspace(-2, 1, 30),
    }
    trees = RandomForestClassifier()
    trees_grid_params = {
        "n_estimators": np.linspace(10, 100, 5).astype(int),
        "criterion": ["gini", "entropy"],
        "max_features": ["sqrt", "log2", .5, None],
        "min_samples_leaf": [1, 3, 5],
    }
    knn = KNeighborsClassifier()
    knn_grid_params = {
        "n_neighbors": [1, 3, 5, 7, 9, 11],
        "p": [1, 2, 3]
    }

    # unite model entities
    search_entities = [
        ('logreg', logreg, logreg_grid_params),
        ('trees', trees, trees_grid_params),
        ('knn', knn, knn_grid_params),
    ]

    # run grid search and write best params and scores to file
    best_params_path = PATH_TO_BEST_MODEL_PARAMS
    with open(best_params_path, 'w') as fout:
        fout.write(f"model_name\tbest_params\taccuracy\n")

        for mname, model, grid_params in search_entities:
            best_params, score = best_model_params(model, grid_params, X, y)
            print(
                f"{mname}, {best_params}, {score}",
                file=sys.stderr,
            )
            fout.write(f"{mname}\t{best_params}\t{score}\n")


if __name__ == "__main__":
    main()
