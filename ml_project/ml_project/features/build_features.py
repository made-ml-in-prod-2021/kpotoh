import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import TransformerMixin

from ml_project.entities import FeatureParams


class FeaturesExtractor(TransformerMixin):
    """ Смешать, но не взбалтывать """

    def __init__(self, feature_params: FeatureParams):
        self.params = feature_params

    def fit(self, df: pd.DataFrame):
        use_scaling = self.params.use_scaling_for_num_features
        self._transformer = self.build_transformer(use_scaling)
        self._transformer.fit(df)
        return self

    def transform(self, df: pd.DataFrame):
        return self._transformer.transform(df)

    @staticmethod
    def build_categorical_pipeline() -> Pipeline:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(
                missing_values=np.nan,
                strategy="most_frequent"
            )),
            ("ohe", OneHotEncoder()),
        ])
        return categorical_pipeline

    @staticmethod
    def build_numerical_pipeline(use_scaling: bool) -> Pipeline:
        operations = [("impute", SimpleImputer(
            missing_values=np.nan,
            strategy="mean")
        )]
        if use_scaling:
            operations.append(("scaler", StandardScaler()))
        num_pipeline = Pipeline(operations)
        return num_pipeline

    def build_transformer(self, use_scaling=False) -> ColumnTransformer:
        transformer = ColumnTransformer(
            [
                (
                    "categorical_pipeline",
                    self.build_categorical_pipeline(),
                    self.params.categorical_features,
                ),
                (
                    "numerical_pipeline",
                    self.build_numerical_pipeline(use_scaling),
                    self.params.numerical_features,
                ),
            ]
        )
        return transformer


def extract_target(df: pd.DataFrame, params: FeatureParams):
    target_col = params.target_col
    return df[target_col].values
