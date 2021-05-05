from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_project.entities import SplittingParams


def read_data(filepath: str):
    data = pd.read_csv(filepath)
    return data


def split_to_train_val(
        data: pd.DataFrame, params: SplittingParams,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_data, val_data = train_test_split(
        data, test_size=params.val_size, random_state=params.random_state
    )
    return train_data, val_data
