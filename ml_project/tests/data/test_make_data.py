import pandas as pd

from ml_project.entities import SplittingParams
from ml_project.data import read_data, split_to_train_val


def test_read_data(path_to_synthetic_data):
    data = read_data(path_to_synthetic_data)
    assert isinstance(data, pd.DataFrame), 'data is not a pd.DataFrame'
    assert data.shape[0] > 0 and data.shape[1] > 0, 'dataframe is empty'


def test_splitting_to_train_val(synthetic_dataset):
    data = synthetic_dataset
    assert isinstance(data, pd.DataFrame)
    val_size = .3
    params = SplittingParams(val_size=val_size, random_state=56)
    train, val = split_to_train_val(data, params)
    assert train.shape[0] + val.shape[0] == data.shape[0]
    assert val_size - .1 < val.shape[0] / data.shape[0] < val_size + .1
