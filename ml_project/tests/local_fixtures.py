import pytest
import numpy as np
import pandas as pd


@pytest.fixture
def synthetic_dataset():
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
    data = synthetic_dataset
    dataset_fio.write(data)
    return dataset_fio
