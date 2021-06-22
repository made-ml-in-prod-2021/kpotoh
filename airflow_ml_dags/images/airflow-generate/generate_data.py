import os

import click
import numpy as np
import pandas as pd


def synthetic_dataset(nrows=100) -> pd.DataFrame:
    """ genereate synthetic dataset independently based on `heart.csv` """
    cat_features = {
        'sex': 2, 'cp': 4, 'fbs': 2, 'restecg': 3,
        'exang': 2, 'slope': 3, 'ca': 5, 'thal': 4,
    }
    num_features = {
        'age': (29, 77), 'trestbps': (94, 200), 'chol': (126, 564),
        'thalach': (71, 202), 'oldpeak': (0, 6.2),
    }
    target = {'target': 2}

    data = dict()
    for fea, nvars in cat_features.items():
        data[fea] = np.random.randint(0, nvars, nrows)
    for fea, (minv, maxv) in num_features.items():
        if fea == 'oldpeak':
            data[fea] = np.random.randint(
                minv * 10, (maxv + 1) * 10, nrows) / 10
        else:
            data[fea] = np.random.randint(minv, maxv + 1, nrows)
    for fea, nvars in target.items():
        data[fea] = np.random.randint(0, nvars, nrows)

    data = pd.DataFrame(data)
    return data


def write_dataset(df: pd.DataFrame, dir_path: str, target_col='target'):
    os.makedirs(dir_path, exist_ok=True)

    path_to_data = os.path.join(dir_path, 'data.csv')
    path_to_target = os.path.join(dir_path, 'target.csv')

    df.drop(target_col, axis=1).to_csv(path_to_data, index=None)
    df[target_col].to_csv(path_to_target, index=None)


@click.command("generate")
@click.argument("dir_path")
def generate(dir_path: str):
    df = synthetic_dataset()
    write_dataset(df, dir_path)


if __name__ == '__main__':
    generate()
