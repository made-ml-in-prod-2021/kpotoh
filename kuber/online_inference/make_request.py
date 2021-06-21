import numpy as np
import pandas as pd
import requests

# adress = "http://0.0.0.0:8000"  # if run docker or scriptonline_inference locally
adress = "http://localhost:41935"  # if run kubernetes


def synthetic_dataset(nrows=50):
    """ genereate synthetic dataset independently based on true one """
    cat_features = {
        'sex': 2, 'cp': 4, 'fbs': 2, 'restecg': 3, 
        'exang':2, 'slope':3, 'ca': 5, 'thal': 4,
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
            data[fea] = np.random.randint(minv*10, (maxv + 1)*10, nrows) / 10
        else:   
            data[fea] = np.random.randint(minv, maxv + 1, nrows)
    for fea, nvars in target.items():
        data[fea] = np.random.randint(0, nvars, nrows)

    data = pd.DataFrame(data)
    return data


if __name__ == "__main__":
    data = synthetic_dataset()
    data = data.drop('target', axis=1)
    request_features = list(data.columns)
    for i in range(500):
        request_data = [
            x.item() if isinstance(x, np.generic) else x for x in data.iloc[i % len(data)].tolist()
        ]
        print(request_data, len(request_data))
        response = requests.get(
            f"{adress}/predict/",
            json={"data": [request_data], "features": request_features},
        )
        print(response.status_code)
        print(response.json())