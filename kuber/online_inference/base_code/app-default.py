import logging
import os
import pickle
from random import randint
from time import sleep
from typing import List, Union, Optional, Tuple

import numpy as np
import pandas as pd
import uvicorn
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger("uvicorn")

N_FEATURES = 13


def load_object(path: str) -> RandomForestClassifier:
    with open(path, "rb") as f:
        return pickle.load(f)


class InputStruct(BaseModel):
    data: List[
        conlist(Union[int, float, None], min_items=N_FEATURES, max_items=N_FEATURES)
    ]
    features: conlist(str, min_items=N_FEATURES, max_items=N_FEATURES)


class OutputStruct(BaseModel):
    predicted_class: int


def make_predict(
    data: List, features: List[str], model: RandomForestClassifier,
) -> List[OutputStruct]:
    data = pd.DataFrame(data, columns=features)
    predicts = model.predict(data)
    logger.debug(f"data: {data};\n prediction: {predicts}")
    return [OutputStruct(predicted_class=int(pred)) for pred in predicts]


app = FastAPI()
app.model: RandomForestClassifier = None


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    model_path = os.getenv("PATH_TO_MODEL", 'online_inference/model.pkl')
    if model_path is None:
        err = f"PATH_TO_MODEL is {model_path}"
        logger.error(err)
        raise RuntimeError(err)

    app.model = load_object(model_path)
    logger.info(f"model loaded from {model_path}")


@app.get("/healz")
def health() -> bool:
    return not (app.model is None)


@app.get("/predict/", response_model=List[OutputStruct])
def predict(request: InputStruct):
    return make_predict(request.data, request.features, app.model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
