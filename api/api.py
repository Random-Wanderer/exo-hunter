from os import name
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

class ItemList(BaseModel):
    instances: List[float]


@app.get("/")
def root():
    return {"greeting": "hello"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/predict")
def predict(inputdata: ItemList):
    extracted_list = inputdata.instances
    X_pred = np.array(extracted_list)
    np.expand_dims(X_pred, axis=0)
    pipeline = joblib.load("rnn_model.joblib")
    y_pred = pipeline.predict(X_pred)
    prediction = y_pred[0]
    if prediction == 0:
        result_prediction = 'This star is likely to NOT have exoplanet'
    result_prediction = 'This star is LIKELY to have exoplanet(s)'
    return {"prediction": result_prediction}
