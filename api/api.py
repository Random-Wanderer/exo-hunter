import os
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from ExoHunter import formatter
from ExoHunter.params import DEFAULT_LEN
from ExoHunter.formatter import Formatter
import lightkurve as lk
from scipy.signal import savgol_filter

app = FastAPI()

class ItemList(BaseModel):
    instances: List[float]

class Kepid(BaseModel):
    kepid: int


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

formatter = Formatter()

@app.post("/predictcurve")
def predictcurve(inputdata: ItemList):
    extracted_list = inputdata.instances
    X_pred = np.array(extracted_list)
    if len(X_pred) > DEFAULT_LEN:
        X_pred = X_pred[:DEFAULT_LEN]
    elif len(X_pred) < DEFAULT_LEN:
        return {"prediction": 'Insufficient data for meaningful answer'}
    X_pred = savgol_filter(X_pred,window_length=251,polyorder=5,)
    X_pred = np.abs(np.fft.fft(X_pred,axis=0))
    X_pred = X_pred[:(len(X_pred) // 2)]
    norm = np.linalg.norm(X_pred)
    X_pred = X_pred/norm
    X_pred = X_pred.T
    X_pred = np.expand_dims(X_pred, axis=0)
    X_pred = np.expand_dims(X_pred, axis=-1)
    pipeline = joblib.load("model2.pkl")
    y_pred = pipeline.predict(X_pred)
    prediction = y_pred[0]
    if prediction == 0:
        result_prediction = 'This star is likely to NOT have exoplanet'
    else:
        result_prediction = 'This star is LIKELY to have exoplanet(s)'
    return {"prediction": result_prediction}


@app.post("/predictid")
def predictid(inputdata:Kepid):
    kepid = inputdata.kepid
    light_curve_data = lk.search_lightcurve(f'kplr{kepid}', mission="Kepler", quarter=[1,2,3,4,5])
    if len(light_curve_data) >= 5:
        light_curve_data = light_curve_data.download_all().stitch()
        light_curve_data = light_curve_data['flux'].value
        #light_curve_data_send = light_curve_data.tolist() [float(a) for a in maskedarray.flat()]
        X_pred = np.array(light_curve_data)
    else:
        return {"prediction": 'Sorry there is not such kepler id or we were unable to retrieve the data'}
    if len(X_pred) > DEFAULT_LEN:
        X_pred = X_pred[:DEFAULT_LEN]
    elif len(X_pred) < DEFAULT_LEN:
        return {"prediction": 'Insufficient data for meaningful answer'}
    X_pred = savgol_filter(X_pred,window_length=251,polyorder=5,)
    X_pred = np.abs(np.fft.fft(X_pred,axis=0))
    X_pred = X_pred[:(len(X_pred) // 2)]
    norm = np.linalg.norm(X_pred)
    X_pred = X_pred/norm
    X_pred = X_pred.T
    X_pred = np.expand_dims(X_pred, axis=0)
    X_pred = np.expand_dims(X_pred, axis=-1)
    pipeline = joblib.load("model2.pkl")
    y_pred = pipeline.predict(X_pred)
    prediction = y_pred[0]
    if prediction == 0:
        result_prediction = 'This star is likely to NOT have exoplanet'
    else:
        result_prediction = 'This star is LIKELY to have exoplanet(s)'
    directory_path = os.getcwd()
    print("My current directory is : " + directory_path)
    database = pd.read_csv('raw_data/keplerid_for_manim.csv')
    database = database[database['kepid'] == kepid]
    solar_mass = list(database['sun_mass (solar_mass)'].values)[0]
    solar_radius = list(database['sun_radius (solar_radii)'].values)[0]
    orbital_period = list(database['orbital_period (days)'].values)
    planet_star_dist = list(database['planet_star_dist (AU)'].values)
    planet_radius = list(database['planet_radius(to_Earth)'].values)
    result= {"prediction": result_prediction,
            "solar_mass": solar_mass,
            "solar_radius": solar_radius,
            "orbital_period": orbital_period,
            "planet_star_rad": planet_star_dist,
            "planet_radius": planet_radius
            #"light_curve_data": light_curve_data_send
            }
    return result
