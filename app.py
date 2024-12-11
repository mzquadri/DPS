from fastapi import FastAPI, Body, Query

from pydantic import BaseModel

import uvicorn

from typing import Dict
import pickle

import json


app = FastAPI()

class PredictedRequest(BaseModel):
    year : int
    month : int



pickle_model = open("Regressionmodel.pkl", "rb")
model = pickle.load(pickle_model)


@app.get("/")
def index():
    return{"Hi I am Zamin"}




@app.post("/deaths/")

def death_values(request:PredictedRequest):
    deathsin_month = model.predict([[request.year, request.month]])
    return {
        "prediction" : float(deathsin_month[0].round(0))
    }




if __name__ == "__main__":

    uvicorn.run(app, host = '127.0.0.1', port = 8000)

    