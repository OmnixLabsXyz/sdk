from fastapi import FastAPI
import pandas as pd
from models.model import OmnixModel

app = FastAPI()
model = OmnixModel()

@app.get("/")
def read_root():
    return {"message": "OmnixLabs API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame(data)
    prediction = model.predict(df)
    return {"prediction": prediction.tolist()}
