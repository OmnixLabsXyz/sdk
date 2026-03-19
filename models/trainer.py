import pandas as pd
from models.model import OmnixModel

def train_model(data_path):
    data = pd.read_csv(data_path)

    X = data.drop("label", axis=1)
    y = data["label"]

    model = OmnixModel()
    model.train(X, y)

    return model
