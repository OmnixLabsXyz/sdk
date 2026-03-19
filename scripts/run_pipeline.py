from pipelines.ingestion import load_data
from pipelines.preprocessing import preprocess
from models.trainer import train_model

def run():
    data = load_data("data/raw/sample.csv")
    clean_data = preprocess(data)

    clean_data.to_csv("data/processed/clean.csv", index=False)

    model = train_model("data/processed/clean.csv")
    print("Model trained successfully")

if __name__ == "__main__":
    run()
