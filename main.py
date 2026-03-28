from src.load_data import load_csv, load_multiple_csvs
from src.train_models import ModelTrainer
from src.preprocess import preprocess_data
from pathlib import Path

def main():
    df_weather = load_csv(Path("data/weather.csv"))
    df_rides = load_csv(Path("data/rides.csv"))

    df_weather, df_rides = preprocess_data(df_weather, df_rides)