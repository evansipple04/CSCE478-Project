from src.load_data import load_csv, load_multiple_csvs
from src.train_models import ModelTrainer
from pathlib import Path

def main():
    df_weather = load_csv(Path("data/weather.csv"))
    df_rides = load_csv(Path("data/rides.csv"))