from src.load_data import load_csv
from src.train_models import ModelTrainer
from src.preprocess import preprocess_data
from src.train_models import ModelTrainer
from pathlib import Path

def main():
    df_weather = load_csv(Path("data/weather.csv"))
    df_rides = load_csv(Path("data/cab_rides.csv"))
    print(f"data before preprocessing: {df_weather.shape[0]} weather rows, {df_rides.shape[0]} rides rows")
    df_weather, df_rides = preprocess_data(df_weather, df_rides)
    print(f"data after preprocessing: {df_weather.shape[0]} weather rows, {df_rides.shape[0]} rides rows")

if __name__ == "__main__":
    main()
    