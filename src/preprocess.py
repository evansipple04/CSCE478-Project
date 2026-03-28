import pandas as pd

def preprocess_weather_data(df):
    """
    Preprocess weather data.
    - Handle missing values
    - Remove duplicates
    - Convert data types if needed
    """
    df = df.drop_duplicates()
    df = df.fillna(df.mean(numeric_only=True))
    return df

def preprocess_rides_data(df):
    """
    Preprocess rides data.
    - Handle missing values
    - Remove duplicates
    - Drop irrelevant columns if needed
    """
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def preprocess_data(df_weather, df_rides):
    """
    Main preprocessing function.
    """
    df_weather = preprocess_weather_data(df_weather)
    df_rides = preprocess_rides_data(df_rides)
    return df_weather, df_rides
