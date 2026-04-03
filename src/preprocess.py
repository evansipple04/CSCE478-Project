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
    # Drop rows with missing prices (target variable), but keep other NaN values for now
    df = df.dropna(subset=['price'])
    # Drop irrelevant columns
    df = df.drop(['id', 'product_id', 'name'], axis=1, errors='ignore')
    return df

def preprocess_data(df_weather, df_rides):
    """
    Main preprocessing function.
    Preprocesses both datasets and merges them on common keys.
    """
    df_weather = preprocess_weather_data(df_weather)
    df_rides = preprocess_rides_data(df_rides)
    
    # Convert rides timestamp from milliseconds to seconds for matching with weather
    df_rides['time_stamp'] = df_rides['time_stamp'] // 1000
    
    # Merge rides with weather on time_stamp and location
    # Weather has 'location' and rides doesn't, so merge on time_stamp only
    df_merged = pd.merge(df_rides, df_weather, on='time_stamp', how='left')
    
    # Drop any remaining NaN values after merge
    df_merged = df_merged.dropna()
    
    return df_merged
