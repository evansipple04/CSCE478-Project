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
    Preprocesses both datasets and merges them on common keys.
    """
    df_weather = preprocess_weather_data(df_weather)
    df_rides = preprocess_rides_data(df_rides)
    
    # Merge datasets on common columns
    # Identify common columns between the two dataframes
    common_cols = list(set(df_weather.columns) & set(df_rides.columns))
    
    if common_cols:
        # Merge on common columns
        df_merged = pd.merge(df_rides, df_weather, on=common_cols, how='left')
    else:
        # If no common columns, try merging on date-like columns
        date_cols_weather = [col for col in df_weather.columns if 'date' in col.lower() or 'time' in col.lower()]
        date_cols_rides = [col for col in df_rides.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if date_cols_weather and date_cols_rides:
            merge_key_weather = date_cols_weather[0]
            merge_key_rides = date_cols_rides[0]
            df_merged = pd.merge(df_rides, df_weather, left_on=merge_key_rides, right_on=merge_key_weather, how='left')
        else:
            # If no date columns found, just concatenate horizontally
            df_merged = pd.concat([df_rides, df_weather], axis=1)
    
    return df_merged, df_merged
