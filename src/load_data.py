import pandas as pd
from pathlib import Path

def load_csv(file_path):
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows from {file_path}")
    return df