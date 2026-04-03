from src.load_data import load_csv
from src.train_models import ModelTrainer
from src.preprocess import preprocess_data
from src.evaluate import evaluate_models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

def main():
    print("="*70)
    print("UBER PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    
    # LOAD
    print("\n[1] LOADING DATA")
    df_weather = load_csv(Path("data/weather.csv"))
    df_rides = load_csv(Path("data/cab_rides.csv"))
    print(f"Before preprocessing: {df_weather.shape[0]} weather, {df_rides.shape[0]} rides")
    
    # PREPROCESS
    print("\n[2] PREPROCESSING")
    df_rides = preprocess_data(df_weather, df_rides)
    print(f"After preprocessing: {df_rides.shape[0]} rides with weather data merged")
    
    # PREPARE DATA
    print("\n[3] PREPARING DATA")
    
    # Encode categorical columns
    df_rides_encoded = df_rides.copy()
    categorical_cols = ['cab_type', 'destination', 'source', 'location']
    
    for col in categorical_cols:
        if col in df_rides_encoded.columns:
            le = LabelEncoder()
            df_rides_encoded[col] = le.fit_transform(df_rides_encoded[col].astype(str))
    
    # Drop target variable to create features
    X = df_rides_encoded.drop(['price'], axis=1, errors='ignore')
    y = df_rides_encoded['price']
    
    print(f"Features: {X.shape[1]} | Samples: {X.shape[0]}")
    print(f"Feature columns: {list(X.columns)}")
    
    # TRAIN/TEST SPLIT
    print("\n[4] SPLITTING DATA")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # SCALE
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train: {X_train_scaled.shape[0]} samples | Test: {X_test_scaled.shape[0]} samples")
    
    # TRAIN MODELS
    print("\n[5] TRAINING MODELS")
    trainer = ModelTrainer()
    trainer.train_both(X_train_scaled, y_train)
    
    # PREDICT
    print("\n[6] MAKING PREDICTIONS")
    pred1 = trainer.predict_model1(X_test_scaled)
    pred2 = trainer.predict_model2(X_test_scaled)
    print(f"Predictions made for {X_test_scaled.shape[0]} test samples")
    
    # EVALUATE
    print("\n[7] MODEL EVALUATION")
    # Get training predictions for overfitting detection
    train_pred1 = trainer.predict_model1(X_train_scaled)
    train_pred2 = trainer.predict_model2(X_train_scaled)
    evaluate_models(y_test, pred1, pred2, y_train, train_pred1, train_pred2)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
