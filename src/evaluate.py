from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # type: ignore
import pandas as pd
import numpy as np

def evaluate_models(y_test, pred1, pred2, y_train=None, train_pred1=None, train_pred2=None):
    """Compare both models with proper metrics.
    
    If training predictions are provided, also shows train vs test comparison
    to detect overfitting.
    """
    results = []
    
    # Linear Regression
    mse1 = mean_squared_error(y_test, pred1)
    rmse1 = np.sqrt(mse1)
    mae1 = mean_absolute_error(y_test, pred1)
    r2_1 = r2_score(y_test, pred1)
    
    results.append({
        'Model': 'Linear Regression',
        'MSE': round(mse1, 2),
        'RMSE': round(rmse1, 2),
        'MAE': round(mae1, 2),
        'R²': round(r2_1, 4)
    })
    
    # Random Forest
    mse2 = mean_squared_error(y_test, pred2)
    rmse2 = np.sqrt(mse2)
    mae2 = mean_absolute_error(y_test, pred2)
    r2_2 = r2_score(y_test, pred2)
    
    results.append({
        'Model': 'Random Forest',
        'MSE': round(mse2, 2),
        'RMSE': round(rmse2, 2),
        'MAE': round(mae2, 2),
        'R²': round(r2_2, 4)
    })
    
    df_results = pd.DataFrame(results)
    print("\nTEST SET RESULTS:")
    print("="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    # Overfitting detection
    if y_train is not None and train_pred1 is not None and train_pred2 is not None:
        train_results = []
        
        # Linear Regression on training data
        train_r2_1 = r2_score(y_train, train_pred1)
        train_results.append({
            'Model': 'Linear Regression',
            'Train R²': round(train_r2_1, 4),
            'Test R²': round(r2_1, 4),
            'Overfitting Gap': round(train_r2_1 - r2_1, 4)
        })
        
        # Random Forest on training data
        train_r2_2 = r2_score(y_train, train_pred2)
        train_results.append({
            'Model': 'Random Forest',
            'Train R²': round(train_r2_2, 4),
            'Test R²': round(r2_2, 4),
            'Overfitting Gap': round(train_r2_2 - r2_2, 4)
        })
        
        df_train = pd.DataFrame(train_results)
        print("\nOVERFITTING DETECTION (Train vs Test):")
        print("="*60)
        print(df_train.to_string(index=False))
        print("="*60)
        print("\nNote: Large gaps indicate overfitting. Gap > 0.1 is concerning.")
    
    best_idx = df_results['R²'].idxmax()
    best = df_results.loc[best_idx]
    print(f"\n✓ BEST MODEL: {best['Model']}")
    print(f"  R² = {best['R²']} (explains {best['R²']*100:.1f}% of variance)")
    print(f"  RMSE = ${best['RMSE']:.2f} (avg error)")