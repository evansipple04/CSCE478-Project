from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np

def evaluate_models(y_test, pred1, pred2):
    """Compare both models with proper metrics."""
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
    print("\n" + "="*60)
    print(df_results.to_string(index=False))
    print("="*60)
    
    best_idx = df_results['R²'].idxmax()
    best = df_results.loc[best_idx]
    print(f"\n✓ BEST MODEL: {best['Model']}")
    print(f"  R² = {best['R²']} (explains {best['R²']*100:.1f}% of variance)")
    print(f"  RMSE = ${best['RMSE']:.2f} (avg error)")