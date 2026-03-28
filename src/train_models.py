from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.ensemble import RandomForestRegressor # type: ignore
from sklearn.metrics import mean_squared_error # type: ignore
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.model1 = LinearRegression()
        self.model2 = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.pred1 = None
        self.pred2 = None
        self.model1_trained = False
        self.model2_trained = False
    
    def train_model1(self, X_train, y_train):
        self.model1.fit(X_train, y_train)
        self.model1_trained = True

        train_r2 = self.model1.score(X_train, y_train)
        print(f"Model 1 (Linear Regression) - R² Score: {train_r2}")
        return self.model1

    def train_model2(self, X_train, y_train):
        self.model2.fit(X_train, y_train)
        self.model2_trained = True
        train_r2 = self.model2.score(X_train, y_train)
        print(f"Model 2 (Random Forest) - R² Score: {train_r2}")
        return self.model2
    
    def train_both(self, X_train, y_train):
        self.train_model1(X_train, y_train)
        self.train_model2(X_train, y_train)
    
    def predict_model1(self, X_test):
        self.pred1 = self.model1.predict(X_test)
        return self.pred1

    def predict_model2(self, X_test):
        self.pred2 = self.model2.predict(X_test)
        return self.pred2
    
    def get_feature_importance(self):
        if self.model2_trained:
            importance = self.model2.feature_importances_
            return importance
        else:
            print("Model 2 is not trained yet.")
            return None