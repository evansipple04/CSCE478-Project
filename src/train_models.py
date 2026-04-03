from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

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
    
    def train_model1(self, X_train, y_train):
        self.model1.fit(X_train, y_train)
        train_r2 = self.model1.score(X_train, y_train)
        print(f"Model 1 (Linear Regression) - R² Score: {train_r2}")

    def train_model2(self, X_train, y_train):
        self.model2.fit(X_train, y_train)
        train_r2 = self.model2.score(X_train, y_train)
        print(f"Model 2 (Random Forest) - R² Score: {train_r2}")
    
    def train_both(self, X_train, y_train):
        self.train_model1(X_train, y_train)
        self.train_model2(X_train, y_train)
    
    def predict_model1(self, X_test):
        return self.model1.predict(X_test)

    def predict_model2(self, X_test):
        return self.model2.predict(X_test)