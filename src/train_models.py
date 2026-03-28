from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

class ModelTrainer:
    def __init__(self):
        self.model1 = LinearRegression()
        self.model2 = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def train_model1(self, X_train, y_train):
        self.model1.fit(X_train, y_train)
        return self.model1

    def train_model2(self, X_train, y_train):
        self.model2.fit(X_train, y_train)
        return self.model2
    
    def predict(self, X_test):
        pred1 = self.model1.predict(X_test)
        pred2 = self.model2.predict(X_test)
        return pred1, pred2