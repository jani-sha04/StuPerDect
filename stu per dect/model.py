import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class StudentPerformancePredictor:
    def __init__(self):
        self.models = {
            'ridge': Ridge(alpha=1.0),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1)
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
    
    def train(self, X_train, y_train, X_val, y_val):
        """Train multiple models and select the best one."""
        best_score = float('-inf')
        
        for name, model in self.models.items():
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            val_score = model.score(X_val, y_val)
            
            if val_score > best_score:
                best_score = val_score
                self.best_model = model
                self.best_model_name = name
        
        # Calculate feature importance for the best model
        if self.best_model_name == 'random_forest':
            self.feature_importance = self.best_model.feature_importances_
    
    def predict(self, X):
        """Make predictions using the best model."""
        return self.best_model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model performance."""
        y_pred = self.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model to disk."""
        joblib.dump(self.best_model, filepath)
    
    @staticmethod
    def load_model(filepath):
        """Load a trained model from disk."""
        return joblib.load(filepath)
