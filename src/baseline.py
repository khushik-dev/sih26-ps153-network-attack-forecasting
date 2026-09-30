"""Baseline Logistic Regression Classifier.
Trains a static logistic regression model on window features for comparison against the temporal world model.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib


class LogisticRegressionBaseline:
    """Static classification baseline."""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit logistic regression model on window feature arrays."""
        # If X is 3D (batch, seq_len, dim), flatten to 2D (batch, seq_len * dim) or take last time step
        if len(X.shape) == 3:
            batch_size, seq_len, dim = X.shape
            X_flat = X[:, -1, :]  # Use latest window state
        else:
            X_flat = X
            
        self.model.fit(X_flat, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if len(X.shape) == 3:
            X_flat = X[:, -1, :]
        else:
            X_flat = X
        return self.model.predict(X_flat)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if len(X.shape) == 3:
            X_flat = X[:, -1, :]
        else:
            X_flat = X
        return self.model.predict_proba(X_flat)[:, 1]

    def save(self, filepath: str):
        joblib.dump(self.model, filepath)

    @staticmethod
    def load(filepath: str) -> 'LogisticRegressionBaseline':
        model = LogisticRegressionBaseline()
        model.model = joblib.load(filepath)
        model.is_fitted = True
        return model
