"""Tests for baseline logistic regression model."""

import pytest
import numpy as np
from src.baseline import LogisticRegressionBaseline


def test_logistic_regression_baseline():
    X = np.random.randn(50, 5, 15)
    y = np.random.choice([0, 1], size=50)
    
    baseline = LogisticRegressionBaseline()
    baseline.fit(X, y)
    
    preds = baseline.predict(X)
    probas = baseline.predict_proba(X)
    
    assert len(preds) == 50
    assert len(probas) == 50
    assert all(p in [0, 1] for p in preds)
