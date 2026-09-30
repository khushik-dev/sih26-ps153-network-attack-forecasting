"""Tests for preprocessing and schema canonicalization."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import NetworkPreprocessor


def test_preprocessor_canonicalization():
    df = pd.DataFrame({
        'Destination Port': [80, 443],
        'Source Port': [1234, 5678],
        'Protocol': [6, 6],
        'Flow Duration': [100, 200],
        'Label': ['BENIGN', 'ATTACK']
    })
    
    preprocessor = NetworkPreprocessor()
    X_scaled = preprocessor.fit_transform(df)
    
    assert X_scaled.shape[0] == 2
    assert preprocessor.is_fitted is True
    assert preprocessor.clean_and_sanitize(df)['label'].tolist() == [0, 1]


def test_string_arrow_label_regression():
    # Test pandas string dtype / arrow dtype labels
    df = pd.DataFrame({
        'dst_port': [80, 443],
        'label': pd.Series(['BENIGN', 'DDoS'], dtype='string')
    })
    preprocessor = NetworkPreprocessor()
    df_clean = preprocessor.clean_and_sanitize(df)
    assert df_clean['label'].tolist() == [0, 1]

