"""Headless smoke test verifying core pipeline against demo CSV and synthetic string label CSV."""

import os
import torch
torch.set_num_threads(1)
import pandas as pd
import numpy as np
from src.preprocessing import NetworkPreprocessor
from src.temporal_state import TemporalStateAggregator
from src.world_model import NetworkWorldModel
from src.forecasting import KStepForecaster


def test_smoke_pipeline_demo_data():
    demo_path = "data/demo/demo_traffic.csv"
    if not os.path.exists(demo_path):
        from train import generate_demo_dataset
        generate_demo_dataset(demo_path)
        
    df = pd.read_csv(demo_path)
    preprocessor = NetworkPreprocessor()
    X_scaled = preprocessor.fit_transform(df)
    
    df_scaled = df.copy()
    df_scaled[preprocessor.feature_columns] = X_scaled
    
    aggregator = TemporalStateAggregator(window_seconds=5)
    state_df = aggregator.aggregate(df_scaled)
    X_seqs, y_next, y_risks = aggregator.build_sequences(state_df, seq_len=5)
    
    assert len(X_seqs) > 0
    
    # Load or instantiate world model
    model_path = "models/lstm_world_model.pt"
    if os.path.exists(model_path):
        model = NetworkWorldModel.load(model_path)
    else:
        model = NetworkWorldModel(input_dim=X_seqs.shape[2], hidden_dim=32, num_layers=1)
        
    forecaster = KStepForecaster(model)
    preds, risks, uncert = forecaster.forecast_trajectory(X_seqs[0], k_steps=3)
    assert len(preds) == 3


def test_smoke_pipeline_synthetic_string_labels():
    # Synthetic CSV with string labels ('BENIGN', 'ATTACK') and pyarrow-backed string dtypes
    df_synth = pd.DataFrame({
        'destination_port': [80, 443, 8080],
        'source_port': [1234, 5678, 9999],
        'protocol': [6, 6, 17],
        'flow_duration': [150, 300, 50],
        'tot_fwd_pkts': [4, 10, 1],
        'tot_bwd_pkts': [2, 8, 0],
        'totlen_fwd_pkts': [200, 1500, 40],
        'totlen_bwd_pkts': [100, 800, 0],
        'syn_flag_cnt': [1, 0, 1],
        'ack_flag_cnt': [1, 1, 0],
        'rst_flag_cnt': [0, 1, 0],
        'label': pd.Series(['BENIGN', 'ATTACK', 'BENIGN'], dtype='string')
    })
    
    preprocessor = NetworkPreprocessor()
    X_scaled = preprocessor.fit_transform(df_synth)
    assert X_scaled.shape[0] == 3
    assert preprocessor.clean_and_sanitize(df_synth)['label'].tolist() == [0, 1, 0]
