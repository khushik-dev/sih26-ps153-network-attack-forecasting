"""Tests for temporal network state aggregation and sequence building."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import NetworkPreprocessor
from src.temporal_state import TemporalStateAggregator


def test_temporal_aggregation_and_sequences():
    df = pd.DataFrame({
        'timestamp': pd.date_range('2026-09-12 00:00:00', periods=20, freq='1s'),
        'src_port': np.random.randint(1000, 5000, 20),
        'dst_port': np.random.randint(1, 80, 20),
        'flow_duration': [100] * 20,
        'tot_fwd_pkts': [5] * 20,
        'totlen_fwd_pkts': [500] * 20,
        'syn_flag_cnt': [1] * 20,
        'label': [0] * 10 + [1] * 10
    })
    
    preprocessor = NetworkPreprocessor()
    df_clean = preprocessor.clean_and_sanitize(df)
    
    aggregator = TemporalStateAggregator(window_seconds=2)
    state_df = aggregator.aggregate(df_clean)
    
    assert len(state_df) > 0
    assert 'flow_count' in state_df.columns
    
    X_seqs, y_next, y_risks = aggregator.build_sequences(state_df, seq_len=3)
    assert len(X_seqs) == len(state_df) - 3
    assert X_seqs.shape[1] == 3
