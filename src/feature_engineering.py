"""Feature engineering pipeline for network flow telemetry.
Handles feature transformations, ratio calculations (SYN/ACK ratios, packet/byte ratios),
and normalization routines. Delegated to by preprocessing.py.
"""

import pandas as pd
import numpy as np


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived telemetry features such as bidirectional ratios and rate indicators."""
    df = df.copy()
    
    # Ensure base numerical columns exist
    for col in ['tot_fwd_pkts', 'tot_bwd_pkts', 'totlen_fwd_pkts', 'totlen_bwd_pkts', 'syn_flag_cnt', 'ack_flag_cnt', 'rst_flag_cnt', 'flow_duration']:
        if col not in df.columns:
            df[col] = 0.0
            
    # Compute downstream/upstream ratios safely
    fwd_bytes = df['totlen_fwd_pkts'].astype(float)
    bwd_bytes = df['totlen_bwd_pkts'].astype(float)
    df['down_up_ratio'] = np.where(fwd_bytes > 0, bwd_bytes / fwd_bytes, 0.0)
    
    # Compute total packet counts and flag ratios
    total_pkts = df['tot_fwd_pkts'].astype(float) + df['tot_bwd_pkts'].astype(float)
    total_flags = df['syn_flag_cnt'].astype(float) + df['ack_flag_cnt'].astype(float) + df['rst_flag_cnt'].astype(float)
    total_flags = np.where(total_flags == 0, 1.0, total_flags)
    
    df['syn_ratio'] = df['syn_flag_cnt'].astype(float) / total_flags
    df['ack_ratio'] = df['ack_flag_cnt'].astype(float) / total_flags
    df['rst_ratio'] = df['rst_flag_cnt'].astype(float) / total_flags
    
    return df
