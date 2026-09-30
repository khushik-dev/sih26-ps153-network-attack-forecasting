"""Preprocessing and schema canonicalization for network traffic flows.
Handles heterogeneous CSV headers (CIC-IDS, UNSW-NB15, generic flow exports),
missing values, infinite values, type casting, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

from src.feature_engineering import compute_derived_features
CANONICAL_COLUMNS = [
    'timestamp',          # Epoch or datetime for chronological sorting
    'src_port',           # Source port
    'dst_port',           # Destination port
    'protocol',           # Protocol (TCP=6, UDP=17, ICMP=1, etc.)
    'flow_duration',      # Duration of flow in microseconds or milliseconds
    'tot_fwd_pkts',       # Total packets in forward direction
    'tot_bwd_pkts',       # Total packets in backward direction
    'totlen_fwd_pkts',    # Total bytes in forward packets
    'totlen_bwd_pkts',    # Total bytes in backward packets
    'fwd_iat_mean',       # Mean inter-arrival time of forward packets
    'bwd_iat_mean',       # Mean inter-arrival time of backward packets
    'fwd_iat_std',        # Std deviation of forward IAT
    'bwd_iat_std',        # Std deviation of backward IAT
    'syn_flag_cnt',       # Count of SYN flags
    'ack_flag_cnt',       # Count of ACK flags
    'rst_flag_cnt',       # Count of RST flags
    'fin_flag_cnt',       # Count of FIN flags
    'psh_flag_cnt',       # Count of PSH flags
    'urg_flag_cnt',       # Count of URG flags
    'mean_pkt_len',       # Mean packet length
    'pkt_len_std',        # Packet length standard deviation
    'down_up_ratio',      # Download to upload byte ratio
    'subflow_fwd_byts',   # Average bytes in a subflow forward
    'init_win_bytes_forward', # Initial TCP window size forward
    'active_mean',        # Mean time flow was active before becoming idle
    'idle_mean',          # Mean time flow was idle
    'label'               # Binary or categorical label (0 = Benign, 1 = Malicious / Attack)
]

# Mapping dictionary for known dataset column name variations (CIC-IDS, UNSW, etc.)
COLUMN_ALIASES = {
    'Destination Port': 'dst_port',
    'Source Port': 'src_port',
    'Protocol': 'protocol',
    'Flow Duration': 'flow_duration',
    'Total Fwd Packets': 'tot_fwd_pkts',
    'Total Backward Packets': 'tot_bwd_pkts',
    'Total Length of Fwd Packets': 'totlen_fwd_pkts',
    'Total Length of Bwd Packets': 'totlen_bwd_pkts',
    'Fwd IAT Mean': 'fwd_iat_mean',
    'Bwd IAT Mean': 'bwd_iat_mean',
    'Fwd IAT Std': 'fwd_iat_std',
    'Bwd IAT Std': 'bwd_iat_std',
    'SYN Flag Count': 'syn_flag_cnt',
    'ACK Flag Count': 'ack_flag_cnt',
    'RST Flag Count': 'rst_flag_cnt',
    'FIN Flag Count': 'fin_flag_cnt',
    'PSH Flag Count': 'psh_flag_cnt',
    'URG Flag Count': 'urg_flag_cnt',
    'Average Packet Size': 'mean_pkt_len',
    'Packet Length Std': 'pkt_len_std',
    'Down/Up Ratio': 'down_up_ratio',
    'Init_Win_bytes_forward': 'init_win_bytes_forward',
    'Active Mean': 'active_mean',
    'Idle Mean': 'idle_mean',
    'Label': 'label',
    'Timestamp': 'timestamp',
    # Lowercase variations
    'destination port': 'dst_port',
    'source port': 'src_port',
    'protocol': 'protocol',
    'flow duration': 'flow_duration',
    'label': 'label'
}


class NetworkPreprocessor:
    """Robust preprocessor for network telemetry CSVs with imputation, cleaning, and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = [c for c in CANONICAL_COLUMNS if c not in ['timestamp', 'label']]
        self.is_fitted = False

    def canonicalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns to canonical names using aliases and strip whitespace."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        rename_map = {}
        for col in df.columns:
            if col in COLUMN_ALIASES:
                rename_map[col] = COLUMN_ALIASES[col]
            elif col.lower().replace(' ', '_') in [c.lower() for c in CANONICAL_COLUMNS]:
                rename_map[col] = col.lower().replace(' ', '_')
                
        df = df.rename(columns=rename_map)
        
        # If timestamp is missing, generate synthetic chronological sequence
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2026-09-12 00:00:00', periods=len(df), freq='1s')
        else:
            ts_parsed = pd.to_datetime(df['timestamp'], errors='coerce')
            fallback_ts = pd.date_range(start='2026-09-12 00:00:00', periods=len(df), freq='1s')
            df['timestamp'] = ts_parsed.combine_first(pd.Series(fallback_ts))
            
        # Ensure all canonical feature columns exist; if missing, fill with 0 or sensible defaults
        for col in CANONICAL_COLUMNS:
            if col not in df.columns:
                if col == 'label':
                    df[col] = 0
                elif col == 'protocol':
                    df[col] = 6  # Default TCP
                else:
                    df[col] = 0.0
                    
        return df

    def clean_and_sanitize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle NaNs, infinite values, variance clamping, and type conversions."""
        df = self.canonicalize_columns(df)
        df = compute_derived_features(df)
        
        # Convert feature columns to numeric
        for col in self.feature_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Replace infinities with NaN
        df[self.feature_columns] = df[self.feature_columns].replace([np.inf, -np.inf], np.nan)
        
        # Impute NaNs with median or zero
        df[self.feature_columns] = df[self.feature_columns].fillna(0.0)
        
        # Clean label column (convert text labels like 'BENIGN'/'ATTACK' or 'DDoS' to 0/1)
        if 'label' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['label']):
                df['label'] = df['label'].astype(str).str.upper().apply(
                    lambda x: 0 if x in ['BENIGN', 'NORMAL', '0', 'FALSE'] else 1
                )
            else:
                df['label'] = df['label'].fillna(0).astype(int).apply(lambda x: 1 if x > 0 else 0)
        else:
            df['label'] = 0
            
        return df

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Clean dataframe, fit scaler on feature columns, and return normalized array."""
        df_clean = self.clean_and_sanitize(df)
        X = df_clean[self.feature_columns].values
        
        # Numerical stability check
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_scaled = self.scaler.fit_transform(X)
        self.is_fitted = True
        return X_scaled

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Clean dataframe and apply fitted scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform().")
        df_clean = self.clean_and_sanitize(df)
        X = df_clean[self.feature_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return self.scaler.transform(X)

    def save(self, filepath: str):
        """Save preprocessor state."""
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> 'NetworkPreprocessor':
        """Load preprocessor state."""
        return joblib.load(filepath)
