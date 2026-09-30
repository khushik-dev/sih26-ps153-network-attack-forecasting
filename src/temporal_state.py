"""Temporal Network State Aggregator.
Aggregates individual flow records into chronological time windows (e.g. 1-second or 5-second bins)
to construct network state vectors S_t containing statistical aggregates and behavior indicators.
"""

import pandas as pd
import numpy as np


class TemporalStateAggregator:
    """Aggregates flow telemetry into chronological network state vectors."""
    
    def __init__(self, window_seconds: int = 5):
        self.window_seconds = window_seconds
        self.state_feature_names = [
            'flow_count',
            'bytes_total',
            'packets_total',
            'mean_duration',
            'mean_iat',
            'iat_variance',
            'syn_ratio',
            'ack_ratio',
            'rst_ratio',
            'unique_dst_ports',
            'unique_src_ports',
            'mean_pkt_length',
            'retransmission_rate',
            'active_mean',
            'idle_mean'
        ]

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate flow dataframe into time windows."""
        df = df.copy()
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(start='2026-09-12 00:00:00', periods=len(df), freq='1s')
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Set timestamp as index for resampling
        df_time = df.set_index('timestamp')
        
        # Resample by window_seconds
        rule = f"{self.window_seconds}s"
        
        agg_rules = {
            'flow_direction': 'count' if 'flow_duration' in df.columns else 'count', # placeholder count
        }
        
        # Build aggregated dataframe window by window
        windows = []
        grouped = df.groupby(pd.Grouper(key='timestamp', freq=rule))
        
        for name, group in grouped:
            if len(group) == 0:
                continue
                
            flow_count = len(group)
            fwd_bytes = group['totlen_fwd_pkts'].sum() if 'totlen_fwd_pkts' in group.columns else 0
            bwd_bytes = group['totlen_bwd_pkts'].sum() if 'totlen_bwd_pkts' in group.columns else 0
            bytes_total = fwd_bytes + bwd_bytes
            
            fwd_pkts = group['tot_fwd_pkts'].sum() if 'tot_fwd_pkts' in group.columns else 0
            bwd_pkts = group['tot_bwd_pkts'].sum() if 'tot_bwd_pkts' in group.columns else 0
            packets_total = fwd_pkts + bwd_pkts
            
            mean_duration = group['flow_duration'].mean() if 'flow_duration' in group.columns else 0.0
            
            fwd_iat = group['fwd_iat_mean'] if 'fwd_iat_mean' in group.columns else pd.Series([0.0])
            mean_iat = fwd_iat.mean()
            iat_variance = fwd_iat.var() if len(fwd_iat) > 1 else 0.0
            if pd.isna(iat_variance):
                iat_variance = 0.0
                
            syn_cnt = group['syn_flag_cnt'].sum() if 'syn_flag_cnt' in group.columns else 0
            ack_cnt = group['ack_flag_cnt'].sum() if 'ack_flag_cnt' in group.columns else 0
            rst_cnt = group['rst_flag_cnt'].sum() if 'rst_flag_cnt' in group.columns else 0
            
            total_flags = max(syn_cnt + ack_cnt + rst_cnt, 1)
            syn_ratio = syn_cnt / total_flags
            ack_ratio = ack_cnt / total_flags
            rst_ratio = rst_cnt / total_flags
            
            unique_dst = group['dst_port'].nunique() if 'dst_port' in group.columns else 1
            unique_src = group['src_port'].nunique() if 'src_port' in group.columns else 1
            
            mean_pkt_len = group['mean_pkt_len'].mean() if 'mean_pkt_len' in group.columns else 0.0
            
            # Retransmission proxy from RST/ACK ratio or explicit retransmissions
            retransmission_rate = min(rst_cnt / max(flow_count, 1), 1.0)
            
            active_mean = group['active_mean'].mean() if 'active_mean' in group.columns else 0.0
            idle_mean = group['idle_mean'].mean() if 'idle_mean' in group.columns else 0.0
            
            # Label determination: if any flow in window is malicious, window is malicious (1)
            window_label = 1 if 'label' in group.columns and group['label'].max() > 0 else 0
            
            windows.append({
                'window_start': name,
                'flow_count': float(flow_count),
                'bytes_total': float(bytes_total),
                'packets_total': float(packets_total),
                'mean_duration': float(mean_duration),
                'mean_iat': float(mean_iat),
                'iat_variance': float(iat_variance),
                'syn_ratio': float(syn_ratio),
                'ack_ratio': float(ack_ratio),
                'rst_ratio': float(rst_ratio),
                'unique_dst_ports': float(unique_dst),
                'unique_src_ports': float(unique_src),
                'mean_pkt_length': float(mean_pkt_len),
                'retransmission_rate': float(retransmission_rate),
                'active_mean': float(active_mean),
                'idle_mean': float(idle_mean),
                'label': int(window_label)
            })
            
        state_df = pd.DataFrame(windows)
        if state_df.empty:
            # Fallback dummy state if empty
            state_df = pd.DataFrame([{
                'window_start': pd.Timestamp.now(),
                **{k: 0.0 for k in self.state_feature_names},
                'label': 0
            }])
            
        return state_df

    def build_sequences(self, state_df: pd.DataFrame, seq_len: int = 5):
        """Construct sliding sequences [S(t-k)...S(t)] and target S(t+1) + risk label."""
        feature_vals = state_df[self.state_feature_names].values
        labels = state_df['label'].values
        
        X_seqs, y_next_states, y_risks = [], [], []
        
        if len(feature_vals) <= seq_len:
            # If not enough windows, pad or repeat
            padding = np.tile(feature_vals[0], (seq_len - len(feature_vals) + 1, 1))
            feature_vals = np.vstack([padding, feature_vals])
            labels = np.pad(labels, (len(padding)-1, 0), 'edge')
            
        for i in range(len(feature_vals) - seq_len):
            seq = feature_vals[i:i+seq_len]
            target_state = feature_vals[i+seq_len]
            # Risk is 1 if target window is malicious or has high anomaly indicators
            target_risk = float(labels[i+seq_len])
            
            X_seqs.append(seq)
            y_next_states.append(target_state)
            y_risks.append(target_risk)
            
        return np.array(X_seqs, dtype=np.float32), np.array(y_next_states, dtype=np.float32), np.array(y_risks, dtype=np.float32)
