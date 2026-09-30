"""Training CLI script for SIH26153.
Generates demo data or ingests CSV traffic telemetry, builds temporal states,
trains the LSTM World Model and Logistic Regression baseline, and serializes artifacts.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.preprocessing import NetworkPreprocessor
from src.temporal_state import TemporalStateAggregator
from src.world_model import NetworkWorldModel
from src.baseline import LogisticRegressionBaseline


def generate_demo_dataset(filepath: str = "data/demo/demo_traffic.csv"):
    """Generate a realistic multi-phase demonstration dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    np.random.seed(42)
    n_rows = 3000
    timestamps = pd.date_range(start='2026-09-12 08:00:00', periods=n_rows, freq='50ms')
    
    phases = []
    # Phase 0: Normal enterprise traffic (0 - 600 rows)
    phases.extend([0] * 600)
    # Phase 1: Reconnaissance / Scanning (600 - 1200 rows)
    phases.extend([1] * 600)
    # Phase 2: Initial Access / Brute Force (1200 - 1800 rows)
    phases.extend([2] * 600)
    # Phase 3: Lateral Movement (1800 - 2400 rows)
    phases.extend([3] * 600)
    # Phase 4: C2 & Exfiltration (2400 - 3000 rows)
    phases.extend([4] * 600)
    
    data = []
    for i, phase in enumerate(phases):
        ts = timestamps[i]
        if phase == 0:
            src_port = np.random.randint(49152, 65535)
            dst_port = np.random.choice([80, 443, 53, 22])
            duration = np.random.exponential(1000)
            fwd_pkts = np.random.randint(2, 10)
            bwd_pkts = np.random.randint(2, 10)
            fwd_bytes = fwd_pkts * np.random.randint(100, 500)
            bwd_bytes = bwd_pkts * np.random.randint(200, 1000)
            syn = 1
            ack = 1
            rst = 0
            label = 0
        elif phase == 1:
            src_port = np.random.randint(40000, 50000)
            dst_port = np.random.randint(1, 1024) # Scanning ports
            duration = np.random.exponential(50)
            fwd_pkts = 1
            bwd_pkts = 0
            fwd_bytes = 40
            bwd_bytes = 0
            syn = 1
            ack = 0
            rst = 0
            label = 1
        elif phase == 2:
            src_port = np.random.randint(50000, 60000)
            dst_port = np.random.choice([22, 3389, 445]) # SSH, RDP, SMB
            duration = np.random.exponential(200)
            fwd_pkts = np.random.randint(10, 50)
            bwd_pkts = np.random.randint(5, 20)
            fwd_bytes = fwd_pkts * 60
            bwd_bytes = bwd_pkts * 50
            syn = 1
            ack = 1
            rst = 1
            label = 1
        elif phase == 3:
            src_port = np.random.randint(45000, 55000)
            dst_port = 445 # SMB lateral
            duration = np.random.exponential(500)
            fwd_pkts = np.random.randint(20, 100)
            bwd_pkts = np.random.randint(20, 100)
            fwd_bytes = fwd_pkts * 200
            bwd_bytes = bwd_pkts * 500
            syn = 1
            ack = 1
            rst = 0
            label = 1
        else:
            src_port = np.random.randint(50000, 55000)
            dst_port = 443 # C2 outbound HTTPS
            duration = np.random.exponential(2000)
            fwd_pkts = np.random.randint(50, 200)
            bwd_pkts = np.random.randint(10, 50)
            fwd_bytes = fwd_pkts * 1400 # Large outbound exfil
            bwd_bytes = bwd_pkts * 100
            syn = 0
            ack = 1
            rst = 0
            label = 1

        data.append({
            'timestamp': ts,
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': 6,
            'flow_duration': duration,
            'tot_fwd_pkts': fwd_pkts,
            'tot_bwd_pkts': bwd_pkts,
            'totlen_fwd_pkts': fwd_bytes,
            'totlen_bwd_pkts': bwd_bytes,
            'fwd_iat_mean': duration / max(fwd_pkts, 1),
            'bwd_iat_mean': duration / max(bwd_pkts, 1),
            'fwd_iat_std': duration * 0.1,
            'bwd_iat_std': duration * 0.1,
            'syn_flag_cnt': syn,
            'ack_flag_cnt': ack,
            'rst_flag_cnt': rst,
            'fin_flag_cnt': 0,
            'psh_flag_cnt': 1,
            'urg_flag_cnt': 0,
            'mean_pkt_len': (fwd_bytes + bwd_bytes) / max(fwd_pkts + bwd_pkts, 1),
            'pkt_len_std': 50.0,
            'down_up_ratio': bwd_bytes / max(fwd_bytes, 1),
            'init_win_bytes_forward': 8192,
            'active_mean': duration * 0.8,
            'idle_mean': duration * 0.2,
            'label': label
        })
        
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Generated demo dataset at {filepath} with {len(df)} flow records.")
    return filepath


def train_pipeline(data_path: str = "data/demo/demo_traffic.csv"):
    """Execute full training pipeline."""
    if not os.path.exists(data_path):
        data_path = generate_demo_dataset(data_path)
        
    print(f"Loading telemetry from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing and scaling features...")
    preprocessor = NetworkPreprocessor()
    X_scaled = preprocessor.fit_transform(df)
    
    # Re-attach scaled features to dataframe for temporal aggregation
    df_scaled = df.copy()
    df_scaled[preprocessor.feature_columns] = X_scaled
    
    print("Aggregating temporal network states (5s windows)...")
    aggregator = TemporalStateAggregator(window_seconds=5)
    state_df = aggregator.aggregate(df_scaled)
    
    print(f"Constructed {len(state_df)} temporal network states S_t.")
    
    seq_len = 5
    X_seqs, y_next, y_risks = aggregator.build_sequences(state_df, seq_len=seq_len)
    
    print(f"Built {len(X_seqs)} training sequences of length {seq_len}.")
    
    # Split train/test chronologically (80/20)
    split_idx = int(len(X_seqs) * 0.8)
    X_train, X_test = X_seqs[:split_idx], X_seqs[split_idx:]
    y_next_train, y_next_test = y_next[:split_idx], y_next[split_idx:]
    y_risk_train, y_risk_test = y_risks[:split_idx], y_risks[split_idx:]
    
    # Train LSTM World Model
    print("Training PyTorch LSTM World Model...")
    input_dim = X_seqs.shape[2]
    world_model = NetworkWorldModel(input_dim=input_dim, hidden_dim=64, num_layers=2)
    
    criterion_state = nn.MSELoss()
    criterion_risk = nn.BCELoss()
    optimizer = torch.optim.Adam(world_model.parameters(), lr=0.002)
    
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_next_train, dtype=torch.float32),
        torch.tensor(y_risk_train[:, np.newaxis], dtype=torch.float32)
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    world_model.train()
    epochs = 25
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y_state, batch_y_risk in dataloader:
            optimizer.zero_grad()
            pred_state, pred_risk = world_model(batch_x)
            
            loss_state = criterion_state(pred_state, batch_y_state)
            loss_risk = criterion_risk(pred_risk, batch_y_risk)
            loss = loss_state + loss_risk
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
            
    # Train Logistic Regression Baseline
    print("Training Logistic Regression Baseline...")
    baseline = LogisticRegressionBaseline()
    baseline.fit(X_train, y_risk_train)
    
    # Save all artifacts
    os.makedirs("models", exist_ok=True)
    world_model.save("models/lstm_world_model.pt")
    baseline.save("models/baseline_lr.joblib")
    preprocessor.save("models/scaler.joblib")
    
    metadata = {
        'seq_len': seq_len,
        'input_dim': input_dim,
        'feature_names': aggregator.state_feature_names,
        'window_seconds': 5
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    print("Training complete! Artifacts saved successfully under models/.")


if __name__ == '__main__':
    train_pipeline()
