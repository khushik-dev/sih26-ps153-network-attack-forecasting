# System Architecture — SIH26153

## Overview
SIH26153 implements a temporal network-state transition model and K-step forward infiltration risk forecasting system designed for Security Operations Center (SOC) environments.

## Component Diagram

```
[ Raw Network Telemetry CSV ] (CIC-IDS / UNSW / NetFlow / Demo)
             │
             ▼
[ src/preprocessing.py ] ──► Column canonicalization, NaN/inf cleaning, StandardScaler
             │
             ▼
[ src/temporal_state.py ] ──► Chronological window aggregation (S_t vectors) & Sequence building [S(t-k)...S(t)]
             │
             ├─────────────────────────────────────────────┐
             ▼                                             ▼
[ src/world_model.py ]                         [ src/baseline.py ]
(PyTorch LSTM World Model)                     (Logistic Regression Baseline)
             │                                             │
             ▼                                             ▼
[ src/forecasting.py ]                         [ Model Evaluation Metrics ]
(K-Step Autoregressive Rollout)                (Precision, Recall, F1, AUC)
             │
             ├───────────────────┬─────────────────────────┐
             ▼                   ▼                         ▼
[ src/attack_mapping.py ]  [ src/explainability.py ]  [ app.py ]
(MITRE ATT&CK Stages)      (Perturbation Attribution)  (Streamlit SOC Dashboard)
```

## Core Modules
- `src/preprocessing.py`: Flow schema canonicalization and feature normalization.
- `src/temporal_state.py`: Sliding window state aggregation.
- `src/world_model.py`: PyTorch LSTM network predicting $\hat{S}_{t+1}$ and infiltration risk.
- `src/forecasting.py`: Autoregressive multi-step rollout.
- `src/attack_mapping.py`: Kill-chain heuristics and stage mapping.
- `src/explainability.py`: Deterministic perturbation impact scoring.
- `app.py`: Streamlit SOC interface.
