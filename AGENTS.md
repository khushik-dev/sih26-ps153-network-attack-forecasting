# AGENTS.md — SIH26153 Project Conventions & Architecture Notes

## Project Overview
- **Project ID**: SIH26153
- **Title**: AI Based Network Attack Forecasting from Network Traffic Data
- **Objective**: Temporal network-state transition modeling and K-step forward infiltration risk forecasting.
- **Target Audience**: SIH evaluation jury, SOC analysts, network defense engineers.

---

## Agent Operational Principles

1. **Authenticity & Technical Integrity**:
   - Never fabricate benchmark results, detection metrics, or dataset claims.
   - If synthetic or simulated network telemetry is used, explicitly label it as **DEMO DATA**.
   - Do not claim a static binary classifier is a "world model". The model must explicitly approximate $P(S_{t+1} \mid S_t, \dots, S_{t-k})$ and support recursive forward rollout.

2. **System Boundaries**:
   - Pure offline execution: zero external cloud AI dependencies, LLM API calls, or telemetry phoning home.
   - Minimal attack surface: uploaded network telemetry CSVs must be validated, sanitized, and handled strictly in read-only analysis passes.
   - No arbitrary command execution from user or data inputs.

3. **Code & Engineering Standards**:
   - Python 3 with clean type hints and defensive bounds checking.
   - Numerical stability: handle divide-by-zero, infinite values, NaN imputation, and variance clamping.
   - Reproducibility: set random seeds across PyTorch, NumPy, and Scikit-Learn.
   - Visual SOC styling: restrained palette (dark mode / SOC theme), clear data hierarchy, actionable security metrics.

4. **Component Map**:
   - `src/preprocessing.py`: Flow schema canonicalization, sanitization, and handling heterogeneous IDS formats.
   - `src/feature_engineering.py`: Port, flag, IAT, and byte/packet telemetry normalization.
   - `src/temporal_state.py`: Rolling chronological window aggregation into network state vector $S_t$.
   - `src/world_model.py`: PyTorch LSTM transition dynamics network predicting next state $\hat{S}_{t+1}$ and infiltration risk.
   - `src/forecasting.py`: Autoregressive K-step forward simulation with prediction uncertainty & risk trajectory.
   - `src/attack_mapping.py`: MITRE ATT&CK kill-chain heuristics distinguishing observed vs forecasted stages.
   - `src/explainability.py`: Deterministic perturbation and gradient sensitivity analysis identifying driving features.
   - `src/baseline.py`: Static Logistic Regression classifier for rigorous baseline comparison.
   - `src/evaluation.py`: Precision, Recall, F1, FPR, ROC AUC, and state transition MSE.
   - `app.py`: Streamlit SOC dashboard.
   - `train.py`: Self-contained training & serialization CLI.
