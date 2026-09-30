# SIH26153 — AI Based Network Attack Forecasting from Network Traffic Data

## Problem Statement & Objective
Traditional Intrusion Detection Systems (IDS) evaluate network flows as isolated, static classification tasks ($\text{flow} \to \text{benign / malicious}$). This reactive approach fails to anticipate multi-stage cyber attacks before complete compromise occurs.

**SIH26153** implements a temporal network-state transition model and K-step forward infiltration risk forecasting engine:
$$P(S_{t+1} \mid S_t, S_{t-1}, \dots, S_{t-k})$$

---

## Architecture & Technology Stack
- **Language**: Python 3.10+
- **Framework**: Streamlit (SOC Dashboard)
- **Deep Learning**: PyTorch (LSTM World Model for state transitions and risk forecasting)
- **Data & ML**: pandas, NumPy, scikit-learn (Logistic Regression baseline & preprocessing scalers)
- **Visualization**: Plotly

---

## Repository Structure
```
SIH26153/
├── app.py                      # Streamlit SOC Dashboard
├── train.py                    # Training CLI & Demo Data Generator
├── requirements.txt            # Python dependencies
├── AGENTS.md                   # Autonomous Engineering Directives
├── data/
│   ├── README.md               # Public dataset instructions (CIC-IDS, UNSW, CTU-13)
│   └── demo/                   # Bundled demo dataset
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Schema canonicalization & scaling
│   ├── feature_engineering.py  # Flow-level feature normalization
│   ├── temporal_state.py       # Rolling time-window aggregator (S_t)
│   ├── world_model.py          # PyTorch LSTM transition network
│   ├── forecasting.py          # K-step autoregressive rollout
│   ├── attack_mapping.py       # MITRE ATT&CK stage heuristics
│   ├── explainability.py       # Perturbation feature attribution
│   ├── baseline.py             # Logistic regression baseline
│   └── evaluation.py           # Precision, Recall, F1, ROC AUC, MSE
├── models/                     # Serialized model checkpoints & metadata
├── tests/                      # pytest test suite
└── docs/                       # Architecture, methodology, demo script, presentation
```

---

## Installation & Local Execution

1. **Clone and Setup Virtual Environment**:
   ```bash
   python3 -m venv .venv --system-site-packages
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Train Models & Generate Demo Data**:
   ```python
   python train.py
   ```

3. **Launch Streamlit SOC Dashboard**:
   ```bash
   streamlit run app.py
   ```

---

## Testing
Run the test suite using pytest:
```bash
python -m pytest
```

---

## Deployment Instructions
The application runs offline without external LLM APIs or cloud AI services.
- **Streamlit Community Cloud**: Push repository to GitHub, link `app.py` as the main entry point, and ensure `requirements.txt` is present.
- **Local SOC Deployment**: Run via systemd service or Docker container using the provided `requirements.txt`.

---

## Known Limitations & Engineering Trade-offs
1. **Heuristic Confidence**: MITRE stage confidence scores in `attack_mapping.py` are rule-based heuristic weights rather than statistically calibrated probabilities.
2. **State Abstraction**: $S_t$ uses 15 statistical aggregates per time window to maintain CPU efficiency and real-time Streamlit responsiveness.
3. **Synthetic Demo Mode**: Included demo telemetry is simulated to demonstrate multi-phase kill-chain progression offline.
