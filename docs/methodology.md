# Methodology — SIH26153

## 1. Problem Formulation
Traditional intrusion detection systems (IDS) evaluate individual traffic flows independently as static classification tasks:
$$\text{flow} \to \text{benign / malicious}$$

This prototype introduces a **temporal world-model framework** that models state transitions across rolling chronological windows:
$$P(S_{t+1} \mid S_t, S_{t-1}, \dots, S_{t-k})$$

## 2. State Vector Representation ($S_t$)
Each network state vector $S_t$ encapsulates statistical traffic behavior over a time window (e.g., 5 seconds):
- Flow volume and byte/packet totals
- Inter-arrival time (IAT) mean and variance
- TCP flag ratios (SYN, ACK, RST)
- Port diversity (unique source and destination ports)
- Retransmission and activity/idle metrics

## 3. World Model & K-Step Forecasting
An LSTM neural network processes temporal sequences $[S_{t-k}, \dots, S_t]$ to predict the subsequent state $\hat{S}_{t+1}$ and infiltration risk probability $r_{t+1} \in [0, 1]$. By feeding $\hat{S}_{t+1}$ back into the sequence buffer recursively, the system performs K-step forward simulation.

## 4. Explainability via Perturbation
To ensure transparency for SOC analysts, the explainability engine perturbs each feature dimension across the historical window and measures the resulting shift in predicted risk, producing ranked feature contribution scores without unstable dependencies.

## 5. Known Limitations
- **Heuristic Stage Mapping**: MITRE stage confidence scores in `attack_mapping.py` are hand-tuned heuristic weights, not statistically calibrated probabilities, and have not been validated against labeled multi-stage attack datasets.
- **Simplified State Space**: The network state vector $S_t$ abstracts flow telemetry into 15 dimensions to ensure CPU-practical inference, omitting granular application-layer payload introspection.
- **Synthetic Demo Data**: The default demonstration dataset is synthetically generated to illustrate chronological attack progression and should be substituted with real CIC-IDS or UNSW captures for production validation.
