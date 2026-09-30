# Presentation Outline — SIH26153 (5 Slides)

## Slide 1: Problem + Motivation
- **Title**: Moving Beyond Static Traffic Classification
- **Content**:
  - Traditional IDS detect attacks reactively at packet/flow level.
  - Advanced persistent threats (APTs) exhibit multi-stage progression over time (Recon → Lateral Movement → Exfil).
  - **Objective**: Temporal network-state transition modeling and K-step forward infiltration risk forecasting.

## Slide 2: Proposed Solution + Architecture
- **Title**: SIH26153 World Model Architecture
- **Content**:
  - Chronological time-window state aggregation ($S_t$).
  - PyTorch LSTM transition dynamics network: $P(S_{t+1} \mid S_t, \dots, S_{t-k})$.
  - Offline execution, zero external cloud dependencies, fully reproducible MVP.

## Slide 3: World Model + Forecasting
- **Title**: K-Step Autoregressive Simulation
- **Content**:
  - Recursive rollout predicting future network states.
  - Actionable risk trajectory scoring ($r_t \to r_{t+K}$).
  - Real-time visualization in Streamlit SOC dashboard.

## Slide 4: Results + Explainability
- **Title**: Transparent SOC Decision Support
- **Content**:
  - MITRE ATT&CK kill-chain stage mapping (distinguishing observed vs. forecasted stages).
  - Perturbation-based feature contribution analysis (top driving features).
  - Rigorous baseline comparison against Logistic Regression.

## Slide 5: Demo + Impact & Future Scope
- **Title**: Demonstration & Real-World Impact
- **Content**:
  - Live Streamlit SOC dashboard demonstration.
  - Proactive defense capability for national security & enterprise SOCs.
  - Future scope: PCAP streaming ingestion and distributed sensor telemetry.
