# 2-Minute Demo Script — SIH26153

## Scene 1: Introduction & Demo Mode Launch (0:00 - 0:25)
- **Action**: Launch Streamlit dashboard (`streamlit run app.py`).
- **Narrative**: "Welcome to our SIH 2026 cybersecurity prototype, SIH26153. Traditional intrusion detection systems rely on static packet classification. Our system introduces a learned temporal world model that forecasts multi-step infiltration risk before compromise occurs."

## Scene 2: Overview & Current Risk Inspection (0:25 - 0:50)
- **Action**: Navigate to **1. Overview**.
- **Narrative**: "Here on the SOC executive dashboard, we see the current network risk score, forecasted risk trajectory over a K-step horizon, and the predicted MITRE ATT&CK stage."

## Scene 3: K-Step Forward Forecasting & Attack Progression (0:50 - 1:20)
- **Action**: Navigate to **3. Forecast** and **4. Attack Progression**.
- **Narrative**: "By performing recursive autoregressive rollout, the LSTM world model predicts future states. We can observe the transition from Reconnaissance to Initial Access and Lateral Movement across future windows."

## Scene 4: Explainability & Baseline Comparison (1:20 - 2:00)
- **Action**: Navigate to **5. Explainability** and **6. Model Performance**.
- **Narrative**: "SOC analysts need to know *why* a forecast was produced. Our perturbation engine highlights driving features like SYN ratios and port diversity. Finally, our evaluation tab contrasts the temporal world model against a traditional logistic regression baseline, demonstrating superior early warning capabilities."
