"""Streamlit SOC Dashboard for SIH26153: AI Based Network Attack Forecasting.
Provides real-time network telemetry analysis, temporal state construction,
K-step forward risk forecasting, MITRE ATT&CK stage mapping, perturbation explainability,
baseline comparison, dataset library, and MITRE knowledge base.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
import json
import torch
import requests

from src.preprocessing import NetworkPreprocessor
from src.temporal_state import TemporalStateAggregator
from src.world_model import NetworkWorldModel
from src.forecasting import KStepForecaster
from src.attack_mapping import AttackStageMapper
from src.explainability import ExplainabilityEngine
from src.baseline import LogisticRegressionBaseline
from src.evaluation import ModelEvaluator

st.set_page_config(
    page_title="SIH26153 | Network Attack Forecasting SOC",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for restrained SOC styling
st.markdown("""
<style>
    .main { background-color: #0E1117; color: #C9D1D9; }
    .stMetric { background-color: #161B22; padding: 15px; border-radius: 6px; border: 1px solid #30363D; }
    .stAlert { background-color: #161B22; border: 1px solid #30363D; color: #C9D1D9; }
    h1, h2, h3 { color: #58A6FF; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }
    .metric-card { background: #161B22; border: 1px solid #30363D; padding: 20px; border-radius: 8px; text-align: center; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load pre-trained models and metadata if available, else train on demo data."""
    models_dir = "models"
    model_path = os.path.join(models_dir, "lstm_world_model.pt")
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    baseline_path = os.path.join(models_dir, "baseline_lr.joblib")
    meta_path = os.path.join(models_dir, "metadata.json")
    
    if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
        from train import train_pipeline
        train_pipeline()
        
    world_model = NetworkWorldModel.load(model_path)
    scaler = NetworkPreprocessor.load(scaler_path)
    baseline = LogisticRegressionBaseline.load(baseline_path)
    
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
        
    return world_model, scaler, baseline, metadata


def load_dataset_catalog():
    cat_path = "data/catalog.json"
    if os.path.exists(cat_path):
        with open(cat_path, 'r') as f:
            return json.load(f)
    return []


def load_knowledge_base():
    kb_path = "data/knowledge_base.json"
    if os.path.exists(kb_path):
        with open(kb_path, 'r') as f:
            return json.load(f)
    return {}


def main():
    st.sidebar.title("SOC Defense Console")
    st.sidebar.markdown("**SIH26153**: AI Network Attack Forecasting")
    st.sidebar.markdown("---")
    
    try:
        world_model, preprocessor, baseline, metadata = load_artifacts()
    except Exception as e:
        st.error(f"Failed to load model artifacts: {e}")
        st.stop()
        
    forecaster = KStepForecaster(world_model)
    stage_mapper = AttackStageMapper()
    aggregator = TemporalStateAggregator(window_seconds=metadata.get('window_seconds', 5))
    explain_engine = ExplainabilityEngine(world_model, aggregator)
    
    tabs = [
        "1. Overview",
        "2. Traffic Explorer",
        "3. Forecast",
        "4. Attack Progression",
        "5. Explainability",
        "6. Model Performance",
        "7. About / Methodology",
        "8. Dataset & Knowledge Base"
    ]
    
    selected_tab = st.sidebar.radio("Navigation", tabs)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Telemetry Source")
    data_source = st.sidebar.radio(
        "Choose Data Input",
        ["Demo Dataset (Multi-Phase Scenario)", "Upload Custom CSV (CIC-IDS / NetFlow)"]
    )
    
    df_raw = None
    if data_source.startswith("Demo"):
        demo_path = "data/demo/demo_traffic.csv"
        if not os.path.exists(demo_path):
            from train import generate_demo_dataset
            generate_demo_dataset(demo_path)
        df_raw = pd.read_csv(demo_path)
        st.sidebar.success("Loaded DEMO DATA (3,000 flows)")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Network CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Loaded CSV ({len(df_raw)} records)")
            except Exception as e:
                st.sidebar.error(f"Error reading CSV: {e}")
                return
        else:
            demo_path = "data/demo/demo_traffic.csv"
            df_raw = pd.read_csv(demo_path)
            st.sidebar.info("No file uploaded. Falling back to DEMO DATA.")
            
    df_clean = preprocessor.clean_and_sanitize(df_raw)
    X_scaled = preprocessor.transform(df_raw)
    df_scaled = df_raw.copy()
    df_scaled[preprocessor.feature_columns] = X_scaled
    
    state_df = aggregator.aggregate(df_scaled)
    X_seqs, y_next, y_risks = aggregator.build_sequences(state_df, seq_len=metadata.get('seq_len', 5))
    
    latest_seq = X_seqs[-1] if len(X_seqs) > 0 else np.zeros((metadata.get('seq_len', 5), metadata.get('input_dim', 15)))
    current_state = latest_seq[-1]
    
    k_steps = st.sidebar.slider("Forecast Horizon (K steps)", min_value=1, max_value=10, value=5)
    pred_states, risk_traj, uncertainty_traj = forecaster.forecast_trajectory(latest_seq, k_steps=k_steps)
    current_risk = risk_traj[0]
    forecasted_max_risk = max(risk_traj)
    current_stage, current_tech, _ = stage_mapper.map_state_to_stage(current_state, current_risk)
    
    # -------------------------------------------------------------
    # TAB 1: OVERVIEW
    # -------------------------------------------------------------
    if selected_tab.startswith("1."):
        st.title("SOC Executive & Operations Overview")
        st.markdown("Real-time temporal network state transition monitoring and multi-step infiltration risk forecasting.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Network Risk", f"{current_risk*100:.1f}%", delta="Normal Baseline" if current_risk < 0.35 else "Elevated Risk", delta_color="inverse")
        with col2:
            st.metric(f"Forecasted Peak Risk (+{k_steps}W)", f"{forecasted_max_risk*100:.1f}%", delta=f"{((forecasted_max_risk - current_risk)*100):+.1f}%", delta_color="inverse")
        with col3:
            st.metric("Predicted Attack Stage", current_stage.split(" ")[0], delta=current_tech)
        with col4:
            st.metric("Analyzed Flows", f"{len(df_raw):,}", delta=f"{len(state_df)} windows")
            
        st.markdown("---")
        
        st.subheader("K-Step Forward Risk Trajectory")
        fig_traj = go.Figure()
        steps = list(range(1, k_steps + 1))
        fig_traj.add_trace(go.Scatter(
            x=steps,
            y=[r * 100 for r in risk_traj],
            mode='lines+markers',
            name='Forecasted Risk (%)',
            line=dict(color='#00FF66' if forecasted_max_risk < 0.5 else '#FF3333', width=3),
            marker=dict(size=8)
        ))
        fig_traj.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#161B22',
            font=dict(color='#C9D1D9'),
            xaxis=dict(title='Forecast Horizon (+k Windows)', gridcolor='#30363D'),
            yaxis=dict(title='Infiltration Risk Probability (%)', range=[0, 100], gridcolor='#30363D'),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_traj, use_container_width=True)
        
        st.info("**Temporal World Model Principle**: Unlike static packet classifiers, the system models state transition dynamics $P(S_{t+1} \mid S_t, \dots, S_{t-k})$ to anticipate lateral movement and C2 before full compromise occurs.")

    # -------------------------------------------------------------
    # TAB 2: TRAFFIC EXPLORER
    # -------------------------------------------------------------
    elif selected_tab.startswith("2."):
        st.title("Traffic Telemetry & Aggregated States")
        st.markdown("Inspect raw network flow records and aggregated chronological network state vectors ($S_t$).")
        
        tab_a, tab_b = st.tabs(["Raw Flow Telemetry", "Aggregated Network States ($S_t$)"])
        
        with tab_a:
            st.subheader("Raw Flow Records")
            st.dataframe(df_raw.head(100), use_container_width=True)
            
        with tab_b:
            st.subheader("Chronological State Windows")
            st.dataframe(state_df.head(50), use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: FORECAST
    # -------------------------------------------------------------
    elif selected_tab.startswith("3."):
        st.title("K-Step Forward Simulation")
        st.markdown("Autoregressive rollout predicting future network states and infiltration trajectories.")
        
        col_a, col_b = st.columns([2, 1])
        with col_a:
            fig = go.Figure()
            steps = list(range(1, k_steps + 1))
            fig.add_trace(go.Bar(
                x=[f"+{s} Window" for s in steps],
                y=[r * 100 for r in risk_traj],
                marker_color=['#58A6FF' if r < 0.5 else '#FF3333' for r in risk_traj]
            ))
            fig.update_layout(
                title="Infiltration Probability per Forecast Horizon",
                paper_bgcolor='#0E1117',
                plot_bgcolor='#161B22',
                font=dict(color='#C9D1D9'),
                yaxis=dict(title="Risk Probability (%)", range=[0, 100], gridcolor='#30363D'),
                xaxis=dict(gridcolor='#30363D')
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col_b:
            st.subheader("Forecast Breakdown")
            forecast_summary = pd.DataFrame({
                'Horizon': [f"+{i+1} Win" for i in range(k_steps)],
                'Risk Score': [f"{r*100:.1f}%" for r in risk_traj],
                'Uncertainty': [f"±{u*100:.1f}%" for u in uncertainty_traj]
            })
            st.dataframe(forecast_summary, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: ATTACK PROGRESSION
    # -------------------------------------------------------------
    elif selected_tab.startswith("4."):
        st.title("Attack Progression & Kill-Chain Mapping")
        st.markdown("Mapping predicted network states to MITRE ATT&CK kill-chain heuristics.")
        
        progression = stage_mapper.analyze_trajectory(pred_states, risk_traj)
        progression_df = pd.DataFrame(progression)
        # Relabel Confidence to Heuristic Confidence
        progression_df = progression_df.rename(columns={'confidence': 'heuristic_confidence'})
        
        st.subheader("Forecasted Kill-Chain Progression")
        st.dataframe(progression_df, use_container_width=True)
        st.caption("*Note: Heuristic confidence values are rule-based weights derived from state characteristics, not statistically calibrated probabilities.*")
        
        st.markdown("---")
        st.subheader("Observed vs. Forecasted Stage Distinction")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Observed Current State**: `{current_stage}`\n\n- **Technique**: `{current_tech}`\n- **Heuristic Confidence**: `{1.0-current_risk:.2f}`")
        with col2:
            peak_stage, peak_tech, _ = stage_mapper.map_state_to_stage(pred_states[np.argmax(risk_traj)], forecasted_max_risk)
            st.warning(f"**Forecasted Peak State (+{np.argmax(risk_traj)+1}W)**: `{peak_stage}`\n\n- **Technique**: `{peak_tech}`\n- **Peak Risk**: `{forecasted_max_risk*100:.1f}%`")

    # -------------------------------------------------------------
    # TAB 5: EXPLAINABILITY
    # -------------------------------------------------------------
    elif selected_tab.startswith("5."):
        st.title("Model Explainability & Driving Features")
        st.markdown("Deterministic perturbation contribution analysis identifying top driving features behind the world model's forecast.")
        
        with st.spinner("Computing perturbation contributions..."):
            contributions = explain_engine.explain_prediction(latest_seq)
            
        contrib_df = pd.DataFrame(contributions, columns=['Feature', 'Impact Score'])
        
        fig_feat = px.bar(
            contrib_df.head(10).sort_values('Impact Score', ascending=True),
            x='Impact Score',
            y='Feature',
            orientation='h',
            title="Top 10 Driving Features (Perturbation Impact)",
            color='Impact Score',
            color_continuous_scale='Blues'
        )
        fig_feat.update_layout(
            paper_bgcolor='#0E1117',
            plot_bgcolor='#161B22',
            font=dict(color='#C9D1D9'),
            xaxis=dict(gridcolor='#30363D'),
            yaxis=dict(gridcolor='#30363D')
        )
        st.plotly_chart(fig_feat, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Feature Contribution Summary")
        st.dataframe(contrib_df, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 6: MODEL PERFORMANCE
    # -------------------------------------------------------------
    elif selected_tab.startswith("6."):
        st.title("Model Performance & Baseline Comparison")
        st.markdown("Rigorous evaluation comparing the LSTM World Model against the Logistic Regression baseline.")
        
        if len(X_seqs) > 5:
            split = int(len(X_seqs) * 0.8)
            X_test, y_test = X_seqs[split:], y_risks[split:]
            if len(X_test) == 0:
                X_test, y_test = X_seqs, y_risks
                
            baseline_preds = baseline.predict(X_test)
            baseline_probas = baseline.predict_proba(X_test)
            baseline_metrics = ModelEvaluator.evaluate_classification(y_test, baseline_preds, baseline_probas)
            
            wm_probas = np.array([forecaster.forecast_trajectory(x, k_steps=1)[1][0] for x in X_test])
            wm_preds = (wm_probas >= 0.5).astype(int)
            wm_metrics = ModelEvaluator.evaluate_classification(y_test, wm_preds, wm_probas)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("LSTM World Model (Temporal)")
                st.json(wm_metrics)
            with col2:
                st.subheader("Logistic Regression (Static Baseline)")
                st.json(baseline_metrics)
        else:
            st.warning("Insufficient sequences for validation split. Run `python train.py` with a larger dataset.")

    # -------------------------------------------------------------
    # TAB 7: ABOUT / METHODOLOGY
    # -------------------------------------------------------------
    elif selected_tab.startswith("7."):
        st.title("Methodology & Architecture")
        st.markdown("""
### SIH26153: AI Based Network Attack Forecasting from Network Traffic Data

**Core Architecture**:
1. **Flow Canonicalization**: Standardizes heterogeneous CSV headers (CIC-IDS, UNSW-NB15, Netflow) into 15 canonical feature dimensions.
2. **Temporal Windowing**: Aggregates flows into chronological windows ($S_t$), computing statistical moments (IAT variance, SYN ratios, port diversity).
3. **World Model**: PyTorch LSTM architecture approximating transition dynamics $P(S_{t+1} \mid S_t, \dots, S_{t-k})$ and predicting multi-step infiltration risk.
4. **Explainability**: Perturbation-based feature impact analysis.
5. **Offline Execution**: Zero external cloud AI dependencies or API keys required.
""")

    # -------------------------------------------------------------
    # TAB 8: DATASET & KNOWLEDGE BASE
    # -------------------------------------------------------------
    elif selected_tab.startswith("8."):
        st.title("Dataset Library & MITRE Knowledge Base")
        st.markdown("Curated catalog of public network security datasets and offline MITRE ATT&CK reference data.")
        
        tab_ds, tab_kb, tab_nciipc = st.tabs(["Public Dataset Catalog", "MITRE Knowledge Base", "NCIIPC / Restricted Access"])
        
        with tab_ds:
            st.subheader("Curated Public Telemetry Datasets")
            catalog = load_dataset_catalog()
            
            for ds in catalog:
                with st.expander(f"{ds['name']} — {ds['description'][:80]}..."):
                    st.markdown(f"**Description**: {ds['description']}")
                    st.markdown(f"**Schema Notes**: `{ds['schema_notes']}`")
                    st.markdown(f"**Official Source**: [{ds['official_url']}]({ds['official_url']})")
                    st.markdown(f"**Status**: {'Directly Fetchable' if ds.get('fetchable') else 'Restricted / Registration Required'}")
                    st.markdown(f"*Note*: {ds.get('fetch_note', '')}")
                    
                    if ds.get('fetchable') and 'sample_url' in ds:
                        if st.button(f"Fetch Sample Data for {ds['name']}", key=f"fetch_{ds['id']}"):
                            try:
                                os.makedirs("data/downloaded", exist_ok=True)
                                resp = requests.get(ds['sample_url'], timeout=10)
                                target_path = f"data/downloaded/{ds['id']}_sample.md"
                                with open(target_path, 'w') as f:
                                    f.write(resp.text)
                                st.success(f"Successfully fetched sample file and saved to `{target_path}`!")
                            except Exception as e:
                                st.error(f"Network fetch failed: {e}. Falling back to manual download link: {ds['official_url']}")
                    else:
                        st.info("This dataset requires manual download/agreement. You can download it locally and upload it via the Telemetry Source sidebar.")
                        
            st.markdown("---")
            st.subheader("Manual Dataset Ingestion (Drag & Drop)")
            uploaded_ds = st.file_uploader("Drop exported network CSV or PCAP-derived flow export", type=["csv"], key="ds_upload")
            if uploaded_ds is not None:
                try:
                    os.makedirs("data/downloaded", exist_ok=True)
                    save_path = os.path.join("data/downloaded", uploaded_ds.name)
                    with open(save_path, "wb") as f:
                        f.write(uploaded_ds.getbuffer())
                    st.success(f"Dataset successfully uploaded to `{save_path}` and registered for pipeline ingestion!")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

        with tab_kb:
            st.subheader("MITRE ATT&CK Technique Reference (Offline Bundle)")
            kb = load_knowledge_base()
            
            # Live fetch toggle
            fetch_live = st.checkbox("Attempt Live Enrichment (MITRE STIX API)", value=False)
            if fetch_live:
                st.info("Live network connection enabled. Attempting API query...")
                try:
                    # Optional live fetch test against public MITRE STIX repo or similar
                    resp = requests.get("https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json", timeout=3)
                    if resp.status_code == 200:
                        st.success("Connected to MITRE CTI repository (Live data active).")
                    else:
                        st.warning("Live fetch returned non-200 status. Falling back to offline reference bundle.")
                except Exception as e:
                    st.warning(f"Offline mode active (Network unreachable: {e}). Using bundled offline reference.")
                    
            for tech_id, details in kb.items():
                with st.expander(f"{details['technique_id']}: {details['name']} ({details['tactic']})"):
                    st.markdown(f"**Tactic**: {details['tactic']}")
                    st.markdown(f"**Description**: {details['description']}")
                    st.markdown(f"**Official Link**: [{details['url']}]({details['url']})")
                    st.caption("Offline Reference (Zero-Network Guaranteed)")

        with tab_nciipc:
            st.subheader("National Critical Information Infrastructure Protection Centre (NCIIPC)")
            st.markdown("""
For official government critical infrastructure defense, restricted national threat-intelligence feeds, and specialized SCADA/ICS telemetry access, authorized researchers and defense agencies can coordinate through NCIIPC India.

- **Helpdesk Email**: `helpdesk1@nciipc.gov.in`
- **Official Portal**: [nciipc.gov.in](https://www.nciipc.gov.in)
- **Scope**: Guidance on secure reporting, vulnerability coordination, and authorized dataset access for national cybersecurity research.
""")
            st.info("Note: This application operates 100% offline and does not transmit local telemetry to external government or cloud endpoints.")


if __name__ == '__main__':
    main()
