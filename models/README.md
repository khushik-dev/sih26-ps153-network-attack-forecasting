# Model Checkpoints and Artifacts

This directory stores serialized model weights, normalization scalers, and metadata schemas.

## Generated Artifacts
When executing `python train.py`, the following files are produced:
- `lstm_world_model.pt`: PyTorch state dict containing learned network state transition weights ($P(S_{t+1} \mid S_{t-W:t})$) and risk prediction head.
- `baseline_lr.joblib`: Scikit-Learn Logistic Regression model trained for static window classification baseline.
- `scaler.joblib`: Fitted `StandardScaler` ensuring consistent z-score normalization across training and live inference.
- `metadata.json`: Feature ordering, window length, sequence history $W$, input dimensions, and training hyperparameters.
