"""Tests for K-step forecasting and world model rollout."""

import pytest
import numpy as np
from src.world_model import NetworkWorldModel
from src.forecasting import KStepForecaster


def test_k_step_forecasting():
    input_dim = 15
    seq_len = 5
    model = NetworkWorldModel(input_dim=input_dim, hidden_dim=32, num_layers=1)
    forecaster = KStepForecaster(model)
    
    initial_seq = np.random.randn(seq_len, input_dim)
    k_steps = 4
    
    predicted_states, risk_traj, uncertainty_traj = forecaster.forecast_trajectory(initial_seq, k_steps=k_steps)
    
    assert len(predicted_states) == k_steps
    assert len(risk_traj) == k_steps
    assert len(uncertainty_traj) == k_steps
    assert predicted_states[0].shape == (input_dim,)
    assert 0.0 <= risk_traj[0] <= 1.0
