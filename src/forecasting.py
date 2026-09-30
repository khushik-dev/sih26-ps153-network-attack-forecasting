"""K-Step Forward Simulation and Attack Risk Forecasting.
Performs autoregressive rollout given the latest observed temporal sequence [S(t-k)...S(t)]
to forecast future states S(t+1), ..., S(t+K) and infiltration risk trajectories.
"""

import torch
import numpy as np
from src.world_model import NetworkWorldModel


class KStepForecaster:
    """Performs K-step forward simulation using the trained world model."""
    
    def __init__(self, model: NetworkWorldModel):
        self.model = model
        self.model.eval()

    def forecast_trajectory(self, initial_sequence: np.ndarray, k_steps: int = 5):
        """Perform K-step forward autoregressive rollout.
        Args:
            initial_sequence: numpy array of shape (seq_len, input_dim)
            k_steps: number of future windows to forecast (K)
        Returns:
            predicted_states: list of predicted state numpy arrays of length K
            risk_trajectory: list of float risk probabilities of length K
            uncertainty_trajectory: list of uncertainty estimates of length K
        """
        self.model.eval()
        seq_len, input_dim = initial_sequence.shape
        current_seq = initial_sequence.copy()
        
        predicted_states = []
        risk_trajectory = []
        uncertainty_trajectory = []
        
        with torch.no_grad():
            for step in range(k_steps):
                x_tensor = torch.tensor(current_seq[np.newaxis, :, :], dtype=torch.float32)
                next_state_tensor, risk_tensor = self.model(x_tensor)
                
                next_state = next_state_tensor.cpu().numpy()[0]
                risk_prob = float(risk_tensor.cpu().numpy()[0, 0])
                
                # Estimate uncertainty as cumulative step distance or residual variance proxy
                uncertainty = float(0.05 * (step + 1) + 0.02 * np.std(next_state))
                
                predicted_states.append(next_state)
                risk_trajectory.append(risk_prob)
                uncertainty_trajectory.append(uncertainty)
                
                # Slide sequence window forward by appending predicted state and dropping oldest
                current_seq = np.vstack([current_seq[1:], next_state])
                
        return predicted_states, risk_trajectory, uncertainty_trajectory
