"""PyTorch LSTM World Model for Network State Transitions.
Approximates P(S_{t+1} | S_t, ..., S_{t-k}) and predicts infiltration risk probability.
"""

import torch
import torch.nn as nn
import numpy as np
import os


class NetworkWorldModel(nn.Module):
    """LSTM-based network state transition dynamics and risk forecasting model."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super(NetworkWorldModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Transition head: predicts next state vector S_{t+1}
        self.state_transition_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Infiltration risk head: predicts attack probability [0, 1]
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass.
        Args:
            x: tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            next_state: tensor of shape (batch_size, input_dim)
            risk_score: tensor of shape (batch_size, 1)
        """
        lstm_out, (hn, cn) = self.lstm(x)
        # Take the output of the last time step
        last_hidden = lstm_out[:, -1, :]
        
        next_state = self.state_transition_head(last_hidden)
        risk_score = self.risk_head(last_hidden)
        
        return next_state, risk_score

    def save(self, filepath: str):
        """Save model weights and configuration."""
        torch.save({
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers
        }, filepath)

    @staticmethod
    def load(filepath: str) -> 'NetworkWorldModel':
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
        model = NetworkWorldModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model
