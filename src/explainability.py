"""Explainability Engine for Network State Predictions.
Implements a robust perturbation-based feature contribution method to explain
why the world model produced a specific risk forecast or state transition.
"""

import numpy as np
import torch
from src.world_model import NetworkWorldModel
from src.temporal_state import TemporalStateAggregator


class ExplainabilityEngine:
    """Computes feature contributions via deterministic perturbation analysis."""
    
    def __init__(self, model: NetworkWorldModel, aggregator: TemporalStateAggregator):
        self.model = model
        self.aggregator = aggregator
        self.model.eval()

    def explain_prediction(self, sequence: np.ndarray) -> list:
        """Compute feature contributions for a given input sequence by perturbing feature columns.
        Args:
            sequence: numpy array of shape (seq_len, input_dim)
        Returns:
            sorted_contributions: list of tuples (feature_name, contribution_score)
        """
        self.model.eval()
        feature_names = self.aggregator.state_feature_names
        input_dim = sequence.shape[1]
        
        # Baseline prediction
        with torch.no_grad():
            x_base = torch.tensor(sequence[np.newaxis, :, :], dtype=torch.float32)
            _, base_risk_tensor = self.model(x_base)
            base_risk = float(base_risk_tensor.cpu().numpy()[0, 0])
            
        contributions = {}
        
        # Perturb each feature dimension across the sequence
        for idx in range(min(len(feature_names), input_dim)):
            feat_name = feature_names[idx]
            perturbed_seq = sequence.copy()
            
            # Perturb by zeroing out or scaling feature values
            perturbed_seq[:, idx] = 0.0
            
            with torch.no_grad():
                x_pert = torch.tensor(perturbed_seq[np.newaxis, :, :], dtype=torch.float32)
                _, pert_risk_tensor = self.model(x_pert)
                pert_risk = float(pert_risk_tensor.cpu().numpy()[0, 0])
                
            # Importance is absolute change in risk when feature is removed/perturbed
            impact = abs(base_risk - pert_risk)
            contributions[feat_name] = impact
            
        # Sort by impact descending
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return sorted_contrib
