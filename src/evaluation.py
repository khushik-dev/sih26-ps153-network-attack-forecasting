"""Evaluation metrics calculation for world model and baseline.
Computes Precision, Recall, F1, False Positive Rate (FPR), ROC AUC, and State Transition MSE.
"""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix, mean_squared_error


class ModelEvaluator:
    """Computes rigorous evaluation metrics for network attack forecasting."""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
        """Compute classification metrics."""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
        
        # Calculate FPR
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            fpr = 0.0
            
        try:
            auc = roc_auc_score(y_true, y_proba)
        except Exception:
            auc = 0.5
            
        accuracy = float(np.mean(y_true == y_pred))
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive_rate': float(fpr),
            'roc_auc': float(auc),
            'accuracy': float(accuracy)
        }

    @staticmethod
    def evaluate_state_transition(y_true_states: np.ndarray, y_pred_states: np.ndarray) -> dict:
        """Compute state transition MSE."""
        mse = mean_squared_error(y_true_states, y_pred_states)
        rmse = np.sqrt(mse)
        return {
            'state_transition_mse': float(mse),
            'state_transition_rmse': float(rmse)
        }
