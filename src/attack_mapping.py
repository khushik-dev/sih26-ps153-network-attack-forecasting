"""MITRE ATT&CK Attack Stage Mapping and Heuristics.
Maps network state feature vectors and risk trajectories to practical attack stages:
- Reconnaissance
- Initial Access
- Lateral Movement
- Command and Control
- Exfiltration
Distinguishes between observed evidence in historical windows and forecasted stages.
"""

import numpy as np


class AttackStageMapper:
    """Maps network telemetry characteristics to MITRE ATT&CK stages."""
    
    STAGES = [
        "Normal Operation",
        "Reconnaissance (Scanning / Probing)",
        "Initial Access (Brute Force / Exploit)",
        "Lateral Movement (Internal Sweeps / SMB)",
        "Command & Control (C2 Beaconing)",
        "Exfiltration (Bulk Outbound Transfer)"
    ]

    def map_state_to_stage(self, state_vector: np.ndarray, risk_score: float) -> tuple:
        """Map a single network state vector and risk score to an ATT&CK stage.
        Returns:
            stage_name (str): Associated stage description
            technique_id (str): MITRE technique reference
            confidence (float): Confidence score [0, 1]
        """
        if risk_score < 0.35:
            return "Normal Operation", "T1078 (Valid Accounts)", 0.85
            
        # Extract features (assuming standard canonical state order)
        # 0: flow_count, 1: bytes_total, 2: packets_total, 3: mean_duration,
        # 4: mean_iat, 5: iat_variance, 6: syn_ratio, 7: ack_ratio, 8: rst_ratio,
        # 9: unique_dst_ports, 10: unique_src_ports, 11: mean_pkt_length, 12: retransmission_rate
        
        syn_ratio = float(state_vector[6]) if len(state_vector) > 6 else 0.0
        unique_dst = float(state_vector[9]) if len(state_vector) > 9 else 0.0
        bytes_tot = float(state_vector[1]) if len(state_vector) > 1 else 0.0
        rst_ratio = float(state_vector[8]) if len(state_vector) > 8 else 0.0
        
        # Heuristics based on network traffic characteristics
        if unique_dst > 15 or syn_ratio > 0.6:
            return "Reconnaissance (Scanning / Probing)", "T1595 (Active Scanning)", min(0.7 + syn_ratio * 0.3, 0.98)
        elif rst_ratio > 0.3 or (syn_ratio > 0.3 and unique_dst < 5):
            return "Initial Access (Brute Force / Exploit)", "T1110 (Brute Force)", min(0.65 + risk_score * 0.3, 0.95)
        elif bytes_tot > 500000 and unique_dst > 5:
            return "Lateral Movement (Internal Sweeps / SMB)", "T1210 (Exploitation of Remote Services)", min(0.75 + risk_score * 0.2, 0.96)
        elif bytes_tot < 50000 and syn_ratio < 0.2:
            return "Command & Control (C2 Beaconing)", "T1071 (Application Layer Protocol)", min(0.7 + risk_score * 0.25, 0.94)
        else:
            return "Exfiltration (Bulk Outbound Transfer)", "T1041 (Exfiltration Over C2 Channel)", min(0.8 + risk_score * 0.15, 0.99)

    def analyze_trajectory(self, trajectory_states: list, risk_trajectory: list) -> list:
        """Analyze a multi-step forecasted trajectory for attack progression."""
        progression = []
        for i, (state, risk) in enumerate(zip(trajectory_states, risk_trajectory)):
            stage, tech, conf = self.map_state_to_stage(state, risk)
            progression.append({
                'step': i + 1,
                'risk': risk,
                'stage': stage,
                'technique': tech,
                'confidence': conf
            })
        return progression
