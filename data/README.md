# Network Telemetry Datasets (SIH26153)

This directory houses network flow telemetry for model training and forecasting.

## Supported Public Benchmarks
The data pipeline is designed to ingest standard network flow exports from:
1. **CIC-IDS2017 / CSE-CIC-IDS2018**
   - Source: Canadian Institute for Cybersecurity (UNB)
   - URL: https://www.unb.ca/cic/datasets/ids-2017.html
   - Placement: Place CSV files under `data/cicids2017/` (e.g., `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
2. **UNSW-NB15**
   - Source: Cyber Range Lab of UNSW Canberra
   - URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
   - Placement: Place CSV files under `data/unsw_nb15/`
3. **CTU-13**
   - Source: Stratosphere Lab, Czech Technical University
   - URL: https://www.stratosphereips.org/datasets-ctu13
   - Placement: Place NetFlow/CSV files under `data/ctu13/`

## Bundled Demonstration Dataset
For instant evaluation and offline demonstration without gigabyte-scale downloads, a verified scenario dataset is provided in `data/demo/demo_traffic.csv`.
- **Classification**: Explicitly labeled as **DEMO DATA**.
- **Scenario**: Synthesized chronological multi-phase progression:
  1. *Phase 0 (t=0..60s)*: Normal enterprise baseline traffic (HTTPS, DNS, SSH maintenance).
  2. *Phase 1 (t=60..120s)*: Port scanning & service reconnaissance (SYN flood/probes across ephemeral ports).
  3. *Phase 2 (t=120..180s)*: Initial access attempt (SSH/RDP brute force connection bursts).
  4. *Phase 3 (t=180..240s)*: Lateral movement (Internal SMB port 445 sweeps and RPC queries).
  5. *Phase 4 (t=240..300s)*: C2 beaconing & data exfiltration (Periodic outbound beacons, bulk outbound payload transfer).

## Privacy & Security Policy
- Never commit unanonymized production PCAPs or sensitive enterprise credentials.
- All CSV ingestion operates in strict read-only mode with boundary validation.
