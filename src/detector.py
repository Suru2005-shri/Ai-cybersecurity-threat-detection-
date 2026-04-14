"""
detector.py
-----------
Real-time (simulated) threat detection and alert engine.

Simulates a SOC (Security Operations Center) detection pipeline:

  Incoming flow → Preprocess → [IsoForest + Binary RF + Multi-class RF]
                              → Combine scores → Severity assessment
                              → Generate ALERT → Log to CSV

Alert severity levels (mirrors CVSS scoring):
  CRITICAL : anomaly_score > 0.9  AND  binary_prob > 0.95
  HIGH     : binary_prob > HIGH_RISK_THRESHOLD (0.80)
  MEDIUM   : binary_prob > MEDIUM_RISK_THRESHOLD (0.50)
  LOW      : IsolationForest flags as anomaly but RF disagrees
  INFO     : normal traffic

Usage
-----
  from src.detector import ThreatDetector
  detector = ThreatDetector()
  alerts = detector.detect_batch(df_new_flows)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import (
    ALERTS_DIR, REPORTS_DIR,
    HIGH_RISK_THRESHOLD, MEDIUM_RISK_THRESHOLD,
    ENGINEERED_FEATURES  # re-exported from preprocessor
)
from src.models import (
    load_isolation_forest,
    load_binary_classifier,
    load_multiclass_classifier,
    isolation_forest_scores,
    predict_isolation_forest,
)
from src.preprocessor import (
    clean_data, engineer_features,
    load_scaler, load_label_encoder,
    scale_features, ENGINEERED_FEATURES
)

# ─── Severity mapping ──────────────────────────────────────────────────────────
SEVERITY_COLORS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
    "INFO":     "🟢",
}


def _assign_severity(binary_prob: float, iso_score: float,
                     iso_pred: int) -> str:
    """Map model outputs to SIEM-style severity level."""
    if iso_score > 0.9 and binary_prob > 0.95:
        return "CRITICAL"
    elif binary_prob > HIGH_RISK_THRESHOLD:
        return "HIGH"
    elif binary_prob > MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"
    elif iso_pred == 1:
        return "LOW"
    else:
        return "INFO"


def _attack_description(attack_type: str) -> str:
    descriptions = {
        "BENIGN":      "Normal network traffic. No threat detected.",
        "DoS":         "Denial-of-Service attack: flooding target with requests.",
        "DDoS":        "Distributed DoS: coordinated flood from multiple sources.",
        "PortScan":    "Port scanning activity: attacker probing for open services.",
        "BruteForce":  "Brute-force login attempt: automated credential guessing.",
        "SQLInjection":"SQL Injection: malicious DB queries embedded in requests.",
        "XSS":         "Cross-Site Scripting: injecting scripts into web responses.",
        "BotNet":      "BotNet beacon: compromised host contacting C2 server.",
        "Infiltration":"Data exfiltration attempt: unusually large outbound transfer.",
    }
    return descriptions.get(attack_type, "Unknown threat pattern detected.")


class ThreatDetector:
    """
    Encapsulates all three models and the alert generation logic.

    Load pre-trained models once, then call detect_batch() or detect_flow().
    """

    def __init__(self):
        print("[DETECTOR] Loading models …")
        self.iso    = load_isolation_forest()
        self.rf_bin = load_binary_classifier()
        self.rf_mul = load_multiclass_classifier()
        self.scaler = load_scaler()
        self.le     = load_label_encoder()
        self.class_names = list(self.le.classes_)
        print("[DETECTOR] All models loaded ✓")

    # ── Single flow ────────────────────────────────────────────────────────────

    def detect_flow(self, flow_dict: dict) -> dict:
        """
        Detect threat in a single network flow.

        Parameters
        ----------
        flow_dict : dict – raw flow features

        Returns
        -------
        dict with severity, attack_type, probabilities, alert message
        """
        df = pd.DataFrame([flow_dict])
        df = clean_data(df)
        df = engineer_features(df)
        X  = scale_features(df, self.scaler, fit=False)

        iso_score  = float(isolation_forest_scores(self.iso, X)[0])
        iso_pred   = int(predict_isolation_forest(self.iso, X)[0])
        bin_prob   = float(self.rf_bin.predict_proba(X)[0][1])
        mul_pred   = int(self.rf_mul.predict(X)[0])
        attack_type = self.class_names[mul_pred]

        severity = _assign_severity(bin_prob, iso_score, iso_pred)

        return {
            "timestamp":      datetime.now().isoformat(),
            "severity":       severity,
            "severity_icon":  SEVERITY_COLORS[severity],
            "attack_type":    attack_type,
            "binary_prob":    round(bin_prob, 4),
            "anomaly_score":  round(iso_score, 4),
            "iso_flag":       bool(iso_pred),
            "description":    _attack_description(attack_type),
            "src_ip":         flow_dict.get("src_ip", "unknown"),
            "dst_ip":         flow_dict.get("dst_ip", "unknown"),
            "dst_port":       int(flow_dict.get("dst_port", 0)),
            "protocol":       flow_dict.get("protocol", "unknown"),
        }

    # ── Batch detection ────────────────────────────────────────────────────────

    def detect_batch(self, df: pd.DataFrame,
                     save_alerts: bool = True) -> pd.DataFrame:
        """
        Run detection on a batch of network flows.

        Parameters
        ----------
        df          : DataFrame with raw flow features
        save_alerts : bool – save alert log to CSV

        Returns
        -------
        DataFrame with all original columns + detection results
        """
        print(f"\n[DETECTOR] Scanning {len(df):,} network flows …")

        df_clean = clean_data(df.copy())
        df_clean = engineer_features(df_clean)
        X        = scale_features(df_clean, self.scaler, fit=False)

        # Model predictions
        iso_scores = isolation_forest_scores(self.iso, X)
        iso_preds  = predict_isolation_forest(self.iso, X)
        bin_probs  = self.rf_bin.predict_proba(X)[:, 1]
        mul_preds  = self.rf_mul.predict(X)
        attack_types = [self.class_names[p] for p in mul_preds]

        # Severity
        severities = [
            _assign_severity(bp, iso_s, iso_p)
            for bp, iso_s, iso_p in zip(bin_probs, iso_scores, iso_preds)
        ]

        # Assemble results
        result_df = df.copy()
        result_df["predicted_attack_type"] = attack_types
        result_df["threat_probability"]    = np.round(bin_probs, 4)
        result_df["anomaly_score"]         = np.round(iso_scores, 4)
        result_df["iso_flag"]              = iso_preds.astype(bool)
        result_df["severity"]              = severities
        result_df["description"]           = [
            _attack_description(a) for a in attack_types
        ]

        # Summary
        sev_counts = result_df["severity"].value_counts()
        print(f"\n{'─'*45}")
        print(f"  DETECTION SUMMARY ({len(df):,} flows analyzed)")
        print(f"{'─'*45}")
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            count = sev_counts.get(sev, 0)
            icon  = SEVERITY_COLORS[sev]
            print(f"  {icon}  {sev:<10}: {count:>6,} flows")
        print(f"{'─'*45}")

        # Save alerts (non-INFO only)
        if save_alerts:
            alerts_df = result_df[result_df["severity"] != "INFO"].copy()
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            alert_path = os.path.join(ALERTS_DIR, f"alerts_{ts}.csv")
            alerts_df.to_csv(alert_path, index=False)
            print(f"\n[INFO] {len(alerts_df):,} alerts saved → {alert_path}")

            # Also save summary JSON
            summary = {
                "scan_timestamp" : datetime.now().isoformat(),
                "total_flows"    : int(len(df)),
                "threat_counts"  : {
                    sev: int(sev_counts.get(sev, 0))
                    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]
                },
                "top_attack_types": result_df[
                    result_df["severity"] != "INFO"
                ]["predicted_attack_type"].value_counts().to_dict(),
            }
            summary_path = os.path.join(REPORTS_DIR, f"scan_summary_{ts}.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"[INFO] Scan summary → {summary_path}")

        return result_df

    # ── Top threats ────────────────────────────────────────────────────────────

    def get_top_threats(self, result_df: pd.DataFrame,
                        n: int = 10) -> pd.DataFrame:
        """Return top N most dangerous flows sorted by threat probability."""
        threat_df = result_df[result_df["severity"].isin(["CRITICAL","HIGH","MEDIUM"])]
        return (
            threat_df
            .sort_values("threat_probability", ascending=False)
            .head(n)[["timestamp", "src_ip", "dst_ip", "dst_port",
                       "protocol", "predicted_attack_type",
                       "threat_probability", "anomaly_score", "severity"]]
        )


# ─── Standalone simulation ─────────────────────────────────────────────────────

def simulate_live_detection(detector: ThreatDetector, n_flows: int = 20) -> list:
    """
    Simulate 'live' real-time flow detection — prints one alert at a time
    as if flows are arriving from the network.

    Used for demo / GitHub proof.
    """
    import time
    from src.data_generator import generate_dataset

    print("\n" + "="*55)
    print("  LIVE THREAT DETECTION SIMULATION")
    print("  Simulating real-time network flow analysis …")
    print("="*55 + "\n")

    df_live = generate_dataset(n_samples=n_flows)
    alerts = []

    for idx, row in df_live.iterrows():
        flow_dict = row.to_dict()
        result = detector.detect_flow(flow_dict)
        alerts.append(result)

        # Only print non-INFO
        if result["severity"] != "INFO":
            print(
                f"[{result['timestamp']}]  "
                f"{result['severity_icon']} {result['severity']:<8}  "
                f"| {result['attack_type']:<14} "
                f"| prob={result['binary_prob']:.3f}  "
                f"| {result['src_ip']} → {result['dst_ip']}:{result['dst_port']}"
            )
        time.sleep(0.05)   # simulate stream delay

    critical = sum(1 for a in alerts if a["severity"] == "CRITICAL")
    high     = sum(1 for a in alerts if a["severity"] == "HIGH")
    print(f"\n[SIMULATION COMPLETE] {critical} CRITICAL, {high} HIGH alerts in {n_flows} flows.")
    return alerts
