"""
data_generator.py
-----------------
Generates a realistic synthetic network traffic dataset that mirrors
the structure of CICIDS-2017 and UNSW-NB15 — two of the most widely
used public cybersecurity datasets in academic and industry research.

WHY SYNTHETIC DATA?
  In a real SOC (Security Operations Center), raw network flows are
  captured using tools like CICFlowMeter or Zeek/Bro and stored as
  CSVs with 80+ features per flow.
  Since we can't capture live corporate traffic, we generate data
  that has the SAME statistical properties as the real datasets.

HOW TO USE REAL DATA INSTEAD:
  1. Download CICIDS 2017 from:
     https://www.unb.ca/cic/datasets/ids-2017.html
  2. Place CSVs in data/raw/
  3. Update data_loader.py to read those CSVs

Run standalone:
  python -m src.data_generator
"""

import os
import numpy as np
import pandas as pd
from src.config import (
    RAW_DIR, RANDOM_SEED, N_SAMPLES, ATTACK_TYPES,
    BINARY_LABEL_MAP, NUMERIC_FEATURES, CATEGORICAL_FEATURES
)

np.random.seed(RANDOM_SEED)

# ─── Per-attack-type statistical profiles ─────────────────────────────────────
# Each entry: (mean_vector, std_vector, weight)
# Features order matches NUMERIC_FEATURES list in config.py

ATTACK_PROFILES = {
    # BENIGN: typical web/API traffic — moderate flow, varied ports
    "BENIGN": {
        "weight": 0.60,
        "flow_duration":      (500_000,  300_000),
        "fwd_packets":        (15,        10),
        "bwd_packets":        (12,         8),
        "fwd_bytes":          (8_000,   5_000),
        "bwd_bytes":          (12_000,  7_000),
        "flow_bytes_per_sec": (3_000,   2_000),
        "flow_pkts_per_sec":  (0.05,    0.03),
        "fwd_iat_mean":       (50_000,  30_000),
        "bwd_iat_mean":       (60_000,  35_000),
        "pkt_len_mean":       (700,      300),
        "pkt_len_std":        (200,      100),
        "fin_flag_count":     (1,          1),
        "syn_flag_count":     (1,          1),
        "rst_flag_count":     (0,          0.2),
        "ack_flag_count":     (14,         8),
        "psh_flag_count":     (3,          2),
        "urg_flag_count":     (0,          0.05),
        "active_mean":        (200_000, 100_000),
        "idle_mean":          (300_000, 150_000),
        "dst_port":           (443,       100),
        "protocol":           ["TCP", "UDP", "TCP", "TCP"],
    },

    # DoS: flood of small packets, very high packet rate
    "DoS": {
        "weight": 0.08,
        "flow_duration":      (10_000,    5_000),
        "fwd_packets":        (500,       200),
        "bwd_packets":        (5,           3),
        "fwd_bytes":          (30_000,  10_000),
        "bwd_bytes":          (500,       200),
        "flow_bytes_per_sec": (800_000, 300_000),
        "flow_pkts_per_sec":  (50,        20),
        "fwd_iat_mean":       (200,        50),
        "bwd_iat_mean":       (5_000,   2_000),
        "pkt_len_mean":       (60,         20),
        "pkt_len_std":        (10,          5),
        "fin_flag_count":     (0,          0.1),
        "syn_flag_count":     (450,       180),
        "rst_flag_count":     (0,          0.5),
        "ack_flag_count":     (0,          1),
        "psh_flag_count":     (0,          0.2),
        "urg_flag_count":     (0,          0.1),
        "active_mean":        (8_000,    3_000),
        "idle_mean":          (2_000,    1_000),
        "dst_port":           (80,         10),
        "protocol":           ["TCP", "TCP", "UDP", "TCP"],
    },

    # DDoS: similar to DoS but from many sources (high bwd traffic)
    "DDoS": {
        "weight": 0.07,
        "flow_duration":      (8_000,    3_000),
        "fwd_packets":        (1_000,     400),
        "bwd_packets":        (800,       300),
        "fwd_bytes":          (60_000,  20_000),
        "bwd_bytes":          (50_000,  15_000),
        "flow_bytes_per_sec": (1_200_000, 400_000),
        "flow_pkts_per_sec":  (120,        40),
        "fwd_iat_mean":       (100,        30),
        "bwd_iat_mean":       (120,        40),
        "pkt_len_mean":       (60,         15),
        "pkt_len_std":        (8,           3),
        "fin_flag_count":     (0,          0.1),
        "syn_flag_count":     (900,       300),
        "rst_flag_count":     (5,           3),
        "ack_flag_count":     (0,          2),
        "psh_flag_count":     (0,          0.3),
        "urg_flag_count":     (0,          0.1),
        "active_mean":        (5_000,    2_000),
        "idle_mean":          (1_000,      500),
        "dst_port":           (80,         20),
        "protocol":           ["UDP", "TCP", "UDP", "ICMP"],
    },

    # PortScan: many short flows to different ports
    "PortScan": {
        "weight": 0.07,
        "flow_duration":      (500,       200),
        "fwd_packets":        (1,           0.5),
        "bwd_packets":        (0,           0.3),
        "fwd_bytes":          (60,          20),
        "bwd_bytes":          (0,           10),
        "flow_bytes_per_sec": (5_000,    2_000),
        "flow_pkts_per_sec":  (2,           1),
        "fwd_iat_mean":       (0,            5),
        "bwd_iat_mean":       (0,            5),
        "pkt_len_mean":       (40,          10),
        "pkt_len_std":        (0,            2),
        "fin_flag_count":     (0,          0.1),
        "syn_flag_count":     (1,          0.2),
        "rst_flag_count":     (1,          0.3),
        "ack_flag_count":     (0,          0.2),
        "psh_flag_count":     (0,          0.1),
        "urg_flag_count":     (0,          0.05),
        "active_mean":        (300,        100),
        "idle_mean":          (100,         50),
        "dst_port":           (32_000,  16_000),   # random high ports
        "protocol":           ["TCP", "TCP", "TCP", "TCP"],
    },

    # BruteForce: many authentication attempts — high SYN, moderate packets
    "BruteForce": {
        "weight": 0.05,
        "flow_duration":      (2_000,    1_000),
        "fwd_packets":        (20,          10),
        "bwd_packets":        (15,           8),
        "fwd_bytes":          (1_200,      500),
        "bwd_bytes":          (900,        400),
        "flow_bytes_per_sec": (60_000,  20_000),
        "flow_pkts_per_sec":  (10,          5),
        "fwd_iat_mean":       (100,         40),
        "bwd_iat_mean":       (150,         60),
        "pkt_len_mean":       (80,          30),
        "pkt_len_std":        (20,          10),
        "fin_flag_count":     (1,           0.5),
        "syn_flag_count":     (5,           2),
        "rst_flag_count":     (2,           1),
        "ack_flag_count":     (18,          8),
        "psh_flag_count":     (4,           2),
        "urg_flag_count":     (0,          0.1),
        "active_mean":        (1_500,      600),
        "idle_mean":          (500,        200),
        "dst_port":           (22,           5),   # SSH port
        "protocol":           ["TCP", "TCP", "TCP", "TCP"],
    },

    # SQLInjection: unusual payload sizes, single port (80/443)
    "SQLInjection": {
        "weight": 0.04,
        "flow_duration":      (100_000,  50_000),
        "fwd_packets":        (6,           3),
        "bwd_packets":        (8,           4),
        "fwd_bytes":          (3_000,    1_500),
        "bwd_bytes":          (15_000,   8_000),
        "flow_bytes_per_sec": (2_000,    1_000),
        "flow_pkts_per_sec":  (0.08,      0.04),
        "fwd_iat_mean":       (15_000,   8_000),
        "bwd_iat_mean":       (12_000,   6_000),
        "pkt_len_mean":       (500,       200),
        "pkt_len_std":        (400,       200),
        "fin_flag_count":     (1,          0.3),
        "syn_flag_count":     (1,          0.3),
        "rst_flag_count":     (0,          0.2),
        "ack_flag_count":     (10,          4),
        "psh_flag_count":     (5,           2),
        "urg_flag_count":     (0,          0.05),
        "active_mean":        (80_000,  40_000),
        "idle_mean":          (20_000,  10_000),
        "dst_port":           (80,         10),
        "protocol":           ["TCP", "TCP", "TCP", "TCP"],
    },

    # XSS: similar to SQLi but smaller payloads
    "XSS": {
        "weight": 0.03,
        "flow_duration":      (80_000,  40_000),
        "fwd_packets":        (5,           3),
        "bwd_packets":        (7,           4),
        "fwd_bytes":          (1_500,     700),
        "bwd_bytes":          (10_000,   5_000),
        "flow_bytes_per_sec": (1_500,     800),
        "flow_pkts_per_sec":  (0.06,      0.03),
        "fwd_iat_mean":       (14_000,   7_000),
        "bwd_iat_mean":       (11_000,   5_500),
        "pkt_len_mean":       (300,       150),
        "pkt_len_std":        (250,       125),
        "fin_flag_count":     (1,          0.3),
        "syn_flag_count":     (1,          0.3),
        "rst_flag_count":     (0,          0.2),
        "ack_flag_count":     (9,           4),
        "psh_flag_count":     (4,           2),
        "urg_flag_count":     (0,          0.05),
        "active_mean":        (60_000,  30_000),
        "idle_mean":          (18_000,   9_000),
        "dst_port":           (443,        10),
        "protocol":           ["TCP", "TCP", "TCP", "TCP"],
    },

    # BotNet: periodic beaconing, consistent intervals
    "BotNet": {
        "weight": 0.04,
        "flow_duration":      (300_000, 100_000),
        "fwd_packets":        (4,           2),
        "bwd_packets":        (3,           2),
        "fwd_bytes":          (400,        150),
        "bwd_bytes":          (350,        130),
        "flow_bytes_per_sec": (200,        100),
        "flow_pkts_per_sec":  (0.015,      0.005),
        "fwd_iat_mean":       (60_000,  10_000),   # very regular intervals
        "bwd_iat_mean":       (60_000,  10_000),
        "pkt_len_mean":       (100,         20),
        "pkt_len_std":        (10,           3),   # very consistent size
        "fin_flag_count":     (1,          0.2),
        "syn_flag_count":     (1,          0.2),
        "rst_flag_count":     (0,          0.1),
        "ack_flag_count":     (3,           1),
        "psh_flag_count":     (1,           0.5),
        "urg_flag_count":     (0,          0.02),
        "active_mean":        (5_000,    2_000),
        "idle_mean":          (250_000, 80_000),
        "dst_port":           (6667,       100),   # IRC command & control
        "protocol":           ["TCP", "UDP", "TCP", "TCP"],
    },

    # Infiltration: low & slow, mimics normal but unusual data exfil
    "Infiltration": {
        "weight": 0.02,
        "flow_duration":      (1_000_000, 300_000),
        "fwd_packets":        (30,          15),
        "bwd_packets":        (100,         40),   # large response = data exfil
        "fwd_bytes":          (2_000,     1_000),
        "bwd_bytes":          (500_000, 200_000),  # large download
        "flow_bytes_per_sec": (500,        200),
        "flow_pkts_per_sec":  (0.1,        0.05),
        "fwd_iat_mean":       (30_000,  15_000),
        "bwd_iat_mean":       (8_000,    3_000),
        "pkt_len_mean":       (4_000,    1_500),   # large packets
        "pkt_len_std":        (1_000,      500),
        "fin_flag_count":     (1,          0.3),
        "syn_flag_count":     (1,          0.3),
        "rst_flag_count":     (0,          0.2),
        "ack_flag_count":     (120,        40),
        "psh_flag_count":     (90,         30),
        "urg_flag_count":     (0,          0.05),
        "active_mean":        (900_000, 200_000),
        "idle_mean":          (50_000,  20_000),
        "dst_port":           (443,        20),
        "protocol":           ["TCP", "TCP", "TCP", "TCP"],
    },
}


def _sample_feature(profile: dict, feature: str, n: int) -> np.ndarray:
    """Sample numeric feature values with clipping to physical bounds."""
    mean, std = profile[feature]
    values = np.random.normal(mean, std, n)
    return np.clip(values, 0, None)   # no negative network metrics


def generate_dataset(n_samples: int = N_SAMPLES,
                     save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic network flow dataset.

    Parameters
    ----------
    n_samples : int  – total flow records
    save_path : str  – if given, save CSV here

    Returns
    -------
    pd.DataFrame with NUMERIC_FEATURES + ['protocol', 'attack_type', 'label_binary']
    """
    records = []
    protocols = ["TCP", "UDP", "ICMP"]

    attack_items = list(ATTACK_PROFILES.items())
    for idx, (attack_type, profile) in enumerate(attack_items):
        n = int(n_samples * profile["weight"])

        rows = {}
        for feat in NUMERIC_FEATURES:
            if feat == "dst_port":
                rows[feat] = _sample_feature(profile, feat, n).astype(int)
                rows[feat] = np.clip(rows[feat], 0, 65535)
            else:
                rows[feat] = _sample_feature(profile, feat, n)

        # Protocol sampling
        proto_choices = profile.get("protocol", ["TCP", "UDP", "ICMP"])
        rows["protocol"] = np.random.choice(proto_choices, n)

        rows["attack_type"]   = attack_type
        rows["label_binary"]  = BINARY_LABEL_MAP[attack_type]

        records.append(pd.DataFrame(rows))

    df = pd.concat(records, ignore_index=True)

    # Add timestamps (simulate 24-hour capture window)
    start_ts = pd.Timestamp("2024-01-15 00:00:00")
    df["timestamp"] = start_ts + pd.to_timedelta(
        np.sort(np.random.uniform(0, 86400, len(df))), unit="s"
    )

    # Add source/destination IP simulation
    def random_ip(n):
        return [f"192.168.{np.random.randint(0,255)}.{np.random.randint(1,255)}"
                for _ in range(n)]

    df["src_ip"] = random_ip(len(df))
    df["dst_ip"] = random_ip(len(df))

    # Shuffle
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\n[INFO] Dataset saved → {save_path}")
        print(f"[INFO] Shape: {df.shape}")
        print(f"\n[INFO] Class distribution:")
        print(df["attack_type"].value_counts().to_string())

    return df


if __name__ == "__main__":
    import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    save = os.path.join(RAW_DIR, "network_traffic.csv")
    generate_dataset(save_path=save)
