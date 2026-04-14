"""
config.py
---------
Central configuration for AI Cybersecurity Threat Detection System.
Edit values here to adapt to different datasets or models.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DIR         = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR   = os.path.join(DATA_DIR, "processed")
SAMPLE_DIR      = os.path.join(DATA_DIR, "sample")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
GRAPHS_DIR      = os.path.join(OUTPUTS_DIR, "graphs")
ALERTS_DIR      = os.path.join(OUTPUTS_DIR, "alerts")
REPORTS_DIR     = os.path.join(OUTPUTS_DIR, "reports")

# ─── Dataset ──────────────────────────────────────────────────────────────────
# We use a synthetic dataset that mirrors the UNSW-NB15 / CICIDS structure
# Columns simulate real network flow features used in industry SOC systems
RANDOM_SEED     = 42
N_SAMPLES       = 50_000          # total synthetic traffic records

# Attack types (mirrors CICIDS 2017 labels)
ATTACK_TYPES = [
    "BENIGN",
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "SQLInjection",
    "XSS",
    "BotNet",
    "Infiltration",
]

# For binary classification
BINARY_LABEL_MAP = {
    "BENIGN": 0,
    "DoS": 1, "DDoS": 1, "PortScan": 1,
    "BruteForce": 1, "SQLInjection": 1, "XSS": 1,
    "BotNet": 1, "Infiltration": 1,
}

# ─── Feature groups ────────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "flow_duration",       # microseconds
    "fwd_packets",         # forward packet count
    "bwd_packets",         # backward packet count
    "fwd_bytes",           # forward bytes total
    "bwd_bytes",           # backward bytes total
    "flow_bytes_per_sec",  # bytes per second
    "flow_pkts_per_sec",   # packets per second
    "fwd_iat_mean",        # inter-arrival time (mean) forward
    "bwd_iat_mean",        # inter-arrival time (mean) backward
    "pkt_len_mean",        # mean packet length
    "pkt_len_std",         # std packet length
    "fin_flag_count",      # TCP FIN flags
    "syn_flag_count",      # TCP SYN flags
    "rst_flag_count",      # TCP RST flags
    "ack_flag_count",      # TCP ACK flags
    "psh_flag_count",      # TCP PSH flags
    "urg_flag_count",      # TCP URG flags
    "active_mean",         # active time mean
    "idle_mean",           # idle time mean
    "dst_port",            # destination port
]

CATEGORICAL_FEATURES = ["protocol"]   # TCP, UDP, ICMP

TARGET_BINARY     = "label_binary"
TARGET_MULTICLASS = "attack_type"

# ─── Model settings ────────────────────────────────────────────────────────────
TEST_SIZE         = 0.20
CONTAMINATION     = 0.08     # IsolationForest: expected fraction of anomalies
N_ESTIMATORS      = 200      # Random Forest / IsolationForest trees

# Alert thresholds
HIGH_RISK_THRESHOLD   = 0.80   # probability above which → HIGH alert
MEDIUM_RISK_THRESHOLD = 0.50   # probability above which → MEDIUM alert

# ─── Ensure directories exist ──────────────────────────────────────────────────
for _dir in [RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR,
             GRAPHS_DIR, ALERTS_DIR, REPORTS_DIR]:
    os.makedirs(_dir, exist_ok=True)
