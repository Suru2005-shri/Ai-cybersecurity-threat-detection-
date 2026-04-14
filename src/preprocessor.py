"""
preprocessor.py
---------------
Data cleaning and feature engineering pipeline.

Steps:
  1. Handle missing / infinite values
  2. Encode categorical features (protocol → integer)
  3. Engineer new features (ratios, flags summary, entropy proxies)
  4. Scale numeric features (StandardScaler)
  5. Return clean X, y for training

Used by:
  - train.py
  - detect.py (inference on new flows)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

from src.config import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    TARGET_BINARY, TARGET_MULTICLASS,
    PROCESSED_DIR, MODELS_DIR
)

SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")


# ─── Step 1: Clean raw data ───────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove / fix data quality issues.

    - Replace inf with NaN, then fill with column median
    - Clip extreme outliers (beyond 99.9th percentile) per numeric col
    - Drop duplicate rows
    """
    df = df.copy()

    # Replace infinite values
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Clip extreme outliers per column (99.9th percentile cap)
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            cap = df[col].quantile(0.999)
            df[col] = df[col].clip(upper=cap)

    # Drop duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    if before != after:
        print(f"[INFO] Removed {before - after} duplicate rows.")

    return df


# ─── Step 2: Feature Engineering ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that help the model detect subtle attack patterns.

    Engineered features:
      bwd_fwd_ratio     – imbalance between upload and download (exfil signal)
      bytes_per_packet  – average payload size
      flag_score        – weighted sum of anomalous TCP flags
      pkt_rate_ratio    – ratio of forward to backward packet rates
      iat_ratio         – ratio of forward to backward inter-arrival times
      port_risk         – categorical risk level of destination port
    """
    df = df.copy()

    # Byte ratio: high bwd/fwd ratio → possible data exfiltration
    total_fwd = df["fwd_bytes"].replace(0, 1)
    total_bwd = df["bwd_bytes"].replace(0, 1)
    df["bwd_fwd_ratio"] = total_bwd / total_fwd

    # Bytes per packet
    total_pkts = (df["fwd_packets"] + df["bwd_packets"]).replace(0, 1)
    total_bytes = df["fwd_bytes"] + df["bwd_bytes"]
    df["bytes_per_packet"] = total_bytes / total_pkts

    # Anomalous TCP flag score
    # SYN flood → high SYN; RST → connection rejection; URG rarely legitimate
    df["flag_score"] = (
        df["syn_flag_count"] * 2.0 +
        df["rst_flag_count"] * 1.5 +
        df["urg_flag_count"] * 3.0 +
        df["fin_flag_count"] * 0.5
    )

    # Packet rate asymmetry
    fwd_rate = (df["fwd_packets"] / (df["flow_duration"].replace(0, 1) / 1e6))
    bwd_rate = (df["bwd_packets"] / (df["flow_duration"].replace(0, 1) / 1e6))
    df["pkt_rate_ratio"] = fwd_rate / (bwd_rate.replace(0, 1))

    # IAT ratio (very consistent IAT → possible botnet beaconing)
    df["iat_ratio"] = df["fwd_iat_mean"] / (df["bwd_iat_mean"].replace(0, 1))

    # Port risk score
    risky_ports = {22: 3, 23: 4, 3389: 4, 445: 4, 6667: 5,
                   21: 3, 25: 2, 139: 3, 80: 1, 443: 1}
    df["port_risk"] = df["dst_port"].map(risky_ports).fillna(1)

    # Protocol encoding (TCP=2, UDP=1, ICMP=0)
    proto_map = {"TCP": 2, "UDP": 1, "ICMP": 0}
    df["protocol_enc"] = df["protocol"].map(proto_map).fillna(1)

    return df


# ─── Step 3: Scale and encode ─────────────────────────────────────────────────

# All feature columns used for ML (after engineering)
ENGINEERED_FEATURES = NUMERIC_FEATURES + [
    "bwd_fwd_ratio", "bytes_per_packet", "flag_score",
    "pkt_rate_ratio", "iat_ratio", "port_risk", "protocol_enc"
]


def fit_scaler(X: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler on training features and save to disk."""
    scaler = StandardScaler()
    scaler.fit(X[ENGINEERED_FEATURES])
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Scaler saved → {SCALER_PATH}")
    return scaler


def load_scaler() -> StandardScaler:
    """Load saved scaler."""
    return joblib.load(SCALER_PATH)


def scale_features(X: pd.DataFrame, scaler: StandardScaler = None,
                   fit: bool = False) -> np.ndarray:
    """Scale features. If fit=True, fits and saves scaler."""
    if fit or scaler is None:
        scaler = fit_scaler(X)
    return scaler.transform(X[ENGINEERED_FEATURES])


def fit_label_encoder(y: pd.Series) -> LabelEncoder:
    """Encode multi-class labels and save encoder."""
    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, ENCODER_PATH)
    print(f"[INFO] Label encoder saved → {ENCODER_PATH}")
    return le


def load_label_encoder() -> LabelEncoder:
    return joblib.load(ENCODER_PATH)


# ─── Full pipeline ────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               fit: bool = True) -> tuple:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    df  : raw DataFrame
    fit : bool – True for training (fit+save scaler), False for inference

    Returns
    -------
    X_scaled : np.ndarray – scaled feature matrix
    y_binary : np.ndarray – binary labels (0=BENIGN, 1=ATTACK)
    y_multi  : np.ndarray – multi-class integer labels
    le       : LabelEncoder – fitted encoder
    scaler   : StandardScaler – fitted scaler
    df_clean : cleaned DataFrame with engineered features
    """
    print("[STEP] Cleaning data …")
    df = clean_data(df)

    print("[STEP] Engineering features …")
    df = engineer_features(df)

    print("[STEP] Scaling features …")
    if fit:
        scaler = fit_scaler(df)
        le     = fit_label_encoder(df[TARGET_MULTICLASS])
    else:
        scaler = load_scaler()
        le     = load_label_encoder()

    X_scaled = scale_features(df, scaler, fit=False)
    y_binary = df[TARGET_BINARY].values
    y_multi  = le.transform(df[TARGET_MULTICLASS])

    print(f"[INFO] Feature matrix: {X_scaled.shape}")
    print(f"[INFO] Classes: {list(le.classes_)}")

    return X_scaled, y_binary, y_multi, le, scaler, df


# ─── Inference helper ─────────────────────────────────────────────────────────

def preprocess_single_flow(flow_dict: dict) -> np.ndarray:
    """
    Preprocess a single network flow dictionary for real-time inference.

    Parameters
    ----------
    flow_dict : dict – keys matching NUMERIC_FEATURES + 'protocol'

    Returns
    -------
    np.ndarray shape (1, n_features)
    """
    df = pd.DataFrame([flow_dict])
    df = engineer_features(df)

    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    scaler = load_scaler()
    return scale_features(df, scaler, fit=False)
