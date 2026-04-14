"""
models.py
---------
Three complementary threat detection models:

  1. IsolationForest  – Unsupervised anomaly detection
       • No labels required
       • Detects flows that are statistically "isolated" from normal traffic
       • Used in real SOCs for zero-day / unknown attack detection

  2. RandomForestClassifier – Supervised binary classification
       • BENIGN (0) vs ATTACK (1)
       • Fast, interpretable, handles imbalanced classes well
       • Feature importance shows WHICH network attributes matter most

  3. RandomForestClassifier – Supervised multi-class classification
       • Identifies the specific attack type (DoS, BruteForce, etc.)
       • Mirrors what commercial IDS products like Snort/Suricata report

All models are saved to disk after training so they can be reloaded for
real-time detection without re-training.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)

from src.config import (
    MODELS_DIR, N_ESTIMATORS, CONTAMINATION, RANDOM_SEED
)

# ─── Model paths ──────────────────────────────────────────────────────────────
ISOFOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
BINARY_RF_PATH = os.path.join(MODELS_DIR, "rf_binary.pkl")
MULTI_RF_PATH  = os.path.join(MODELS_DIR, "rf_multiclass.pkl")


# ─── 1. Isolation Forest (Unsupervised Anomaly Detection) ────────────────────

def train_isolation_forest(X_train: np.ndarray) -> IsolationForest:
    """
    Train Isolation Forest on BENIGN traffic only.

    In production, you train on a clean baseline of normal traffic, then
    detect deviations. Here we train on the full dataset to match the
    contamination proportion.

    Returns
    -------
    IsolationForest – fitted model
    """
    print("\n[MODEL 1] Training Isolation Forest (unsupervised) …")
    iso = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    iso.fit(X_train)
    joblib.dump(iso, ISOFOREST_PATH)
    print(f"[INFO] IsolationForest saved → {ISOFOREST_PATH}")
    return iso


def predict_isolation_forest(iso: IsolationForest,
                              X: np.ndarray) -> np.ndarray:
    """
    Predict anomaly labels.
    Returns array where 1 = anomaly (attack), 0 = normal.
    IsolationForest uses: +1 = inlier (normal), -1 = outlier (anomaly)
    """
    raw = iso.predict(X)
    # Convert: -1 → 1 (anomaly), +1 → 0 (normal)
    return (raw == -1).astype(int)


def isolation_forest_scores(iso: IsolationForest,
                             X: np.ndarray) -> np.ndarray:
    """
    Return anomaly scores (lower = more anomalous).
    Normalized to [0, 1] where 1 = most anomalous.
    """
    raw_scores = iso.decision_function(X)   # more negative = more anomalous
    # Normalize to [0,1], invert so higher = more suspicious
    norm = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8)
    return 1 - norm


# ─── 2. Random Forest Binary Classifier (BENIGN vs ATTACK) ───────────────────

def train_binary_classifier(X_train: np.ndarray,
                             y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train supervised binary threat classifier.

    class_weight='balanced' handles the BENIGN >> ATTACK imbalance.
    """
    print("\n[MODEL 2] Training Random Forest Binary Classifier …")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, BINARY_RF_PATH)
    print(f"[INFO] Binary RF saved → {BINARY_RF_PATH}")
    return rf


# ─── 3. Random Forest Multi-class Classifier (Attack Type) ───────────────────

def train_multiclass_classifier(X_train: np.ndarray,
                                 y_train: np.ndarray) -> RandomForestClassifier:
    """
    Train supervised multi-class attack-type classifier.
    """
    print("\n[MODEL 3] Training Random Forest Multi-class Classifier …")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=25,
        min_samples_split=4,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        verbose=0,
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, MULTI_RF_PATH)
    print(f"[INFO] Multi-class RF saved → {MULTI_RF_PATH}")
    return rf


# ─── Load helpers ─────────────────────────────────────────────────────────────

def load_isolation_forest() -> IsolationForest:
    return joblib.load(ISOFOREST_PATH)


def load_binary_classifier() -> RandomForestClassifier:
    return joblib.load(BINARY_RF_PATH)


def load_multiclass_classifier() -> RandomForestClassifier:
    return joblib.load(MULTI_RF_PATH)


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_binary(model, X_test: np.ndarray, y_test: np.ndarray,
                    model_name: str = "Binary RF") -> dict:
    """Full binary evaluation report."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] \
             if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=["BENIGN", "ATTACK"],
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n{'='*55}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred,
                                target_names=["BENIGN", "ATTACK"]))
    if auc:
        print(f"  AUC-ROC: {auc:.4f}")

    return {
        "accuracy": acc,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


def evaluate_multiclass(model, X_test: np.ndarray, y_test: np.ndarray,
                         class_names: list) -> dict:
    """Full multi-class evaluation report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"  Multi-class RF — Attack Type Classifier")
    print(f"{'='*55}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


def get_feature_importance(model: RandomForestClassifier,
                            feature_names: list) -> dict:
    """Return feature importances sorted descending."""
    importances = model.feature_importances_
    return dict(sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    ))
