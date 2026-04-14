"""
train.py
--------
End-to-end training pipeline:

  1. Generate (or load) dataset
  2. Preprocess + feature engineering
  3. Train IsolationForest (unsupervised)
  4. Train Binary RF (BENIGN vs ATTACK)
  5. Train Multi-class RF (attack type)
  6. Evaluate all three models
  7. Save evaluation metrics to JSON

Usage
-----
  python main.py --mode train
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DIR, REPORTS_DIR, TEST_SIZE, RANDOM_SEED, N_SAMPLES
)
from src.data_generator import generate_dataset
from src.preprocessor import preprocess
from src.models import (
    train_isolation_forest,
    train_binary_classifier,
    train_multiclass_classifier,
    predict_isolation_forest,
    evaluate_binary,
    evaluate_multiclass,
    get_feature_importance,
)
from src.preprocessor import ENGINEERED_FEATURES


def train(n_samples: int = N_SAMPLES,
          force_regenerate: bool = False) -> dict:
    """
    Full training pipeline.

    Parameters
    ----------
    n_samples        : int  – size of synthetic dataset
    force_regenerate : bool – regenerate even if CSV exists

    Returns
    -------
    dict of all evaluation metrics
    """

    # ── 1. Dataset ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(RAW_DIR, "network_traffic.csv")
    if os.path.exists(csv_path) and not force_regenerate:
        import pandas as pd
        print(f"[INFO] Loading existing dataset from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"[INFO] Generating synthetic dataset ({n_samples:,} flows) …")
        df = generate_dataset(n_samples=n_samples, save_path=csv_path)

    print(f"\n[INFO] Dataset shape : {df.shape}")
    print(f"[INFO] Attack dist   :\n{df['attack_type'].value_counts().to_string()}\n")

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    X, y_bin, y_mul, le, scaler, df_clean = preprocess(df, fit=True)

    # ── 3. Train/test split ────────────────────────────────────────────────────
    (X_train, X_test,
     y_bin_train, y_bin_test,
     y_mul_train, y_mul_test) = train_test_split(
        X, y_bin, y_mul,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_bin,      # preserve class ratio
    )
    print(f"\n[INFO] Train size : {X_train.shape[0]:,}")
    print(f"[INFO] Test size  : {X_test.shape[0]:,}")

    # ── 4. Train Isolation Forest ──────────────────────────────────────────────
    # Train only on BENIGN samples (index where y_bin_train == 0)
    benign_mask = y_bin_train == 0
    iso = train_isolation_forest(X_train[benign_mask])

    iso_preds_test = predict_isolation_forest(iso, X_test)
    iso_results    = evaluate_binary(
        iso, X_test, y_bin_test,
        model_name="Isolation Forest"
    )
    # Override with our re-mapped predictions for proper reporting
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    iso_results["accuracy"]          = accuracy_score(y_bin_test, iso_preds_test)
    iso_results["confusion_matrix"]  = confusion_matrix(y_bin_test, iso_preds_test)
    iso_results["y_pred"]            = iso_preds_test

    # ── 5. Train Binary RF ─────────────────────────────────────────────────────
    rf_bin = train_binary_classifier(X_train, y_bin_train)
    bin_results = evaluate_binary(rf_bin, X_test, y_bin_test,
                                  model_name="Random Forest Binary")

    # ── 6. Train Multi-class RF ────────────────────────────────────────────────
    rf_mul = train_multiclass_classifier(X_train, y_mul_train)
    mul_results = evaluate_multiclass(rf_mul, X_test, y_mul_test,
                                      class_names=list(le.classes_))

    # ── 7. Feature importance ──────────────────────────────────────────────────
    fi = get_feature_importance(rf_bin, ENGINEERED_FEATURES)
    print("\n[INFO] Top 10 most important features:")
    for feat, score in list(fi.items())[:10]:
        bar = "█" * int(score * 300)
        print(f"  {feat:<25} {score:.4f}  {bar}")

    # ── 8. Save metrics ────────────────────────────────────────────────────────
    metrics = {
        "isolation_forest": {
            "accuracy": float(iso_results["accuracy"]),
        },
        "binary_rf": {
            "accuracy": float(bin_results["accuracy"]),
            "auc_roc":  float(bin_results["auc"]) if bin_results["auc"] else None,
            "f1_attack": float(bin_results["report"]["ATTACK"]["f1-score"]),
        },
        "multiclass_rf": {
            "accuracy": float(mul_results["accuracy"]),
        },
        "feature_importance": {k: float(v) for k, v in list(fi.items())[:15]},
        "train_samples": int(X_train.shape[0]),
        "test_samples":  int(X_test.shape[0]),
        "n_features":    int(X_train.shape[1]),
        "class_names":   list(le.classes_),
    }

    metrics_path = os.path.join(REPORTS_DIR, "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[INFO] Metrics saved → {metrics_path}")

    print("\n" + "="*55)
    print("  TRAINING COMPLETE — RESULTS SUMMARY")
    print("="*55)
    print(f"  IsolationForest  Accuracy : {metrics['isolation_forest']['accuracy']*100:.2f}%")
    print(f"  Binary RF        Accuracy : {metrics['binary_rf']['accuracy']*100:.2f}%")
    print(f"  Binary RF        AUC-ROC  : {metrics['binary_rf']['auc_roc']:.4f}")
    print(f"  Multi-class RF   Accuracy : {metrics['multiclass_rf']['accuracy']*100:.2f}%")
    print("="*55)

    return {
        "metrics":     metrics,
        "iso":         iso,
        "rf_bin":      rf_bin,
        "rf_mul":      rf_mul,
        "le":          le,
        "scaler":      scaler,
        "X_test":      X_test,
        "y_bin_test":  y_bin_test,
        "y_mul_test":  y_mul_test,
        "bin_results": bin_results,
        "mul_results": mul_results,
        "iso_results": iso_results,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    train()
