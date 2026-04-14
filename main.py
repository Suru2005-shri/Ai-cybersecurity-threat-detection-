"""
main.py
=======
CLI entry point for AI Cybersecurity Threat Detection System.

Usage
-----
  python main.py --mode generate     # generate synthetic dataset only
  python main.py --mode train        # train all three models
  python main.py --mode detect       # run batch detection on dataset
  python main.py --mode simulate     # live detection simulation (terminal)
  python main.py --mode visualize    # generate all graphs (requires trained models)
  python main.py --mode all          # full pipeline: generate→train→detect→visualize
  python main.py --mode report       # print evaluation summary from saved metrics
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="AI Cybersecurity Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode",
        choices=["generate", "train", "detect", "simulate", "visualize", "all", "report"],
        required=True,
    )
    p.add_argument("--samples", type=int, default=50_000,
                   help="Number of synthetic flow records to generate (default: 50000)")
    p.add_argument("--flows",   type=int, default=30,
                   help="Number of flows for live simulation (default: 30)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Generate ────────────────────────────────────────────────────────────
    if args.mode in ("generate", "all"):
        print("\n[STEP] Generating synthetic dataset …")
        from src.config import RAW_DIR
        from src.data_generator import generate_dataset
        save = os.path.join(RAW_DIR, "network_traffic.csv")
        df = generate_dataset(n_samples=args.samples, save_path=save)

    # ── Train ────────────────────────────────────────────────────────────────
    if args.mode in ("train", "all"):
        print("\n[STEP] Training models …")
        from src.train import train
        results = train(n_samples=args.samples)

    # ── Detect ───────────────────────────────────────────────────────────────
    if args.mode in ("detect", "all"):
        print("\n[STEP] Running batch threat detection …")
        import pandas as pd
        from src.config import RAW_DIR, OUTPUTS_DIR
        from src.detector import ThreatDetector

        csv_path = os.path.join(RAW_DIR, "network_traffic.csv")
        if not os.path.exists(csv_path):
            print("[ERROR] Dataset not found. Run --mode generate first.")
            sys.exit(1)

        df = pd.read_csv(csv_path).head(5000)   # use 5k for speed demo
        detector = ThreatDetector()
        result_df = detector.detect_batch(df, save_alerts=True)

        top = detector.get_top_threats(result_df, n=10)
        print("\n[TOP 10 HIGHEST RISK FLOWS]")
        print(top.to_string(index=False))

    # ── Simulate ─────────────────────────────────────────────────────────────
    if args.mode == "simulate":
        from src.detector import ThreatDetector, simulate_live_detection
        detector = ThreatDetector()
        simulate_live_detection(detector, n_flows=args.flows)

    # ── Visualize ─────────────────────────────────────────────────────────────
    if args.mode in ("visualize", "all"):
        print("\n[STEP] Generating visualizations …")
        _run_visualize()

    # ── Report ───────────────────────────────────────────────────────────────
    if args.mode == "report":
        import json
        from src.config import REPORTS_DIR
        mpath = os.path.join(REPORTS_DIR, "training_metrics.json")
        if not os.path.exists(mpath):
            print("[ERROR] No metrics found. Run --mode train first.")
            sys.exit(1)
        with open(mpath) as f:
            m = json.load(f)
        print("\n" + "="*55)
        print("  EVALUATION REPORT")
        print("="*55)
        print(f"  IsolationForest  Accuracy  : {m['isolation_forest']['accuracy']*100:.2f}%")
        print(f"  Binary RF        Accuracy  : {m['binary_rf']['accuracy']*100:.2f}%")
        print(f"  Binary RF        AUC-ROC   : {m['binary_rf']['auc_roc']:.4f}")
        print(f"  Binary RF        F1-Attack  : {m['binary_rf']['f1_attack']:.4f}")
        print(f"  Multi-class RF   Accuracy  : {m['multiclass_rf']['accuracy']*100:.2f}%")
        print(f"\n  Top Features:")
        for feat, score in list(m["feature_importance"].items())[:8]:
            print(f"    {feat:<28} {score:.4f}")
        print("="*55)


def _run_visualize():
    """Run visualization with all required data."""
    import json
    import pandas as pd
    import numpy as np
    from src.config import RAW_DIR, REPORTS_DIR
    from src.models import (
        load_binary_classifier, load_multiclass_classifier, load_isolation_forest,
        isolation_forest_scores, evaluate_binary, evaluate_multiclass
    )
    from src.preprocessor import preprocess, load_label_encoder, ENGINEERED_FEATURES
    from src.visualize import generate_all_visuals

    # Load dataset
    csv = os.path.join(RAW_DIR, "network_traffic.csv")
    df  = pd.read_csv(csv)
    X, y_bin, y_mul, le, scaler, df_clean = preprocess(df, fit=False)

    from sklearn.model_selection import train_test_split
    from src.config import TEST_SIZE, RANDOM_SEED
    _, X_test, _, y_bin_test, _, y_mul_test = train_test_split(
        X, y_bin, y_mul, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=y_bin
    )

    rf_bin = load_binary_classifier()
    rf_mul = load_multiclass_classifier()
    iso    = load_isolation_forest()

    bin_results = evaluate_binary(rf_bin, X_test, y_bin_test)
    bin_results["y_true"] = y_bin_test

    mul_results = evaluate_multiclass(rf_mul, X_test, y_mul_test, list(le.classes_))
    mul_results["y_true"] = y_mul_test

    iso_preds  = (iso.predict(X_test) == -1).astype(int)
    iso_scores = isolation_forest_scores(iso, X_test)
    iso_results = {"y_true": y_bin_test, "y_pred": iso_preds}

    with open(os.path.join(REPORTS_DIR, "training_metrics.json")) as f:
        metrics = json.load(f)

    generate_all_visuals(df, metrics, bin_results, mul_results,
                         iso_results, list(le.classes_), iso_scores)


if __name__ == "__main__":
    main()
