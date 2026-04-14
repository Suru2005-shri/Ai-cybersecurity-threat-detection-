"""
visualize.py
------------
All visualization outputs for the threat detection system.

Generates (all saved to outputs/graphs/):
  1. dataset_overview.png          – class distribution, feature correlations
  2. feature_importance.png        – top ML features for threat detection
  3. confusion_matrix_binary.png   – binary classifier CM
  4. confusion_matrix_multi.png    – multi-class classifier CM
  5. roc_curve.png                 – AUC-ROC curve
  6. anomaly_scores.png            – IsolationForest score distribution
  7. threat_timeline.png           – simulated 24-hour attack timeline
  8. attack_heatmap.png            – attack type vs time heatmap
  9. threat_dashboard.png          – executive summary dashboard
  10. alert_severity_donut.png     – severity breakdown donut chart
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from src.config import GRAPHS_DIR, ATTACK_TYPES

# ─── Palette ──────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL_BG = "#161b22"
BORDER   = "#30363d"
WHITE    = "#e6edf3"
MUTED    = "#8b949e"

SEV_COLORS = {
    "CRITICAL": "#ff4444",
    "HIGH":     "#ff8c00",
    "MEDIUM":   "#ffd700",
    "LOW":      "#4da6ff",
    "INFO":     "#2ecc71",
}

ATTACK_COLORS = {
    "BENIGN":      "#2ecc71",
    "DoS":         "#e74c3c",
    "DDoS":        "#c0392b",
    "PortScan":    "#3498db",
    "BruteForce":  "#e67e22",
    "SQLInjection":"#9b59b6",
    "XSS":         "#f39c12",
    "BotNet":      "#1abc9c",
    "Infiltration":"#e91e63",
}


def _save(fig, filename):
    path = os.path.join(GRAPHS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓ {filename}")
    return path


def _styled_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.set_title(title, color=WHITE, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=10)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(alpha=0.12, color=WHITE)
    return ax


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dataset overview
# ──────────────────────────────────────────────────────────────────────────────

def plot_dataset_overview(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Dataset Overview — Network Traffic Analysis",
                 color=WHITE, fontsize=16, fontweight="bold", y=1.02)

    # ── Attack type distribution ──
    ax = axes[0]
    counts = df["attack_type"].value_counts()
    colors = [ATTACK_COLORS.get(c, "#555") for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, alpha=0.88)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_width() + 100, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", color=MUTED, fontsize=9)
    _styled_ax(ax, "Flow Count by Category", "Count", "Attack Type")

    # ── Binary split ──
    ax = axes[1]
    bin_counts = df["label_binary"].value_counts()
    labels = ["BENIGN", "ATTACK"]
    vals   = [bin_counts.get(0,0), bin_counts.get(1,0)]
    colors2 = ["#2ecc71", "#e74c3c"]
    wedges, texts, autos = ax.pie(
        vals, labels=labels, colors=colors2, autopct="%1.1f%%",
        startangle=90,
        textprops={"color": WHITE, "fontsize": 11},
        wedgeprops={"linewidth": 2, "edgecolor": BG},
    )
    for at in autos: at.set_fontweight("bold")
    ax.set_facecolor(PANEL_BG)
    ax.set_title("BENIGN vs ATTACK Split", color=WHITE, fontsize=12, fontweight="bold")

    # ── Packet rate distribution ──
    ax = axes[2]
    for atype in ["BENIGN", "DoS", "DDoS", "PortScan"]:
        subset = df[df["attack_type"] == atype]["flow_pkts_per_sec"]
        subset = subset[subset < subset.quantile(0.99)]
        ax.hist(subset, bins=40, alpha=0.65,
                color=ATTACK_COLORS.get(atype, "#aaa"), label=atype)
    _styled_ax(ax, "Packet Rate Distribution\n(key feature)", "Packets/sec", "Count")
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=BORDER, fontsize=9)

    plt.tight_layout()
    return _save(fig, "dataset_overview.png")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Feature importance
# ──────────────────────────────────────────────────────────────────────────────

def plot_feature_importance(fi_dict: dict) -> str:
    items = list(fi_dict.items())[:15]
    features, scores = zip(*items)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(BG)

    # Gradient colors by importance
    cmap = plt.colormaps["RdYlGn"]
    norm_scores = np.array(scores) / max(scores)
    colors = [cmap(s) for s in norm_scores]

    bars = ax.barh(range(len(features)), scores, color=colors, alpha=0.9)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, color=WHITE, fontsize=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", color=MUTED, fontsize=9)

    _styled_ax(ax, "Top 15 Features — Random Forest Importance Score",
               "Importance Score", "Feature")
    ax.invert_yaxis()

    plt.tight_layout()
    return _save(fig, "feature_importance.png")


# ──────────────────────────────────────────────────────────────────────────────
# 3 & 4. Confusion matrices
# ──────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_binary(cm: np.ndarray, title: str = "Binary Classifier") -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle(f"Confusion Matrix — {title}", color=WHITE, fontsize=14, fontweight="bold")

    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    for ax, data, fmt, subtitle in zip(
        axes, [cm, cm_pct], ["d", ".1f"],
        ["Raw Counts", "Normalised (%)"]
    ):
        ax.set_facecolor(PANEL_BG)
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=["BENIGN", "ATTACK"],
                    yticklabels=["BENIGN", "ATTACK"],
                    ax=ax, linewidths=0.5, linecolor=BG,
                    annot_kws={"size": 15, "weight": "bold"})
        ax.set_title(subtitle, color=WHITE, fontsize=12, pad=8)
        ax.set_xlabel("Predicted", color=MUTED)
        ax.set_ylabel("Actual", color=MUTED)
        ax.tick_params(colors=WHITE)

    plt.tight_layout()
    return _save(fig, "confusion_matrix_binary.png")


def plot_confusion_matrix_multi(cm: np.ndarray, class_names: list) -> str:
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    cm_pct = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5, linecolor=BG,
                annot_kws={"size": 11, "weight": "bold"})
    ax.set_title("Multi-class Attack Classifier — Confusion Matrix (%)",
                 color=WHITE, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Attack Type", color=MUTED, fontsize=11)
    ax.set_ylabel("Actual Attack Type",    color=MUTED, fontsize=11)
    ax.tick_params(colors=WHITE)

    plt.tight_layout()
    return _save(fig, "confusion_matrix_multi.png")


# ──────────────────────────────────────────────────────────────────────────────
# 5. ROC Curve
# ──────────────────────────────────────────────────────────────────────────────

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    ax.plot(fpr, tpr, color="#3498db", lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0,1],[0,1], color="#555", ls="--", lw=1, label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.10, color="#3498db")

    opt = np.argmax(tpr - fpr)
    ax.scatter(fpr[opt], tpr[opt], color="#e74c3c", s=90, zorder=5,
               label=f"Optimal threshold ({fpr[opt]:.3f}, {tpr[opt]:.3f})")

    _styled_ax(ax, "ROC Curve — Threat Detector",
               "False Positive Rate", "True Positive Rate (Detection Rate)")
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=BORDER, fontsize=10)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    plt.tight_layout()
    return _save(fig, "roc_curve.png")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Anomaly score distribution
# ──────────────────────────────────────────────────────────────────────────────

def plot_anomaly_scores(scores: np.ndarray, y_true: np.ndarray) -> str:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    benign_scores  = scores[y_true == 0]
    attack_scores  = scores[y_true == 1]

    ax.hist(benign_scores, bins=60, alpha=0.7, color="#2ecc71",
            label=f"BENIGN (n={len(benign_scores):,})", density=True)
    ax.hist(attack_scores, bins=60, alpha=0.7, color="#e74c3c",
            label=f"ATTACK (n={len(attack_scores):,})", density=True)
    ax.axvline(0.5, color="#ffd700", ls="--", lw=2, label="Alert threshold (0.5)")

    _styled_ax(ax, "IsolationForest Anomaly Score Distribution",
               "Anomaly Score (higher = more suspicious)", "Density")
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=BORDER, fontsize=10)

    # Annotation
    ax.text(0.75, ax.get_ylim()[1] * 0.85, "⚠ HIGH RISK ZONE",
            color="#e74c3c", fontsize=12, fontweight="bold", ha="center")

    plt.tight_layout()
    return _save(fig, "anomaly_scores.png")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Threat timeline
# ──────────────────────────────────────────────────────────────────────────────

def plot_threat_timeline(df: pd.DataFrame) -> str:
    """Simulated 24-hour attack timeline from timestamp column."""
    df = df.copy()
    if "timestamp" not in df.columns:
        return ""

    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))
    fig.patch.set_facecolor(BG)
    fig.suptitle("24-Hour Network Threat Timeline",
                 color=WHITE, fontsize=16, fontweight="bold")

    # ── Top: Total flows per hour ──
    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    benign_h = df[df["label_binary"]==0].groupby("hour").size()
    attack_h = df[df["label_binary"]==1].groupby("hour").size()
    hours = range(24)
    ax.bar(hours, [benign_h.get(h,0) for h in hours],
           color="#2ecc71", alpha=0.8, label="Benign")
    ax.bar(hours, [attack_h.get(h,0) for h in hours],
           bottom=[benign_h.get(h,0) for h in hours],
           color="#e74c3c", alpha=0.8, label="Attack")
    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, color=MUTED, fontsize=8)
    _styled_ax(ax, "Network Flows per Hour (Benign vs Attack)",
               "Hour of Day", "Flow Count")
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=BORDER)

    # ── Bottom: Attack type breakdown ──
    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    attack_df = df[df["label_binary"]==1]
    bottom = np.zeros(24)
    for atype in ATTACK_TYPES[1:]:   # skip BENIGN
        counts = attack_df[attack_df["attack_type"]==atype].groupby("hour").size()
        vals = [counts.get(h,0) for h in range(24)]
        ax.bar(hours, vals, bottom=bottom,
               color=ATTACK_COLORS.get(atype, "#aaa"), alpha=0.85, label=atype)
        bottom += np.array(vals, dtype=float)

    ax.set_xlim(-0.5, 23.5)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                       rotation=45, color=MUTED, fontsize=8)
    _styled_ax(ax, "Attack Type Timeline",
               "Hour of Day", "Attack Count")
    ax.legend(facecolor=PANEL_BG, labelcolor=WHITE, edgecolor=BORDER,
              fontsize=9, loc="upper left", ncol=2)

    plt.tight_layout()
    return _save(fig, "threat_timeline.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Alert severity donut
# ──────────────────────────────────────────────────────────────────────────────

def plot_severity_donut(severity_counts: dict) -> str:
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL_BG)

    severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    values     = [severity_counts.get(s, 0) for s in severities]
    colors     = [SEV_COLORS[s] for s in severities]

    wedges, texts, autos = ax.pie(
        values,
        labels=severities,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.78,
        wedgeprops={"linewidth": 3, "edgecolor": BG, "width": 0.55},
        textprops={"color": WHITE, "fontsize": 12, "fontweight": "bold"},
    )
    for at in autos:
        at.set_color(BG); at.set_fontweight("bold"); at.set_fontsize(10)

    total = sum(values)
    ax.text(0, 0, f"{total:,}\nALERTS", ha="center", va="center",
            color=WHITE, fontsize=16, fontweight="bold")

    ax.set_title("Alert Severity Distribution",
                 color=WHITE, fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return _save(fig, "alert_severity_donut.png")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Executive Dashboard
# ──────────────────────────────────────────────────────────────────────────────

def plot_executive_dashboard(metrics: dict, severity_counts: dict,
                              df: pd.DataFrame) -> str:
    """Full-page SOC executive dashboard."""
    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(BG)
    fig.suptitle("AI Cybersecurity Threat Detection — SOC Dashboard",
                 color=WHITE, fontsize=22, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(3, 5, figure=fig,
                           hspace=0.55, wspace=0.40,
                           left=0.04, right=0.98,
                           top=0.91, bottom=0.06)

    # ── Metric cards (row 0) ──────────────────────────────────────────────────
    kpis = [
        ("Binary RF\nAccuracy",   f"{metrics['binary_rf']['accuracy']*100:.2f}%",   "#3498db"),
        ("AUC-ROC",                f"{metrics['binary_rf']['auc_roc']:.4f}",          "#9b59b6"),
        ("Multi-class\nAccuracy", f"{metrics['multiclass_rf']['accuracy']*100:.2f}%","#2ecc71"),
        ("CRITICAL\nAlerts",      f"{severity_counts.get('CRITICAL',0):,}",           "#ff4444"),
        ("HIGH\nAlerts",          f"{severity_counts.get('HIGH',0):,}",               "#ff8c00"),
    ]
    for col, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(PANEL_BG)
        ax.axis("off")
        ax.text(0.5, 0.62, value,  transform=ax.transAxes,
                ha="center", va="center", fontsize=22, fontweight="bold", color=color)
        ax.text(0.5, 0.20, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color=MUTED)
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_edgecolor(color); spine.set_linewidth(2)

    # ── Attack distribution bar (row 1, cols 0-1) ────────────────────────────
    ax = fig.add_subplot(gs[1, :2])
    counts = df["attack_type"].value_counts()
    colors = [ATTACK_COLORS.get(c, "#aaa") for c in counts.index]
    ax.barh(counts.index, counts.values, color=colors, alpha=0.85)
    _styled_ax(ax, "Traffic Breakdown by Category", "Flow Count", "")
    ax.set_facecolor(PANEL_BG)

    # ── Severity donut (row 1, col 2) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ax.set_facecolor(PANEL_BG)
    sevs = ["CRITICAL","HIGH","MEDIUM","LOW"]
    vals = [severity_counts.get(s,1) for s in sevs]
    wedges, _, autos = ax.pie(
        vals, colors=[SEV_COLORS[s] for s in sevs],
        autopct="%1.0f%%", startangle=90,
        wedgeprops={"linewidth":2,"edgecolor":BG,"width":0.55},
        pctdistance=0.75,
        textprops={"color":BG,"fontsize":9,"fontweight":"bold"},
    )
    ax.text(0,0,f"{sum(vals):,}\nAlerts", ha="center",va="center",
            color=WHITE, fontsize=11, fontweight="bold")
    ax.set_title("Severity Mix", color=WHITE, fontsize=11, fontweight="bold")

    # ── Feature importance (row 1, cols 3-4) ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 3:])
    ax.set_facecolor(PANEL_BG)
    fi   = metrics.get("feature_importance", {})
    feats = list(fi.keys())[:8]
    fscores = [fi[f] for f in feats]
    cmap = plt.colormaps["RdYlGn"]
    norm = np.array(fscores) / (max(fscores) + 1e-8)
    ax.barh(feats, fscores, color=[cmap(n) for n in norm], alpha=0.88)
    ax.invert_yaxis()
    _styled_ax(ax, "Top Features (RF Importance)", "Score", "")
    ax.set_facecolor(PANEL_BG)

    # ── 24-hr timeline (row 2, full width) ───────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    ax.set_facecolor(PANEL_BG)
    if "timestamp" in df.columns:
        df_t = df.copy()
        df_t["hour"] = pd.to_datetime(df_t["timestamp"]).dt.hour
        benign_h = df_t[df_t["label_binary"]==0].groupby("hour").size()
        attack_h = df_t[df_t["label_binary"]==1].groupby("hour").size()
        hours = range(24)
        ax.bar(hours, [benign_h.get(h,0) for h in hours],
               color="#2ecc71", alpha=0.7, label="Benign", width=0.8)
        ax.bar(hours, [attack_h.get(h,0) for h in hours],
               bottom=[benign_h.get(h,0) for h in hours],
               color="#e74c3c", alpha=0.8, label="Attack", width=0.8)
        ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(24))
        ax.set_xticklabels([f"{h:02d}:00" for h in range(24)],
                           rotation=45, color=MUTED, fontsize=8)
        _styled_ax(ax, "24-Hour Traffic Timeline", "Hour", "Flows")
        ax.legend(facecolor=PANEL_BG, labelcolor=WHITE,
                  edgecolor=BORDER, fontsize=9)

    return _save(fig, "threat_dashboard.png")


# ──────────────────────────────────────────────────────────────────────────────
# Master function — generate all outputs
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_visuals(df, metrics, bin_results, mul_results, iso_results,
                          class_names, iso_scores_arr):
    """Generate every visualization in one call."""
    print("\n[VIZ] Generating all visualizations …")

    from src.models import get_feature_importance, load_binary_classifier
    from src.preprocessor import ENGINEERED_FEATURES

    rf_bin = load_binary_classifier()
    fi     = get_feature_importance(rf_bin, ENGINEERED_FEATURES)

    # Severity simulation
    y_prob = bin_results["y_prob"]
    sev_counts = {
        "CRITICAL": int(np.sum(y_prob > 0.95)),
        "HIGH":     int(np.sum((y_prob > 0.80) & (y_prob <= 0.95))),
        "MEDIUM":   int(np.sum((y_prob > 0.50) & (y_prob <= 0.80))),
        "LOW":      int(np.sum((y_prob > 0.30) & (y_prob <= 0.50))),
        "INFO":     int(np.sum(y_prob <= 0.30)),
    }

    plot_dataset_overview(df)
    plot_feature_importance(fi)
    plot_confusion_matrix_binary(bin_results["confusion_matrix"], "Random Forest")
    plot_confusion_matrix_multi(mul_results["confusion_matrix"], class_names)
    plot_roc_curve(bin_results.get("y_true", []),
                   bin_results.get("y_prob", []))
    plot_anomaly_scores(iso_scores_arr, iso_results.get("y_true", []))
    plot_threat_timeline(df)
    plot_severity_donut(sev_counts)
    plot_executive_dashboard(metrics, sev_counts, df)

    print("[VIZ] All visualizations saved to outputs/graphs/")
