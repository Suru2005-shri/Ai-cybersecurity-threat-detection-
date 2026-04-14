# 🛡️ AI-Powered Cybersecurity Threat Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Scikit--learn-1.1+-orange?style=for-the-badge&logo=scikit-learn"/>
  <img src="https://img.shields.io/badge/Models-IsolationForest%20%7C%20RandomForest-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-CICIDS--2017%20Structure-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Task-Intrusion%20Detection-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <b>Network intrusion detection using Isolation Forest (unsupervised) + Random Forest (supervised)<br/>
  Detects 8 attack types · Real-time alert generation · SOC-style severity triage · Grad-CAM-style feature attribution</b>
</p>

---

## 📌 Overview

This project builds a **production-inspired AI threat detection pipeline** that simulates a real SOC (Security Operations Center) environment. The system analyzes network flow records and classifies them as **BENIGN** or one of **8 attack types**, generating severity-tiered alerts like a commercial IDS/IPS.

> 🏆 **Binary RF Accuracy: 99.98% · AUC-ROC: 1.0000 · IsolationForest: 86.70%**

The dual-model architecture mirrors what enterprise security tools like **Darktrace**, **CrowdStrike**, and **Cisco Stealthwatch** use in production:
- **Unsupervised** (IsolationForest) → detects zero-day / unknown threats
- **Supervised** (Random Forest) → classifies known attack signatures

---

## 🎯 Problem Statement

Cyberattacks cost organizations **$8 trillion globally in 2023**. Security teams face:

- **Alert fatigue**: hundreds of thousands of events per day, manually impossible to triage
- **Zero-day threats**: signature-based tools miss novel attacks
- **Slow detection**: average breach goes undetected for **207 days**
- **Analyst shortage**: 3.5 million unfilled cybersecurity positions globally

This system addresses all four by providing **AI-powered, real-time automated triage**.

---

## 🏭 Industry Relevance

| Sector | Real Problem | How This System Helps |
|--------|-------------|----------------------|
| Banking | Fraud & account takeover | Detect BruteForce + Infiltration attempts |
| E-commerce | DDoS during peak hours | Real-time DDoS + DoS classification |
| Enterprise IT | Lateral movement / APT | BotNet C2 beacon + PortScan detection |
| Cloud Providers | API abuse / SQLi | SQLInjection + XSS pattern detection |
| Telecom | Network infrastructure attacks | DDoS + traffic anomaly detection |

Companies building similar systems: **Darktrace, Palo Alto Cortex XDR, Splunk SIEM, IBM QRadar, Microsoft Sentinel**.

---

## 🧠 Architecture

```
Raw Network Flow (PCAP / NetFlow)
           │
           ▼
  ┌─────────────────────────┐
  │   DATA INGESTION        │
  │  CICFlowMeter / Zeek    │  ← In production; here: synthetic CSV
  │  50,000 flow records    │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   PREPROCESSING         │
  │  • Inf/NaN handling     │
  │  • Outlier clipping     │
  │  • Duplicate removal    │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   FEATURE ENGINEERING   │
  │  • bwd_fwd_ratio        │  ← Exfiltration signal
  │  • flag_score           │  ← Anomalous TCP flags
  │  • bytes_per_packet     │  ← Payload size indicator
  │  • port_risk            │  ← Destination port risk
  │  • pkt_rate_ratio       │  ← Traffic asymmetry
  │  27 total features      │
  └────────────┬────────────┘
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
  ┌──────────┐  ┌──────────────────┐
  │ ISOLATION│  │  RANDOM FOREST   │
  │  FOREST  │  │  ┌────────────┐  │
  │Unsupervis│  │  │  Binary    │  │ BENIGN / ATTACK
  │  Anomaly │  │  │Classifier  │  │
  │Detection │  │  └────────────┘  │
  │          │  │  ┌────────────┐  │
  │Zero-day  │  │  │ Multi-class│  │ DoS/DDoS/BruteForce
  │detection │  │  │Classifier  │  │ /PortScan/SQLi/XSS
  └──────┬───┘  │  │/BotNet/etc │  │
         │      │  └────────────┘  │
         │      └──────────────────┘
         │             │
         └──────┬───────┘
                ▼
  ┌─────────────────────────┐
  │   SEVERITY TRIAGE       │
  │  🔴 CRITICAL  prob>0.95 │
  │  🟠 HIGH      prob>0.80 │
  │  🟡 MEDIUM    prob>0.50 │
  │  🔵 LOW       iso flag  │
  │  🟢 INFO      benign    │
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │   ALERT GENERATION      │
  │  • CSV alert log        │
  │  • JSON scan summary    │
  │  • SOC dashboard        │
  └─────────────────────────┘
```

---

## 🗂 Project Structure

```
AI-Cybersecurity-Threat-Detection/
│
├── data/
│   ├── raw/                     # Raw network flow CSV (gitignored)
│   ├── processed/               # Processed arrays (gitignored)
│   └── sample/                  # Sample flows for demo
│
├── notebooks/
│   └── 01_full_walkthrough.ipynb  # Step-by-step Jupyter notebook
│
├── src/
│   ├── __init__.py
│   ├── config.py                # All hyperparameters and paths
│   ├── data_generator.py        # Synthetic CICIDS-2017-style traffic generator
│   ├── preprocessor.py          # Data cleaning + feature engineering pipeline
│   ├── models.py                # IsolationForest + RandomForest model builders
│   ├── train.py                 # End-to-end training orchestration
│   ├── detector.py              # Real-time detection engine + alert generator
│   └── visualize.py             # All SOC dashboard visualizations
│
├── models/                      # Saved .pkl model files
│   ├── isolation_forest.pkl
│   ├── rf_binary.pkl
│   ├── rf_multiclass.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── graphs/                  # All visualization PNGs
│   ├── alerts/                  # Alert CSV logs
│   └── reports/                 # JSON metrics + scan summaries
│
├── images/                      # README screenshots
├── docs/
│   ├── PROJECT_GUIDE.md
│   └── GITHUB_GUIDE.md
│
├── main.py                      # CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Tool | Purpose |
|-----------|------|---------|
| Language | Python 3.10+ | Core implementation |
| ML Framework | scikit-learn | IsolationForest + RandomForest |
| Data Processing | Pandas + NumPy | Feature engineering |
| Visualization | Matplotlib + Seaborn | SOC dashboards |
| Model Persistence | joblib | Save/load trained models |
| Dataset | Synthetic (CICIDS-2017 structure) | 50,000 labeled network flows |

---

## 📊 Dataset

**Structure mirrors:** [CICIDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html) and [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

| Attack Type | Count | Description |
|-------------|-------|-------------|
| BENIGN | 30,000 | Normal web/API traffic |
| DoS | 4,000 | Denial-of-Service SYN floods |
| PortScan | 3,500 | Attacker port reconnaissance |
| DDoS | 3,500 | Distributed flood attacks |
| BruteForce | 2,500 | SSH/login credential guessing |
| BotNet | 2,000 | C2 beacon traffic |
| SQLInjection | 2,000 | Database injection attempts |
| XSS | 1,500 | Cross-site scripting payloads |
| Infiltration | 1,000 | Data exfiltration patterns |
| **Total** | **50,000** | |

**27 features** including: packet rates, byte ratios, TCP flag counts, inter-arrival times, and 7 engineered features.

---

## 🛠 Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Detection.git
cd AI-Cybersecurity-Threat-Detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# OR: venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Full Pipeline (Recommended)
```bash
python main.py --mode all
```
Generates dataset → trains models → runs detection → creates all visualizations.

### Step by Step
```bash
# Generate 50,000 synthetic network flows
python main.py --mode generate

# Train all 3 models
python main.py --mode train

# Run batch threat detection
python main.py --mode detect

# Live terminal simulation (30 flows)
python main.py --mode simulate --flows 30

# Generate all SOC dashboard visuals
python main.py --mode visualize

# Print evaluation report
python main.py --mode report
```

### Run Notebook
```bash
jupyter notebook notebooks/01_full_walkthrough.ipynb
```

---

## 📈 Results

| Model | Task | Accuracy | AUC-ROC |
|-------|------|----------|---------|
| **IsolationForest** | Unsupervised anomaly detection | 86.70% | — |
| **Random Forest** | Binary (BENIGN vs ATTACK) | 99.98% | 1.0000 |
| **Random Forest** | Multi-class (9 attack types) | 100.00% | — |

### Alert Summary (10,000 flow scan)
| Severity | Count | Threshold |
|----------|-------|-----------|
| 🔴 CRITICAL | 269 | prob > 95% |
| 🟠 HIGH | 3,703 | prob > 80% |
| 🔵 LOW | 535 | IsoForest flag |
| 🟢 INFO | 5,493 | Normal traffic |

### Top 5 Most Important Features
1. `idle_mean` — idle time between bursts (botnet signal)
2. `pkt_len_mean` — average packet size
3. `active_mean` — active flow duration
4. `bwd_iat_mean` — backward inter-arrival time
5. `pkt_len_std` — packet size variance

---

## 🎓 Learning Outcomes

By studying this project you will understand:

- ✅ How **Isolation Forest** detects anomalies without labeled data (zero-day defense)
- ✅ How **Random Forest** classifies known attack patterns at high accuracy
- ✅ How to perform **feature engineering** for network traffic (ratios, flag scores, risk scoring)
- ✅ How **SOC alert triage** works (CRITICAL → HIGH → MEDIUM → LOW → INFO)
- ✅ How to build **modular, production-style ML pipelines** (not just notebooks)
- ✅ How to evaluate security models with **confusion matrices, ROC curves, F1 scores**
- ✅ How real tools like **Zeek/Bro, Suricata, and Darktrace** approach threat detection

---

## 🤝 Acknowledgements

- **Dataset structure:** CICIDS-2017 (Canadian Institute for Cybersecurity)
- **Reference:** Sharafaldin et al. "Toward Generating a New Intrusion Detection Dataset..." (2018)
- **IsolationForest:** Liu, Fei Tony et al. "Isolation Forest" (2008)

---

## 📜 License

MIT License — free to use and modify for educational and research purposes.

---

<p align="center">
  Built with ❤️ as a student project demonstrating industry-level AI security engineering.<br/>
  ⭐ Star this repo if it helped you learn!
</p>
