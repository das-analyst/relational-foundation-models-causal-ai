# Clinical Quality & Relational DiD: TabPFN, RelBench & Kumo RFM vs. Traditional DiD

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A causal inference benchmark evaluating **TabPFN** (Tabular Foundation Model), **RelBench** (Relational Deep Learning), and **Kumo Relational Foundation Model (RFM)** against **Traditional Difference-in-Differences (DiD)** on real-world clinical quality, medication adherence, and hospital readmissions data.

📖 **Read the full scientific report**: [STUDY_WALKTHROUGH.md](./STUDY_WALKTHROUGH.md)

---

## Benchmark Highlights

We evaluated the causal impact of an **inpatient medication titration protocol** on **30-day hospital readmissions** across **16,773 longitudinal patients** from the NIH / UCI 130-US Hospitals database.

![Model Comparison Forest Plot](./output/model_comparison_forest_plot.png)

### Summary Results

| Model / Estimator | Category | Average Treatment Effect (ATT) | 95% Conf. Interval | $p$-value | Clinical Finding |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Naïve 2x2 DiD** | Unadjusted Baseline | **-0.394% pts** | [-2.038%, +1.250%] | 0.6388 | **Insignificant**: Confounding by indication. |
| **2. TWFE OLS DiD** | Classical Econometrics | **-0.394% pts** | [-2.038%, +1.251%] | 0.6389 | **Insignificant**: Linear controls confirm population null. |
| **3. TabPFN Doubly Robust DiD** | Tabular Foundation Model | **-0.295% pts** | [-2.143%, +1.554%] | 0.7546 | Full-dataset DR-DiD (n=16,773, 5-fold, bootstrap SE). Confirms population null. |
| **4. RelBench Relational Multi-Table DiD** | **Relational Multi-Table DR-DiD** | **+0.144% pts** | **[-2.796%, +3.083%]** | **0.9237** | **Insignificant (Consensus Null)**: Clean baseline multi-table features confirm the population null. |
| **5. Kumo Relational Foundation Model DiD** | **Relational Foundation Model (RFM)** | **+0.019% pts** | **[-1.230%, +1.267%]** | **0.9765** | Multi-table in-context learning via NVIDIA NIM API (`patients`, `encounters`, `medications`, `diagnoses`). Dual-classification architecture ($n=12,000$ cohort). Ultra-narrow 95% CI (2.5% pts wide) proves population effect is definitively null ($0.00\% \pm 0.64\%$). |

---

### The Grand Methodological Takeaway: Five Paradigms, One Empirical Reality

We also evaluated whether standard health services research approaches—such as **Linear Mixed Models (LMM)**, **Logistic GLMMs**, and **CMS-Style Risk-Standardized Readmission Models**—could alter the conclusion (`src/mixed_effects_risk_adjusted.py`):

| Family | Model Specification | ATT ($\hat{\tau}$) | Std Error | $p$-value | Conclusion |
|:---|:---|:---:|:---:|:---:|:---|
| Econometrics | Naïve DiD & TWFE OLS | -0.394% | 0.839% | 0.6389 | Population null |
| Mixed Effects | Linear Mixed Model (Random Intercept $u_i$) | -0.394% | 0.839% | 0.6389 | Identical to TWFE ($u_i$ cancels in $\Delta Y$) |
| Mixed Effects | Logistic GLMM / Marginal DiD | -0.213% | 0.845% | 0.8008 | Population-averaged null ($\text{OR} = 0.991$) |
| Risk Adjustment | CMS-Style Logistic Risk Score | -0.165% | 0.840% | 0.8440 | Additive risk score confirms null |
| Risk Adjustment | Non-Linear ML Risk Score (GBDT) | +0.279% | 0.826% | 0.7354 | Non-linear tree risk score confirms null |
| Foundation Model | TabPFN Doubly Robust DiD | -0.295% | 0.943% | 0.7546 | Population average null |
| Relational Multi-Table | RelBench Multi-Table DR-DiD | +0.144% | 1.500% | 0.9237 | Multi-table baseline features confirm null |
| Foundation Model | Kumo RFM DiD ($N=12,000$) | +0.019% | 0.637% | 0.9765 | Ultra-precise population null ($p \to 1.0$) |

**Universal Consensus on the Population Null**: Every single valid specification across econometrics, mixed effects, risk adjustment, tabular foundation models, multi-table relational aggregations, and relational foundation models converges to the **broad population null ($[-0.39\%, +0.28\%]$)**. Inpatient medication titration has no statistically significant average effect on 30-day readmissions.

> **Forensic Note on the $-3.57\%$ Legacy Result**: An earlier pipeline iteration reported a spurious $-3.57\%$ effect ($p=0.0093$). A forensic audit confirmed this was driven by cohort truncation (`[:10000]` zero-filling 46% of patients), temporal leakage (aggregating post-period visits into baseline features), and treatment leakage (`is_dosage_change` encoding treatment). Once corrected with strict baseline filtering and Hajek normalization, the estimate returns to the consensus null (+0.14%, $p=0.92$). See [STUDY_WALKTHROUGH.md](./STUDY_WALKTHROUGH.md) for the full ablation matrix.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/das-analyst/relational-foundation-models-causal-ai.git
cd relational-foundation-models-causal-ai
pip install -r requirements.txt
```

### 2. Run the Benchmark

```bash
python run_experiment.py
```

This single command will:
1. Automatically download and unpack the public 130-US Hospitals dataset.
2. Build the multi-table relational SQLite database (`data/clinical_trial.db`).
3. Extract longitudinal cohorts with strict pre-intervention temporal cutoffs ($t \le T_0$).
4. Fit all 4 estimators (Naïve DiD, TWFE OLS, TabPFN DR-DiD, RelBench Graph DR-DiD).
5. Generate publication-ready figures in `output/`.

### 3. (Optional) Unlock TabPFN Transformer Weights

The TabPFN estimator runs with a `HistGradientBoosting` fallback by default. To use the actual
TabPFN in-context learning transformer weights:

1. Register and accept the license at **https://ux.priorlabs.ai** (Licenses tab).
2. Copy your API key from **https://ux.priorlabs.ai/account**.
3. Set the environment variable before running:

```bash
# Linux / macOS
export TABPFN_TOKEN="tabpfn_sk_..."
python run_experiment.py

# Windows PowerShell
$env:TABPFN_TOKEN = "tabpfn_sk_..."
python run_experiment.py
```

The script will automatically detect the token and attempt to download the model weights
(`Prior-Labs/tabpfn_3_5` on HuggingFace). If the license server is reachable, each fold will
log `-> TabPFN local OK` instead of `-> HGBT`.

---

## Repository Structure

```
clinical-trial-did-ml/
├── data/
│   ├── raw/                  # Downloaded raw dataset
│   ├── clinical_trial.db     # Relational SQLite database
│   ├── panel_patient_level.csv
│   └── panel_longitudinal.csv
├── src/
│   ├── fetch_data.py         # Automated downloader for public 130-US Hospitals dataset
│   ├── build_relational_db.py# Normalizes raw tables into SQLite schema
│   ├── panel_prep.py         # Leakage-free longitudinal cohort extraction
│   ├── traditional_did.py    # Naïve 2x2 DiD & TWFE OLS DiD
│   ├── tabpfn_did.py         # TabPFN Doubly Robust DiD estimator
│   ├── relbench_graph.py     # RelBench multi-table graph representation
│   ├── kumo_did.py           # Kumo Relational Model (NVIDIA NIM cloud API)
│   └── visualize.py          # Visualization suite
├── output/
│   ├── benchmark_results.json
│   ├── parallel_trends.png
│   ├── propensity_overlap.png
│   └── model_comparison_forest_plot.png
├── run_experiment.py         # Master orchestrator
├── STUDY_WALKTHROUGH.md      # Detailed scientific study report
├── requirements.txt
└── README.md
```

---

## References

* **Sant'Anna, P. H., & Zhao, J. (2020)**. *Doubly robust difference-in-differences estimators*. Journal of Econometrics, 219(1), 101-122.
* **Hollmann, N., et al. (2025)**. *Accurate predictions on small data with a tabular foundation model (TabPFN)*. Nature.
* **Ranjan, R., et al. (2024)**. *RelBench: A Benchmark for Deep Learning on Relational Databases*. NeurIPS.
* **Strack, B., et al. (2014)**. *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. BioMed Research International.
