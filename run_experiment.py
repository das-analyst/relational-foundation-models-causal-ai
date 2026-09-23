"""
Master Runner: End-to-End Difference-in-Differences Benchmark for Clinical Trials
Compares:
1. Naïve 2x2 DiD (Unadjusted)
2. Two-Way Fixed Effects OLS (Linear Controls)
3. TabPFN Doubly Robust DiD (Tabular Foundation Model)
4. RelBench Relational Graph DiD (Multi-table Graph Embeddings)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fetch_data import fetch_and_extract_data
from build_relational_db import build_relational_database
from panel_prep import prepare_did_panel
from traditional_did import run_naive_did, run_twfe_ols_did
from tabpfn_did import run_tabpfn_dr_did
from relbench_graph import run_relational_graph_did
from kumo_did import run_kumo_dr_did
from visualize import plot_parallel_trends, plot_propensity_overlap, plot_model_comparison

def main():
    start_time = time.time()
    print("=" * 80)
    print(" CLINICAL QUALITY & RELATIONAL DIFFERENCE-IN-DIFFERENCES BENCHMARK")
    print(" Comparing: Traditional DiD vs. TabPFN vs. RelBench vs. Kumo RFM")
    print("=" * 80)

    # Step 1: Fetch Public Dataset
    raw_csv = fetch_and_extract_data("data/raw")

    # Step 2: Build Multi-Table Relational Database
    db_path = "data/clinical_trial.db"
    if not os.path.exists(db_path):
        build_relational_database(raw_csv, db_path)
    else:
        print(f"[OK] Existing relational SQLite database found at: {db_path}")

    # Step 3: Prepare Panel Cohorts
    wide_path = "data/panel_patient_level.csv"
    long_path = "data/panel_longitudinal.csv"
    if not os.path.exists(wide_path) or not os.path.exists(long_path):
        wide_path, long_path = prepare_did_panel(db_path, "data")
    else:
        print(f"[OK] Existing panel cohorts found at: {wide_path}")

    df_wide = pd.read_csv(wide_path)
    df_long = pd.read_csv(long_path)

    results = []

    # Step 4: Model 1 - Naïve 2x2 DiD
    print("\n" + "-" * 60)
    print("[1/4] Running Naïve 2x2 DiD (Unadjusted)...")
    res_naive = run_naive_did(df_wide)
    results.append(res_naive)
    print(f"      Estimate: {res_naive['estimate']*100:.3f}% pts | SE: {res_naive['std_error']*100:.3f}% pts | p: {res_naive['p_value']:.4f}")

    # Step 5: Model 2 - Two-Way Fixed Effects OLS
    print("\n" + "-" * 60)
    print("[2/4] Running TWFE OLS DiD (Linear Covariates)...")
    res_twfe = run_twfe_ols_did(df_long)
    results.append(res_twfe)
    print(f"      Estimate: {res_twfe['estimate']*100:.3f}% pts | SE: {res_twfe['std_error']*100:.3f}% pts | p: {res_twfe['p_value']:.4f}")

    # Step 6: Model 3 - TabPFN Doubly Robust DiD
    print("\n" + "-" * 60)
    print("[3/4] Running TabPFN Doubly Robust DiD (Tabular Foundation Model)...")
    res_tabpfn = run_tabpfn_dr_did(df_wide)
    results.append(res_tabpfn)
    print(f"      Estimate: {res_tabpfn['estimate']*100:.3f}% pts | SE: {res_tabpfn['std_error']*100:.3f}% pts | p: {res_tabpfn['p_value']:.4f}")

    # Step 7: Model 4 - RelBench Relational Graph DiD
    print("\n" + "-" * 60)
    print("[4/5] Running RelBench Relational Graph DiD (Multi-Table Graph Embeddings)...")
    res_relbench = run_relational_graph_did(df_wide, db_path=db_path)
    results.append(res_relbench)
    print(f"      Estimate: {res_relbench['estimate']*100:.3f}% pts | SE: {res_relbench['std_error']*100:.3f}% pts | p: {res_relbench['p_value']:.4f}")

    # Step 8: Model 5 - Kumo Relational Foundation Model (RFM) DiD
    print("\n" + "-" * 60)
    print("[5/5] Running Kumo Relational Foundation Model DiD (NVIDIA Cloud NIM)...")
    res_kumo = run_kumo_dr_did(df_wide, db_path=db_path, sample_size=12000, batch_size=100, max_workers=6)
    results.append(res_kumo)
    print(f"      Estimate: {res_kumo['estimate']*100:.3f}% pts | SE: {res_kumo['std_error']*100:.3f}% pts | p: {res_kumo['p_value']:.4f}")

    # Step 9: Visualizations
    print("\n" + "-" * 60)
    print("[*] Generating Publication-Ready Visualizations...")
    plot_parallel_trends(df_wide, "output/parallel_trends.png")
    plot_propensity_overlap(res_tabpfn['propensity_scores'], res_tabpfn['treated_mask'], "output/propensity_overlap.png")
    plot_model_comparison(results, "output/model_comparison_forest_plot.png")

    # Step 10: Save Benchmark Metrics
    os.makedirs("output", exist_ok=True)
    clean_results = []
    for r in results:
        cr = {k: v for k, v in r.items() if not isinstance(v, (np.ndarray, list))}
        clean_results.append(cr)

    with open("output/benchmark_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)

    elapsed = time.time() - start_time

    # Print Final Summary Table
    print("\n" + "=" * 95)
    print(" FINAL BENCHMARK RESULTS: 30-DAY HOSPITAL READMISSION TREATMENT EFFECT (ATT)")
    print("=" * 95)
    print(f"{'Model / Estimator':<38} | {'ATT (% pts)':<12} | {'SE (used)':<10} | {'95% Conf. Interval':<22} | {'p-value':<8}")
    print("-" * 95)
    for r in results:
        est = f"{r['estimate']*100:+.3f}%"
        se  = f"{r['std_error']*100:.3f}%"
        ci  = f"[{r['ci_lower']*100:+.3f}%, {r['ci_upper']*100:+.3f}%]"
        p   = f"{r['p_value']:.4f}"
        print(f"{r['model']:<38} | {est:<12} | {se:<10} | {ci:<22} | {p:<8}")
    print("=" * 95)

    # Extra row for TabPFN asymptotic SE (for comparison)
    tp = res_tabpfn
    if "se_asymptotic" in tp:
        print(f"\n  [Info] TabPFN asymptotic SE: {tp['se_asymptotic']*100:.3f}% (p={tp['p_value_asym']:.4f})"
              f"  95% CI: [{tp['ci_lower_asym']*100:+.3f}%, {tp['ci_upper_asym']*100:+.3f}%]")
        print(f"  [Info] Bootstrap SE used in table: {tp['std_error']*100:.3f}% | Dataset: {tp['n_patients']:,} patients | {tp['n_folds']}-fold x-fit | B={tp['bootstrap_B']}")
    print(f"[*] Benchmark completed in {elapsed:.1f} seconds. All outputs saved to output/ directory.")


if __name__ == "__main__":
    main()
