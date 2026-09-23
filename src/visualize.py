"""
Visualization suite for Difference-in-Differences and Model Benchmark:
1. Parallel trends trajectory
2. Propensity score overlap (positivity assumption)
3. Model comparison forest plot (Point estimates & 95% Confidence Intervals)
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def plot_parallel_trends(df_wide: pd.DataFrame, output_path: str = "output/parallel_trends.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate means
    treated = df_wide[df_wide['treatment'] == 1]
    control = df_wide[df_wide['treatment'] == 0]

    y_t0_treat = treated['readmitted_30d_pre'].mean() * 100
    y_t1_treat = treated['readmitted_30d_post'].mean() * 100

    y_t0_ctrl = control['readmitted_30d_pre'].mean() * 100
    y_t1_ctrl = control['readmitted_30d_post'].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot trajectories
    periods = ['Baseline (T0)', 'Follow-Up (T1)']
    ax.plot(periods, [y_t0_treat, y_t1_treat], 'o-', color='#D9381E', linewidth=2.5, markersize=8, label='Treated (Medication Titration Protocol)')
    ax.plot(periods, [y_t0_ctrl, y_t1_ctrl], 's--', color='#1B4965', linewidth=2.5, markersize=8, label='Control (Standard Unchanged Regime)')

    # Annotate points
    ax.text(0, y_t0_treat + 0.3, f"{y_t0_treat:.1f}%", ha='center', fontweight='bold', color='#D9381E')
    ax.text(1, y_t1_treat + 0.3, f"{y_t1_treat:.1f}%", ha='center', fontweight='bold', color='#D9381E')
    ax.text(0, y_t0_ctrl - 0.7, f"{y_t0_ctrl:.1f}%", ha='center', fontweight='bold', color='#1B4965')
    ax.text(1, y_t1_ctrl - 0.7, f"{y_t1_ctrl:.1f}%", ha='center', fontweight='bold', color='#1B4965')

    ax.set_ylabel('30-Day Readmission Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Clinical Quality Trajectory: 30-Day Readmissions (Pre vs Post)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(frameon=True, loc='upper left', fontsize=10)
    ax.set_ylim([0, max(y_t0_treat, y_t1_treat, y_t0_ctrl, y_t1_ctrl) + 5])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Saved parallel trends plot to: {output_path}")

def plot_propensity_overlap(propensity_scores: np.ndarray, treated_mask: np.ndarray, output_path: str = "output/propensity_overlap.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)

    e_treat = propensity_scores[treated_mask == 1]
    e_ctrl = propensity_scores[treated_mask == 0]

    bins = np.linspace(0, 1, 35)
    ax.hist(e_treat, bins=bins, alpha=0.6, color='#D9381E', density=True, label=f'Treated (N={len(e_treat):,})')
    ax.hist(e_ctrl, bins=bins, alpha=0.5, color='#1B4965', density=True, label=f'Control (N={len(e_ctrl):,})')

    ax.set_xlabel('Estimated Propensity Score P(D=1 | X)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title('Common Support & Positivity Verification (Propensity Overlap)', fontsize=13, fontweight='bold', pad=15)
    ax.legend(frameon=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Saved propensity overlap plot to: {output_path}")

def plot_model_comparison(results_list: list, output_path: str = "output/model_comparison_forest_plot.png"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)

    models = [r['model'] for r in results_list]
    estimates = [r['estimate'] * 100 for r in results_list] # convert to percentage points
    ci_lowers = [r['ci_lower'] * 100 for r in results_list]
    ci_uppers = [r['ci_upper'] * 100 for r in results_list]

    y_pos = np.arange(len(models))

    # Error bars (symmetric or asymmetric)
    xerr = [
        [est - low for est, low in zip(estimates, ci_lowers)],
        [up - est for est, up in zip(estimates, ci_uppers)]
    ]

    colors = ['#8E9AAF', '#4A90E2', '#50C878', '#2E7D32']

    ax.axvline(0, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)

    for i in range(len(models)):
        ax.errorbar(estimates[i], y_pos[i], xerr=[[xerr[0][i]], [xerr[1][i]]], 
                    fmt='o', color=colors[i % len(colors)], ecolor=colors[i % len(colors)],
                    elinewidth=2.5, capsize=5, markersize=8, label=models[i])
        # Annotate estimate value
        sign = "+" if estimates[i] > 0 else ""
        ax.text(estimates[i], y_pos[i] + 0.22, f"{sign}{estimates[i]:.2f}% pts (p={results_list[i]['p_value']:.3f})", 
                ha='center', fontsize=9.5, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models, fontsize=11, fontweight='bold')
    ax.invert_yaxis()  # top-down order
    ax.set_xlabel('Estimated Average Treatment Effect on Treated (ATT) on 30d Readmission (% points)', fontsize=11, fontweight='bold')
    ax.set_title('Head-to-Head Comparison: Traditional DiD vs. TabPFN vs. RelBench', fontsize=13, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[OK] Saved forest plot to: {output_path}")

if __name__ == "__main__":
    dummy_results = [
        {"model": "1. Naïve 2x2 DiD (Unadjusted)", "estimate": -0.015, "ci_lower": -0.025, "ci_upper": -0.005, "p_value": 0.003},
        {"model": "2. TWFE OLS DiD (Linear Controls)", "estimate": -0.021, "ci_lower": -0.031, "ci_upper": -0.011, "p_value": 0.000},
        {"model": "3. TabPFN Doubly Robust DiD", "estimate": -0.028, "ci_lower": -0.039, "ci_upper": -0.017, "p_value": 0.000},
        {"model": "4. RelBench Relational Graph DiD", "estimate": -0.032, "ci_lower": -0.043, "ci_upper": -0.021, "p_value": 0.000}
    ]
    plot_model_comparison(dummy_results)
