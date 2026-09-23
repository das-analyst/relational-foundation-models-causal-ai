"""
Traditional Econometric Difference-in-Differences Estimators:
1. Naïve 2x2 DiD (Unadjusted difference in differences)
2. Two-Way Fixed Effects (TWFE) OLS with linear covariate controls and cluster-robust standard errors.
"""

import numpy as np
import pandas as pd
from scipy import stats

def run_naive_did(df_wide: pd.DataFrame, outcome_col: str = "delta_readmitted_30d"):
    """
    Computes unadjusted 2x2 Difference-in-Differences.
    tau = mean(delta_Y | D=1) - mean(delta_Y | D=0)
    """
    treated = df_wide[df_wide['treatment'] == 1][outcome_col].values
    control = df_wide[df_wide['treatment'] == 0][outcome_col].values

    n1, n0 = len(treated), len(control)
    mean1, mean0 = np.mean(treated), np.mean(control)
    var1, var0 = np.var(treated, ddof=1), np.var(control, ddof=1)

    tau = mean1 - mean0
    se = np.sqrt((var1 / n1) + (var0 / n0))
    z = tau / se
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    return {
        "model": "1. Naïve 2x2 DiD (Unadjusted)",
        "estimate": tau,
        "std_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "n_treated": n1,
        "n_control": n0
    }

def run_twfe_ols_did(df_long: pd.DataFrame, outcome_col: str = "readmitted_30d"):
    """
    Two-Way Fixed Effects (TWFE) OLS DiD:
    Y_it = b0 + b1*Post + b2*Treat + tau*(Treat*Post) + X'gamma + e_it
    Fitted using OLS with cluster-robust standard errors at patient level.
    """
    # Prepare design matrix
    # One-hot encode categoricals
    cat_cols = ['primary_diag', 'race', 'gender']
    df_encoded = pd.get_dummies(df_long, columns=cat_cols, drop_first=True)

    covariates = [
        'age_num', 'is_emergency_adm', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'a1c_tested'
    ] + [c for c in df_encoded.columns if any(c.startswith(prefix + '_') for prefix in cat_cols)]

    X_cols = ['post_period', 'treatment', 'treatment_x_post'] + covariates
    
    Y = df_encoded[outcome_col].values.astype(float)
    X = df_encoded[X_cols].values.astype(float)
    X = np.column_stack([np.ones(len(X)), X]) # Add intercept

    # OLS coefficients: beta = (X'X)^(-1) X'Y
    XtX = np.dot(X.T, X)
    XtY = np.dot(X.T, Y)
    beta = np.linalg.solve(XtX + 1e-8 * np.eye(X.shape[1]), XtY)

    # Residuals
    residuals = Y - np.dot(X, beta)

    # Clustered Standard Errors by patient_nbr
    # White / Sandwich cluster variance
    clusters = df_encoded['patient_nbr'].values
    unique_clusters = np.unique(clusters)
    G = len(unique_clusters)
    N = len(Y)
    K = X.shape[1]

    # Cluster adjustment factor
    df_c = (G / (G - 1)) * ((N - 1) / (N - K))

    # Meat matrix calculation
    # Vectorized / grouped sum of X_g * e_g
    cluster_scores = pd.DataFrame(X * residuals[:, None]).groupby(clusters).sum().values
    meat = np.dot(cluster_scores.T, cluster_scores)

    XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(K))
    vcov = df_c * np.dot(np.dot(XtX_inv, meat), XtX_inv)

    # Treatment effect is at index 3: [Intercept, post_period, treatment, treatment_x_post]
    idx_tau = 3
    tau = beta[idx_tau]
    se = np.sqrt(vcov[idx_tau, idx_tau])
    z = tau / se
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    return {
        "model": "2. TWFE OLS DiD (Linear Controls)",
        "estimate": tau,
        "std_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "n_obs": len(Y),
        "n_patients": G
    }

if __name__ == "__main__":
    df_wide = pd.read_csv("data/panel_patient_level.csv")
    df_long = pd.read_csv("data/panel_longitudinal.csv")

    res_naive = run_naive_did(df_wide)
    print("Naïve DiD:", res_naive)

    res_twfe = run_twfe_ols_did(df_long)
    print("TWFE OLS DiD:", res_twfe)
