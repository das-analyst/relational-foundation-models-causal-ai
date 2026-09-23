"""
TabPFN-powered Doubly Robust Difference-in-Differences (DR-DiD).
Uses the TabPFN foundation model (prior-fitted transformer) to estimate:
1. Propensity score: e(X) = P(D=1 | X)
2. Baseline counterfactual trajectory: m_0(X) = E[Delta Y | X, D=0]
Applies the Sant'Anna & Zhao (2020) doubly robust influence-function estimator.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

def run_tabpfn_dr_did(df_wide: pd.DataFrame, 
                      outcome_col: str = "delta_readmitted_30d",
                      sample_size: int = 2500,
                      random_state: int = 42):
    """
    Fits Doubly Robust DiD using TabPFN (or high-capacity gradient boosting fallback).
    """
    # Check if TabPFN is installed
    try:
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        tabpfn_available = True
        print("[*] TabPFN Foundation Model loaded successfully.")
    except Exception as e:
        tabpfn_available = False
        print(f"[!] TabPFN import note: {e}. Using optimized ensemble.")

    # Prepare features
    cat_cols = ['primary_diag', 'race', 'gender']
    df_encoded = pd.get_dummies(df_wide, columns=cat_cols, drop_first=True)

    covariate_cols = [
        'age_num', 'is_emergency_adm', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'a1c_tested'
    ] + [c for c in df_encoded.columns if any(c.startswith(prefix + '_') for prefix in cat_cols)]

    # Subsample if dataset is very large to match TabPFN optimal context size
    if len(df_encoded) > sample_size:
        print(f"[*] Subsampling {sample_size:,} patients (from {len(df_encoded):,}) for TabPFN transformer context...")
        df_sample = df_encoded.sample(n=sample_size, random_state=random_state).copy()
    else:
        df_sample = df_encoded.copy()

    X = df_sample[covariate_cols].values.astype(float)
    D = df_sample['treatment'].values.astype(int)
    dY = df_sample[outcome_col].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-fitting to prevent regularization bias
    n_splits = 3
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    e_hat = np.zeros(len(D))
    m0_hat = np.zeros(len(D))

    print(f"[*] Running {n_splits}-fold cross-fitting for nuisance estimation...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, D)):
        X_tr, D_tr, dY_tr = X_scaled[train_idx], D[train_idx], dY[train_idx]
        X_val = X_scaled[val_idx]

        # 1. Propensity Score Model: e(X) = P(D=1 | X)
        if tabpfn_available:
            try:
                clf = TabPFNClassifier(device='cpu')
                clf.fit(X_tr, D_tr)
                # TabPFN predict_proba
                probs = clf.predict_proba(X_val)
                e_val = probs[:, 1]
            except Exception as ex:
                print(f"    [TabPFN Classifier fallback on fold {fold}: {ex}]")
                from sklearn.ensemble import HistGradientBoostingClassifier
                clf = HistGradientBoostingClassifier(random_state=random_state)
                clf.fit(X_tr, D_tr)
                e_val = clf.predict_proba(X_val)[:, 1]
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(random_state=random_state)
            clf.fit(X_tr, D_tr)
            e_val = clf.predict_proba(X_val)[:, 1]

        # Clip propensity scores for numerical stability (common support)
        e_hat[val_idx] = np.clip(e_val, 0.02, 0.98)

        # 2. Outcome Model for Controls: m0(X) = E[dY | X, D=0]
        ctrl_mask = (D_tr == 0)
        X_tr_ctrl = X_tr[ctrl_mask]
        dY_tr_ctrl = dY_tr[ctrl_mask]

        if tabpfn_available:
            try:
                reg = TabPFNRegressor(device='cpu')
                reg.fit(X_tr_ctrl, dY_tr_ctrl)
                m0_hat[val_idx] = reg.predict(X_val)
            except Exception as ex:
                print(f"    [TabPFN Regressor fallback on fold {fold}: {ex}]")
                from sklearn.ensemble import HistGradientBoostingRegressor
                reg = HistGradientBoostingRegressor(random_state=random_state)
                reg.fit(X_tr_ctrl, dY_tr_ctrl)
                m0_hat[val_idx] = reg.predict(X_val)
        else:
            from sklearn.ensemble import HistGradientBoostingRegressor
            reg = HistGradientBoostingRegressor(random_state=random_state)
            reg.fit(X_tr_ctrl, dY_tr_ctrl)
            m0_hat[val_idx] = reg.predict(X_val)

    # 3. Sant'Anna & Zhao (2020) Doubly Robust DiD Estimator
    # Influence function for ATT:
    # psi_i = D_i * (dY_i - m0(X_i)) - (e(X_i) * (1 - D_i) / (1 - e(X_i))) * (dY_i - m0(X_i))
    # Normalized by mean(D)
    mean_D = np.mean(D)
    weight_ctrl = e_hat / (1.0 - e_hat)

    # Doubly robust influence function
    psi_treated = D * (dY - m0_hat)
    psi_control = (1 - D) * weight_ctrl * (dY - m0_hat)
    psi = (psi_treated - psi_control) / mean_D

    tau = np.mean(psi)
    # Influence-function based standard error
    se = np.std(psi, ddof=1) / np.sqrt(len(psi))
    z = tau / se
    p_val = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ci_lower = tau - 1.96 * se
    ci_upper = tau + 1.96 * se

    return {
        "model": "3. TabPFN Doubly Robust DiD",
        "estimate": tau,
        "std_error": se,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "n_patients": len(df_sample),
        "propensity_mean": float(np.mean(e_hat)),
        "propensity_scores": e_hat,
        "treated_mask": D
    }

if __name__ == "__main__":
    df = pd.read_csv("data/panel_patient_level.csv")
    res = run_tabpfn_dr_did(df)
    print("TabPFN DR-DiD Result:", res)
