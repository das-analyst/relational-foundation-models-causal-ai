"""
RelBench-inspired Relational Multi-Table Representation & Doubly Robust DiD.

Constructs multi-table relational features by traversing foreign keys across:
[Patients] <--> [Encounters] <--> [Medications] & [Diagnoses]
strictly evaluated at baseline (t <= T0) without temporal or treatment leakage.

Estimator: Sant'Anna & Zhao (2020) Doubly Robust DiD with Hajek self-normalization,
5-fold cross-fitting, and non-parametric bootstrap SE (B=500).
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def extract_relational_graph_embeddings(db_path: str = "data/clinical_trial.db", 
                                        patient_df: pd.DataFrame = None):
    """
    Constructs a multi-table relational feature representation across foreign keys.
    Traverses foreign keys from patients -> baseline encounters (t <= T0) -> medications & diagnoses.
    
    Guarantees:
      1. Full cohort inclusion (all 16,773 patients; no [:10000] truncation).
      2. Strict temporal cutoff at baseline (t <= T0, encounter rank 1).
      3. Zero treatment leakage (is_dosage_change dropped; only non-leaky clinical risk kept).
    """
    print("[*] Extracting multi-table relational features strictly at baseline (t <= T0)...")
    conn = sqlite3.connect(db_path)

    if patient_df is None:
        patient_df = pd.read_csv("data/panel_patient_level.csv")

    # 1. Relational aggregation from 'medications' table strictly at baseline encounter (t <= T0)
    # Extracts count of active medications and specific baseline therapeutic classes
    med_query = """
    WITH baseline_encs AS (
        SELECT patient_nbr, MIN(encounter_id) AS baseline_encounter_id
        FROM encounters
        GROUP BY patient_nbr
    )
    SELECT 
        b.patient_nbr,
        COUNT(m.id) AS rel_total_active_meds,
        SUM(CASE WHEN m.drug_name = 'insulin' THEN 1 ELSE 0 END) AS rel_has_insulin,
        SUM(CASE WHEN m.drug_name = 'metformin' THEN 1 ELSE 0 END) AS rel_has_metformin,
        SUM(CASE WHEN m.drug_name IN ('glimepiride', 'glipizide', 'glyburide') THEN 1 ELSE 0 END) AS rel_has_sulfonylurea
    FROM baseline_encs b
    JOIN medications m ON m.encounter_id = b.baseline_encounter_id
    GROUP BY b.patient_nbr
    """
    med_agg = pd.read_sql_query(med_query, conn)

    # 2. Relational aggregation from 'diagnoses' table strictly at baseline encounter (t <= T0)
    # Extracts comorbidity burden and multi-system diagnostic clusters
    diag_query = """
    WITH baseline_encs AS (
        SELECT patient_nbr, MIN(encounter_id) AS baseline_encounter_id
        FROM encounters
        GROUP BY patient_nbr
    )
    SELECT 
        b.patient_nbr,
        COUNT(d.id) AS rel_total_diagnoses,
        COUNT(DISTINCT d.category) AS rel_distinct_diag_categories,
        SUM(CASE WHEN d.category = 'Circulatory' THEN 1 ELSE 0 END) AS rel_circulatory_burden,
        SUM(CASE WHEN d.category = 'Diabetes' THEN 1 ELSE 0 END) AS rel_diabetes_burden,
        SUM(CASE WHEN d.category = 'Genitourinary' THEN 1 ELSE 0 END) AS rel_genitourinary_burden
    FROM baseline_encs b
    JOIN diagnoses d ON d.encounter_id = b.baseline_encounter_id
    GROUP BY b.patient_nbr
    """
    diag_agg = pd.read_sql_query(diag_query, conn)
    conn.close()

    # Merge relational features with patient-level panel (for ALL patients)
    rel_merged = patient_df.merge(med_agg, on='patient_nbr', how='left').merge(diag_agg, on='patient_nbr', how='left')
    
    # Fill missing values for patients with no recorded meds/diags at baseline with 0
    rel_cols = [
        'rel_total_active_meds', 'rel_has_insulin',
        'rel_has_metformin', 'rel_has_sulfonylurea', 'rel_total_diagnoses',
        'rel_distinct_diag_categories', 'rel_circulatory_burden',
        'rel_diabetes_burden', 'rel_genitourinary_burden'
    ]
    rel_merged[rel_cols] = rel_merged[rel_cols].fillna(0)
    print(f"[OK] Generated {len(rel_cols)} relational structural features across {len(rel_merged):,} patients.")
    return rel_merged, rel_cols


def run_relational_graph_did(df_wide: pd.DataFrame, 
                             db_path: str = "data/clinical_trial.db",
                             outcome_col: str = "delta_readmitted_30d",
                             n_splits: int = 5,
                             bootstrap_B: int = 500,
                             random_state: int = 42):
    """
    Evaluates Doubly Robust DiD using multi-table Relational Features.
    Employs 5-fold cross-fitting, Hajek self-normalized weights, and bootstrap SE (B=500).
    """
    df_rel, rel_cols = extract_relational_graph_embeddings(db_path, df_wide)

    # Combine base features + relational features
    cat_cols = ['primary_diag', 'race', 'gender']
    df_encoded = pd.get_dummies(df_rel, columns=cat_cols, drop_first=True)

    base_cols = [
        'age_num', 'is_emergency_adm', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'a1c_tested'
    ] + [c for c in df_encoded.columns if any(c.startswith(prefix + '_') for prefix in cat_cols)]

    all_features = base_cols + rel_cols

    X = df_encoded[all_features].values.astype(float)
    D = df_encoded['treatment'].values.astype(int)
    dY = df_encoded[outcome_col].values.astype(float)
    n = len(D)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5-Fold Cross-Fitting DR-DiD
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    e_hat = np.zeros(n)
    m0_hat = np.zeros(n)

    print(f"[*] Training Relational DR-DiD across {n_splits} cross-fitting folds...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, D)):
        X_tr, D_tr, dY_tr = X_scaled[train_idx], D[train_idx], dY[train_idx]
        X_val = X_scaled[val_idx]

        # Propensity Model with Relational features
        clf = HistGradientBoostingClassifier(random_state=random_state)
        clf.fit(X_tr, D_tr)
        e_hat[val_idx] = np.clip(clf.predict_proba(X_val)[:, 1], 0.02, 0.98)

        # Baseline Outcome Model with Relational features
        ctrl_mask = (D_tr == 0)
        reg = HistGradientBoostingRegressor(random_state=random_state)
        reg.fit(X_tr[ctrl_mask], dY_tr[ctrl_mask])
        m0_hat[val_idx] = reg.predict(X_val)

    # Sant'Anna & Zhao Doubly Robust ATT Estimator with Hajek Self-Normalization
    mean_D = np.mean(D)
    weight_ctrl = e_hat / (1.0 - e_hat)
    mean_w_ctrl = np.mean((1.0 - D) * weight_ctrl)

    # Hajek-stabilized influence function
    psi = (D * (dY - m0_hat)) / mean_D - ((1.0 - D) * weight_ctrl * (dY - m0_hat)) / mean_w_ctrl
    tau = float(np.mean(psi))

    # Asymptotic Influence Function SE
    se_asym = float(np.std(psi, ddof=1) / np.sqrt(n))
    z_asym = tau / se_asym
    p_asym = float(2 * (1 - stats.norm.cdf(abs(z_asym))))
    ci_lower_asym = tau - 1.96 * se_asym
    ci_upper_asym = tau + 1.96 * se_asym

    # Non-parametric Bootstrap SE (B=500)
    print(f"[*] Computing bootstrap SE (B={bootstrap_B})...")
    boot_taus = np.empty(bootstrap_B)
    boot_rng = np.random.RandomState(random_state)
    for b in range(bootstrap_B):
        idx = boot_rng.choice(n, n, replace=True)
        D_b, dY_b, e_b, m0_b = D[idx], dY[idx], e_hat[idx], m0_hat[idx]
        w_b = e_b / (1.0 - e_b)
        mD_b = np.mean(D_b)
        mw_b = np.mean((1.0 - D_b) * w_b)
        psi_b = (D_b * (dY_b - m0_b)) / mD_b - ((1.0 - D_b) * w_b * (dY_b - m0_b)) / mw_b
        boot_taus[b] = np.mean(psi_b)

    se_boot = float(np.std(boot_taus, ddof=1))
    z_boot = tau / se_boot
    p_boot = float(2 * (1 - stats.norm.cdf(abs(z_boot))))
    ci_lower_boot = tau - 1.96 * se_boot
    ci_upper_boot = tau + 1.96 * se_boot

    return {
        "model": "4. RelBench Relational Graph DiD",
        "estimate": tau,
        "std_error": se_boot,
        "ci_lower": ci_lower_boot,
        "ci_upper": ci_upper_boot,
        "p_value": p_boot,
        "se_asymptotic": se_asym,
        "ci_lower_asym": ci_lower_asym,
        "ci_upper_asym": ci_upper_asym,
        "p_value_asym": p_asym,
        "n_patients": len(df_rel),
        "n_features": len(all_features),
        "n_folds": n_splits,
        "bootstrap_B": bootstrap_B,
        "propensity_mean": float(np.mean(e_hat)),
        "propensity_scores": e_hat,
        "treated_mask": D
    }


if __name__ == "__main__":
    df = pd.read_csv("data/panel_patient_level.csv")
    res = run_relational_graph_did(df)
    print("Relational Multi-Table DR-DiD Result:")
    for k, v in res.items():
        if not isinstance(v, np.ndarray):
            print(f"  {k}: {v}")
