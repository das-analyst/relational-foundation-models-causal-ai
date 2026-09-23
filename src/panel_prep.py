"""
Panel Preparation for Difference-in-Differences Analysis.
Extracts longitudinal patient cohorts (Pre T0 vs Post T1), defines the clinical medication
titration intervention, and constructs leakage-free pre-intervention covariate vectors.
"""

import os
import sqlite3
import pandas as pd
import numpy as np

def prepare_did_panel(db_path: str = "data/clinical_trial.db", output_dir: str = "data"):
    os.makedirs(output_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)

    print("[*] Querying longitudinal patient trajectories from database...")
    # Find patients with at least 2 encounters
    query = """
    SELECT 
        e.encounter_id,
        e.patient_nbr,
        e.admission_type_id,
        e.discharge_disposition_id,
        e.admission_source_id,
        e.time_in_hospital,
        e.num_lab_procedures,
        e.num_procedures,
        e.num_medications,
        e.number_outpatient,
        e.number_emergency,
        e.number_inpatient,
        e.number_diagnoses,
        e.A1Cresult,
        e.change,
        e.diabetesMed,
        e.readmitted_30d,
        p.race,
        p.gender,
        p.age
    FROM encounters e
    JOIN patients p ON e.patient_nbr = p.patient_nbr
    ORDER BY e.patient_nbr, e.encounter_id ASC
    """
    df = pd.read_sql_query(query, conn)
    print(f"[OK] Total records queried: {len(df):,}")

    # Count encounters per patient
    enc_counts = df.groupby('patient_nbr')['encounter_id'].count()
    multi_visit_patients = enc_counts[enc_counts >= 2].index
    print(f"[OK] Patients with >= 2 encounters: {len(multi_visit_patients):,}")

    # Filter to first 2 encounters for each patient (T0 = baseline pre-period, T1 = follow-up post-period)
    df_cohort = df[df['patient_nbr'].isin(multi_visit_patients)].copy()
    df_cohort['visit_rank'] = df_cohort.groupby('patient_nbr')['encounter_id'].rank(method='first').astype(int)
    df_cohort = df_cohort[df_cohort['visit_rank'].isin([1, 2])].copy()

    # Query diagnosis info for T0
    diag_query = """
    SELECT encounter_id, category AS primary_diag
    FROM diagnoses
    WHERE position = 1
    """
    diag_df = pd.read_sql_query(diag_query, conn)
    conn.close()

    df_cohort = df_cohort.merge(diag_df, on='encounter_id', how='left')
    df_cohort['primary_diag'] = df_cohort['primary_diag'].fillna('Other')

    # Pivot to patient-level wide format
    t0_df = df_cohort[df_cohort['visit_rank'] == 1].copy()
    t1_df = df_cohort[df_cohort['visit_rank'] == 2].copy()

    # Intervention / Treatment definition:
    # Active Clinical Medication Titration Protocol at baseline (change == 'Ch')
    # Treated (D = 1): Medication was actively adjusted/titrated at baseline
    # Control (D = 0): Medication was maintained without adjustment (change == 'No')
    t0_df['treatment'] = (t0_df['change'] == 'Ch').astype(int)
    
    # Merge pre (T0) and post (T1)
    merged = t0_df.merge(
        t1_df[['patient_nbr', 'readmitted_30d', 'time_in_hospital']],
        on='patient_nbr',
        suffixes=('_pre', '_post')
    )

    # Calculate change in outcome (Delta Y)
    merged['delta_readmitted_30d'] = merged['readmitted_30d_post'] - merged['readmitted_30d_pre']
    merged['delta_time_in_hospital'] = merged['time_in_hospital_post'] - merged['time_in_hospital_pre']

    # Encode pre-intervention covariates (X) strictly at t <= T0
    # Clean up race and gender
    merged['race'] = merged['race'].replace('?', 'Unknown')
    merged['gender'] = merged['gender'].replace('Unknown/Invalid', 'Female')
    
    # Clean age: convert brackets to numeric proxy
    age_map = {
        '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
        '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
        '[80-90)': 85, '[90-100)': 95
    }
    merged['age_num'] = merged['age'].map(age_map).fillna(60)

    # A1C tested binary indicator
    merged['a1c_tested'] = (merged['A1Cresult'] != 'None').astype(int)

    # Emergency admission indicator (1 = Emergency, 2 = Urgent)
    merged['is_emergency_adm'] = merged['admission_type_id'].isin([1, 2]).astype(int)

    # Select and rename final modeling features
    feature_cols = [
        'patient_nbr', 'treatment',
        'readmitted_30d_pre', 'readmitted_30d_post', 'delta_readmitted_30d',
        'time_in_hospital_pre', 'time_in_hospital_post', 'delta_time_in_hospital',
        'age_num', 'is_emergency_adm', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_emergency', 'number_inpatient', 'number_diagnoses',
        'a1c_tested', 'primary_diag', 'race', 'gender'
    ]
    wide_panel = merged[feature_cols].copy()
    wide_path = os.path.join(output_dir, "panel_patient_level.csv")
    wide_panel.to_csv(wide_path, index=False)
    print(f"[OK] Saved patient-level wide panel: {wide_path} ({len(wide_panel):,} patients)")

    # Also build long-format panel for Traditional TWFE OLS DiD
    records_pre = wide_panel.copy()
    records_pre['post_period'] = 0
    records_pre['readmitted_30d'] = records_pre['readmitted_30d_pre']
    records_pre['time_in_hospital'] = records_pre['time_in_hospital_pre']

    records_post = wide_panel.copy()
    records_post['post_period'] = 1
    records_post['readmitted_30d'] = records_post['readmitted_30d_post']
    records_post['time_in_hospital'] = records_post['time_in_hospital_post']

    long_panel = pd.concat([records_pre, records_post], ignore_index=True)
    long_panel['treatment_x_post'] = long_panel['treatment'] * long_panel['post_period']
    long_path = os.path.join(output_dir, "panel_longitudinal.csv")
    long_panel.to_csv(long_path, index=False)
    print(f"[OK] Saved longitudinal long panel: {long_path} ({len(long_panel):,} observation rows)")

    # Print summary of cohort
    n_treated = (wide_panel['treatment'] == 1).sum()
    n_control = (wide_panel['treatment'] == 0).sum()
    print(f"\n[*] Clinical Cohort Summary:")
    print(f"    - Treated Patients (Medication Titration): {n_treated:,} ({n_treated/len(wide_panel)*100:.1f}%)")
    print(f"    - Control Patients (Medication Maintained): {n_control:,} ({n_control/len(wide_panel)*100:.1f}%)")
    print(f"    - Baseline 30d Readmission Rate (Treated): {wide_panel[wide_panel['treatment']==1]['readmitted_30d_pre'].mean()*100:.2f}%")
    print(f"    - Baseline 30d Readmission Rate (Control): {wide_panel[wide_panel['treatment']==0]['readmitted_30d_pre'].mean()*100:.2f}%")

    return wide_path, long_path

if __name__ == "__main__":
    prepare_did_panel()
