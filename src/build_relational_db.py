"""
Relational Database Builder: Normalizes flat 130-US Hospitals dataset into SQLite schema.
Tables: patients, encounters, medications, diagnoses.
"""

import os
import sqlite3
import pandas as pd

MEDICATION_COLS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

def map_icd9_category(code):
    if pd.isna(code) or str(code).strip() == '?' or str(code).strip() == '':
        return 'Missing'
    code_str = str(code).strip()
    if code_str.startswith('V') or code_str.startswith('E'):
        return 'External/Supplemental'
    try:
        val = float(code_str)
        if 390 <= val <= 459 or val == 785:
            return 'Circulatory'
        elif 460 <= val <= 519 or val == 786:
            return 'Respiratory'
        elif 520 <= val <= 579 or val == 787:
            return 'Digestive'
        elif int(val) == 250:
            return 'Diabetes'
        elif 800 <= val <= 999:
            return 'Injury'
        elif 710 <= val <= 739:
            return 'Musculoskeletal'
        elif 580 <= val <= 629 or val == 788:
            return 'Genitourinary'
        elif 140 <= val <= 239:
            return 'Neoplasms'
        else:
            return 'Other'
    except ValueError:
        return 'Other'

def build_relational_database(csv_path: str = "data/raw/diabetic_data.csv", db_path: str = "data/clinical_trial.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source file {csv_path} not found. Run fetch_data.py first.")

    print(f"[*] Reading raw clinical CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[OK] Read {len(df):,} total encounters.")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1. Patients Table
    print("[*] Building 'patients' table...")
    cur.execute("DROP TABLE IF EXISTS patients")
    cur.execute("""
        CREATE TABLE patients (
            patient_nbr INTEGER PRIMARY KEY,
            race TEXT,
            gender TEXT,
            age TEXT
        )
    """)
    patients_df = df[['patient_nbr', 'race', 'gender', 'age']].drop_duplicates(subset=['patient_nbr'])
    patients_df.to_sql('patients', conn, if_exists='append', index=False)
    print(f"[OK] Created 'patients' table with {len(patients_df):,} unique patients.")

    # 2. Encounters Table
    print("[*] Building 'encounters' table...")
    cur.execute("DROP TABLE IF EXISTS encounters")
    cur.execute("""
        CREATE TABLE encounters (
            encounter_id INTEGER PRIMARY KEY,
            patient_nbr INTEGER,
            admission_type_id INTEGER,
            discharge_disposition_id INTEGER,
            admission_source_id INTEGER,
            time_in_hospital INTEGER,
            num_lab_procedures INTEGER,
            num_procedures INTEGER,
            num_medications INTEGER,
            number_outpatient INTEGER,
            number_emergency INTEGER,
            number_inpatient INTEGER,
            number_diagnoses INTEGER,
            max_glu_serum TEXT,
            A1Cresult TEXT,
            change TEXT,
            diabetesMed TEXT,
            readmitted TEXT,
            readmitted_30d INTEGER,
            FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr)
        )
    """)
    enc_cols = [
        'encounter_id', 'patient_nbr', 'admission_type_id', 'discharge_disposition_id',
        'admission_source_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
        'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'readmitted'
    ]
    encounters_df = df[enc_cols].copy()
    encounters_df['readmitted_30d'] = (encounters_df['readmitted'] == '<30').astype(int)
    encounters_df.to_sql('encounters', conn, if_exists='append', index=False)
    print(f"[OK] Created 'encounters' table with {len(encounters_df):,} encounters.")

    # 3. Medications Relational Table
    print("[*] Building normalized 'medications' table...")
    cur.execute("DROP TABLE IF EXISTS medications")
    cur.execute("""
        CREATE TABLE medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encounter_id INTEGER,
            patient_nbr INTEGER,
            drug_name TEXT,
            status TEXT,
            is_prescribed INTEGER,
            is_dosage_change INTEGER,
            FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr)
        )
    """)
    med_dfs = []
    for drug in MEDICATION_COLS:
        sub_df = df[['encounter_id', 'patient_nbr', drug]].copy()
        sub_df.rename(columns={drug: 'status'}, inplace=True)
        sub_df['drug_name'] = drug
        sub_df['is_prescribed'] = (sub_df['status'] != 'No').astype(int)
        sub_df['is_dosage_change'] = sub_df['status'].isin(['Up', 'Down']).astype(int)
        # Only keep records where drug was either prescribed or changed
        sub_df = sub_df[sub_df['is_prescribed'] == 1]
        med_dfs.append(sub_df)
    
    all_meds_df = pd.concat(med_dfs, ignore_index=True)
    all_meds_df.to_sql('medications', conn, if_exists='append', index=False)
    print(f"[OK] Created 'medications' table with {len(all_meds_df):,} active drug records.")

    # 4. Diagnoses Relational Table
    print("[*] Building normalized 'diagnoses' table...")
    cur.execute("DROP TABLE IF EXISTS diagnoses")
    cur.execute("""
        CREATE TABLE diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encounter_id INTEGER,
            patient_nbr INTEGER,
            position INTEGER,
            icd9_code TEXT,
            category TEXT,
            FOREIGN KEY (encounter_id) REFERENCES encounters(encounter_id),
            FOREIGN KEY (patient_nbr) REFERENCES patients(patient_nbr)
        )
    """)
    diag_records = []
    for pos, col in [(1, 'diag_1'), (2, 'diag_2'), (3, 'diag_3')]:
        sub_df = df[['encounter_id', 'patient_nbr', col]].copy()
        sub_df.rename(columns={col: 'icd9_code'}, inplace=True)
        sub_df['position'] = pos
        sub_df['category'] = sub_df['icd9_code'].apply(map_icd9_category)
        diag_records.append(sub_df)
    
    all_diags_df = pd.concat(diag_records, ignore_index=True)
    all_diags_df.to_sql('diagnoses', conn, if_exists='append', index=False)
    print(f"[OK] Created 'diagnoses' table with {len(all_diags_df):,} diagnosis records.")

    # Create Indexes for fast relational querying
    print("[*] Creating relational database indexes...")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_enc_patient ON encounters(patient_nbr)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_enc ON medications(encounter_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_med_patient ON medications(patient_nbr)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_enc ON diagnoses(encounter_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_diag_patient ON diagnoses(patient_nbr)")

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Relational SQLite database ready at: {db_path}")

if __name__ == "__main__":
    build_relational_database()
