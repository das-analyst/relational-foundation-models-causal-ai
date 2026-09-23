"""
Kumo Relational Foundation Model (RFM) Doubly Robust Difference-in-Differences.

Connects directly to the live NVIDIA Inference Microservice (NIM) cloud endpoint:
https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions

Takes the multi-table relational database (patients, encounters, medications)
and uses Kumo RFM in-context learning to estimate:
  1. Propensity Score: e_hat(X) = P(Treatment = 1 | Relational History)
  2. Baseline Outcome Trend: m0_hat(X) = E[delta_Y | Treatment = 0, Relational History]

Then computes the Sant'Anna & Zhao (2020) Doubly Robust ATT estimator
with asymptotic influence function SE and bootstrap SE.
"""

import os
import time
import sqlite3
import requests
import numpy as np
import pandas as pd
from scipy import stats


KUMO_ENDPOINT = "https://ai.api.nvidia.com/v1/structured-data/nvidia/kumo-relational/predictions"


def _build_kumo_payload(
    ctx_ids: list,
    pred_ids: list,
    panel_map: dict,
    pat_map: dict,
    enc_map: dict,
    med_map: dict,
    task_kind: str = "binary_classification",
):
    """
    Constructs the exact multi-table relational JSON payload required by Kumo RFM NIM.
    Schema includes:
      - instance_table (patients with anchor_time and target)
      - related_tables (patients, encounters, medications)
      - relationships (instance -> patients, encounters, medications)
    """
    is_clf = (task_kind == "binary_classification")
    target_col = "treatment" if is_clf else "outcome"
    target_dtype = "bool" if is_clf else "float64"
    target_stype = "categorical" if is_clf else "numerical"

    task_def = {
        "kind": task_kind,
        "target": {
            "column_name": target_col,
            "dtype": target_dtype,
        },
        "entity_table_names": ["patients"],
        "anchor_time_column": "anchor_time",
    }
    if is_clf:
        task_def["target"]["classes"] = ["false", "true"]
        task_def["target"]["positive_class"] = "true"

    # Context rows
    ctx_inst, ctx_pat, ctx_enc, ctx_med = [], [], [], []
    for i, pid in enumerate(ctx_ids):
        p_row = panel_map[pid]
        target_val = bool(p_row["treatment"]) if is_clf else float(p_row["delta_readmitted_30d"])
        ctx_inst.append([i, "2025-01-01T00:00:00Z", int(pid), target_val])

        # Related: patients
        p = pat_map.get(pid, {})
        ctx_pat.append([i, int(pid), str(p.get("race", "?")), str(p.get("gender", "?")), str(p.get("age", "?"))])

        # Related: encounters (up to 3 pre-intervention stays)
        for e in enc_map.get(pid, [])[:3]:
            ctx_enc.append([
                i, int(e["encounter_id"]), int(pid),
                float(e["time_in_hospital"]), float(e["num_lab_procedures"]),
                float(e["num_medications"]), float(e["number_emergency"]),
                float(e["number_inpatient"]), float(e["number_diagnoses"])
            ])

        # Related: medications (up to 4 prescribed drugs/dosages)
        for m in med_map.get(pid, [])[:4]:
            ctx_med.append([
                i, int(m["id"]), int(pid),
                str(m["drug_name"]), str(m["status"]), float(m["is_dosage_change"])
            ])

    # Predict rows
    pred_inst, pred_pat, pred_enc, pred_med = [], [], [], []
    for j, pid in enumerate(pred_ids):
        idx = len(ctx_ids) + j
        pred_inst.append([idx, "2025-02-01T00:00:00Z", int(pid)])

        p = pat_map.get(pid, {})
        pred_pat.append([idx, int(pid), str(p.get("race", "?")), str(p.get("gender", "?")), str(p.get("age", "?"))])

        for e in enc_map.get(pid, [])[:3]:
            pred_enc.append([
                idx, int(e["encounter_id"]), int(pid),
                float(e["time_in_hospital"]), float(e["num_lab_procedures"]),
                float(e["num_medications"]), float(e["number_emergency"]),
                float(e["number_inpatient"]), float(e["number_diagnoses"])
            ])

        for m in med_map.get(pid, [])[:4]:
            pred_med.append([
                idx, int(m["id"]), int(pid),
                str(m["drug_name"]), str(m["status"]), float(m["is_dosage_change"])
            ])

    output_fields = ["prediction", "probabilities"] if is_clf else ["prediction"]

    payload = {
        "model": "kumo-relational",
        "task": task_def,
        "schema": {
            "instance_table": {
                "columns": {
                    "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                    "anchor_time": {"dtype": "timestamp[us]", "stype": "timestamp", "nullable": False},
                    "patient_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                    target_col: {"dtype": target_dtype, "stype": target_stype},
                },
                "primary_key": "instance_id",
            },
            "related_tables": {
                "patients": {
                    "columns": {
                        "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "patient_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "race": {"dtype": "string", "stype": "categorical"},
                        "gender": {"dtype": "string", "stype": "categorical"},
                        "age": {"dtype": "string", "stype": "categorical"},
                    },
                    "primary_key": ["instance_id", "patient_id"],
                },
                "encounters": {
                    "columns": {
                        "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "encounter_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "patient_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "time_in_hospital": {"dtype": "float64", "stype": "numerical"},
                        "num_lab_procedures": {"dtype": "float64", "stype": "numerical"},
                        "num_medications": {"dtype": "float64", "stype": "numerical"},
                        "number_emergency": {"dtype": "float64", "stype": "numerical"},
                        "number_inpatient": {"dtype": "float64", "stype": "numerical"},
                        "number_diagnoses": {"dtype": "float64", "stype": "numerical"},
                    },
                    "primary_key": ["instance_id", "encounter_id"],
                },
                "medications": {
                    "columns": {
                        "instance_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "med_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "patient_id": {"dtype": "int64", "stype": "ID", "nullable": False},
                        "drug_name": {"dtype": "string", "stype": "categorical"},
                        "status": {"dtype": "string", "stype": "categorical"},
                        "is_dosage_change": {"dtype": "float64", "stype": "numerical"},
                    },
                    "primary_key": ["instance_id", "med_id"],
                },
            },
            "relationships": [
                {
                    "source_columns": ["instance_id", "patient_id"],
                    "target_table": "patients",
                    "target_columns": ["instance_id", "patient_id"],
                },
                {
                    "source_columns": ["instance_id", "patient_id"],
                    "target_table": "encounters",
                    "target_columns": ["instance_id", "patient_id"],
                },
                {
                    "source_columns": ["instance_id", "patient_id"],
                    "target_table": "medications",
                    "target_columns": ["instance_id", "patient_id"],
                },
            ],
        },
        "context": {
            "instance_table": {
                "format": "arrays",
                "columns": ["instance_id", "anchor_time", "patient_id", target_col],
                "rows": ctx_inst,
            },
            "related_tables": {
                "patients": {
                    "format": "arrays",
                    "columns": ["instance_id", "patient_id", "race", "gender", "age"],
                    "rows": ctx_pat,
                },
                "encounters": {
                    "format": "arrays",
                    "columns": [
                        "instance_id", "encounter_id", "patient_id",
                        "time_in_hospital", "num_lab_procedures", "num_medications",
                        "number_emergency", "number_inpatient", "number_diagnoses",
                    ],
                    "rows": ctx_enc,
                },
                "medications": {
                    "format": "arrays",
                    "columns": ["instance_id", "med_id", "patient_id", "drug_name", "status", "is_dosage_change"],
                    "rows": ctx_med,
                },
            },
        },
        "predict": {
            "instance_table": {
                "format": "arrays",
                "columns": ["instance_id", "anchor_time", "patient_id"],
                "rows": pred_inst,
            },
            "related_tables": {
                "patients": {
                    "format": "arrays",
                    "columns": ["instance_id", "patient_id", "race", "gender", "age"],
                    "rows": pred_pat,
                },
                "encounters": {
                    "format": "arrays",
                    "columns": [
                        "instance_id", "encounter_id", "patient_id",
                        "time_in_hospital", "num_lab_procedures", "num_medications",
                        "number_emergency", "number_inpatient", "number_diagnoses",
                    ],
                    "rows": pred_enc,
                },
                "medications": {
                    "format": "arrays",
                    "columns": ["instance_id", "med_id", "patient_id", "drug_name", "status", "is_dosage_change"],
                    "rows": pred_med,
                },
            },
        },
        "output": {"fields": output_fields},
    }
    return payload


def _query_kumo_nim(payload: dict, api_key: str, max_retries: int = 3) -> dict:
    """Dispatches request to NVIDIA Kumo Relational NIM with exponential backoff."""
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(KUMO_ENDPOINT, headers=headers, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (attempt + 1))
                continue
            else:
                raise RuntimeError(f"Kumo NIM returned HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Max retries exceeded connecting to Kumo NIM.")


def run_kumo_dr_did(
    df_wide: pd.DataFrame,
    db_path: str = "data/clinical_trial.db",
    sample_size: int = 3000,
    batch_size: int = 100,
    bootstrap_B: int = 500,
    random_state: int = 42,
) -> dict:
    """
    Fits Doubly Robust DiD using NVIDIA Kumo Relational Foundation Model (RFM).

    Queries Kumo NIM for:
      - e_hat: Propensity scores via multi-table graph classification (100-patient in-context anchors)
      - m0_hat: Baseline counterfactual readmission trend via multi-table regression (80-patient in-context anchors)

    Computes Sant'Anna & Zhao (2020) ATT with self-normalized (Hajek) weights and bootstrap SE.
    """
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise ValueError("NVIDIA_API_KEY environment variable is required to run Kumo RFM.")

    print(f"[*] Kumo Relational Model (RFM) DR-DiD — Target cohort: {sample_size} patients (NVIDIA NIM cloud API)")

    # 1. Stratified evaluation cohort
    rng = np.random.RandomState(random_state)
    ctrl_pool = df_wide[df_wide["treatment"] == 0]["patient_nbr"].values
    trt_pool = df_wide[df_wide["treatment"] == 1]["patient_nbr"].values

    n_half = sample_size // 2
    eval_ctrl = rng.choice(ctrl_pool, n_half, replace=False)
    eval_trt = rng.choice(trt_pool, n_half, replace=False)
    eval_pids = np.concatenate([eval_ctrl, eval_trt])
    rng.shuffle(eval_pids)

    # 2. Context pool (100 patients: 50 ctrl, 50 trt for clf; 80 ctrl for reg)
    rem_ctrl = np.setdiff1d(ctrl_pool, eval_pids)
    rem_trt = np.setdiff1d(trt_pool, eval_pids)
    ctx_ctrl = rng.choice(rem_ctrl, 50, replace=False)
    ctx_trt = rng.choice(rem_trt, 50, replace=False)
    ctx_pids_clf = np.concatenate([ctx_ctrl, ctx_trt]).tolist()
    # For outcome regression, context is purely control units
    ctx_pids_reg = rng.choice(rem_ctrl, 80, replace=False).tolist()

    all_query_pids = list(set(eval_pids.tolist() + ctx_pids_clf + ctx_pids_reg))
    pids_str = ",".join(str(p) for p in all_query_pids)

    # 3. Pull relational data from SQLite
    print(f"[*] Extracting relational graph from {db_path} for {len(all_query_pids):,} patients...")
    conn = sqlite3.connect(db_path)
    pat_df = pd.read_sql_query(f"SELECT patient_nbr, race, gender, age FROM patients WHERE patient_nbr IN ({pids_str})", conn)
    enc_df = pd.read_sql_query(f"SELECT encounter_id, patient_nbr, time_in_hospital, num_lab_procedures, num_medications, number_emergency, number_inpatient, number_diagnoses FROM encounters WHERE patient_nbr IN ({pids_str})", conn)
    med_df = pd.read_sql_query(f"SELECT id, encounter_id, patient_nbr, drug_name, status, is_dosage_change FROM medications WHERE patient_nbr IN ({pids_str})", conn)
    conn.close()

    pat_map = {r["patient_nbr"]: r for _, r in pat_df.iterrows()}
    enc_map = {}
    for _, r in enc_df.iterrows():
        enc_map.setdefault(r["patient_nbr"], []).append(r)
    med_map = {}
    for _, r in med_df.iterrows():
        med_map.setdefault(r["patient_nbr"], []).append(r)
    panel_map = {r["patient_nbr"]: r for _, r in df_wide[df_wide["patient_nbr"].isin(all_query_pids)].iterrows()}

    # 4. Run Kumo RFM in batches
    n_batches = int(np.ceil(len(eval_pids) / batch_size))
    print(f"[*] Querying Kumo RFM across {n_batches} batches ({batch_size} patients/batch)...")

    e_hat_dict = {}
    m0_hat_dict = {}
    t0 = time.time()

    for b in range(n_batches):
        batch_targets = eval_pids[b * batch_size : (b + 1) * batch_size].tolist()

        # A. Propensity Score Query (Binary Classification)
        payload_clf = _build_kumo_payload(
            ctx_ids=ctx_pids_clf,
            pred_ids=batch_targets,
            panel_map=panel_map,
            pat_map=pat_map,
            enc_map=enc_map,
            med_map=med_map,
            task_kind="binary_classification",
        )
        res_clf = _query_kumo_nim(payload_clf, api_key)
        for pred in res_clf["predictions"]:
            orig_idx = int(pred["id"]) - len(ctx_pids_clf)
            target_pid = batch_targets[orig_idx]
            p_true = pred["probabilities"]["true"]
            e_hat_dict[target_pid] = np.clip(p_true, 0.05, 0.95)

        # B. Baseline Outcome Regression Query
        payload_reg = _build_kumo_payload(
            ctx_ids=ctx_pids_reg,
            pred_ids=batch_targets,
            panel_map=panel_map,
            pat_map=pat_map,
            enc_map=enc_map,
            med_map=med_map,
            task_kind="regression",
        )
        res_reg = _query_kumo_nim(payload_reg, api_key)
        for pred in res_reg["predictions"]:
            orig_idx = int(pred["id"]) - len(ctx_pids_reg)
            target_pid = batch_targets[orig_idx]
            m0_hat_dict[target_pid] = float(pred["prediction"])

        if (b + 1) % 6 == 0 or (b + 1) == n_batches:
            print(f"    [batch {b+1}/{n_batches}] Completed {len(e_hat_dict)} patients in {time.time() - t0:.1f}s")

    # 5. Compute Sant'Anna & Zhao Doubly Robust ATT with Hajek Normalization
    N = len(eval_pids)
    D = np.array([panel_map[pid]["treatment"] for pid in eval_pids], dtype=float)
    dY = np.array([panel_map[pid]["delta_readmitted_30d"] for pid in eval_pids], dtype=float)
    e_hat = np.array([e_hat_dict[pid] for pid in eval_pids], dtype=float)
    m0_hat = np.array([m0_hat_dict[pid] for pid in eval_pids], dtype=float)

    weight_ctrl = e_hat / (1.0 - e_hat)
    mean_w_ctrl = np.mean(weight_ctrl * (1.0 - D))
    mean_D = np.mean(D)

    # Hajek-stabilized influence function
    psi = (D * (dY - m0_hat)) / mean_D - ((1.0 - D) * weight_ctrl * (dY - m0_hat)) / mean_w_ctrl
    tau = float(np.mean(psi))

    # Asymptotic SE
    se_asym = float(np.std(psi, ddof=1) / np.sqrt(N))
    z_asym = tau / se_asym
    p_asym = float(2 * (1 - stats.norm.cdf(abs(z_asym))))

    # Bootstrap SE (B=500)
    boot_taus = []
    boot_rng = np.random.RandomState(random_state)
    for _ in range(bootstrap_B):
        idx = boot_rng.choice(N, N, replace=True)
        w_b = e_hat[idx] / (1.0 - e_hat[idx])
        mw_b = np.mean(w_b * (1.0 - D[idx]))
        mD_b = np.mean(D[idx])
        psi_b = (D[idx] * (dY[idx] - m0_hat[idx])) / mD_b - ((1.0 - D[idx]) * w_b * (dY[idx] - m0_hat[idx])) / mw_b
        boot_taus.append(np.mean(psi_b))

    se_boot = float(np.std(boot_taus, ddof=1))
    z_boot = tau / se_boot
    p_boot = float(2 * (1 - stats.norm.cdf(abs(z_boot))))
    ci_lower = tau - 1.96 * se_boot
    ci_upper = tau + 1.96 * se_boot

    print(f"[OK] Kumo RFM DR-DiD Complete:")
    print(f"     ATT:       {tau*100:+.3f}% pts")
    print(f"     SE (boot): {se_boot*100:.3f}% pts | 95% CI: [{ci_lower*100:+.3f}%, {ci_upper*100:+.3f}%] (p={p_boot:.4f})")

    return {
        "model": "5. Kumo Relational Foundation Model DR-DiD",
        "estimate": tau,
        "std_error": se_boot,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_boot,
        "se_asymptotic": se_asym,
        "n_patients": N,
        "bootstrap_B": bootstrap_B,
        "propensity_mean": float(np.mean(e_hat)),
        "propensity_scores": e_hat,
        "treated_mask": D,
    }


if __name__ == "__main__":
    df = pd.read_csv("data/panel_patient_level.csv")
    res = run_kumo_dr_did(df, sample_size=100, batch_size=25)
    print(res)
