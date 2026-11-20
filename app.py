import os
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
from google import genai
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="ICU Sepsis Admission Risk Triage & GenAI Copilot", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = "/home/aakash/cllg/SD/s3/mp" if os.path.exists("/home/aakash/cllg/SD/s3/mp") else BASE_DIR

BUNDLE_PATH = os.path.join(DATA_DIR, "sepsis_triage_bundle.joblib")
HOSP_A_CSV = os.path.join(DATA_DIR, "hospital_A_cleaned.csv")
HOSP_B_10_CSV = os.path.join(DATA_DIR, "hospital_B_10pct.csv")
HOSP_B_50_CSV = os.path.join(DATA_DIR, "hospital_B_50pct.csv")
HOSP_B_100_CSV = os.path.join(DATA_DIR, "hospital_B_100pct.csv")

RAW_COLS = [
    "Patient_ID", "ICULOS", "SepsisLabel", "HospAdmTime", "Age", "Gender", "Unit1", "Unit2",
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2",
    "Lactate", "Creatinine", "BUN", "Bilirubin_total", "Bilirubin_direct",
    "AST", "Alkalinephos", "Glucose", "PTT", "Potassium", "Chloride", "TroponinI",
    "Platelets", "pH", "BaseExcess", "HCO3", "Hgb", "Hct",
    "Calcium", "Magnesium", "Phosphate", "SaO2", "Fibrinogen", "WBC"
]

TABLE_COLS = [
    "Patient_ID", "Patient_Name", "ICULOS", "HospAdmTime", "Age", "Gender", "Unit1", "Unit2",
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "FiO2",
    "Lactate", "Creatinine", "Platelets", "WBC"
]


def safe_predict_proba_pos(model, X):
    probs = model.predict_proba(X)
    if probs.ndim == 1:
        return np.asarray(probs, dtype=float)
    if probs.shape[1] == 1:
        cls = model.classes_[0] if hasattr(model, "classes_") and len(model.classes_) > 0 else 0
        return np.ones(len(X), dtype=float) if cls == 1 else np.zeros(len(X), dtype=float)
    return np.asarray(probs[:, 1], dtype=float)


def generate_patient_id(existing_ids=None):
    existing = set(str(x) for x in (existing_ids or []))
    for _ in range(500):
        pid = f"100{random.randint(100, 999)}"
        if pid not in existing:
            return pid
    return f"100{random.randint(1000, 9999)}"


def get_john_doe_rows(patient_id):
    return [
        {
            "Patient_ID": str(patient_id), "Patient_Name": "John Doe", "ICULOS": 1,
            "HospAdmTime": -84.0, "Age": 72.0, "Gender": 1.0, "Unit1": 1.0, "Unit2": 0.0,
            "HR": 102.0, "O2Sat": 95.0, "Temp": 38.0, "SBP": 114.0, "MAP": 70.0,
            "DBP": 56.0, "Resp": 21.0, "FiO2": 0.40,
            "Lactate": 2.2, "Creatinine": 1.5, "Platelets": 148.0, "WBC": 14.9,
        },
        {
            "Patient_ID": str(patient_id), "Patient_Name": "John Doe", "ICULOS": 6,
            "HospAdmTime": -84.0, "Age": 72.0, "Gender": 1.0, "Unit1": 1.0, "Unit2": 0.0,
            "HR": 118.0, "O2Sat": 92.0, "Temp": 38.6, "SBP": 96.0, "MAP": 62.0,
            "DBP": 49.0, "Resp": 26.0, "FiO2": 0.50,
            "Lactate": 3.0, "Creatinine": 1.9, "Platelets": 124.0, "WBC": 17.2,
        },
    ]


def get_new_patient_rows(patient_id, patient_name=None):
    pname = patient_name or f"Patient_{patient_id}"
    return [
        {
            "Patient_ID": str(patient_id), "Patient_Name": pname, "ICULOS": 1,
            "HospAdmTime": np.nan, "Age": np.nan, "Gender": np.nan, "Unit1": np.nan, "Unit2": np.nan,
            "HR": np.nan, "O2Sat": np.nan, "Temp": np.nan, "SBP": np.nan, "MAP": np.nan,
            "DBP": np.nan, "Resp": np.nan, "FiO2": np.nan,
            "Lactate": np.nan, "Creatinine": np.nan, "Platelets": np.nan, "WBC": np.nan,
        },
        {
            "Patient_ID": str(patient_id), "Patient_Name": pname, "ICULOS": 6,
            "HospAdmTime": np.nan, "Age": np.nan, "Gender": np.nan, "Unit1": np.nan, "Unit2": np.nan,
            "HR": np.nan, "O2Sat": np.nan, "Temp": np.nan, "SBP": np.nan, "MAP": np.nan,
            "DBP": np.nan, "Resp": np.nan, "FiO2": np.nan,
            "Lactate": np.nan, "Creatinine": np.nan, "Platelets": np.nan, "WBC": np.nan,
        },
    ]


def build_workstream5_cohort(df_raw, is_train=True, train_stats=None):
    df = df_raw.copy()
    df["Patient_ID"] = df["Patient_ID"].astype(str)
    if "SepsisLabel" not in df.columns:
        df["SepsisLabel"] = 0
    for col in RAW_COLS:
        if col not in df.columns:
            df[col] = np.nan

    df = df.drop(columns=["Unnamed: 0", "Hour", "EtCO2"], errors="ignore")

    clip_ranges = {
        "Temp": (25.0, 44.0), "HR": (30, 300), "SBP": (30, 250),
        "MAP": (20, 200), "DBP": (20, 150), "O2Sat": (20, 100),
        "FiO2": (0.21, 1.0), "pH": (6.5, 7.8)
    }
    for col, (c_min, c_max) in clip_ranges.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=c_min, upper=c_max)

    patients = df.groupby("Patient_ID").agg(
        min_iculos=("ICULOS", "min"),
        Future_Sepsis=("SepsisLabel", "max"),
        hosp_adm_time=("HospAdmTime", "first"),
        Age=("Age", "first"),
        Gender=("Gender", "first"),
        Unit1=("Unit1", "first"),
        Unit2=("Unit2", "first"),
    )
    sepsis_rows = df[df["SepsisLabel"] == 1]
    patients["first_sepsis_iculos"] = sepsis_rows.groupby("Patient_ID")["ICULOS"].min() if not sepsis_rows.empty else np.nan

    adm_cohort = patients[patients["min_iculos"] <= 6].copy()
    adm_cohort["Present_On_Arrival_Sepsis"] = (adm_cohort["first_sepsis_iculos"] <= 6).astype(int)
    adm_cohort["HospAdmTime_is_positive"] = (adm_cohort["hosp_adm_time"] > 0).astype(int)

    if is_train:
        valid_neg = adm_cohort.loc[adm_cohort["hosp_adm_time"] <= 0, "hosp_adm_time"]
        h_median = float(valid_neg.median()) if not valid_neg.empty else -2.84
        temp_clean = adm_cohort["hosp_adm_time"].fillna(h_median).clip(upper=0.0)
        h_1st_pctl = float(np.percentile(temp_clean, 1)) if len(temp_clean) > 0 else -657.37
        train_stats = {"h_median": h_median, "h_1st_pctl": h_1st_pctl}
    else:
        h_median = train_stats["h_median"]
        h_1st_pctl = train_stats["h_1st_pctl"]

    adm_cohort["hosp_adm_time_clean"] = adm_cohort["hosp_adm_time"].fillna(h_median).clip(lower=h_1st_pctl, upper=0.0)
    adm_cohort["ICU_Unit"] = np.select(
        [adm_cohort["Unit1"] == 1, adm_cohort["Unit2"] == 1],
        ["MICU", "SICU"],
        default="Unknown_Unit",
    )

    cohort_model = adm_cohort[adm_cohort["Present_On_Arrival_Sepsis"] == 0].copy()
    df_06 = df[(df["Patient_ID"].isin(cohort_model.index)) & (df["ICULOS"] <= 6)].copy()

    vitals = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
    vitals_agg = df_06.groupby("Patient_ID")[vitals].agg(["first", "max", "min", "mean", "std"])
    vitals_agg.columns = [f"{col}_{stat}" for col, stat in vitals_agg.columns]

    df_06["Temp_dev"] = (df_06["Temp"] - 37.0).abs()
    vitals_agg["Temp_max_dev_37"] = df_06.groupby("Patient_ID")["Temp_dev"].max()

    last_v = df_06.groupby("Patient_ID")[["HR", "MAP", "Resp", "SBP", "O2Sat"]].last()
    first_v = df_06.groupby("Patient_ID")[["HR", "MAP", "Resp", "SBP", "O2Sat"]].first()
    delta_v = last_v - first_v
    delta_v.columns = [f"{col}_Delta_1_to_6" for col in delta_v.columns]

    df_06["Shock_Index"] = df_06["HR"] / (df_06["SBP"] + 1e-4)
    df_06["Pulse_Pressure"] = df_06["SBP"] - df_06["DBP"]
    df_06["SF_Ratio"] = df_06["O2Sat"] / (df_06["FiO2"] + 1e-4)

    comp_agg = df_06.groupby("Patient_ID").agg(
        Max_Shock_Index=("Shock_Index", "max"),
        Mean_Shock_Index=("Shock_Index", "mean"),
        Min_Pulse_Pressure=("Pulse_Pressure", "min"),
        Min_SF_Ratio=("SF_Ratio", "min"),
    )

    max_labs = [
        "Lactate", "Creatinine", "BUN", "Bilirubin_total", "Bilirubin_direct",
        "AST", "Alkalinephos", "Glucose", "PTT", "Potassium", "Chloride", "TroponinI"
    ]
    min_labs = [
        "Platelets", "pH", "BaseExcess", "HCO3", "Hgb", "Hct",
        "Calcium", "Magnesium", "Phosphate", "SaO2", "Fibrinogen"
    ]

    lab_max_df = df_06.groupby("Patient_ID")[max_labs].max()
    lab_max_df.columns = [f"{col}_worst" for col in lab_max_df.columns]
    lab_min_df = df_06.groupby("Patient_ID")[min_labs].min()
    lab_min_df.columns = [f"{col}_worst" for col in lab_min_df.columns]

    df_06["WBC_dev"] = (df_06["WBC"] - 8.5).abs()
    wbc_agg = df_06.groupby("Patient_ID").agg(WBC_max_dev_8_5=("WBC_dev", "max"))

    unit_dummies = pd.get_dummies(cohort_model["ICU_Unit"], prefix="ICU_Unit", dtype=int)
    for expected_u in ["ICU_Unit_MICU", "ICU_Unit_SICU", "ICU_Unit_Unknown_Unit"]:
        if expected_u not in unit_dummies.columns:
            unit_dummies[expected_u] = 0
    unit_dummies = unit_dummies[["ICU_Unit_MICU", "ICU_Unit_SICU", "ICU_Unit_Unknown_Unit"]]

    demo_df = pd.concat([cohort_model[["Age", "Gender", "hosp_adm_time_clean"]], unit_dummies], axis=1)
    X_phys = pd.concat([demo_df, vitals_agg, delta_v, comp_agg, lab_max_df, lab_min_df, wbc_agg], axis=1)
    drop_cols = [c for c in X_phys.columns if c.startswith(("SBP_", "DBP_", "Hct_", "HCO3_"))]
    X_pruned = X_phys.drop(columns=drop_cols)
    y = cohort_model["Future_Sepsis"]
    return adm_cohort, cohort_model, X_pruned, y, train_stats


def build_triage_summary(y_true, y_prob, max_missed_frac=0.10, high_risk_top_frac=0.15):
    df_t = pd.DataFrame({"y": y_true, "prob": y_prob})
    septic_probs = df_t.loc[df_t["y"] == 1, "prob"]
    t_low = float(np.percentile(septic_probs, max_missed_frac * 100)) if len(septic_probs) > 0 else 0.0353
    t_high = float(np.percentile(y_prob, (1.0 - high_risk_top_frac) * 100)) if len(y_prob) > 0 else 0.0605

    buckets = np.select(
        [y_prob < t_low, y_prob >= t_high],
        ["1_Low_Risk", "3_High_Risk"],
        default="2_Medium_Risk",
    )
    df_t["Bucket"] = buckets

    rows = []
    for b in ["1_Low_Risk", "2_Medium_Risk", "3_High_Risk"]:
        sub = df_t[df_t["Bucket"] == b]
        n, s = len(sub), int(sub["y"].sum())
        rows.append({
            "Risk_Tier": b,
            "Share_%": f"{(n / max(len(y_true), 1)) * 100:.1f}%",
            "Total_Patients": n,
            "Septic_Cases": s,
            "Actual_Sepsis_Rate_%": f"{(s / max(n, 1)) * 100:.1f}%",
            "Share_of_All_Septic_%": f"{(s / max(int(y_true.sum()), 1)) * 100:.1f}%",
        })
    return pd.DataFrame(rows), t_low, t_high


def clinical_context_for_feature(feat_name):
    name = feat_name.lower()
    if "lactate" in name or "baseexcess" in name or "ph" in name:
        return "Tissue perfusion & metabolic acid-base balance"
    if "map" in name or "shock_index" in name or "pulse_pressure" in name or "sbp" in name:
        return "Hemodynamic stability & organ perfusion pressure"
    if "sf_ratio" in name or "o2sat" in name or "resp" in name or "fio2" in name:
        return "Respiratory drive & oxygenation efficiency"
    if "creatinine" in name or "bun" in name:
        return "Renal filtration / organ dysfunction marker"
    if "platelets" in name or "wbc" in name or "hgb" in name or "hct" in name:
        return "Hematologic / inflammatory host response"
    if "hosp_adm_time" in name:
        return "Pre-ICU hospital stay length (prior exposure window)"
    if "icu_unit" in name:
        return "ICU care pathway context"
    if "age" in name or "temp" in name:
        return "Baseline host vulnerability & thermoregulation"
    return "0-6h physiological summary feature"


@st.cache_resource
def load_and_fit_pipeline():
    metrics_records = []
    triage_tables = {}

    if os.path.exists(BUNDLE_PATH):
        loaded = joblib.load(BUNDLE_PATH)
        return {
            "train_stats": loaded["train_stats"],
            "feature_columns": loaded["feature_columns"],
            "calibrated_model": loaded["calibrated_model"],
            "background_df": loaded["background_df"],
            "t_low": float(loaded["t_low"]),
            "t_high": float(loaded["t_high"]),
            "metrics_df": loaded.get("metrics_df", pd.DataFrame(metrics_records)),
            "triage_tables": loaded.get("triage_tables", triage_tables),
        }

    if os.path.exists(HOSP_A_CSV):
        df_A = pd.read_csv(HOSP_A_CSV, dtype={"Patient_ID": str})
    else:
        rng = np.random.default_rng(42)
        rows = []
        for pid in range(1, 201):
            is_sep = int(pid % 5 == 0)
            for h in range(1, 9):
                r = {c: np.nan for c in RAW_COLS}
                r.update({
                    "Patient_ID": f"{pid:06d}",
                    "ICULOS": h,
                    "SepsisLabel": 1 if (is_sep and h >= 7) else 0,
                    "HospAdmTime": -90.0 if is_sep else -12.0,
                    "Age": 73.0 if is_sep else 56.0,
                    "Gender": 1.0,
                    "Unit1": 1.0,
                    "Unit2": 0.0,
                    "HR": (112.0 + min(h, 6) * 2 + float(rng.normal(0, 1.5))) if is_sep else 80.0,
                    "O2Sat": (93.0 - min(h, 6) * 0.5) if is_sep else 98.0,
                    "Temp": 38.4 if is_sep else 37.0,
                    "SBP": (102.0 - min(h, 6) * 2) if is_sep else 124.0,
                    "MAP": (66.0 - min(h, 6) * 1.5) if is_sep else 84.0,
                    "DBP": 54.0 if is_sep else 68.0,
                    "Resp": 24.0 if is_sep else 16.0,
                    "FiO2": 0.45 if is_sep else 0.21,
                    "Lactate": 2.8 if is_sep else 1.1,
                    "Creatinine": 1.8 if is_sep else 0.9,
                    "Platelets": 135.0 if is_sep else 220.0,
                    "WBC": 15.8 if is_sep else 7.8,
                })
                rows.append(r)
        df_A = pd.DataFrame(rows, columns=RAW_COLS)

    _, _, X_A, y_A, train_stats_A = build_workstream5_cohort(df_A, is_train=True)
    strat = y_A if y_A.nunique() > 1 else None
    X_A_tr, X_A_te, y_A_tr, y_A_te = train_test_split(
        X_A, y_A, test_size=0.20, random_state=42, stratify=strat
    )

    monotone_dict = {
        "hosp_adm_time_clean": -1, "Age": 1, "Lactate_worst": 1,
        "Max_Shock_Index": 1, "Min_SF_Ratio": -1, "MAP_min": -1, "Creatinine_worst": 1,
    }
    monotone_cst = [monotone_dict.get(c, 0) for c in X_A_tr.columns]

    base_hgb = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05, max_depth=4, max_leaf_nodes=15, min_samples_leaf=40,
        monotonic_cst=monotone_cst, random_state=42,
    )
    calibrated_hgb = CalibratedClassifierCV(estimator=base_hgb, method="isotonic", cv=3)
    calibrated_hgb.fit(X_A_tr, y_A_tr)
    probs_A_te = safe_predict_proba_pos(calibrated_hgb, X_A_te)
    p_a, r_a, _ = precision_recall_curve(y_A_te, probs_A_te)
    tab_a, t_low_a, t_high_a = build_triage_summary(y_A_te.values, probs_A_te)
    triage_tables["Hospital A (Holdout)"] = tab_a
    auroc_a = round(float(roc_auc_score(y_A_te, probs_A_te)), 4) if y_A_te.nunique() > 1 else None
    metrics_records.append({
        "Stage": "Hospital A (Holdout)",
        "AUPRC": round(float(auc(r_a, p_a)), 4),
        "AUROC": auroc_a,
        "Baseline_Prevalence": round(float(y_A_te.mean()), 4),
    })

    active_model = calibrated_hgb
    t_low, t_high = t_low_a, t_high_a
    background_df = X_A_te

    if os.path.exists(HOSP_B_50_CSV):
        df_B50 = pd.read_csv(HOSP_B_50_CSV, dtype={"Patient_ID": str})
        _, _, X_B50, y_B50, _ = build_workstream5_cohort(df_B50, is_train=False, train_stats=train_stats_A)
        X_B50 = X_B50.reindex(columns=X_A_tr.columns, fill_value=0)
        if y_B50.nunique() > 1:
            final_calibrated_hgb = CalibratedClassifierCV(estimator=base_hgb, method="isotonic", cv=3)
            final_calibrated_hgb.fit(X_B50, y_B50)
            active_model = final_calibrated_hgb
            background_df = X_B50

    if os.path.exists(HOSP_B_100_CSV):
        df_B100 = pd.read_csv(HOSP_B_100_CSV, dtype={"Patient_ID": str})
        _, _, X_B100, y_B100, _ = build_workstream5_cohort(df_B100, is_train=False, train_stats=train_stats_A)
        X_B100 = X_B100.reindex(columns=X_A_tr.columns, fill_value=0)
        probs_B100 = safe_predict_proba_pos(active_model, X_B100)
        p_b, r_b, _ = precision_recall_curve(y_B100, probs_B100)
        tab_b, t_low_b, t_high_b = build_triage_summary(y_B100.values, probs_B100)
        triage_tables["Hospital B (100% Release)"] = tab_b
        t_low, t_high = t_low_b, t_high_b
        background_df = X_B100
        auroc_b = round(float(roc_auc_score(y_B100, probs_B100)), 4) if y_B100.nunique() > 1 else None
        metrics_records.append({
            "Stage": "Hospital B (100% Release)",
            "AUPRC": round(float(auc(r_b, p_b)), 4),
            "AUROC": auroc_b,
            "Baseline_Prevalence": round(float(y_B100.mean()), 4),
        })

    return {
        "train_stats": train_stats_A,
        "feature_columns": list(X_A_tr.columns),
        "calibrated_model": active_model,
        "background_df": background_df,
        "t_low": t_low,
        "t_high": t_high,
        "metrics_df": pd.DataFrame(metrics_records),
        "triage_tables": triage_tables,
    }


def init_small_patient_table():
    if "small_patient_df" not in st.session_state:
        if os.path.exists(HOSP_B_10_CSV):
            b10 = pd.read_csv(HOSP_B_10_CSV, dtype={"Patient_ID": str})
            pids = b10["Patient_ID"].dropna().astype(str).unique()[:3]
            df = b10[b10["Patient_ID"].astype(str).isin(pids)].copy()
            df["Patient_Name"] = "Hospital_B_10pct_Patient"
        else:
            df = pd.DataFrame(get_john_doe_rows("100901"))
        for col in TABLE_COLS:
            if col not in df.columns:
                df[col] = np.nan
        st.session_state.small_patient_df = df[TABLE_COLS].copy()


def generate_handover_note(patient_id, patient_name, bucket, prob, top3_df):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        drivers_txt = ", ".join(f"{r['Feature']} ({r['Value']:.2f})" for _, r in top3_df.iterrows())
        return (
            f"Patient {patient_name} (ID: {patient_id}) is assigned to {bucket} with an estimated "
            f"0-6h sepsis risk of {prob * 100:.1f}%, acting as a decision-support estimate requiring "
            f"clinical verification. Key physiological drivers are {drivers_txt}. "
            f"Conduct Sepsis-3 bedside assessment and serial monitoring per clinical protocol."
        )

    client = genai.Client(api_key=api_key)
    driver_lines = "\n".join(
        f"- {r['Feature']}: {r['Value']:.2f} (SHAP={r['SHAP_Impact']:+.4f}; {r['Clinical_Domain']})"
        for _, r in top3_df.iterrows()
    )
    prompt = f"""You are an ICU clinical decision support assistant translating ML triage output for clinicians.
Use ONLY the structured evidence below. Do NOT invent vital signs, lab values, diagnoses, or citations.
Treat model risk scores as decision-support estimates requiring clinical confirmation, not a confirmed diagnosis.

Patient Name: {patient_name}
Patient ID: {patient_id}
Assigned Risk Tier: {bucket}
Estimated Sepsis Probability (0-6h window summary -> future sepsis): {prob * 100:.1f}%
Top Physiological Risk Drivers (local SHAP):
{driver_lines}

Write a concise 3-sentence clinical handover note that:
1) States the assigned risk tier and estimated sepsis probability with decision-support caveat;
2) Cites the exact physiological driver names and values above;
3) Suggests standard Sepsis-3 bedside monitoring/re-evaluation steps without inventing unprovided measurements."""

    resp = client.models.generate_content(model="gemini-3.1-flash-lite", contents=prompt)
    return (resp.text or "").strip()


pipeline = load_and_fit_pipeline()
init_small_patient_table()

st.title("ICU Sepsis Admission Risk Triage & GenAI Copilot")
st.caption(
    "Decision-support interface combining calibrated 0-6h physiology modeling (HistGradientBoostingClassifier + SHAP) "
    "with an evidence-bounded GenAI clinical handover copilot (gemini-3.1-flash-lite). Not a standalone diagnosis."
)

tab1, tab2 = st.tabs(["1. Live Patient Triage & GenAI Copilot", "2. Cohort & Transfer Metrics"])

with tab1:
    b1, b2, b3 = st.columns([1.4, 1.4, 2.2])

    with b1:
        if st.button("Load Dummy Patient (John Doe)", use_container_width=True):
            existing = st.session_state.small_patient_df["Patient_ID"].dropna().astype(str).tolist()
            new_pid = generate_patient_id(existing)
            jd_df = pd.DataFrame(get_john_doe_rows(new_pid))[TABLE_COLS]
            st.session_state.small_patient_df = pd.concat(
                [st.session_state.small_patient_df, jd_df], ignore_index=True
            )
            st.success(f"Added John Doe (ID: {new_pid}) to the patient table.")
            st.rerun()

    with b2:
        if st.button("New Patient Data", use_container_width=True):
            existing = st.session_state.small_patient_df["Patient_ID"].dropna().astype(str).tolist()
            new_pid = generate_patient_id(existing)
            new_df = pd.DataFrame(get_new_patient_rows(new_pid))[TABLE_COLS]
            st.session_state.small_patient_df = pd.concat(
                [st.session_state.small_patient_df, new_df], ignore_index=True
            )
            st.info(f"Created new patient entry with ID {new_pid}. Fill only available fields and leave others blank.")
            st.rerun()

    with b3:
        uploaded_csv = st.file_uploader("Upload Patient CSV (optional; sparse/missing values allowed)", type=["csv"])
        if uploaded_csv is not None:
            up_df = pd.read_csv(uploaded_csv, dtype={"Patient_ID": str})
            if "Patient_Name" not in up_df.columns:
                up_df["Patient_Name"] = "Uploaded Patient"
            for col in TABLE_COLS:
                if col not in up_df.columns:
                    up_df[col] = np.nan
            st.session_state.small_patient_df = pd.concat(
                [st.session_state.small_patient_df, up_df[TABLE_COLS]], ignore_index=True
            ).drop_duplicates(subset=["Patient_ID", "ICULOS"], keep="last")

    st.subheader("Patient Cohort Table (Small Working Set — Sparse Inputs Supported)")
    st.caption("Edit cells directly or leave unmeasured vitals/labs blank. New patients are appended to this table and downloadable as CSV.")

    try:
        edited_df = st.data_editor(
            st.session_state.small_patient_df,
            num_rows="dynamic",
            use_container_width=True,
            key="small_patient_editor",
        )
        st.session_state.small_patient_df = edited_df.copy()
    except Exception as e:
        st.warning(f"Interactive table editor fallback active ({e}). Showing read-only view.")
        st.dataframe(st.session_state.small_patient_df, use_container_width=True)

    st.download_button(
        label="Download Patient Table (.csv)",
        data=st.session_state.small_patient_df.to_csv(index=False).encode("utf-8"),
        file_name="small_patients_0_6h.csv",
        mime="text/csv",
    )

    patient_ids = st.session_state.small_patient_df["Patient_ID"].dropna().astype(str).unique().tolist()
    selected_pid = st.selectbox("Select Patient ID to Score & Explain", options=patient_ids)

    if st.button("Run Triage & Copilot", type="primary"):
        sub_df = st.session_state.small_patient_df[
            st.session_state.small_patient_df["Patient_ID"].astype(str) == str(selected_pid)
        ].copy()

        p_name = (
            sub_df["Patient_Name"].dropna().astype(str).iloc[0]
            if "Patient_Name" in sub_df.columns and not sub_df["Patient_Name"].dropna().empty
            else f"Patient_{selected_pid}"
        )

        for col in TABLE_COLS:
            if col not in ("Patient_ID", "Patient_Name"):
                sub_df[col] = pd.to_numeric(sub_df[col], errors="coerce")
        for col in RAW_COLS:
            if col not in sub_df.columns:
                sub_df[col] = 0 if col == "SepsisLabel" else np.nan
        sub_df = sub_df[RAW_COLS]

        _, _, X_new, _, _ = build_workstream5_cohort(
            sub_df, is_train=False, train_stats=pipeline["train_stats"]
        )

        if X_new.empty:
            st.warning("Ensure this patient has at least one row with ICULOS <= 6.")
        else:
            feat_cols = pipeline["feature_columns"]
            X_aligned = X_new.reindex(columns=feat_cols, fill_value=0)
            model = pipeline["calibrated_model"]
            prob = float(safe_predict_proba_pos(model, X_aligned)[0])
            t_low, t_high = pipeline["t_low"], pipeline["t_high"]

            if prob < t_low:
                bucket = "1_Low_Risk"
            elif prob >= t_high:
                bucket = "3_High_Risk"
            else:
                bucket = "2_Medium_Risk"

            fitted_est = (
                model.calibrated_classifiers_[0].estimator
                if hasattr(model, "calibrated_classifiers_")
                else model
            )
            bg_sample = pipeline["background_df"].reindex(columns=feat_cols, fill_value=0).sample(
                min(100, len(pipeline["background_df"])), random_state=42
            )
            explainer = shap.Explainer(lambda x: safe_predict_proba_pos(fitted_est, x), bg_sample)
            p_shap = explainer(X_aligned).values[0]

            shap_df = pd.DataFrame({
                "Feature": feat_cols,
                "Clinical_Domain": [clinical_context_for_feature(c) for c in feat_cols],
                "Value": X_aligned.iloc[0].values,
                "SHAP_Impact": p_shap,
            }).sort_values("SHAP_Impact", ascending=False)

            top3_df = shap_df.head(3).copy()

            st.divider()
            m1, m2, m3 = st.columns(3)
            m1.metric("Patient", f"{p_name} ({selected_pid})")
            m2.metric("Assigned Risk Tier", bucket)
            m3.metric("Estimated Sepsis Probability", f"{prob * 100:.1f}%")

            st.subheader("Top Physiological Risk Drivers (SHAP & Clinical Context)")
            st.dataframe(top3_df, use_container_width=True, hide_index=True)

            st.subheader("GenAI Clinical Handover Note")
            with st.spinner("Generating evidence-bounded clinical handover note..."):
                note = generate_handover_note(
                    patient_id=selected_pid,
                    patient_name=p_name,
                    bucket=bucket,
                    prob=prob,
                    top3_df=top3_df,
                )
            st.info(note)

with tab2:
    st.subheader("Model Discrimination & Calibration Summary")
    st.dataframe(pipeline["metrics_df"], use_container_width=True, hide_index=True)

    for stage_name, t_df in pipeline["triage_tables"].items():
        st.subheader(f"Safety-Constrained Triage Table — {stage_name}")
        st.dataframe(t_df, use_container_width=True, hide_index=True)
