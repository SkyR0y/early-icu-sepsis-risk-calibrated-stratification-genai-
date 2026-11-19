# Early ICU Sepsis Stratification with Calibrated Gradient Boosting and GenAI Handover Support

An end-to-end clinical decision-support prototype combining a calibrated 0–6 hour ICU physiology machine learning model (`HistGradientBoostingClassifier` with monotonic constraints and isotonic calibration), local SHAP feature attribution, and an evidence-bounded Generative AI clinical handover copilot (`gemini-3.1-flash-lite` via `google-genai`) inside an interactive multi-tab Streamlit frontend (`app.py`).

> **Disclaimer:** Model probabilities, risk tiers, SHAP drivers, and GenAI handover notes are decision-support summaries only and require clinical confirmation and Sepsis-3 assessment. They are not a standalone medical diagnosis.

## Project Structure

- `app.py` — Multi-tab Streamlit application featuring:
  - **Tab 1 (Live Patient Triage & Copilot):** Load a **John Doe** dummy patient, create **New Patient Data** with auto-generated IDs (`100xxx`) and sparse/missing inputs (`NaN`), upload patient CSVs, download the working patient table as `.csv`, and run ML triage + SHAP + `gemini-3.1-flash-lite` handover note generation.
  - **Tab 2 (Cohort & Transfer Metrics):** Displays discrimination/calibration metrics and safety-constrained risk triage tables across evaluated stages.
- `requirements.txt` — Pinned Python dependencies for local and Streamlit Community Cloud environments.
- `hospital_B_10pct.csv` — Lightweight Hospital B 10% release dataset used for cohort/triage testing and patient table workflows.

## Pipeline Overview (Workstreams 5 & 8)

1. **0–6h Cohort & Feature Engineering (`ICULOS <= 6`):** Filters admission cohorts (`min(ICULOS) <= 6`), flags Present-on-Arrival sepsis (`ICULOS <= 6`) separately from future sepsis modeling, clips implausible measurements, cleans pre-ICU `HospAdmTime`, consolidates `ICU_Unit` (`MICU`, `SICU`, `Unknown_Unit`), and extracts physiology-first vital sign summaries, `1 -> 6` deltas, Shock Index, Pulse Pressure, `S/F` Ratio, and worst lab values while avoiding clinician order-flag leakage.
2. **Calibrated Monotonic ML Model:** Uses `HistGradientBoostingClassifier` with biological monotonic constraints and `CalibratedClassifierCV(..., method="isotonic", cv=3)` to output calibrated probabilities and stratify patients into `1_Low_Risk`, `2_Medium_Risk`, and `3_High_Risk` tiers under a `<=10%` missed septic case safety rule.
3. **Local SHAP + GenAI Clinical Copilot:** Computes local SHAP feature impacts for the selected patient and passes the structured risk tier, calibrated probability, and top SHAP drivers into an evidence-bounded prompt for `gemini-3.1-flash-lite` (`google-genai`) to generate a concise 3-sentence clinical handover note without inventing unprovided measurements.

## Running Locally

```bash
python -m venv .venv
source .venv/bin/activate        # On Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export GOOGLE_API_KEY="your_gemini_api_key"
streamlit run app.py