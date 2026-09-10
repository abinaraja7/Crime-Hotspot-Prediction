# Explainable-AI Crime Hotspot Dashboard

Decision-support artefact for *An Explainable AI-driven Spatial Intelligence Framework for
Crime Hotspot Prediction* (City of London Police open data).

## What it shows (six components + filtering)
1. **Predicted risk map** — relative risk *ranking* per 250 m cell for a chosen out-of-sample month.
2. **Operational top-k targeting** — the ranked patrol list with real hit-rate and PAI at a chosen coverage.
3. **Local explanation** — exact additive Logistic-Regression attribution for any selected cell ("why flagged").
4. **Global drivers** — standardised coefficients / odds ratios across the City.
5. **Model performance & trust** — real pooled walk-forward metrics and a spatial error map.
6. **Filters** — target month and patrol coverage (sidebar).

## Selected model and the calibration caveat
The operating model is **Logistic Regression**, chosen on the balanced evidence of Notebook 6
(best and most stable ranking; operational parity; best interpretability). Its probabilities are
**not well calibrated** (Brier 0.064), so the dashboard presents risk as a **relative ranking for
top-k targeting** and does **not** display these values as precise probabilities of crime.
Recalibration is noted as future work; it is not implemented here.

## Data provenance
All values are computed live from the real artefacts produced by the notebooks and stored in
`../processed/`: `models/fitted_models.joblib`, `model_table_250m.csv`,
`model_oof_predictions.csv`, `grid_250m_params.csv`, `crimes_with_grid250.csv`,
`model_selection_decision.json`. Nothing is hard-coded or fabricated.

## Run
```bash
pip install streamlit scikit-learn xgboost pandas numpy matplotlib joblib
# from this dashboard/ folder:
streamlit run app.py
```
If your data folder is elsewhere, set `COL_DATA_DIR` to the folder that contains `processed/`.
