# An Automated, Data-driven Approach to Crime Pattern Detection and Prediction

An explainable, reproducible system for short-term crime **hotspot prediction and decision support**
in the City of London, using open UK police data. MSc Data Science dissertation project.

**Author:** Mallika Abina (25931942) · 
**Supervisor:** Liangxiu Han
**Manchester Metropolitan University**, Department of Computing and Mathematics · 2026

---

## Overview

The system is built as eight connected components — data acquisition, preprocessing, crime-pattern
analysis, spatial representation & hotspot detection, feature engineering, machine-learning modelling,
explainability, and a decision-support dashboard — implemented as seven reproducible notebooks and a
Streamlit application. Crime is aggregated to a 250 m grid; a hotspot is defined empirically as a cell in
the top 20 % (P80) of occupied cells within a month; and next-month hotspot risk is predicted under a
leakage-safe expanding-window walk-forward scheme. Four models are compared (persistence baseline,
Logistic Regression, Random Forest, XGBoost); **Logistic Regression** is selected on the balance of
discrimination, stability, operational usefulness, calibration and interpretability, and explained with
exact additive (coefficient-based) attribution.

Risk is presented as a **relative ranking** for top-k patrol targeting, **not** as a calibrated
probability of crime. No causal or policing-effectiveness claims are made.

## Repository structure

```
Notebook_01_Data_Collection_and_Preprocessing.ipynb   # NB1  raw -> 8,435 analysis-ready records
Notebook_02_Exploratory_Data_Analysis.ipynb           # NB2  composition, concentration (Gini/Lorenz)
Notebook_03_Spatial_Analysis.ipynb                    # NB3  250 m grid, KDE, P80 hotspots, 500 m check
Notebook_04_Feature_Engineering.ipynb                 # NB4  temporal/spatial + OSM features (924 x 9)
Notebook_05_Model_Development.ipynb                   # NB5  4-model ladder, walk-forward validation
Notebook_06_Evaluation_and_Explainable_AI.ipynb       # NB6  metrics, calibration, PAI, model selection
Notebook_07_Dashboard.ipynb                           # NB7  dashboard walkthrough on real artefacts
dashboard/
    app.py              # Streamlit user interface
    dashboard_core.py   # data/logic layer (loads the real artefacts)
    README.md
processed/              # persisted artefacts written by the notebooks (model table, params, models, metrics)
requirements.txt        # exact package versions used
```

> The raw street-level CSVs are **open data** from https://data.police.uk (City of London Police,
> January 2025 – January 2026) and OpenStreetMap contributors; they are not redistributed here.

## Requirements

Python 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: numpy, pandas, scikit-learn, xgboost, matplotlib, shap, streamlit, joblib.
Random seeds are fixed (42) for reproducibility.

## How to run

**Notebooks** — run NB1 → NB7 in order (each writes artefacts to `processed/` that the next consumes):

```bash
jupyter notebook
```

**Dashboard** — from the `dashboard/` folder:

```bash
pip install -r ../requirements.txt
streamlit run app.py
```

The dashboard reads the real artefacts produced by the notebooks and presents six components — a
risk-ranking map, operational top-k targeting, local and global explanations, a performance/trust
panel, and month/coverage filters.

## Key results (out-of-sample, 7 walk-forward test months)

| Model | ROC-AUC | PR-AUC | Brier |
|---|---|---|---|
| Persistence | 0.855 | 0.599 | 0.066 |
| Logistic Regression (selected) | 0.974 | 0.866 | 0.064 |
| Random Forest | 0.970 | 0.822 | 0.047 |
| XGBoost | 0.964 | 0.842 | 0.059 |

All learned models beat the persistence baseline on ranking; Logistic Regression is the strongest and
most stable ranker, and the most interpretable. Random Forest is best calibrated (Brier 0.047) — a
stated limitation, secondary to ranking for top-k targeting.

## Ethics

The project used only open, aggregated, anonymised data and received institutional ethical approval
(EthOS Project ID 92128; Review Reference 2026-92128-70264). Recorded crime is treated as reported/
detected crime, not true incidence.
