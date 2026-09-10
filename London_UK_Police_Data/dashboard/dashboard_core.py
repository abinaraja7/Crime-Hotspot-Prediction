"""
dashboard_core.py
Pure data/logic layer for the Explainable-AI Crime Hotspot dashboard.

Every value returned by this module is computed from the REAL artefacts produced by
Notebooks 3-6 (fitted Logistic Regression model, the modelling table, the walk-forward
out-of-fold predictions, and the crime grid). Nothing is hard-coded or fabricated.

Design decisions carried from the notebooks:
  * The selected model is Logistic Regression (Notebook 6, evidence-based).
  * The model's probabilities are NOT well calibrated (Brier 0.064). The dashboard therefore
    presents a RELATIVE RISK RANKING for top-k targeting, and must NOT present these values as
    precise probabilities of crime. See CALIBRATION_CAVEAT.
  * Risk shown for a month uses the genuine out-of-sample (walk-forward) prediction for that
    month where available, so the map reflects honest out-of-sample performance.
"""
from pathlib import Path
import os, json, math
import numpy as np
import pandas as pd
import joblib

SELECTED_MODEL = "LogisticRegression"

CALIBRATION_CAVEAT = (
    "Risk is shown as a RELATIVE RANKING (percentile / decile / rank) for top-k patrol "
    "targeting. The selected Logistic Regression model is a strong risk *ranker* but its raw "
    "scores are NOT well-calibrated probabilities (Brier 0.064; see Notebook 6). These values "
    "must not be read as precise probabilities of crime. Recalibration is noted as future work."
)

def _base_dir():
    env = os.environ.get("COL_DATA_DIR")
    if env: return Path(env)
    # default: the folder that contains this dashboard/ directory
    return Path(__file__).resolve().parents[1]

def load_artifacts(base_dir=None):
    base = Path(base_dir) if base_dir else _base_dir()
    proc = base / "processed"
    art = joblib.load(proc / "models" / "fitted_models.joblib")
    model_tbl = pd.read_csv(proc / "model_table_250m.csv")
    oof = pd.read_csv(proc / "model_oof_predictions.csv")
    params = pd.read_csv(proc / "grid_250m_params.csv").iloc[0]
    crimes = pd.read_csv(proc / "crimes_with_grid250.csv", parse_dates=["month"])
    crimes["ym"] = crimes["month"].dt.to_period("M").astype(str)
    decision = json.loads((proc / "model_selection_decision.json").read_text())
    return dict(proc=proc, art=art, model_tbl=model_tbl, oof=oof, params=params,
                crimes=crimes, decision=decision,
                features=art["features"],
                grid=dict(ncol=int(params["ncol"]), nrow=int(params["nrow"]),
                          size=int(params["cell_size_m"]), x0=float(params["x0"]),
                          y0=float(params["y0"]), hot_pct=int(params["hotspot_percentile"])))

def available_months(A):
    """Months for which genuine out-of-sample LR predictions exist (the honest display set)."""
    return sorted(A["oof"][A["oof"].model == SELECTED_MODEL]["month"].unique())

def risk_table(A, month):
    """Per-cell relative risk for a month, from the real out-of-sample LR prediction.
    Returns rank, percentile and decile — deliberately NOT labelled a probability."""
    d = A["oof"][(A["oof"].model == SELECTED_MODEL) & (A["oof"].month == month)].copy()
    d = d.rename(columns={"y_prob": "risk_score", "y_true": "was_hotspot"})
    d["rank"] = d["risk_score"].rank(ascending=False, method="first").astype(int)
    d["percentile"] = (100 * d["risk_score"].rank(pct=True)).round(1)
    d["decile"] = pd.qcut(d["risk_score"].rank(method="first"), 10,
                          labels=[f"D{i}" for i in range(1, 11)])
    cr = A["crimes"].groupby(["cell250", "ym"]).size()
    d["crimes_that_month"] = [int(cr.get((c, month), 0)) for c in d["cell"]]
    return d.sort_values("risk_score", ascending=False).reset_index(drop=True)

def top_k_targeting(A, month, coverage):
    """Operational readout for patrolling the top `coverage` fraction of cells by risk."""
    d = risk_table(A, month)
    k = max(1, int(round(coverage * len(d))))
    top = d.head(k)
    N = d["crimes_that_month"].sum(); n = top["crimes_that_month"].sum()
    hit = (n / N) if N else 0.0; area = k / len(d)
    return dict(k=k, n_cells=len(d), coverage=area, hit_rate=hit,
                pai=(hit / area) if area else float("nan"),
                crimes_captured=int(n), crimes_total=int(N),
                cells=top[["cell", "rank", "risk_score", "percentile",
                           "was_hotspot", "crimes_that_month"]])

def local_explanation(A, month, cell):
    """Exact additive Logistic-Regression attribution for one cell-month.
    phi_i = coef_i * (x_std_i - mean_std_i); base + sum(phi) = model log-odds (exact)."""
    feats = A["features"]; scaler = A["art"]["scaler"]; lr = A["art"]["logreg"]
    row = A["model_tbl"][(A["model_tbl"].month == month) & (A["model_tbl"].cell == cell)]
    if row.empty:
        return None
    x = row[feats].values.astype(float)
    xs = scaler.transform(x)[0]
    coef = lr.coef_[0]
    mean_s = scaler.transform(A["model_tbl"][feats].values).mean(axis=0)
    phi = coef * (xs - mean_s)
    base = lr.intercept_[0] + float((mean_s * coef).sum())
    logodds = base + float(phi.sum())
    prob = 1.0 / (1.0 + math.exp(-logodds))
    contrib = (pd.DataFrame({"feature": feats, "value": x[0], "contribution": phi})
               .sort_values("contribution", key=lambda s: s.abs(), ascending=False)
               .reset_index(drop=True))
    return dict(cell=int(cell), month=month, base=base, logodds=logodds,
                risk_score=prob, contributions=contrib)

def global_drivers(A):
    """Standardised coefficients + odds ratios — the exact, faithful global explanation for LR."""
    feats = A["features"]; lr = A["art"]["logreg"]; coef = lr.coef_[0]
    g = pd.DataFrame({"feature": feats, "std_coef": coef,
                      "odds_ratio_per_SD": np.exp(coef)})
    return g.sort_values("std_coef", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

def performance_summary(A):
    """Real pooled out-of-sample metrics for all models (recomputed from saved OOF preds)."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                                 precision_score, recall_score, brier_score_loss)
    rows = {}
    for m in ["Persistence", "LogisticRegression", "RandomForest", "XGBoost"]:
        d = A["oof"][A["oof"].model == m]
        if d.empty: continue
        y = d.y_true.values; p = d.y_prob.values; pred = (p >= 0.5).astype(int)
        rows[m] = dict(ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p),
                       Precision=precision_score(y, pred, zero_division=0),
                       Recall=recall_score(y, pred, zero_division=0),
                       F1=f1_score(y, pred, zero_division=0), Brier=brier_score_loss(y, p))
    return pd.DataFrame(rows).T.round(3)

def spatial_errors(A, month=None):
    """Per-cell out-of-sample error type for the selected model (pooled or a single month)."""
    d = A["oof"][A["oof"].model == SELECTED_MODEL].copy()
    if month: d = d[d.month == month]
    d["pred"] = (d.y_prob >= 0.5).astype(int)
    d["etype"] = np.where((d.pred == 1) & (d.y_true == 0), "FP",
                 np.where((d.pred == 0) & (d.y_true == 1), "FN",
                 np.where((d.pred == 1) & (d.y_true == 1), "TP", "TN")))
    return d

def cell_centroid_lonlat(A, cell):
    m = A["model_tbl"][A["model_tbl"].cell == cell]
    if not m.empty and "centroid_lon" in m.columns:
        return float(m.iloc[0]["centroid_lon"]), float(m.iloc[0]["centroid_lat"])
    return None, None

if __name__ == "__main__":
    A = load_artifacts()
    mos = available_months(A)
    print("SELECTED MODEL:", SELECTED_MODEL)
    print("Out-of-sample months available:", mos)
    rt = risk_table(A, mos[-1])
    print(f"\nRisk table {mos[-1]}: {len(rt)} cells; top cell rank1 risk_score={rt.iloc[0].risk_score:.3f} "
          f"pct={rt.iloc[0].percentile}")
    tk = top_k_targeting(A, mos[-1], 0.20)
    print(f"Top-20% targeting {mos[-1]}: PAI={tk['pai']:.2f}, hit_rate={tk['hit_rate']:.1%}, "
          f"captured {tk['crimes_captured']}/{tk['crimes_total']} crimes in {tk['k']} cells")
    top_cell = int(rt.iloc[0].cell)
    le = local_explanation(A, mos[-1], top_cell)
    print(f"\nLocal explanation cell {top_cell} {mos[-1]}: risk_score={le['risk_score']:.3f}")
    print(le["contributions"].head(4).to_string(index=False))
    print("\nGlobal drivers (top 3):")
    print(global_drivers(A).head(3).to_string(index=False))
    print("\nPerformance summary:")
    print(performance_summary(A).to_string())
    print("\nCALIBRATION CAVEAT:\n ", CALIBRATION_CAVEAT[:120], "...")
