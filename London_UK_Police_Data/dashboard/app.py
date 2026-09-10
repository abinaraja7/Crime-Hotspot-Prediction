"""
Explainable-AI Crime Hotspot Dashboard  (Streamlit)
An Explainable AI-driven Spatial Intelligence Framework for Crime Hotspot Prediction.

Run:
    pip install streamlit scikit-learn xgboost pandas numpy matplotlib joblib
    streamlit run app.py

All values are computed live from the real artefacts of Notebooks 3-6 via dashboard_core.
The selected model is Logistic Regression (Notebook 6). Risk is presented as a RELATIVE
RANKING for top-k targeting, never as a precise probability of crime (see calibration note).
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import streamlit as st
import dashboard_core as core

st.set_page_config(page_title="Crime Hotspot — Explainable AI", layout="wide")

@st.cache_resource
def _load():
    return core.load_artifacts()

A = _load()
G = A["grid"]; MONTHS = core.available_months(A)

# ---------- sidebar: filters (Component 6) ----------
st.sidebar.title("Filters")
month = st.sidebar.select_slider("Target month (out-of-sample)", options=MONTHS, value=MONTHS[-1])
coverage = st.sidebar.slider("Patrol coverage (top-k cells)", 0.05, 0.50, 0.20, 0.05)
st.sidebar.info(core.CALIBRATION_CAVEAT)

st.title("Explainable-AI Crime Hotspot Dashboard — City of London")
st.caption(f"Selected model: **{core.SELECTED_MODEL}** (evidence-based, Notebook 6). "
           f"Showing genuine out-of-sample (walk-forward) risk for {month}.")

def grid_map(values, title, cmap="YlOrRd", flagged=None, cat_colors=None):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    grid = np.full((G["nrow"], G["ncol"]), np.nan)
    for cell, v in values.items():
        grid[cell // G["ncol"], cell % G["ncol"]] = v
    if cat_colors is None:
        ax.imshow(grid, origin="lower", cmap=cmap,
                  extent=[G["x0"], G["x0"]+G["ncol"]*G["size"], G["y0"], G["y0"]+G["nrow"]*G["size"]],
                  norm=mcolors.PowerNorm(0.6), aspect="equal")
    if flagged is not None:
        for cell in flagged:
            ci, ri = cell % G["ncol"], cell // G["ncol"]
            ax.add_patch(plt.Rectangle((G["x0"]+ci*G["size"], G["y0"]+ri*G["size"]),
                                       G["size"], G["size"], fill=False, edgecolor="#123", lw=1.6))
    ax.set_title(title, fontsize=10); ax.grid(False)
    ax.set_xlabel("Easting (m)"); ax.set_ylabel("Northing (m)")
    return fig

rt = core.risk_table(A, month)
tk = core.top_k_targeting(A, month, coverage)

# ---------- Component 1: predicted risk map ----------
st.header("1 · Predicted risk map (relative ranking)")
c1, c2 = st.columns([3, 2])
with c1:
    st.pyplot(grid_map(dict(zip(rt.cell, rt.percentile)),
                       f"Relative risk percentile per 250 m cell — {month}",
                       flagged=list(tk["cells"].cell)))
    st.caption("Colour = risk *percentile* (relative ranking). Outlined cells = current top-k patrol set. "
               "Values are not probabilities of crime.")
with c2:
    st.metric("Cells flagged (top-k)", f"{tk['k']} / {tk['n_cells']}  ({tk['coverage']:.0%})")
    st.metric("Crime captured (hit rate)", f"{tk['hit_rate']:.0%}",
              help="Share of that month's actual crime falling in the flagged cells.")
    st.metric("PAI (targeting efficiency)", f"{tk['pai']:.2f}")

# ---------- Component 2: operational top-k targeting ----------
st.header("2 · Operational top-k targeting list")
show = tk["cells"].copy()
show["risk_score (uncalibrated)"] = show["risk_score"].round(3)
show = show.rename(columns={"was_hotspot": "actual_hotspot", "crimes_that_month": "crimes"})
st.dataframe(show[["rank", "cell", "percentile", "risk_score (uncalibrated)", "actual_hotspot", "crimes"]],
             use_container_width=True, hide_index=True)

# ---------- Component 3: local explanation ----------
st.header("3 · Why is this cell flagged? (local explanation)")
sel_cell = st.selectbox("Select a cell (ranked by risk)", options=list(rt.cell),
                        format_func=lambda c: f"cell {c}  (rank {int(rt[rt.cell==c]['rank'].iloc[0])})")
le = core.local_explanation(A, month, sel_cell)
if le:
    cc = le["contributions"].head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.barh(range(len(cc)), cc["contribution"],
            color=["#B03A3A" if v > 0 else "#2E5496" for v in cc["contribution"]])
    ax.set_yticks(range(len(cc))); ax.set_yticklabels(cc["feature"], fontsize=8)
    ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("contribution to log-odds (exact, additive)")
    ax.set_title(f"cell {sel_cell} — {month}   (relative risk score {le['risk_score']:.2f}, uncalibrated)",
                 fontsize=9.5)
    st.pyplot(fig)
    st.caption("Exact additive Logistic-Regression attribution: base + Σ contributions = the model's "
               "log-odds. This is the faithful explanation for a linear model (SHAP not forced).")

# ---------- Component 4: global drivers ----------
st.header("4 · Global drivers of risk across the City")
gd = core.global_drivers(A)
fig, ax = plt.subplots(figsize=(7.5, 3.4))
s = gd.iloc[::-1]
ax.barh(range(len(s)), s["std_coef"], color=["#B03A3A" if v > 0 else "#2E5496" for v in s["std_coef"]])
ax.set_yticks(range(len(s))); ax.set_yticklabels(s["feature"], fontsize=8); ax.axvline(0, color="k", lw=0.6)
ax.set_xlabel("standardised coefficient (log-odds per 1 SD)")
st.pyplot(fig)
st.dataframe(gd.round(3), use_container_width=True, hide_index=True)

# ---------- Component 5: model performance & trust ----------
st.header("5 · Model performance & trust (out-of-sample)")
perf = core.performance_summary(A)
c1, c2 = st.columns([2, 3])
with c1:
    st.dataframe(perf, use_container_width=True)
    st.caption("Real pooled walk-forward metrics. LR leads on ranking (ROC/PR-AUC); trees calibrate "
               "better (Brier). Calibration is a stated limitation — risk is used as ranking, not probability.")
with c2:
    err = core.spatial_errors(A)  # pooled
    dom = (err.groupby(["cell", "etype"]).size().unstack(fill_value=0)
           .reindex(columns=["TN", "TP", "FP", "FN"], fill_value=0))
    def _dom(r): return "correct" if (r["FP"]==0 and r["FN"]==0) else ("FP" if r["FP"]>=r["FN"] else "FN")
    domc = dom.apply(_dom, axis=1)
    cmap = {"correct": 0, "FP": 1, "FN": 2}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    palette = ["#DDDDDD", "#C8912B", "#B03A3A"]
    for cell, lab in domc.items():
        ci, ri = cell % G["ncol"], cell // G["ncol"]
        ax.add_patch(plt.Rectangle((G["x0"]+ci*G["size"], G["y0"]+ri*G["size"]),
                                   G["size"], G["size"], facecolor=palette[cmap[lab]], edgecolor="white", lw=0.5))
    ax.set_xlim(G["x0"], G["x0"]+G["ncol"]*G["size"]); ax.set_ylim(G["y0"], G["y0"]+G["nrow"]*G["size"])
    ax.set_aspect("equal"); ax.grid(False); ax.set_title("Where the model erred (pooled out-of-sample)", fontsize=10)
    st.pyplot(fig)
    st.caption("Grey = always correct, gold = over-prediction (FP), red = missed hotspot (FN).")

st.divider()
st.caption("Built from real artefacts of Notebooks 3–6. No dashboard value is hard-coded. "
           "Risk is a relative ranking for targeting; it is not a calibrated probability of crime.")
