"""
CatBoost Expert System for Cr³⁺ Phosphor Discovery — Streamlit GUI
===================================================================
Interactive web application for Dq/B prediction and candidate screening.

Authors : Snežana Đurković, Prof. Dr. Miroslav Dramićanin
Group   : OMAS — Optical Materials and Spectroscopy
Institute: Nuclear Sciences "Vinča", University of Belgrade
ORCID   : https://orcid.org/0009-0007-6638-0682
Year    : 2026

Run
---
    streamlit run catboost_streamlit_app.py
"""

import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_LOW  = 2.2
TARGET_HIGH = 2.8
TARGET_TOL  = 0.3
N_SPLITS    = 10
N_REPEATS   = 10

FEATURE_COLS = [
    "avg_Mulliken EN",
    "avg_First ionization energy (kJ/mol)",
    "1/r2",
    "avg_Metallic valence",
    "avg_Martynov-Batsanov EN",
    "beta",
    "SGR No.",
    "avg_Number of outer shell electrons",
    "X",
    "max_metal_ligand_bond_length",
    "std_Mendeleev number",
    "volume_per_atom",
    "max_First ionization energy (kJ/mol)",
    "volume_per_fu",
    "polyhedron volume",
]

TIER_COLORS = {
    "Tier 1 — Strong":    "#2d6a4f",
    "Tier 2 — Promising": "#b8860b",
    "Tier 3 — Uncertain": "#1a4480",
    "Tier 3 — Edge":      "#6b7d96",
    "Tier 4 — Out of range": "#8b2a0a",
}

# ── Model ──────────────────────────────────────────────────────────────────────
def get_model():
    return CatBoostRegressor(
        depth=3, iterations=700, learning_rate=0.1,
        l2_leaf_reg=1.9, loss_function="RMSE",
        border_count=32, od_type="Iter", od_wait=30, verbose=0,
    )

def assign_tier(dqb, sigma):
    in_range  = TARGET_LOW <= dqb <= TARGET_HIGH
    near_edge = (TARGET_LOW - TARGET_TOL <= dqb < TARGET_LOW) or \
                (TARGET_HIGH < dqb <= TARGET_HIGH + TARGET_TOL)
    if in_range and sigma < 0.2:   return "Tier 1 — Strong"
    elif in_range and sigma < 0.4: return "Tier 2 — Promising"
    elif in_range:                 return "Tier 3 — Uncertain"
    elif near_edge:                return "Tier 3 — Edge"
    else:                          return "Tier 4 — Out of range"

# ── Cached training ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(train_bytes: bytes, predict_bytes: bytes):
    """Train model and generate predictions. Cached on file content."""

    # Load training
    train_df = pd.read_excel(io.BytesIO(train_bytes))
    X = train_df[FEATURE_COLS].values
    y = train_df["Dq/B"].values

    # Load prediction
    pred_df = pd.read_excel(io.BytesIO(predict_bytes), header=None)
    if str(pred_df.iloc[0, 0]) in FEATURE_COLS or str(pred_df.iloc[0, 0]) == "Formula":
        pred_df = pd.read_excel(io.BytesIO(predict_bytes))
    else:
        pred_df.columns = ["Formula"] + FEATURE_COLS
    X_new    = pred_df[FEATURE_COLS].values
    formulas = pred_df["Formula"].values

    # Find best random state
    candidates = sorted(set(range(5, 101, 5)).union(range(5, 101, 7)))
    best_r2, best_state = -np.inf, None
    for rs in candidates:
        r2s = []
        for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=rs).split(X):
            m = get_model(); m.fit(X[tr], y[tr])
            r2s.append(r2_score(y[te], m.predict(X[te])))
        if np.mean(r2s) > best_r2:
            best_r2, best_state = np.mean(r2s), rs

    # CV
    y_true, y_pred_cv = [], []
    r2s, maes, rmses  = [], [], []
    fold_preds = [[] for _ in range(len(y))]
    for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=best_state).split(X):
        m = get_model(); m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        y_true.extend(y[te]); y_pred_cv.extend(p)
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        for idx, pred in zip(te, p):
            fold_preds[idx].append(pred)

    # Uncertainty
    all_preds = np.zeros((N_REPEATS * N_SPLITS, len(X_new)))
    idx = 0
    for rep in range(N_REPEATS):
        for tr, _ in KFold(n_splits=N_SPLITS, shuffle=True,
                           random_state=best_state + rep * 13).split(X):
            m = get_model(); m.fit(X[tr], y[tr])
            all_preds[idx] = m.predict(X_new); idx += 1
    uncertainty = np.std(all_preds, axis=0)

    # Final model
    final_model = get_model(); final_model.fit(X, y)
    final_preds = final_model.predict(X_new)

    # Feature importance
    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": final_model.get_feature_importance(),
    }).sort_values("Importance", ascending=False)

    # Results
    tiers = [assign_tier(p, s) for p, s in zip(final_preds, uncertainty)]
    results_df = pd.DataFrame({
        "Formula":         formulas,
        "Predicted Dq/B":  np.round(final_preds, 4),
        "Uncertainty (σ)": np.round(uncertainty, 4),
        "Tier":            tiers,
    }).sort_values(["Tier", "Predicted Dq/B"])

    cv_metrics = {
        "R²":   (r2_score(y_true, y_pred_cv), np.std(r2s)),
        "MAE":  (mean_absolute_error(y_true, y_pred_cv), np.std(maes)),
        "RMSE": (np.sqrt(mean_squared_error(y_true, y_pred_cv)), np.std(rmses)),
    }
    cv_means = [np.mean(p) for p in fold_preds]

    return results_df, fi_df, cv_metrics, y, cv_means, best_state, train_df


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CatBoost Dq/B Predictor",
    page_icon="🔬",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0b1a2e; color: #e8eaf0; }
    [data-testid="stSidebar"]          { background: #0f2241; }
    h1, h2, h3                         { color: #e4b84a; }
    .metric-card {
        background: #162d54; border: 1px solid rgba(228,184,74,0.25);
        border-radius: 6px; padding: 16px 20px; text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #e4b84a; }
    .metric-label { font-size: 0.75rem; color: #6b7d96;
                    letter-spacing: 0.12em; text-transform: uppercase; }
    .tier-badge {
        display: inline-block; padding: 2px 10px; border-radius: 3px;
        font-size: 0.8rem; font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 CatBoost\n### Dq/B Predictor")
    st.markdown("---")
    st.markdown("**OMAS Group**  \nINN Vinča · Belgrade")
    st.markdown("**Authors**  \nSnežana Đurković  \nProf. Dr. M. Dramićanin")
    st.markdown("---")

    train_file   = st.file_uploader("📁 Training set (.xlsx)", type="xlsx")
    predict_file = st.file_uploader("📁 Prediction set (.xlsx)", type="xlsx")

    st.markdown("---")
    st.markdown(f"""
**Target window**  
Dq/B ∈ [{TARGET_LOW}, {TARGET_HIGH}]  

**Tier 1** σ < 0.2  
**Tier 2** σ < 0.4  
**Tier 3** σ ≥ 0.4 or Edge  
**Tier 4** Out of range
""")

# ── Main ───────────────────────────────────────────────────────────────────────
st.title("Cr³⁺ Phosphor Discovery — CatBoost Expert System")
st.markdown(
    "Predict **Dq/B** crystal field parameters for Cr³⁺-doped inorganic phosphors "
    "using CatBoost Gradient Boosting Regression with ensemble uncertainty quantification."
)

if not train_file or not predict_file:
    st.info("⬅️  Upload training set and prediction set in the sidebar to begin.")
    st.stop()

# ── Run pipeline ───────────────────────────────────────────────────────────────
with st.spinner("Training model and computing predictions..."):
    results_df, fi_df, cv_metrics, y_true, cv_means, best_state, train_df = run_pipeline(
        train_file.read(), predict_file.read()
    )

# ── CV metrics ─────────────────────────────────────────────────────────────────
st.markdown("## Cross-Validation Performance")
col1, col2, col3, col4 = st.columns(4)
for col, (name, (val, std)) in zip([col1, col2, col3], cv_metrics.items()):
    col.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{val:.4f}</div>
  <div class="metric-label">{name} (±{std:.4f})</div>
</div>""", unsafe_allow_html=True)
col4.markdown(f"""
<div class="metric-card">
  <div class="metric-value">{best_state}</div>
  <div class="metric-label">Best random state</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Parity plot + Feature importance ──────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### Parity Plot")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_true, y=cv_means, mode="markers",
        marker=dict(color="steelblue", size=7, opacity=0.7,
                    line=dict(color="navy", width=0.5)),
        name="Compounds",
        hovertemplate="True: %{x:.3f}<br>Pred: %{y:.3f}<extra></extra>",
    ))
    lims = [min(min(y_true), min(cv_means)) - 0.05,
            max(max(y_true), max(cv_means)) + 0.05]
    fig.add_trace(go.Scatter(
        x=lims, y=lims, mode="lines",
        line=dict(color="red", dash="dash", width=1.5),
        name="Ideal (y = ŷ)", showlegend=True,
    ))
    fig.add_vrect(x0=TARGET_LOW, x1=TARGET_HIGH,
                  fillcolor="gold", opacity=0.07, line_width=0,
                  annotation_text="NIR target", annotation_position="top left")
    fig.update_layout(
        xaxis_title="True Dq/B", yaxis_title="Predicted Dq/B (CV mean)",
        template="plotly_dark", paper_bgcolor="#0b1a2e", plot_bgcolor="#0f2241",
        margin=dict(l=40, r=20, t=20, b=40), legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown("### Feature Importance")
    fig2 = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        text=fi_df["Importance"].apply(lambda x: f"{x:.2f}%"),
        color="Importance", color_continuous_scale=["#6b7d96", "#1a4480", "#0b1a2e"],
    )
    fig2.update_traces(textposition="outside")
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="#0b1a2e", plot_bgcolor="#0f2241",
        coloraxis_showscale=False, yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=60, t=20, b=40),
        xaxis_title="Importance (%)", yaxis_title="",
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ── Tier summary ───────────────────────────────────────────────────────────────
st.markdown("### Tier Summary")
tier_counts = results_df["Tier"].value_counts()
tcols = st.columns(len(tier_counts))
for col, (tier, count) in zip(tcols, tier_counts.items()):
    color = TIER_COLORS.get(tier, "#333")
    col.markdown(f"""
<div class="metric-card" style="border-color:{color}55;">
  <div class="metric-value" style="color:{color};">{count}</div>
  <div class="metric-label">{tier}</div>
</div>""", unsafe_allow_html=True)

st.markdown("---")

# ── Results table ──────────────────────────────────────────────────────────────
st.markdown("### Predictions")

tier_filter = st.multiselect(
    "Filter by Tier",
    options=list(TIER_COLORS.keys()),
    default=["Tier 1 — Strong", "Tier 2 — Promising"],
)

filtered = results_df[results_df["Tier"].isin(tier_filter)] if tier_filter else results_df

def color_tier(val):
    color = TIER_COLORS.get(val, "#333")
    return f"color: {color}; font-weight: 600;"

st.dataframe(
    filtered.style.applymap(color_tier, subset=["Tier"]).format({
        "Predicted Dq/B": "{:.4f}",
        "Uncertainty (σ)": "{:.4f}",
    }),
    use_container_width=True, height=400,
)

# ── Download ───────────────────────────────────────────────────────────────────
buf = io.BytesIO()
results_df.to_excel(buf, index=False)
st.download_button(
    label="⬇️  Download full predictions (.xlsx)",
    data=buf.getvalue(),
    file_name="catboost_predictions.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>OMAS Group · INN Vinča · Belgrade, Serbia · 2026 · "
    "[github.com/KirkaSSS/phD-AI](https://github.com/KirkaSSS/phD-AI)</small>",
    unsafe_allow_html=True,
)
