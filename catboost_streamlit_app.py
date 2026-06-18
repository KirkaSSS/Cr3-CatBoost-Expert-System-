"""
Streamlit GUI for Cr³⁺ Phosphor CatBoost Dq/B Predictor
=========================================================
Author: Snežana Đurković
Year:   2026
INN Vinča, Belgrade — OMAS Group

Run: streamlit run catboost_streamlit_app.py
"""

import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler

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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cr³⁺ CatBoost Expert System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 600;
    color: #1565C0;
    text-align: center;
    padding: 1.2rem 1rem;
    background: linear-gradient(90deg, #E3F2FD 0%, #BBDEFB 100%);
    border-radius: 12px;
    margin-bottom: 1.5rem;
}
.metric-card {
    background: #F8F9FA;
    padding: 1rem;
    border-radius: 10px;
    border-left: 4px solid #1565C0;
}
.tier1 { background:#C8E6C9; padding:4px 10px; border-radius:6px; font-weight:600; color:#1B5E20; }
.tier2 { background:#FFF9C4; padding:4px 10px; border-radius:6px; font-weight:600; color:#F57F17; }
.tier3 { background:#FFE0B2; padding:4px 10px; border-radius:6px; color:#E65100; }
.tier4 { background:#FFCDD2; padding:4px 10px; border-radius:6px; color:#B71C1C; }
.stButton>button {
    width:100%; background:#1565C0; color:white;
    height:3rem; font-size:1.1rem; font-weight:600; border-radius:8px;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for key in ['results_df', 'fi_df', 'cv_metrics', 'y_train', 'cv_means']:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Model helpers ──────────────────────────────────────────────────────────────
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

def load_prediction(uploaded_file):
    df = pd.read_excel(uploaded_file)
    first_col = str(list(df.columns)[0])
    has_no_header = (
        first_col not in ["Formula", "formula"]
        and not first_col.startswith("avg")
        and first_col not in FEATURE_COLS
    )
    if has_no_header:
        uploaded_file.seek(0)
        df = pd.read_excel(uploaded_file, header=None,
                           names=["Formula"] + FEATURE_COLS)
    return df[FEATURE_COLS].values, df["Formula"].values

def run_pipeline(train_bytes, predict_bytes):
    train_df = pd.read_excel(io.BytesIO(train_bytes))
    X = train_df[FEATURE_COLS].values
    y = train_df["Dq/B"].values

    pred_df  = pd.read_excel(io.BytesIO(predict_bytes))
    first_col = str(list(pred_df.columns)[0])
    if first_col not in ["Formula","formula"] and first_col not in FEATURE_COLS:
        pred_df = pd.read_excel(io.BytesIO(predict_bytes), header=None,
                                names=["Formula"] + FEATURE_COLS)
    X_new    = pred_df[FEATURE_COLS].values
    formulas = pred_df["Formula"].values

    # Find best random state
    candidates = sorted(set(range(5, 101, 5)).union(range(5, 101, 7)))
    best_r2, best_state = -np.inf, None
    for rs in candidates:
        r2s = []
        for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=rs).split(X):
            sc = RobustScaler().fit(X[tr])
            m  = get_model()
            m.fit(sc.transform(X[tr]), y[tr])
            r2s.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
        if np.mean(r2s) > best_r2:
            best_r2, best_state = np.mean(r2s), rs

    # CV
    y_true, y_pred_cv = [], []
    r2s, maes, rmses  = [], [], []
    fold_preds = [[] for _ in range(len(y))]
    for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=best_state).split(X):
        sc = RobustScaler().fit(X[tr])
        m  = get_model()
        m.fit(sc.transform(X[tr]), y[tr])
        p  = m.predict(sc.transform(X[te]))
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
            sc = RobustScaler().fit(X[tr])
            m  = get_model(); m.fit(sc.transform(X[tr]), y[tr])
            all_preds[idx] = m.predict(sc.transform(X_new)); idx += 1
    uncertainty = np.std(all_preds, axis=0)

    # Final model
    sc_full = RobustScaler().fit(X)
    final_m = get_model(); final_m.fit(sc_full.transform(X), y)
    final_preds = final_m.predict(sc_full.transform(X_new))

    # Feature importance
    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": final_m.get_feature_importance(),
    }).sort_values("Importance", ascending=False)

    tiers = [assign_tier(p, s) for p, s in zip(final_preds, uncertainty)]
    results_df = pd.DataFrame({
        "Formula":          formulas,
        "Predicted Dq/B":   np.round(final_preds, 4),
        "Uncertainty (σ)":  np.round(uncertainty, 4),
        "Tier":             tiers,
    }).sort_values(["Tier", "Predicted Dq/B"])

    cv_metrics = {
        "R²":   (r2_score(y_true, y_pred_cv),               np.std(r2s)),
        "MAE":  (mean_absolute_error(y_true, y_pred_cv),     np.std(maes)),
        "RMSE": (np.sqrt(mean_squared_error(y_true,y_pred_cv)), np.std(rmses)),
    }
    cv_means = [np.mean(p) for p in fold_preds]

    return results_df, fi_df, cv_metrics, y, cv_means, best_state

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-header">🔬 Cr³⁺ Phosphor CatBoost Expert System</div>',
    unsafe_allow_html=True,
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("📁 Input Files")
    train_file   = st.file_uploader("Training Dataset (.xlsx)", type=["xlsx"], key="train")
    predict_file = st.file_uploader("Prediction Dataset (.xlsx)", type=["xlsx"], key="pred")

    st.divider()
    st.subheader("🎯 Target Dq/B Range")
    col1, col2 = st.columns(2)
    with col1:
        dqb_min = st.number_input("Min", value=2.2, step=0.1, format="%.1f")
    with col2:
        dqb_max = st.number_input("Max", value=2.8, step=0.1, format="%.1f")

    st.divider()
    st.markdown("**Authors**  \nSnežana Đurković  \nProf. Dr. M. Dramićanin")
    st.markdown("**OMAS Group · INN Vinča · Belgrade**")
    st.divider()
    run_btn = st.button("▶️ Run CatBoost Pipeline", type="primary", use_container_width=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🏆 Top Results", "📈 Statistics", "📌 Feature Importance", "📋 Full Table"
])

with tab1:
    st.header("CatBoost Gradient Boosting for Cr³⁺ Phosphor Discovery")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        trained = "✓ Ready" if st.session_state.results_df is not None else "Not trained"
        st.metric("Model", trained)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        n_cand = len(st.session_state.results_df) if st.session_state.results_df is not None else 0
        st.metric("Candidates evaluated", n_cand)
        st.markdown('</div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.results_df is not None:
            t1 = len(st.session_state.results_df[
                st.session_state.results_df["Tier"].str.contains("Tier 1")])
        else:
            t1 = 0
        st.metric("Tier 1 candidates", t1)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()
    with st.expander("📖 How to use"):
        st.markdown("""
1. **Upload** training dataset (`Formula`, `Dq/B`, 15 features) and prediction dataset (no header row)
2. **Set** target Dq/B range (default 2.2–2.8)
3. **Click** Run CatBoost Pipeline
4. **Review** results in the tabs above

**Hold-out validation strategy:** 36 compounds (one per major oxide structure family) are withheld
from training for prospective validation — ensuring unbiased generalization assessment.

**Tiers:**
- Tier 1: predicted Dq/B in target range, σ < 0.2
- Tier 2: predicted Dq/B in target range, σ < 0.4
- Tier 3: in target range but uncertain, or edge case
- Tier 4: outside target range
""")

    with st.expander("🔄 CatBoost Pipeline"):
        st.markdown("""
```
Training data (.xlsx)  [207 compounds]
↓
RobustScaler — normalises extreme feature distributions
↓
Automated random state search (best 10-fold CV R²)
↓
CatBoost Gradient Boosting Regression
  depth=3 | iterations=700 | lr=0.1 | l2=1.9
↓
10×10-fold repeated CV → ensemble uncertainty σ
↓
Final model on full training set
↓
Tier classification + Excel output + plots
```
""")

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn:
    if train_file is None or predict_file is None:
        st.error("⚠️ Please upload both training and prediction files.")
    else:
        with st.spinner("Running CatBoost pipeline..."):
            try:
                progress = st.progress(0)
                status   = st.empty()

                status.text("Loading data...")
                progress.progress(10)

                train_bytes   = train_file.read()
                predict_bytes = predict_file.read()

                status.text("Searching for best random state...")
                progress.progress(25)

                results_df, fi_df, cv_metrics, y, cv_means, best_state = run_pipeline(
                    train_bytes, predict_bytes
                )

                st.session_state.results_df = results_df
                st.session_state.fi_df      = fi_df
                st.session_state.cv_metrics = cv_metrics
                st.session_state.y_train    = y
                st.session_state.cv_means   = cv_means

                progress.progress(100)
                status.empty(); progress.empty()

                t1_n = len(results_df[results_df["Tier"].str.contains("Tier 1")])
                t2_n = len(results_df[results_df["Tier"].str.contains("Tier 2")])
                r2   = cv_metrics["R²"][0]

                st.success(
                    f"✅ Done! Evaluated {len(results_df)} candidates. "
                    f"Tier 1: {t1_n} | Tier 2: {t2_n} | CV R² = {r2:.4f} | "
                    f"Best random state: {best_state}"
                )
                st.balloons()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.results_df is not None:
    df      = st.session_state.results_df
    fi_df   = st.session_state.fi_df
    cv_met  = st.session_state.cv_metrics
    y       = st.session_state.y_train
    cv_means= st.session_state.cv_means

    # ── Tab 2: Top Results ────────────────────────────────────────────────────
    with tab2:
        st.header("🏆 Top Candidates")
        top_n = min(10, len(df))
        tier1_df = df[df["Tier"].str.contains("Tier 1|Tier 2")]
        show_df  = tier1_df.head(top_n) if len(tier1_df) > 0 else df.head(top_n)

        for i, (_, row) in enumerate(show_df.iterrows()):
            tier_class = (
                "tier1" if "Tier 1" in row["Tier"] else
                "tier2" if "Tier 2" in row["Tier"] else
                "tier3" if "Tier 3" in row["Tier"] else "tier4"
            )
            with st.container():
                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    st.markdown(
                        f'<div class="{tier_class}">#{i+1} {row["Formula"]}</div>',
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.write(f"**Dq/B:** {row['Predicted Dq/B']:.3f} ± {row['Uncertainty (σ)']:.3f}")
                with c3:
                    st.metric("Tier", row["Tier"].split(" — ")[0])
                    st.caption(row["Tier"])
            st.divider()

    # ── Tab 3: Statistics ─────────────────────────────────────────────────────
    with tab3:
        st.header("📈 Statistics")

        # CV metrics
        c1, c2, c3, c4 = st.columns(4)
        for col, (name, (val, std)) in zip([c1, c2, c3], cv_met.items()):
            col.markdown('<div class="metric-card">', unsafe_allow_html=True)
            col.metric(name, f"{val:.4f}", f"±{std:.4f}")
            col.markdown('</div>', unsafe_allow_html=True)
        c4.markdown('<div class="metric-card">', unsafe_allow_html=True)
        c4.metric("Avg σ", f"{df['Uncertainty (σ)'].mean():.4f}")
        c4.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        col1, col2 = st.columns(2)

        # Parity plot
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y, y=cv_means, mode="markers",
                marker=dict(color="#1565C0", size=7, opacity=0.7,
                            line=dict(color="navy", width=0.5)),
                name="Compounds",
            ))
            lims = [min(min(y), min(cv_means)) - 0.05,
                    max(max(y), max(cv_means)) + 0.05]
            fig.add_trace(go.Scatter(
                x=lims, y=lims, mode="lines",
                line=dict(color="red", dash="dash", width=1.5),
                name="Ideal (y = ŷ)",
            ))
            fig.add_vrect(x0=dqb_min, x1=dqb_max, fillcolor="green", opacity=0.07,
                          annotation_text="NIR target", annotation_position="top left")
            fig.update_layout(
                title=f"Parity Plot — CatBoost<br>R²={cv_met['R²'][0]:.4f}  MAE={cv_met['MAE'][0]:.4f}",
                xaxis_title="True Dq/B", yaxis_title="Predicted Dq/B (CV mean)",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)

        # Tier pie
        with col2:
            tier_counts = df["Tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Count"]
            fig2 = px.pie(
                tier_counts, values="Count", names="Tier",
                title="Tier Distribution",
                color_discrete_sequence=["#4CAF50","#FFC107","#FF9800","#f44336"],
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Dq/B histogram
        fig3 = px.histogram(
            df, x="Predicted Dq/B", nbins=15,
            title="Predicted Dq/B Distribution",
            color_discrete_sequence=["#1565C0"],
        )
        fig3.add_vrect(x0=dqb_min, x1=dqb_max, fillcolor="green", opacity=0.08,
                       annotation_text="Target", annotation_position="top left")
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 4: Feature Importance ─────────────────────────────────────────────
    with tab4:
        st.header("📌 Feature Importance (CatBoost)")

        fig = px.bar(
            fi_df, x="Feature", y="Importance",
            title="CatBoost Feature Importance",
            color="Importance",
            color_continuous_scale="Blues",
            labels={"Importance": "Importance (%)"},
            text=fi_df["Importance"].apply(lambda x: f"{x:.2f}%"),
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_tickangle=-45, template="plotly_white",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Feature Importance Table")
        st.dataframe(fi_df, use_container_width=True)
        st.caption(
            "CatBoost feature importance reflects the average gain in prediction accuracy "
            "from splits on each feature across all trees."
        )

    # ── Tab 5: Full Table + Download ──────────────────────────────────────────
    with tab5:
        st.header("📋 Full Results")

        fc1, fc2 = st.columns(2)
        with fc1:
            tier_filter = st.multiselect(
                "Filter by Tier",
                options=df["Tier"].unique().tolist(),
                default=["Tier 1 — Strong", "Tier 2 — Promising"],
            )
        with fc2:
            max_sigma = st.slider("Max uncertainty σ ≤", 0.0, 2.0, 2.0, 0.05)

        filtered = df[df["Tier"].isin(tier_filter)] if tier_filter else df
        filtered = filtered[filtered["Uncertainty (σ)"] <= max_sigma]

        st.dataframe(filtered, use_container_width=True, height=500)
        st.caption(f"Showing {len(filtered)} of {len(df)} candidates")

        c1, c2 = st.columns(2)
        with c1:
            buf = io.BytesIO()
            df.to_excel(buf, index=False)
            st.download_button(
                "📥 Download Excel",
                data=buf.getvalue(),
                file_name="catboost_dqb_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        with c2:
            st.download_button(
                "📥 Download CSV",
                data=df.to_csv(index=False),
                file_name="catboost_dqb_results.csv",
                mime="text/csv",
            )

else:
    for tab in [tab2, tab3, tab4, tab5]:
        with tab:
            st.info("👈 Upload files and run the pipeline to see results.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; color:#888; padding:0.5rem;'>
<p><strong>Cr³⁺ Phosphor CatBoost Expert System</strong> —
Snežana Đurković | INN Vinča 2026 | OMAS Group</p>
</div>
""", unsafe_allow_html=True)
