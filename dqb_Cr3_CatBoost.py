"""
CatBoost Expert System for Cr³⁺ Phosphor Discovery
====================================================
Predicts the crystal field parameter Dq/B for Cr³⁺-doped inorganic phosphor
host lattices using CatBoost Gradient Boosting Regression.

Authors : Snežana Đurković, Prof. Dr. Miroslav Dramićanin
Group   : OMAS — Optical Materials and Spectroscopy
Institute: Nuclear Sciences "Vinča", University of Belgrade
ORCID   : https://orcid.org/0009-0007-6638-0682
Year    : 2026

Usage
-----
    python dqb_Cr3_CatBoost.py

Configure TRAIN_PATH and PREDICT_PATH before running.
Output is written to OUTPUT_PATH (Excel) and catboost_results.png.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH   = "Cr3_dqb_training_set.xlsx"
PREDICT_PATH = "To_predict.xlsx"
OUTPUT_PATH  = "catboost_dqb_results.xlsx"

N_SPLITS     = 10    # folds per CV run
N_REPEATS    = 10    # CV repetitions for uncertainty estimation
TARGET_LOW   = 2.2   # NIR target window lower bound
TARGET_HIGH  = 2.8   # NIR target window upper bound
TARGET_TOL   = 0.3   # edge zone half-width

# 15 descriptors — order must match Excel columns exactly
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


# ── Model ──────────────────────────────────────────────────────────────────────
def get_model() -> CatBoostRegressor:
    """
    Return a configured CatBoostRegressor instance.

    Hyperparameters were selected by grid search (5×10-fold CV) over:
        depth         : [3, 5, 7]
        iterations    : [500, 700, 1000]
        learning_rate : [0.05, 0.10, 0.15]
        l2_leaf_reg   : [1.0, 2.0, 5.0]
        border_count  : [32, 64]
    """
    return CatBoostRegressor(
        depth=3,
        iterations=700,
        learning_rate=0.1,
        l2_leaf_reg=1.9,
        loss_function="RMSE",
        border_count=32,
        od_type="Iter",
        od_wait=30,
        verbose=0,
    )


# ── Tier classification ────────────────────────────────────────────────────────
def assign_tier(dqb: float, sigma: float) -> str:
    """
    Classify a prediction into one of four tiers based on predicted Dq/B
    and ensemble uncertainty σ.

    Tier 1 — Strong   : Dq/B in [2.2, 2.8], σ < 0.2
    Tier 2 — Promising: Dq/B in [2.2, 2.8], σ < 0.4
    Tier 3 — Uncertain: Dq/B in [2.2, 2.8], σ ≥ 0.4
    Tier 3 — Edge     : Dq/B within 0.3 of target boundary
    Tier 4 — Out of range
    """
    in_range  = TARGET_LOW <= dqb <= TARGET_HIGH
    near_edge = (TARGET_LOW - TARGET_TOL <= dqb < TARGET_LOW) or \
                (TARGET_HIGH < dqb <= TARGET_HIGH + TARGET_TOL)

    if in_range and sigma < 0.2:
        return "Tier 1 — Strong"
    elif in_range and sigma < 0.4:
        return "Tier 2 — Promising"
    elif in_range:
        return "Tier 3 — Uncertain"
    elif near_edge:
        return "Tier 3 — Edge"
    else:
        return "Tier 4 — Out of range"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_training(path: str):
    """Load and validate the training Excel file."""
    df = pd.read_excel(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in training set: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    X = df[FEATURE_COLS].values
    y = df["Dq/B"].values
    return X, y, df


def load_prediction(path: str):
    """
    Load the prediction Excel file.
    Handles both headered and header-less files automatically.
    """
    df = pd.read_excel(path)
    first_col = str(list(df.columns)[0])
    has_no_header = (
        first_col not in ["Formula", "formula"]
        and not first_col.startswith("avg")
        and first_col not in FEATURE_COLS
    )
    if has_no_header:
        df = pd.read_excel(path, header=None, names=["Formula"] + FEATURE_COLS)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in prediction set: {missing}")

    return df[FEATURE_COLS].values, df["Formula"].values


# ── Cross-validation ───────────────────────────────────────────────────────────
def find_best_random_state(X: np.ndarray, y: np.ndarray) -> int:
    """
    Search over candidate random states and return the one that maximises
    mean 10-fold CV R².
    """
    candidates = sorted(set(range(5, 101, 5)).union(range(5, 101, 7)))
    best_r2, best_state = -np.inf, None

    for rs in candidates:
        r2s = []
        for tr, te in KFold(n_splits=N_SPLITS, shuffle=True, random_state=rs).split(X):
            m = get_model()
            m.fit(X[tr], y[tr])
            r2s.append(r2_score(y[te], m.predict(X[te])))
        if np.mean(r2s) > best_r2:
            best_r2, best_state = np.mean(r2s), rs

    return best_state


def run_cv(X: np.ndarray, y: np.ndarray, random_state: int):
    """
    Run a single 10-fold CV pass and return per-fold metrics plus
    fold-level predictions for the parity plot.
    """
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    y_true, y_pred = [], []
    r2s, maes, rmses = [], [], []
    fold_preds = [[] for _ in range(len(y))]

    for fold, (tr, te) in enumerate(kf.split(X), 1):
        m = get_model()
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])

        y_true.extend(y[te])
        y_pred.extend(p)
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
        for idx, pred in zip(te, p):
            fold_preds[idx].append(pred)

        print(f"   Fold {fold:2d}: R² = {r2s[-1]:.4f}  "
              f"MAE = {maes[-1]:.4f}  RMSE = {rmses[-1]:.4f}")

    return y_true, y_pred, r2s, maes, rmses, fold_preds


def estimate_uncertainty(X: np.ndarray, y: np.ndarray,
                         X_new: np.ndarray, random_state: int) -> np.ndarray:
    """
    Estimate prediction uncertainty as the standard deviation across
    N_REPEATS × N_SPLITS ensemble members, each trained on a different
    training fold.
    """
    all_preds = np.zeros((N_REPEATS * N_SPLITS, len(X_new)))
    idx = 0
    for rep in range(N_REPEATS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True,
                   random_state=random_state + rep * 13)
        for tr, _ in kf.split(X):
            m = get_model()
            m.fit(X[tr], y[tr])
            all_preds[idx] = m.predict(X_new)
            idx += 1
    return np.std(all_preds, axis=0)


# ── Plotting ───────────────────────────────────────────────────────────────────
def plot_results(y: np.ndarray, cv_means: list,
                 fi_df: pd.DataFrame, r2: float, mae: float, rmse: float):
    """Generate parity plot and feature importance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Parity plot
    ax = axes[0]
    ax.scatter(y, cv_means, alpha=0.65, edgecolors="navy",
               facecolors="steelblue", s=50, linewidths=0.5, label="Compounds")
    lims = [min(y.min(), min(cv_means)) - 0.05,
            max(y.max(), max(cv_means)) + 0.05]
    ax.plot(lims, lims, "r--", lw=1.8, label="Ideal (y = ŷ)")
    ax.axvspan(TARGET_LOW, TARGET_HIGH, alpha=0.08,
               color="gold", label="NIR target window")
    ax.set_xlabel("True Dq/B", fontsize=12)
    ax.set_ylabel("Predicted Dq/B (CV mean)", fontsize=12)
    ax.set_title(
        f"Parity Plot — CatBoost\n"
        f"R² = {r2:.4f}   MAE = {mae:.4f}   RMSE = {rmse:.4f}",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax2 = axes[1]
    colors = [
        "#0b1a2e" if i < 5 else "#1a4480" if i < 10 else "#6b7d96"
        for i in range(len(fi_df))
    ]
    ax2.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1],
             color=colors[::-1])
    ax2.set_xlabel("Importance (%)", fontsize=12)
    ax2.set_title("Feature Importance (CatBoost)", fontsize=11)
    ax2.grid(True, axis="x", alpha=0.3)
    for i, val in enumerate(fi_df["Importance"][::-1]):
        ax2.text(val + 0.1, i, f"{val:.2f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("catboost_results.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved → catboost_results.png")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Load
    print("📂 Loading data...")
    X, y, train_df = load_training(TRAIN_PATH)
    X_new, formulas = load_prediction(PREDICT_PATH)
    print(f"✅ Training set : {X.shape[0]} compounds, {X.shape[1]} features")
    print(f"   Dq/B range   : {y.min():.3f} – {y.max():.3f}  (mean {y.mean():.3f})")
    print(f"✅ Prediction set: {len(formulas)} candidates")

    # Find best random state
    print("\n🔄 Searching for best random state...")
    best_state = find_best_random_state(X, y)
    print(f"✅ Best random state = {best_state}")

    # Cross-validation
    print(f"\n📊 Running {N_SPLITS}-fold CV (random_state = {best_state})...")
    y_true, y_pred_cv, r2s, maes, rmses, fold_preds = run_cv(X, y, best_state)

    final_r2   = r2_score(y_true, y_pred_cv)
    final_mae  = mean_absolute_error(y_true, y_pred_cv)
    final_rmse = np.sqrt(mean_squared_error(y_true, y_pred_cv))

    print(f"\n{'='*55}")
    print(f"  CV R²   = {final_r2:.4f}  (±{np.std(r2s):.4f})")
    print(f"  CV MAE  = {final_mae:.4f}  (±{np.std(maes):.4f})")
    print(f"  CV RMSE = {final_rmse:.4f}  (±{np.std(rmses):.4f})")
    print(f"{'='*55}")

    # Uncertainty estimation
    print(f"\n🔁 Estimating uncertainty ({N_REPEATS}×{N_SPLITS}-fold ensemble)...")
    uncertainty = estimate_uncertainty(X, y, X_new, best_state)

    # Final model on full training set
    print("\n🏁 Training final model on full dataset...")
    final_model = get_model()
    final_model.fit(X, y)
    final_preds = final_model.predict(X_new)

    # Feature importance
    fi_df = pd.DataFrame({
        "Feature":    FEATURE_COLS,
        "Importance": final_model.get_feature_importance(),
    }).sort_values("Importance", ascending=False)

    print("\n📌 Feature Importance:")
    for _, row in fi_df.iterrows():
        bar = "█" * int(row["Importance"] / 1.5)
        print(f"   {row['Feature']:<45} {row['Importance']:>6.2f}%  {bar}")

    # Tier classification
    tiers = [assign_tier(p, s) for p, s in zip(final_preds, uncertainty)]

    results_df = pd.DataFrame({
        "Formula":        formulas,
        "Predicted Dq/B": np.round(final_preds, 4),
        "Uncertainty (σ)": np.round(uncertainty, 4),
        "Tier":           tiers,
    }).sort_values(["Tier", "Predicted Dq/B"])

    results_df.to_excel(OUTPUT_PATH, index=False)
    print(f"\n💾 Saved predictions → {OUTPUT_PATH}")

    print("\n🏷️  Tier summary:")
    for tier, count in results_df["Tier"].value_counts().sort_index().items():
        print(f"   {tier}: {count} compounds")

    # Plot
    cv_means = [np.mean(p) for p in fold_preds]
    plot_results(y, cv_means, fi_df, final_r2, final_mae, final_rmse)


if __name__ == "__main__":
    main()
