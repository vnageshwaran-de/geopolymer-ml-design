#!/usr/bin/env python3
"""
One-part geopolymer compressive-strength ML pipeline
====================================================
Reproduces the Section 4 machine-learning demonstration of the manuscript
"Machine Learning-Assisted Design of One-Part Geopolymer Composites".

Data
----
Genuine one-part geopolymer dataset: 80 fly-ash/GGBS one-part geopolymer paste
mixtures (solid sodium-metasilicate activator, just-add-water) compiled from
twelve open-literature sources by:
  Faridmehr, Sahraei, Nehdi & Valerievich, "Optimization of Fly Ash-Slag
  One-Part Geopolymers with Improved Properties", Materials 2023, 16, 2348,
  doi:10.3390/ma16062348  (dataset = Table 1 of that open-access article).

Expected CSV columns (see onepart_geopolymer_dataset.csv):
  SourceRef, MixNo, FlyAsh_pct, GGBS_pct, Na2O_pct, Water_Binder,
  Curing_temp_C, Strength_MPa

What it does
------------
1. Loads the dataset and reports QC (ranges, missing, source-study grouping).
2. Trains four regressors: Linear, SVR (RBF), Random Forest, XGBoost.
3. Evaluates at two levels:
     - random 80/20 split + 5-fold CV        -> in-sample interpolation
     - leave-one-source-out CV (LOSO)          -> transfer to an unseen study
4. Computes TreeSHAP feature importance for the XGBoost model.
5. Writes: ml_model_comparison.csv, shap_importance.csv, and
           ml_results.png (parity + SHAP panels).

Usage
-----
    python geopolymer_ml_pipeline.py --data onepart_geopolymer_dataset.csv
    python geopolymer_ml_pipeline.py --data onepart_geopolymer_dataset.csv --outdir out

Dependencies: pandas, numpy, scikit-learn, xgboost, shap, matplotlib
"""
from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
import xgboost as xgb

RANDOM_STATE = 42
# Raw descriptors present in the CSV.
RAW_FEATURES = ["FlyAsh_pct", "GGBS_pct", "Na2O_pct", "Water_Binder", "Curing_temp_C"]
# Modelling features: the collinear fly-ash/GGBS pair (Pearson r = -0.99) is
# collapsed to a single slag fraction so SHAP attribution is stable.
FEATURES = ["SlagFraction", "Na2O_pct", "Water_Binder", "Curing_temp_C"]
TARGET = "Strength_MPa"
GROUP = "SourceRef"


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RAW_FEATURES + [TARGET, GROUP] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    # Feature engineering: slag fraction = GGBS / (fly ash + GGBS).
    denom = (df["FlyAsh_pct"] + df["GGBS_pct"]).replace(0, np.nan)
    df["SlagFraction"] = (df["GGBS_pct"] / denom).fillna(0.0)
    print(f"Loaded {len(df)} mixtures from {os.path.basename(path)}")
    print(f"  source studies: {df[GROUP].nunique()}")
    print(f"  strength range: {df[TARGET].min():.1f}-{df[TARGET].max():.1f} MPa "
          f"(mean {df[TARGET].mean():.1f})")
    print(f"  missing values: {int(df[FEATURES + [TARGET]].isna().sum().sum())}")
    return df


def build_models():
    return {
        "Linear regression": (LinearRegression(), True),
        "Support vector regression": (SVR(C=100, gamma="scale", epsilon=0.5), True),
        "Random forest": (RandomForestRegressor(n_estimators=500, random_state=RANDOM_STATE), False),
        "XGBoost": (xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                                     subsample=0.85, random_state=RANDOM_STATE), False),
    }


def pipe(est, scale):
    return Pipeline([("sc", StandardScaler()), ("m", est)]) if scale else est


def evaluate(df):
    X = df[FEATURES].copy()
    y = df[TARGET].values
    groups = df[GROUP].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    kf = KFold(5, shuffle=True, random_state=RANDOM_STATE)
    logo = LeaveOneGroupOut()

    results, fitted, preds = {}, {}, {}
    for name, (est, scale) in build_models().items():
        p = pipe(clone(est), scale)
        p.fit(Xtr, ytr)
        pr = p.predict(Xte)
        cv = cross_val_score(pipe(clone(est), scale), X, y, cv=kf, scoring="r2")
        # leave-one-source-out: predict each held-out study, score once on pooled predictions
        yhat = np.zeros_like(y, dtype=float)
        for trn, tst in logo.split(X, y, groups):
            gp = pipe(clone(est), scale)
            gp.fit(X.iloc[trn], y[trn])
            yhat[tst] = gp.predict(X.iloc[tst])
        results[name] = dict(
            Test_R2=r2_score(yte, pr),
            Test_RMSE=np.sqrt(mean_squared_error(yte, pr)),
            Test_MAE=mean_absolute_error(yte, pr),
            CV_R2=cv.mean(),
            LOSO_R2=r2_score(y, yhat),
            LOSO_RMSE=np.sqrt(mean_squared_error(y, yhat)),
        )
        fitted[name], preds[name] = p, pr
    R = pd.DataFrame(results).T.sort_values("LOSO_R2", ascending=False)
    return R, fitted, preds, (Xte, yte, y)


def compute_shap(xgb_pipeline, Xte):
    import shap
    sv = shap.TreeExplainer(xgb_pipeline).shap_values(Xte)
    return pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)


def make_figure(preds, yte, y, shap_imp, results, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"XGBoost": "#1f77b4", "Random forest": "#e6820e"}
    mk = {"XGBoost": "o", "Random forest": "s"}
    for name in ["XGBoost", "Random forest"]:
        ax_a.scatter(yte, preds[name], s=42, alpha=0.7, color=colors[name], marker=mk[name],
                     edgecolor="none", label=f"{name} (R\u00b2={results[name]['Test_R2']:.2f})")
    lim = [y.min() - 5, y.max() + 5]
    ax_a.plot(lim, lim, "--", color="0.4", lw=1.2, zorder=0)
    ax_a.set_xlim(lim); ax_a.set_ylim(lim)
    ax_a.set_xlabel("Observed 28-d compressive strength (MPa)")
    ax_a.set_ylabel("Predicted strength (MPa)")
    ax_a.set_title("Predicted vs. observed on held-out mixes", fontsize=11, loc="left")
    ax_a.legend(frameon=False, fontsize=9, loc="upper left")

    lab = {"SlagFraction": "Slag fraction GGBS/(FA+GGBS)", "FlyAsh_pct": "Fly ash content",
           "Na2O_pct": "Na\u2082O dosage (activator)",
           "Water_Binder": "Water/binder ratio", "GGBS_pct": "GGBS content",
           "Curing_temp_C": "Curing temperature"}
    labels = [lab[c] for c in shap_imp.index]
    chem = {"Fly ash content", "Na\u2082O dosage (activator)", "GGBS content"}
    bc = ["#2ca02c" if l in chem else "#8a8a8a" for l in labels]
    yp = np.arange(len(shap_imp))[::-1]
    ax_b.barh(yp, shap_imp.values, color=bc)
    for yv, v in zip(yp, shap_imp.values):
        ax_b.text(v + 0.3, yv, f"{v:.1f}", va="center", fontsize=8.5)
    ax_b.set_yticks(yp); ax_b.set_yticklabels(labels)
    ax_b.set_xlabel("Mean |SHAP value| (MPa impact on prediction)")
    ax_b.set_title("Composition/activator terms (green) dominate", fontsize=11, loc="left")
    ax_b.set_xlim(0, shap_imp.max() * 1.18)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="One-part geopolymer strength ML pipeline")
    ap.add_argument("--data", default="onepart_geopolymer_dataset.csv",
                    help="Path to the one-part dataset CSV")
    ap.add_argument("--outdir", default=".", help="Output directory")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = load_data(args.data)
    R, fitted, preds, (Xte, yte, y) = evaluate(df)
    print("\n=== Model comparison (sorted by LOSO R2) ===")
    print(R.round(3).to_string())
    R.round(4).to_csv(os.path.join(args.outdir, "ml_model_comparison.csv"))

    shap_imp = compute_shap(fitted["XGBoost"], Xte)
    print("\n=== SHAP mean|value| (MPa) ===")
    print(shap_imp.round(3).to_string())
    shap_imp.round(4).to_frame("mean_abs_shap_MPa").to_csv(
        os.path.join(args.outdir, "shap_importance.csv"))

    fig_path = os.path.join(args.outdir, "ml_results.png")
    make_figure(preds, yte, y, shap_imp, R.to_dict("index"), fig_path)
    print(f"\nWrote: ml_model_comparison.csv, shap_importance.csv, {os.path.basename(fig_path)}")


if __name__ == "__main__":
    main()
