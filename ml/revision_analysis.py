#!/usr/bin/env python3
"""Revision analyses for reviewer 2 comments (MDPI Materials).
R1: reproduce headline numbers.
A. Duplicate/replicate analysis + noise floor + dedup sensitivity (point 1)
B. Nested CV (tuned) comparison + reduced-capacity XGBoost + early stopping + learning curves (point 2)
C. Bootstrap 95% CIs on LOSO R2/RMSE (point 7)
D. Variance decomposition of LOSO error (point 4)
"""
import json, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold, LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.base import clone
import xgboost as xgb

RS = 42
df = pd.read_csv("dataset.csv")
denom = (df["FlyAsh_pct"] + df["GGBS_pct"]).replace(0, np.nan)
df["SlagFraction"] = (df["GGBS_pct"] / denom).fillna(0.0)
FEATURES = ["SlagFraction", "Na2O_pct", "Water_Binder", "Curing_temp_C"]
RAW5 = ["FlyAsh_pct", "GGBS_pct", "Na2O_pct", "Water_Binder", "Curing_temp_C"]
X, y, groups = df[FEATURES], df["Strength_MPa"].values, df["SourceRef"].values
logo = LeaveOneGroupOut()
out = {}

def loso_pred(est, scale, Xd=None, yd=None, gd=None):
    Xd = X if Xd is None else Xd; yd = y if yd is None else yd; gd = groups if gd is None else gd
    yhat = np.zeros_like(yd, dtype=float)
    for trn, tst in logo.split(Xd, yd, gd):
        p = Pipeline([("sc", StandardScaler()), ("m", clone(est))]) if scale else clone(est)
        p.fit(Xd.iloc[trn], yd[trn]); yhat[tst] = p.predict(Xd.iloc[tst])
    return yhat

def xgbm(n=400, d=3, lr=0.05):
    return xgb.XGBRegressor(n_estimators=n, learning_rate=lr, max_depth=d, subsample=0.85, random_state=RS)

MODELS = {
    "Linear": (LinearRegression(), True),
    "SVR": (SVR(C=100, gamma="scale", epsilon=0.5), True),
    "RF": (RandomForestRegressor(n_estimators=500, random_state=RS), False),
    "XGB": (xgbm(), False),
}

# ---- R1: reproduce headline ----
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RS)
rep = {}
for name, (est, sc) in MODELS.items():
    p = Pipeline([("sc", StandardScaler()), ("m", clone(est))]) if sc else clone(est)
    p.fit(Xtr, ytr); pr = p.predict(Xte)
    yhat = loso_pred(est, sc)
    rep[name] = dict(test_r2=round(r2_score(yte, pr), 3),
                     test_rmse=round(float(np.sqrt(mean_squared_error(yte, pr))), 2),
                     loso_r2=round(r2_score(y, yhat), 3),
                     loso_rmse=round(float(np.sqrt(mean_squared_error(y, yhat))), 2))
out["reproduce"] = rep

# ---- A: duplicates ----
d5 = df.groupby(RAW5)
sizes = d5.size()
dup_groups = sizes[sizes > 1]
n_dup_records = int(dup_groups.sum())
n_unique_vectors = int(len(sizes))
spread = d5["Strength_MPa"].agg(["count", "min", "max", "std"])
spread_d = spread[spread["count"] > 1]
max_range = float((spread_d["max"] - spread_d["min"]).max())
# replicate noise floor: pooled within-group std among duplicate groups
wss = float(np.sqrt(((spread_d["std"] ** 2) * (spread_d["count"] - 1)).sum() / (spread_d["count"] - 1).sum()))
out["duplicates"] = dict(n_records=80, n_unique_5d_vectors=n_unique_vectors,
                         n_records_in_dup_groups=n_dup_records, n_dup_groups=int(len(dup_groups)),
                         max_within_group_range_MPa=round(max_range, 1),
                         replicate_noise_floor_RMSE_MPa=round(wss, 2))
# dedup sensitivity: average duplicates -> rerun LOSO for XGB + Linear
agg = df.groupby(RAW5 + ["SourceRef"], as_index=False).agg(Strength_MPa=("Strength_MPa", "mean"))
denom2 = (agg["FlyAsh_pct"] + agg["GGBS_pct"]).replace(0, np.nan)
agg["SlagFraction"] = (agg["GGBS_pct"] / denom2).fillna(0.0)
Xa, ya, ga = agg[FEATURES], agg["Strength_MPa"].values, agg["SourceRef"].values
dedup = {}
for name in ["Linear", "XGB"]:
    est, sc = MODELS[name]
    yh = loso_pred(est, sc, Xa, ya, ga)
    dedup[name] = dict(n=len(agg), loso_r2=round(r2_score(ya, yh), 3),
                       loso_rmse=round(float(np.sqrt(mean_squared_error(ya, yh))), 2))
out["dedup_sensitivity"] = dedup

# ---- B: nested CV (tuned within each LOSO fold via inner GroupKFold) ----
grids = {
    "Linear": (LinearRegression(), True, {}),
    "SVR": (SVR(), True, {"m__C": [1, 10, 100], "m__epsilon": [0.1, 0.5, 1.0]}),
    "RF": (RandomForestRegressor(random_state=RS), False, {"m__n_estimators": [100, 300], "m__max_depth": [3, 6, None]}),
    "XGB": (xgb.XGBRegressor(random_state=RS, subsample=0.85), False,
            {"m__n_estimators": [50, 100, 300], "m__max_depth": [2, 3], "m__learning_rate": [0.05, 0.1]}),
}
nested = {}
for name, (est, sc, grid) in grids.items():
    yhat = np.zeros_like(y, dtype=float)
    for trn, tst in logo.split(X, y, groups):
        base = Pipeline([("sc", StandardScaler()), ("m", clone(est))]) if sc else Pipeline([("m", clone(est))])
        if grid:
            inner = GroupKFold(n_splits=3)
            gs = GridSearchCV(base, grid, cv=inner.split(X.iloc[trn], y[trn], groups[trn]), scoring="r2", n_jobs=-1)
            gs.fit(X.iloc[trn], y[trn]); mdl = gs.best_estimator_
        else:
            mdl = base.fit(X.iloc[trn], y[trn])
        yhat[tst] = mdl.predict(X.iloc[tst])
    nested[name] = dict(loso_r2=round(r2_score(y, yhat), 3),
                        loso_rmse=round(float(np.sqrt(mean_squared_error(y, yhat))), 2))
out["nested_cv"] = nested

# reduced capacity + early stopping XGBoost
red = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=2, subsample=0.85,
                       random_state=RS, early_stopping_rounds=30)
yhat = np.zeros_like(y, dtype=float); best_iters = []
for trn, tst in logo.split(X, y, groups):
    Xt, Xv, yt, yv = train_test_split(X.iloc[trn], y[trn], test_size=0.2, random_state=RS)
    m = clone(red); m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    best_iters.append(int(m.best_iteration) + 1)
    yhat[tst] = m.predict(X.iloc[tst])
out["early_stopping_xgb"] = dict(loso_r2=round(r2_score(y, yhat), 3),
                                 loso_rmse=round(float(np.sqrt(mean_squared_error(y, yhat))), 2),
                                 median_best_n_trees=int(np.median(best_iters)),
                                 iqr_trees=[int(np.percentile(best_iters, 25)), int(np.percentile(best_iters, 75))])

# learning curve over capacity (LOSO R2 vs n_estimators x depth)
lc = []
for d_ in [2, 3]:
    for n_ in [10, 25, 50, 100, 200, 400, 800]:
        yh = loso_pred(xgbm(n=n_, d=d_), False)
        lc.append(dict(depth=d_, n_estimators=n_, loso_r2=round(r2_score(y, yh), 3),
                       loso_rmse=round(float(np.sqrt(mean_squared_error(y, yh))), 2)))
out["learning_curve"] = lc

# ---- C: bootstrap CIs on pooled LOSO (record-level and cluster/study-level) ----
est, sc = MODELS["XGB"]
yhat = loso_pred(est, sc)
rng = np.random.default_rng(RS)
r2s, rmses = [], []
n = len(y)
for _ in range(4000):
    idx = rng.integers(0, n, n)
    if np.var(y[idx]) == 0: continue
    r2s.append(r2_score(y[idx], yhat[idx]))
    rmses.append(np.sqrt(mean_squared_error(y[idx], yhat[idx])))
studies = np.unique(groups)
cr2, crm = [], []
for _ in range(4000):
    ss = rng.choice(studies, len(studies), replace=True)
    idx = np.concatenate([np.where(groups == s)[0] for s in ss])
    if np.var(y[idx]) == 0: continue
    cr2.append(r2_score(y[idx], yhat[idx]))
    crm.append(np.sqrt(mean_squared_error(y[idx], yhat[idx])))
pct = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
out["bootstrap"] = dict(
    loso_r2_point=round(r2_score(y, yhat), 3),
    loso_rmse_point=round(float(np.sqrt(mean_squared_error(y, yhat))), 2),
    record_boot_r2_95ci=pct(r2s), record_boot_rmse_95ci=pct(rmses),
    cluster_boot_r2_95ci=pct(cr2), cluster_boot_rmse_95ci=pct(crm))

# ---- D: variance decomposition of LOSO error ----
err = yhat - y
tot_mse = float(np.mean(err ** 2))
rows = []
between = 0.0; within = 0.0
for s in studies:
    m = groups == s
    b = float(np.mean(err[m]))          # systematic per-study offset (bias)
    w = float(np.mean((err[m] - b) ** 2))  # within-study residual
    between += m.sum() * b * b
    within += m.sum() * w
    rows.append(dict(study=int(s), n=int(m.sum()), bias_MPa=round(b, 2),
                     within_rmse_MPa=round(np.sqrt(w), 2)))
between /= n; within /= n
# noise floor share (replicate variance among identical vectors)
noise_var = wss ** 2
out["variance_decomposition"] = dict(
    total_loso_mse=round(tot_mse, 1),
    between_study_bias_share=round(between / tot_mse, 3),
    within_study_share=round(within / tot_mse, 3),
    replicate_noise_floor_var=round(noise_var, 1),
    replicate_noise_share_of_mse=round(noise_var / tot_mse, 3),
    per_study=rows)

print(json.dumps(out, indent=1))
with open("revision_results.json", "w") as f:
    json.dump(out, f, indent=1)
