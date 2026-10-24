"""XGBoost baseline with walk-forward evaluation.

Two models, refitted every --refit-every days on all data before each test block:
  - regressor: predicts the whole price path log(p[t+k]/p[t]) for k = 1..HORIZON (multi-output)
  - classifier: probability the price drops at least CHEAPER_DELTA within HORIZON days

Leakage control: labels look HORIZON days ahead, so training rows stop HORIZON days before
each test block starts ("purging"). Otherwise the training labels would overlap the test period.

Writes data/processed/preds_xgb.parquet (date, pid, pred_lr_1..H, pred_cheaper), which the
backtest reads. The transformer writes the same format, so the two are interchangeable.

Usage:
    python -m basket.model_xgb --data data/processed --test-start 2025-07-01 --refit-every 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .config import HORIZON
from .evaluate import calibration_table, classification_report, regression_report
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LR_COLS = [f"y_lr_{k}" for k in range(1, HORIZON + 1)]
PRED_COLS = [f"pred_lr_{k}" for k in range(1, HORIZON + 1)]


def default_test_start(table: pd.DataFrame, frac: float = 0.25) -> pd.Timestamp:
    dates = np.sort(table["date"].unique())
    return pd.Timestamp(dates[int(len(dates) * (1 - frac))])


def _fit(train: pd.DataFrame, seed: int, n_estimators: int) -> tuple[xgb.XGBRegressor, xgb.XGBClassifier]:
    common = dict(tree_method="hist", enable_categorical=True, max_depth=6, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, min_child_weight=20, random_state=seed,
                  n_estimators=n_estimators, max_cat_to_onehot=8)
    reg_rows = train.dropna(subset=LR_COLS)
    reg = xgb.XGBRegressor(objective="reg:squarederror", **common)
    reg.fit(reg_rows[FEATURES], reg_rows[LR_COLS].to_numpy())
    clf_rows = train.dropna(subset=["y_cheaper"])
    clf = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
    clf.fit(clf_rows[FEATURES], clf_rows["y_cheaper"].astype(int))
    return reg, clf


def walk_forward(table: pd.DataFrame, test_start: pd.Timestamp, refit_every: int = 30,
                 train_stride: int = 1, n_estimators: int = 300, seed: int = 0) -> tuple[pd.DataFrame, dict]:
    table = table.sort_values(["date", "pid"]).reset_index(drop=True)
    last_date = table["date"].max()
    block_starts = pd.date_range(test_start, last_date, freq=f"{refit_every}D")
    preds, base_rates = [], []
    for i, start in enumerate(block_starts):
        end = block_starts[i + 1] if i + 1 < len(block_starts) else last_date + pd.Timedelta(days=1)
        cutoff = start - pd.Timedelta(days=HORIZON)          # purge overlapping labels
        train = table[table["date"] < cutoff]
        if train_stride > 1:
            keep_dates = np.sort(train["date"].unique())[::train_stride]
            train = train[train["date"].isin(keep_dates)]
        test = table[(table["date"] >= start) & (table["date"] < end)]
        if train.empty or test.empty:
            continue
        reg, clf = _fit(train, seed, n_estimators)
        out = test[["date", "pid"]].copy()
        out[PRED_COLS] = reg.predict(test[FEATURES]).reshape(len(test), HORIZON)
        out["pred_cheaper"] = clf.predict_proba(test[FEATURES])[:, 1]
        preds.append(out)
        base_rates.append(float(train["y_cheaper"].mean()))
        print(f"block {start.date()}..{(end - pd.Timedelta(days=1)).date()}: "
              f"trained on {len(train):,} rows before {cutoff.date()}, predicted {len(test):,}")
    if not preds:
        raise ValueError("No test blocks. Is --test-start inside the data range?")
    preds = pd.concat(preds, ignore_index=True)
    importance = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return preds, {"base_rate": float(np.mean(base_rates)), "importance": importance}


def report(table: pd.DataFrame, preds: pd.DataFrame, base_rate: float) -> dict:
    merged = preds.merge(table[["date", "pid"] + LR_COLS + ["y_cheaper"]], on=["date", "pid"])
    reg = regression_report(merged[LR_COLS].to_numpy(), merged[PRED_COLS].to_numpy())
    clf = classification_report(merged["y_cheaper"].to_numpy(), merged["pred_cheaper"].to_numpy(), base_rate)
    cal = calibration_table(merged["y_cheaper"].to_numpy(), merged["pred_cheaper"].to_numpy())
    return {"regression": reg, "classification": clf, "calibration": cal}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--test-start", default=None, help="YYYY-MM-DD (default: last 25%% of dates)")
    ap.add_argument("--refit-every", type=int, default=30)
    ap.add_argument("--train-stride", type=int, default=1, help="use every Nth training day to save memory")
    ap.add_argument("--n-estimators", type=int, default=300)
    args = ap.parse_args()

    data = Path(args.data)
    table = pd.read_parquet(data / "features.parquet")
    test_start = pd.Timestamp(args.test_start) if args.test_start else default_test_start(table)
    print(f"test period starts {test_start.date()}")
    preds, info = walk_forward(table, test_start, args.refit_every, args.train_stride, args.n_estimators)
    preds.to_parquet(data / "preds_xgb.parquet", index=False)

    rep = report(table, preds, info["base_rate"])
    print("\nPrice path forecast (log ratio), skill vs 'no change':")
    print(rep["regression"].to_string(index=False))
    print("\nWill it be >= 5% cheaper within the horizon?")
    print(json.dumps(rep["classification"], indent=2))
    print("\nCalibration:")
    print(rep["calibration"].to_string())
    print("\nTop features (last refit):")
    print(info["importance"].head(12).round(3).to_string())
    with open(data / "report_xgb.json", "w") as fh:
        json.dump({"classification": rep["classification"],
                   "regression": rep["regression"].to_dict(orient="records"),
                   "test_start": str(test_start.date())}, fh, indent=2)


if __name__ == "__main__":
    main()
