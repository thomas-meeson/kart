"""Metrics shared by every model, so XGBoost and the transformer are scored identically.

Prices are sticky, so "no change" is a very strong baseline for the price path. Every
regression number is reported as skill relative to that naive forecast:
    skill = 1 - RMSE(model) / RMSE(naive)    (> 0 means better than naive)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from .config import HORIZON


def regression_report(y_true: np.ndarray, y_pred: np.ndarray, ks=(1, 3, 7, 14)) -> pd.DataFrame:
    """y_true, y_pred: (n, HORIZON) log price ratios. NaNs in y_true are ignored per column."""
    rows = []
    for k in ks:
        if k > HORIZON:
            continue
        yt, yp = y_true[:, k - 1], y_pred[:, k - 1]
        ok = ~np.isnan(yt)
        yt, yp = yt[ok], yp[ok]
        rmse_m = float(np.sqrt(np.mean((yt - yp) ** 2)))
        rmse_n = float(np.sqrt(np.mean(yt ** 2)))
        moved = yt != 0
        rows.append({
            "k_days": k, "n": int(ok.sum()), "rmse_model": rmse_m, "rmse_naive": rmse_n,
            "skill_vs_naive": 1 - rmse_m / rmse_n if rmse_n > 0 else np.nan,
            "mae_model": float(np.mean(np.abs(yt - yp))), "mae_naive": float(np.mean(np.abs(yt))),
            "share_price_moved": float(moved.mean()),
            # Direction accuracy only on rows where the price actually moved.
            "direction_acc_when_moved": float(np.mean(np.sign(yp[moved]) == np.sign(yt[moved]))) if moved.any() else np.nan,
        })
    return pd.DataFrame(rows).round(4)


def classification_report(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict:
    ok = ~np.isnan(y)
    y, p = y[ok].astype(int), np.clip(p[ok], 1e-6, 1 - 1e-6)
    base = np.full_like(p, np.clip(base_rate, 1e-6, 1 - 1e-6))
    out = {
        "n": int(ok.sum()), "positive_rate": float(y.mean()),
        "log_loss_model": log_loss(y, p, labels=[0, 1]), "log_loss_base_rate": log_loss(y, base, labels=[0, 1]),
        "brier_model": brier_score_loss(y, p), "brier_base_rate": brier_score_loss(y, base),
        "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """When the model says 30%, does it happen about 30% of the time?"""
    ok = ~np.isnan(y)
    df = pd.DataFrame({"y": y[ok], "p": p[ok]})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("y", "size"), mean_predicted=("p", "mean"),
                                                 observed_rate=("y", "mean")).round(3)
