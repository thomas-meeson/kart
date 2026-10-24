"""Promotion detection, leak-free features and forward-looking labels.

The dataset has no promo flag, so promotions are inferred: a product is "on promo" when its
price is at least PROMO_THRESHOLD below its trailing reference price (a high quantile of the
last REF_WINDOW days). Every feature uses only information available on day t.

Labels look forward and are only used for training and evaluation:
    y_lr_k      log(p[t+k] / p[t])            for k = 1..HORIZON
    y_cheaper   1 if min(p[t+1..t+HORIZON]) <= p[t] * (1 - CHEAPER_DELTA)

Usage:
    python -m basket.features --data data/processed
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CHEAPER_DELTA, HORIZON, PROMO_THRESHOLD, REF_QUANTILE, REF_WINDOW
from .load_data import to_wide

NUMERIC_FEATURES = [
    "log_price", "depth", "on_promo", "days_into_promo", "days_since_change", "days_since_promo_start",
    "days_since_promo_end", "promo_starts_90", "changes_90", "last_promo_depth", "max_depth_90",
    "vol_28", "ret_7", "ret_28", "price_vs_min_90", "retailer_promo_share", "retailer_promo_share_chg_7",
    "category_promo_share", "dow", "dom", "month",
]
CATEGORICAL_FEATURES = ["retailer", "category", "own_brand"]
LABEL_COLUMNS = [f"y_lr_{k}" for k in range(1, HORIZON + 1)] + ["y_cheaper"]


def _days_since(flag: pd.DataFrame, cap: int = 365) -> pd.DataFrame:
    """For each cell, days since `flag` was last True in that column (capped)."""
    rows = np.arange(len(flag), dtype=np.float32)[:, None]
    last = np.where(flag.to_numpy(), rows, np.nan)
    last = pd.DataFrame(last).ffill().to_numpy()
    out = np.where(np.isnan(last), cap, np.minimum(rows - last, cap))
    return pd.DataFrame(out.astype(np.float32), index=flag.index, columns=flag.columns)


def _run_length(flag: pd.DataFrame) -> pd.DataFrame:
    """Number of consecutive True days up to and including today (0 if today is False)."""
    f = flag.to_numpy()
    out = np.zeros(f.shape, dtype=np.float32)
    for i in range(len(f)):
        out[i] = np.where(f[i], (out[i - 1] if i else 0) + 1, 0)
    return pd.DataFrame(out, index=flag.index, columns=flag.columns)


def _group_share(flag: pd.DataFrame, valid: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Per day, the fraction of products in each column's group that are flagged."""
    num = flag.astype(np.float32).where(valid).T.groupby(groups.to_numpy()).sum().T
    den = valid.astype(np.float32).T.groupby(groups.to_numpy()).sum().T
    share = (num / den.replace(0, np.nan))
    return share[groups.to_numpy()].set_axis(flag.columns, axis=1)


def compute_wide(panel: pd.DataFrame, products: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """All features and labels as dates x pids matrices."""
    P = to_wide(panel, "price").astype(np.float32)
    valid = P.notna()
    meta = products.set_index("pid").loc[P.columns]

    ref = P.rolling(REF_WINDOW, min_periods=7).quantile(REF_QUANTILE)
    depth = (1 - P / ref).clip(lower=-1)
    promo = (depth >= PROMO_THRESHOLD) & valid
    promo_prev = promo.shift(1, fill_value=False)
    promo_start = promo & ~promo_prev
    promo_end = ~promo & promo_prev & valid
    changed = (P.diff().abs() > 1e-6) & valid

    logP = np.log(P)
    f: dict[str, pd.DataFrame] = {}
    f["log_price"] = logP
    f["depth"] = depth
    f["on_promo"] = promo.astype(np.float32)
    f["days_into_promo"] = _run_length(promo)
    f["days_since_change"] = _days_since(changed)
    f["days_since_promo_start"] = _days_since(promo_start)
    f["days_since_promo_end"] = _days_since(promo_end)
    f["promo_starts_90"] = promo_start.astype(np.float32).rolling(90, min_periods=1).sum()
    f["changes_90"] = changed.astype(np.float32).rolling(90, min_periods=1).sum()
    f["last_promo_depth"] = depth.where(promo).ffill().fillna(0)
    f["max_depth_90"] = depth.rolling(90, min_periods=1).max()
    f["vol_28"] = logP.diff().rolling(28, min_periods=7).std()
    f["ret_7"] = logP - logP.shift(7)
    f["ret_28"] = logP - logP.shift(28)
    f["price_vs_min_90"] = logP - logP.rolling(90, min_periods=1).min()

    f["retailer_promo_share"] = _group_share(promo, valid, meta["retailer"])
    f["retailer_promo_share_chg_7"] = f["retailer_promo_share"] - f["retailer_promo_share"].shift(7)
    f["category_promo_share"] = _group_share(promo, valid, meta["retailer"].astype(str) + "|" + meta["category"].astype(str))

    idx = P.index
    for name, values in [("dow", idx.dayofweek), ("dom", idx.day), ("month", idx.month)]:
        f[name] = pd.DataFrame(np.repeat(np.asarray(values, dtype=np.float32)[:, None], P.shape[1], axis=1),
                               index=idx, columns=P.columns)

    # Labels (forward-looking).
    future = [P.shift(-k) for k in range(1, HORIZON + 1)]
    for k, Fk in enumerate(future, start=1):
        f[f"y_lr_{k}"] = np.log(Fk / P)
    stack = np.stack([Fk.to_numpy() for Fk in future])            # (H, dates, pids)
    n_future = np.sum(~np.isnan(stack), axis=0)
    with np.errstate(all="ignore"):
        fwd_min = np.nanmin(np.where(np.isnan(stack), np.inf, stack), axis=0)
    cheaper = (fwd_min <= P.to_numpy() * (1 - CHEAPER_DELTA)).astype(np.float32)
    cheaper[(n_future < HORIZON - 3) | np.isnan(P.to_numpy())] = np.nan
    f["y_cheaper"] = pd.DataFrame(cheaper, index=idx, columns=P.columns)

    f["price"] = P
    f["ref_price"] = ref
    return f


def make_table(wide: dict[str, pd.DataFrame], products: pd.DataFrame, stride: int = 1) -> pd.DataFrame:
    """Long table: one row per (date, pid) with a price. `stride` > 1 subsamples days to save memory."""
    P = wide["price"]
    rows = np.arange(0, len(P), stride)
    sub = P.iloc[rows]
    di, pj = np.nonzero(sub.notna().to_numpy())
    out = pd.DataFrame({"date": sub.index[di], "pid": sub.columns.to_numpy()[pj].astype(np.int32)})
    for name in NUMERIC_FEATURES + LABEL_COLUMNS + ["price", "ref_price"]:
        out[name] = wide[name].iloc[rows].to_numpy()[di, pj].astype(np.float32)
    meta = products.set_index("pid")[CATEGORICAL_FEATURES]
    out = out.join(meta, on="pid")
    for c in CATEGORICAL_FEATURES:
        out[c] = out[c].astype(str).astype("category")
    return out


def promo_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Quick sanity check for the pitch: how often products are on promo, by retailer."""
    g = table.groupby("retailer", observed=True)
    return pd.DataFrame({
        "share_days_on_promo": g["on_promo"].mean(),
        "mean_depth_when_on_promo": table[table["on_promo"] > 0].groupby("retailer", observed=True)["depth"].mean(),
        "share_cheaper_within_horizon": g["y_cheaper"].mean(),
        "rows": g.size(),
    }).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    data = Path(args.data)
    products = pd.read_parquet(data / "products.parquet")
    panel = pd.read_parquet(data / "panel.parquet")
    wide = compute_wide(panel, products)
    table = make_table(wide, products, args.stride)
    table.to_parquet(data / "features.parquet", index=False)
    print(f"wrote {data / 'features.parquet'}: {len(table):,} rows")
    print(promo_summary(table).to_string())


if __name__ == "__main__":
    main()
