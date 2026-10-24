"""Demo backend: given a shopping list and nearby stores, say where and when to buy each item.

Prints JSON for the frontend. Uses the latest day that has predictions unless --date is given.

Usage:
    python -m basket.recommend --data data/processed --staples data/staples.csv \
        --preds preds_xgb.parquet --items milk_semi_4pt,heinz_beans_415g,pasta_fusilli_500g \
        --visit Tesco=1,Aldi=3,Sainsburys=4 --home Tesco
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import Market
from .config import HORIZON
from .matching import load_matches
from .optimise import expected_costs, solve_plan


def recommend(m: Market, preds_path: Path, items: list[str], visit: dict[str, float], home: str | None,
              date: pd.Timestamp | None = None, delay: float = 0.02) -> dict:
    preds = m.load_preds(preds_path)
    pr = pd.read_parquet(preds_path, columns=["date", "pid", "pred_cheaper"])
    date = date or pd.Timestamp(pr["date"].max())
    t = int(np.searchsorted(m.dates, date))
    if t >= len(m.dates) or m.dates[t] != date:
        raise ValueError(f"{date.date()} is not in the data")
    p_cheaper = pr[pr["date"] == date].set_index("pid")["pred_cheaper"]

    unknown = [i for i in items if i not in m.items]
    if unknown:
        raise ValueError(f"unknown items {unknown}; options: {m.items}")
    idx = np.array([m.items.index(i) for i in items])
    visit_arr = np.array([visit.get(s, np.inf) for s in m.stores], dtype=float)
    today = m.cost[t][idx].copy()
    today[:, ~np.isfinite(visit_arr)] = np.nan
    needby = m.shelf[idx]
    pred = np.nan_to_num(preds[t][idx], nan=0.0)
    exp_cost = expected_costs(today, pred, int(needby.max()) + 1)
    # Items with no price today at any nearby store can't be planned.
    plannable = np.isfinite(exp_cost).any(axis=(1, 2))
    plan = solve_plan(exp_cost[plannable], needby[plannable], np.where(np.isfinite(visit_arr), visit_arr, 0.0), delay)
    planned_idx = np.nonzero(plannable)[0]

    out_items = []
    for k, (i_pos) in enumerate(planned_idx):
        s, d = plan.choice[k]
        row = {
            "item": items[i_pos],
            "buy_at": m.stores[s],
            "buy_in_days": int(d),
            "expected_cost": round(float(exp_cost[i_pos, s, d]), 2),
            "prices_today": {m.stores[j]: round(float(today[i_pos, j]), 2)
                             for j in range(len(m.stores)) if np.isfinite(today[i_pos, j])},
            "p_cheaper_within_14d": {m.stores[j]: round(float(p_cheaper.get(m.pid[idx[i_pos], j], np.nan)), 3)
                                     for j in range(len(m.stores)) if np.isfinite(today[i_pos, j])},
            "expected_path": {m.stores[j]: [round(float(x), 2) for x in exp_cost[i_pos, j, :HORIZON + 1]
                                            if np.isfinite(x)]
                              for j in range(len(m.stores)) if np.isfinite(today[i_pos, j])},
        }
        out_items.append(row)

    home_total = None
    if home in m.stores:
        h = m.stores.index(home)
        hp = today[planned_idx, h]
        if np.isfinite(hp).all():
            home_total = round(float(hp.sum() + visit_arr[h]), 2)
    plan_total = round(sum(r["expected_cost"] for r in out_items)
                       + sum(visit_arr[s] for s, _ in plan.visits), 2)
    return {
        "date": str(date.date()),
        "items": out_items,
        "not_available_nearby": [items[i] for i in np.nonzero(~plannable)[0]],
        "trips": sorted([{"store": m.stores[s], "in_days": int(d)} for s, d in plan.visits],
                        key=lambda x: (x["in_days"], x["store"])),
        "expected_total_incl_travel": plan_total,
        "home_store_total_today_incl_travel": home_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--staples", default="data/staples.csv")
    ap.add_argument("--matches", default=None)
    ap.add_argument("--preds", default="preds_xgb.parquet")
    ap.add_argument("--items", default=None, help="comma-separated item ids (default: all)")
    ap.add_argument("--visit", default="", help="Store=cost pairs, e.g. Tesco=1,Aldi=3. Unlisted stores = not nearby")
    ap.add_argument("--home", default=None)
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    staples = pd.read_csv(args.staples)
    products = pd.read_parquet(data / "products.parquet")
    panel = pd.read_parquet(data / "panel.parquet")
    matches = load_matches(args.matches or data / "match_candidates.csv", staples, products, panel)
    m = Market(data, staples, matches)
    visit = {k: float(v) for k, v in (p.split("=") for p in args.visit.split(",") if p)} or \
            {s: 3.0 for s in m.stores}
    items = args.items.split(",") if args.items else m.items
    preds_path = Path(args.preds) if Path(args.preds).exists() else data / args.preds
    res = recommend(m, preds_path, items, visit, args.home, pd.Timestamp(args.date) if args.date else None)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
