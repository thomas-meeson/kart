"""Lead-lag between retailers: when one chain changes a regular price, do others follow?

Tesco and Sainsbury's run "Aldi Price Match" schemes, so own-label prices there may follow Aldi
with a delay. That's the grocery version of a lead-lag relationship between markets, and a
leading retailer's move is a usable predictive signal for the followers.

Method, for each matched item and each ordered retailer pair (A, B):
  - regular price = the price with promo days replaced by the last non-promo price
  - an "event" is a regular price change of more than 1%
  - for each event at B, check whether A moved in the same direction in the previous L days
    (A led) or the following L days (B led)
  - asymmetry = share A led - share B led; clearly positive means A tends to move first

Usage:
    python -m basket.analysis --data data/processed --staples data/staples.csv --lag 14
"""
from __future__ import annotations

import argparse
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROMO_THRESHOLD
from .features import compute_wide
from .matching import load_matches


def regular_price_events(price: np.ndarray, depth: np.ndarray, min_change: float = 0.01) -> list[tuple[int, int]]:
    reg = pd.Series(np.where(depth >= PROMO_THRESHOLD, np.nan, price)).ffill().to_numpy()
    lr = np.diff(np.log(reg))
    days = np.nonzero(np.abs(np.nan_to_num(lr)) > min_change)[0] + 1
    return [(int(d), int(np.sign(lr[d - 1]))) for d in days]


def lead_lag(wide: dict[str, pd.DataFrame], matches: pd.DataFrame, lag: int = 14) -> pd.DataFrame:
    P, D = wide["price"], wide["depth"]
    col = {p: j for j, p in enumerate(P.columns)}
    Pn, Dn = P.to_numpy(), D.to_numpy()
    counts: dict[tuple[str, str], list[int]] = {}
    for _, grp in matches.groupby("item"):
        events = {r.retailer: regular_price_events(Pn[:, col[r.pid]], Dn[:, col[r.pid]])
                  for r in grp.itertuples() if r.pid in col}
        for a, b in permutations(events, 2):
            ea = events[a]
            c = counts.setdefault((a, b), [0, 0, 0])
            for t, sgn in events[b]:
                c[0] += 1
                c[1] += any(sgn == s and t - lag <= u < t for u, s in ea)
                c[2] += any(sgn == s and t < u <= t + lag for u, s in ea)
    rows = [{"leader": a, "follower": b, "follower_events": n,
             "share_leader_moved_first": led / n if n else np.nan,
             "share_follower_moved_first": fol / n if n else np.nan}
            for (a, b), (n, led, fol) in counts.items()]
    out = pd.DataFrame(rows)
    out["asymmetry"] = out["share_leader_moved_first"] - out["share_follower_moved_first"]
    return out.sort_values("asymmetry", ascending=False).round(3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--staples", default="data/staples.csv")
    ap.add_argument("--matches", default=None)
    ap.add_argument("--lag", type=int, default=14)
    args = ap.parse_args()
    data = Path(args.data)
    products = pd.read_parquet(data / "products.parquet")
    panel = pd.read_parquet(data / "panel.parquet")
    staples = pd.read_csv(args.staples)
    matches = load_matches(args.matches or data / "match_candidates.csv", staples, products, panel)
    res = lead_lag(compute_wide(panel, products), matches, args.lag)
    print(res.to_string(index=False))
    res.to_csv(data / "lead_lag.csv", index=False)


if __name__ == "__main__":
    main()
