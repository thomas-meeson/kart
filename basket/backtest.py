"""Walk-forward backtest: how much would a shopper have saved by following each policy?

Each scenario is a random shopper on a random day in the test period with a random shopping
list (drawn from the confirmed staples), a need-by day per item (from shelf_days) and a set of
nearby stores with travel costs. Policies compared, all scored on the real prices that followed:

    home_store       buy everything today at your nearest store (what most people do)
    compare_today    best mix of stores using today's prices only (what a comparison site does)
    model:<name>     re-plan every day using forecasts, carry out today's purchases only
    oracle           perfect knowledge of future prices (the best anyone could do)

Your purchases don't change supermarket prices, so this backtest is an honest test of the
decision, not just the forecast. The headline numbers:
    saving vs home_store and vs compare_today, with 95% confidence intervals
    share of the timing gap captured = (compare_today - model) / (compare_today - oracle)

Usage:
    python -m basket.backtest --data data/processed --staples data/staples.csv \
        --matches data/processed/match_candidates.csv --preds xgb=preds_xgb.parquet \
        --preds transformer=preds_transformer.parquet --n-scenarios 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HORIZON, RETAILERS
from .load_data import to_wide
from .matching import load_matches
from .optimise import expected_costs, solve_plan


class Market:
    """True costs per (day, item, store) for the confirmed matches."""

    def __init__(self, data_dir: Path, staples: pd.DataFrame, matches: pd.DataFrame):
        panel = pd.read_parquet(data_dir / "panel.parquet")
        P = to_wide(panel, "price")
        self.dates = P.index
        self.items = sorted(matches["item"].unique())
        self.stores = [r for r in RETAILERS if r in set(matches["retailer"])]
        n_i, n_s = len(self.items), len(self.stores)
        self.pid = np.full((n_i, n_s), -1, dtype=np.int64)
        self.mult = np.ones((n_i, n_s))
        for _, r in matches.iterrows():
            i, s = self.items.index(r["item"]), self.stores.index(r["retailer"])
            self.pid[i, s], self.mult[i, s] = int(r["pid"]), float(r["multiplier"])
        self.cost = np.full((len(self.dates), n_i, n_s), np.nan)
        cols = {c: j for j, c in enumerate(P.columns)}
        arr = P.to_numpy()
        for i in range(n_i):
            for s in range(n_s):
                if self.pid[i, s] in cols:
                    self.cost[:, i, s] = arr[:, cols[self.pid[i, s]]] * self.mult[i, s]
        shelf = staples.set_index("item")["shelf_days"]
        self.shelf = np.array([int(min(shelf.get(it, 7), HORIZON)) for it in self.items])

    def load_preds(self, path: Path) -> np.ndarray:
        """(days, items, stores, HORIZON) predicted log ratios; NaN where no prediction."""
        pr = pd.read_parquet(path)
        pr = pr[pr["pid"].isin(set(self.pid[self.pid >= 0].tolist()))]
        out = np.full(self.cost.shape + (HORIZON,), np.nan, dtype=np.float32)
        t_index = pd.Series(np.arange(len(self.dates)), index=self.dates)
        pr = pr[pr["date"].isin(self.dates)]
        t = t_index.loc[pr["date"]].to_numpy()
        vals = pr[[f"pred_lr_{k}" for k in range(1, HORIZON + 1)]].to_numpy(dtype=np.float32)
        loc = {int(p): (i, s) for (i, s), p in np.ndenumerate(self.pid) if p >= 0}
        for row, (tt, p) in enumerate(zip(t, pr["pid"].to_numpy())):
            i, s = loc[int(p)]
            out[tt, i, s] = vals[row]
        return out


def make_scenarios(m: Market, test_start: pd.Timestamp, n: int, items_per_list: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    t_first = int(np.searchsorted(m.dates, test_start))
    t_last = len(m.dates) - 1 - int(m.shelf.max())
    if t_last <= t_first:
        raise ValueError("Test period is shorter than the longest shelf_days.")
    scenarios, tries = [], 0
    while len(scenarios) < n and tries < n * 50:
        tries += 1
        t0 = int(rng.integers(t_first, t_last + 1))
        visit = rng.uniform(2.0, 6.0, len(m.stores))
        home = int(rng.integers(len(m.stores)))
        visit[home] = 1.0
        nearby = rng.random(len(m.stores)) > 0.3
        nearby[home] = True
        visit = np.where(nearby, visit, np.inf)
        # Items available at some nearby store on every day until their need-by day.
        ok = []
        for i in range(len(m.items)):
            window = m.cost[t0:t0 + m.shelf[i] + 1, i, :][:, nearby]
            if np.isfinite(window).any(axis=1).all():
                ok.append(i)
        if len(ok) < items_per_list:
            continue
        items = np.sort(rng.choice(ok, items_per_list, replace=False))
        scenarios.append({"t0": t0, "items": items, "needby": m.shelf[items], "visit": visit, "home": home})
    if len(scenarios) < n:
        print(f"warning: only built {len(scenarios)} scenarios (not enough items available together)")
    return scenarios


def _masked_cost(m: Market, t: int, items, visit) -> np.ndarray:
    c = m.cost[t][items].copy()
    c[:, ~np.isfinite(visit)] = np.nan
    return c


def policy_home_store(m: Market, sc: dict) -> dict:
    today = _masked_cost(m, sc["t0"], sc["items"], sc["visit"])
    spend, stores = 0.0, set()
    for k in range(len(sc["items"])):
        s = sc["home"] if np.isfinite(today[k, sc["home"]]) else int(np.nanargmin(today[k]))
        spend += today[k, s]
        stores.add(s)
    spend += sum(sc["visit"][s] for s in stores)
    return {"spend": spend, "visits": len(stores), "wait_days": 0.0}


def policy_mpc(m: Market, sc: dict, preds: np.ndarray | None, delay: float) -> dict:
    """Re-plan every day. preds=None means 'assume prices stay the same' (compare_today)."""
    remaining = list(range(len(sc["items"])))
    spend, visits, wait = 0.0, 0, 0.0
    visit_finite = np.where(np.isfinite(sc["visit"]), sc["visit"], 0.0)
    for d in range(int(sc["needby"].max()) + 1):
        if not remaining:
            break
        t = sc["t0"] + d
        items = sc["items"][remaining]
        today = _masked_cost(m, t, items, sc["visit"])
        nb = sc["needby"][remaining] - d
        if preds is None:
            pred = np.zeros(today.shape + (HORIZON,))
        else:
            pred = np.nan_to_num(preds[t][items], nan=0.0)
        exp_cost = expected_costs(today, pred, int(nb.max()) + 1)
        plan = solve_plan(exp_cost, nb, visit_finite, delay)
        bought = [k for k, (_, dd) in plan.choice.items() if dd == 0]
        stores = {plan.choice[k][0] for k in bought}
        spend += sum(today[k, plan.choice[k][0]] for k in bought) + sum(sc["visit"][s] for s in stores)
        visits += len(stores)
        wait += d * len(bought)
        remaining = [r for j, r in enumerate(remaining) if j not in set(bought)]
    if remaining:
        raise RuntimeError("items left unbought after their need-by day")
    return {"spend": spend, "visits": visits, "wait_days": wait / len(sc["items"])}


def policy_oracle(m: Market, sc: dict, delay: float) -> dict:
    n_days = int(sc["needby"].max()) + 1
    c = np.stack([_masked_cost(m, sc["t0"] + d, sc["items"], sc["visit"]) for d in range(n_days)], axis=2)
    visit_finite = np.where(np.isfinite(sc["visit"]), sc["visit"], 0.0)
    plan = solve_plan(c, sc["needby"], visit_finite, delay)
    spend = sum(c[i, s, d] for i, (s, d) in plan.choice.items()) + sum(sc["visit"][s] for s, _ in plan.visits)
    wait = np.mean([d for (_, d) in plan.choice.values()])
    return {"spend": spend, "visits": len(plan.visits), "wait_days": wait}


def _ci(x: np.ndarray) -> str:
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan
    return f"{x.mean():.2%} (95% CI {x.mean() - 1.96 * se:.2%} to {x.mean() + 1.96 * se:.2%})"


def run(m: Market, scenarios: list[dict], preds: dict[str, np.ndarray], delay: float) -> pd.DataFrame:
    rows = []
    for n, sc in enumerate(scenarios):
        res = {"home_store": policy_home_store(m, sc), "compare_today": policy_mpc(m, sc, None, delay),
               "oracle": policy_oracle(m, sc, delay)}
        for name, arr in preds.items():
            res[f"model:{name}"] = policy_mpc(m, sc, arr, delay)
        for pol, r in res.items():
            rows.append({"scenario": n, "date": m.dates[sc["t0"]].date(), "policy": pol, **r})
        if (n + 1) % 25 == 0:
            print(f"  {n + 1}/{len(scenarios)} scenarios")
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> str:
    wide = results.pivot(index="scenario", columns="policy", values="spend")
    lines = ["Mean spend per shopping list (prices + travel):"]
    summary = results.groupby("policy")[["spend", "visits", "wait_days"]].mean().round(2)
    lines.append(summary.to_string())
    gap = (wide["compare_today"] - wide["oracle"]).to_numpy()
    lines.append(f"\nMost that timing could add (oracle vs compare_today): "
                 f"{_ci(gap / wide['compare_today'].to_numpy())} of spend")
    lines.append(f"compare_today vs home_store saving: "
                 f"{_ci(1 - wide['compare_today'].to_numpy() / wide['home_store'].to_numpy())}")
    for col in [c for c in wide.columns if c.startswith("model:")]:
        lines.append(f"\n{col}")
        lines.append(f"  saving vs home_store:    {_ci(1 - wide[col].to_numpy() / wide['home_store'].to_numpy())}")
        lines.append(f"  saving vs compare_today: {_ci(1 - wide[col].to_numpy() / wide['compare_today'].to_numpy())}")
        captured = (wide["compare_today"] - wide[col]).sum() / gap.sum() if gap.sum() > 0 else np.nan
        lines.append(f"  share of timing gap captured: {captured:.1%}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--staples", default="data/staples.csv")
    ap.add_argument("--matches", default=None, help="confirmed match file (default: <data>/match_candidates.csv)")
    ap.add_argument("--preds", action="append", default=[], help="name=file.parquet (repeatable)")
    ap.add_argument("--test-start", default=None)
    ap.add_argument("--n-scenarios", type=int, default=200)
    ap.add_argument("--items-per-list", type=int, default=12)
    ap.add_argument("--delay", type=float, default=0.02, help="impatience cost in £ per item per day waited")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    staples = pd.read_csv(args.staples)
    products = pd.read_parquet(data / "products.parquet")
    panel = pd.read_parquet(data / "panel.parquet")
    matches = load_matches(args.matches or data / "match_candidates.csv", staples, products, panel)
    m = Market(data, staples, matches)
    print(f"{len(m.items)} items across {len(m.stores)} stores")

    preds = {}
    for spec in args.preds:
        name, _, file = spec.partition("=")
        path = Path(file) if Path(file).exists() else data / file
        preds[name] = m.load_preds(path)
    if args.test_start:
        test_start = pd.Timestamp(args.test_start)
    elif args.preds:
        first = min(pd.read_parquet(Path(f.partition("=")[2]) if Path(f.partition("=")[2]).exists()
                                    else data / f.partition("=")[2], columns=["date"])["date"].min() for f in args.preds)
        test_start = pd.Timestamp(first)
    else:
        test_start = m.dates[int(len(m.dates) * 0.75)]
    print(f"scenarios start on or after {test_start.date()}")

    scenarios = make_scenarios(m, test_start, args.n_scenarios, args.items_per_list, args.seed)
    results = run(m, scenarios, preds, args.delay)
    results.to_csv(data / "backtest_results.csv", index=False)
    text = summarise(results)
    print("\n" + text)
    (data / "backtest_summary.txt").write_text(text)


if __name__ == "__main__":
    main()
