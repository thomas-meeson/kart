"""Match a list of generic shopping items ("staples") to real products at each retailer.

Automatic matching across retailers is hard (different names and pack sizes), so this
script proposes the top candidates and a human confirms them. Workflow:

    1. Write data/staples.csv (see data/staples_example.csv): item, query, category, unit, qty, shelf_days
    2. python -m basket.matching propose --data data/processed --staples data/staples.csv
       -> writes data/processed/match_candidates.csv with the top 3 candidates per item and retailer
    3. Open it, check each item, and set `selected` to 1 for the right product (0 for the rest).
       If none is right, leave all 0 and that item is treated as unavailable at that retailer.
    4. Everything downstream reads the confirmed file through load_matches().

Comparing on unit price: if the staple says unit=l, qty=2.272 and the product's unit is "l",
the cost used is unit_price * qty, so a 1-pint and a 4-pint milk compare fairly.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# Words that identify the retailer rather than the product. Extend as you find more.
RETAILER_WORDS = {
    "tesco", "tescos", "sainsbury", "sainsburys", "asda", "morrisons", "aldi", "finest", "taste",
    "difference", "extra", "special", "everyday", "essentials", "savers", "hubbards", "stockwell",
    "the", "and", "with", "of",
}

PINT_LITRES = 0.568


def _short_unit(u: str) -> str:
    return "l" if u.startswith("lit") or u == "ltr" else u


def normalise_name(name: str) -> str:
    """Lowercase, standardise sizes (4 pints -> 2.272l, 1 litre -> 1l), drop retailer words."""
    s = str(name).lower().replace("&", " and ")
    s = re.sub(r"(\d+)\s*pints?\b", lambda m: f"{int(m.group(1)) * PINT_LITRES:.3f}l", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*(kg|ml|litres?|ltr|g|l)\b", lambda m: f"{m.group(1)}{_short_unit(m.group(2))}", s)
    s = re.sub(r"[^a-z0-9. ]", " ", s)
    words = [w for w in s.split() if w not in RETAILER_WORDS]
    return " ".join(words)


def _latest_prices(panel: pd.DataFrame) -> pd.DataFrame:
    last = panel.sort_values("date").groupby("pid").tail(1).set_index("pid")
    ratio = (panel["unit_price"] / panel["price"]).groupby(panel["pid"]).median()
    return last[["price", "unit_price"]].assign(unit_per_price=ratio)


def propose(products: pd.DataFrame, panel: pd.DataFrame, staples: pd.DataFrame,
            top_k: int = 3, min_score: float = 70.0) -> pd.DataFrame:
    latest = _latest_prices(panel)
    prods = products.join(latest, on="pid")
    prods["norm"] = prods["name"].map(normalise_name)
    rows = []
    for _, st in staples.iterrows():
        query = normalise_name(st["query"])
        for retailer, group in prods.groupby("retailer"):
            cand = group
            if isinstance(st.get("category"), str) and st["category"]:
                in_cat = cand[cand["category"] == st["category"]]
                cand = in_cat if len(in_cat) else cand
            if cand.empty:
                continue
            hits = process.extract(query, cand["norm"].tolist(), scorer=fuzz.token_set_ratio, limit=top_k)
            for rank, (_, score, pos) in enumerate(hits, start=1):
                p = cand.iloc[pos]
                rows.append({
                    "item": st["item"], "retailer": retailer, "rank": rank, "pid": int(p["pid"]),
                    "name": p["name"], "score": round(score, 1), "unit": p["unit"],
                    "price": p["price"], "unit_price": p["unit_price"],
                    "selected": int(rank == 1 and score >= min_score),
                })
    return pd.DataFrame(rows)


def load_matches(match_path: str | Path, staples: pd.DataFrame, products: pd.DataFrame,
                 panel: pd.DataFrame) -> pd.DataFrame:
    """Confirmed matches with a cost multiplier so that cost_t = price_t * multiplier.

    Returns columns: item, retailer, pid, multiplier, shelf_days.
    """
    m = pd.read_csv(match_path)
    m = m[m["selected"] == 1]
    dup = m.duplicated(["item", "retailer"], keep=False)
    if dup.any():
        raise ValueError(f"More than one product selected for:\n{m[dup][['item', 'retailer', 'name']]}")
    ratio = (panel["unit_price"] / panel["price"]).groupby(panel["pid"]).median()
    m = m.merge(staples[["item", "unit", "qty", "shelf_days"]], on="item", how="left", suffixes=("_prod", ""))
    prod_unit = products.set_index("pid")["unit"]
    mult = []
    for _, r in m.iterrows():
        same_unit = str(prod_unit.get(r["pid"], "")).lower() == str(r["unit"]).lower()
        rr = ratio.get(r["pid"], np.nan)
        if same_unit and pd.notna(r["qty"]) and pd.notna(rr) and rr > 0:
            mult.append(float(rr * r["qty"]))     # unit_price * qty, expressed per unit of pack price
        else:
            mult.append(1.0)                      # fall back to comparing pack prices
    m["multiplier"] = mult
    return m[["item", "retailer", "pid", "multiplier", "shelf_days"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["propose"])
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--staples", default="data/staples.csv")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--min-score", type=float, default=70.0)
    args = ap.parse_args()
    data = Path(args.data)
    products = pd.read_parquet(data / "products.parquet")
    panel = pd.read_parquet(data / "panel.parquet")
    staples = pd.read_csv(args.staples)
    out = propose(products, panel, staples, args.top_k, args.min_score)
    path = data / "match_candidates.csv"
    out.to_csv(path, index=False)
    n_items = staples["item"].nunique()
    auto = out[out["selected"] == 1].groupby("item")["retailer"].nunique()
    print(f"wrote {path}. Auto-selected matches for {len(auto)}/{n_items} items; "
          f"average {auto.mean():.1f} retailers per item. Now check it by hand.")


if __name__ == "__main__":
    main()
