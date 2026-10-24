"""Generate fake data in the same format as the Kaggle CSVs, to test the pipeline end to end.

Not for results. It has the structures we expect in the real data (sticky regular prices,
promotions that start on a fixed weekday and last whole weeks, Aldi price matching with a lag,
missing scrape days) so every stage can be checked before the real download finishes.

Usage:
    python -m basket.make_synthetic --out data/synthetic_raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OWN_PREFIX = {"Tesco": "Tesco", "Sainsburys": "Sainsbury's", "ASDA": "ASDA", "Morrisons": "Morrisons", "Aldi": "Aldi"}
FILE_LABEL = {"Tesco": "Tesco", "Sainsburys": "Sains", "ASDA": "ASDA", "Morrisons": "Morrisons", "Aldi": "Aldi"}
LEVEL = {"Aldi": 0.86, "ASDA": 0.95, "Tesco": 1.0, "Morrisons": 1.02, "Sainsburys": 1.05}
PROMO_RATE = {"Aldi": 0.2, "ASDA": 0.6, "Tesco": 1.0, "Morrisons": 0.9, "Sainsburys": 1.0}

# (template, category, unit, qty_for_unit_price, base_price, branded)
STAPLES = [
    ("{own} Semi Skimmed Milk 2.272L 4 Pints", "fresh_food", "l", 2.272, 1.65, False),
    ("{own} British Free Range Eggs 12 Pack", "fresh_food", "unit", 12, 3.10, False),
    ("{own} Fusilli Pasta 500G", "food_cupboard", "kg", 0.5, 0.75, False),
    ("{own} Basmati Rice 1kg", "food_cupboard", "kg", 1.0, 1.95, False),
    ("{own} Chopped Tomatoes 400G", "food_cupboard", "kg", 0.4, 0.55, False),
    ("{own} British Chicken Breast Fillets 640G", "fresh_food", "kg", 0.64, 4.40, False),
    ("{own} Beef Mince 5% Fat 500G", "fresh_food", "kg", 0.5, 3.50, False),
    ("{own} Salted Block Butter 250G", "fresh_food", "kg", 0.25, 2.05, False),
    ("{own} Mature Cheddar 400G", "fresh_food", "kg", 0.4, 3.15, False),
    ("{own} Greek Style Natural Yogurt 500G", "fresh_food", "kg", 0.5, 1.30, False),
    ("{own} White Potatoes 2.5kg", "fresh_food", "kg", 2.5, 1.45, False),
    ("{own} Plain Flour 1.5kg", "food_cupboard", "kg", 1.5, 0.95, False),
    ("{own} Smooth Orange Juice 1L", "drinks", "l", 1.0, 1.20, False),
    ("Heinz Baked Beanz In Tomato Sauce 415G", "food_cupboard", "kg", 0.415, 1.40, True),
    ("Warburtons Toastie White Bread 800G", "fresh_food", "kg", 0.8, 1.65, True),
    ("PG Tips 160 Tea Bags 464G", "drinks", "unit", 160, 5.25, True),
    ("Hellmann's Real Mayonnaise 400ml", "food_cupboard", "l", 0.4, 3.20, True),
    ("Kellogg's Corn Flakes 500G", "food_cupboard", "kg", 0.5, 2.60, True),
    ("Lurpak Slightly Salted Spreadable 500G", "fresh_food", "kg", 0.5, 5.25, True),
    ("Coca-Cola Original Taste 2L", "drinks", "l", 2.0, 2.25, True),
    ("McVitie's Digestives Biscuits 400G", "food_cupboard", "kg", 0.4, 1.95, True),
    ("Cathedral City Mature Cheddar 350G", "fresh_food", "kg", 0.35, 3.85, True),
    ("Andrex Classic Clean Toilet Tissue 9 Rolls", "household", "unit", 9, 6.50, True),
    ("Fairy Original Washing Up Liquid 640ml", "household", "l", 0.64, 2.50, True),
    ("Nescafe Original Instant Coffee 200G", "drinks", "kg", 0.2, 6.80, True),
]
STAPLE_ITEMS = [
    ("milk_semi_4pt", "semi skimmed milk 4 pints", 1), ("eggs_free_range_12", "free range eggs 12", 7),
    ("pasta_fusilli_500g", "fusilli pasta 500g", 14), ("rice_basmati_1kg", "basmati rice 1kg", 14),
    ("tomatoes_chopped_400g", "chopped tomatoes 400g", 14), ("chicken_breast_640g", "chicken breast fillets 640g", 2),
    ("beef_mince_500g", "beef mince 5% fat 500g", 2), ("butter_salted_250g", "salted butter 250g", 7),
    ("cheddar_mature_400g", "mature cheddar 400g", 7), ("yogurt_greek_500g", "greek style yogurt 500g", 5),
    ("potatoes_2_5kg", "white potatoes 2.5kg", 5), ("flour_plain_1_5kg", "plain flour 1.5kg", 14),
    ("orange_juice_1l", "smooth orange juice 1l", 5), ("heinz_beans_415g", "heinz baked beanz 415g", 14),
    ("warburtons_toastie_800g", "warburtons toastie white bread 800g", 2), ("pg_tips_160", "pg tips 160 tea bags", 14),
    ("hellmanns_mayo_400ml", "hellmann's real mayonnaise 400ml", 14), ("kelloggs_cornflakes_500g", "kellogg's corn flakes 500g", 14),
    ("lurpak_spreadable_500g", "lurpak slightly salted spreadable 500g", 7), ("coca_cola_2l", "coca-cola original 2l", 14),
    ("mcvities_digestives_400g", "mcvitie's digestives 400g", 14), ("cathedral_city_350g", "cathedral city mature cheddar 350g", 7),
    ("andrex_9_rolls", "andrex classic clean toilet tissue 9 rolls", 14), ("fairy_640ml", "fairy original washing up liquid 640ml", 14),
    ("nescafe_200g", "nescafe original instant coffee 200g", 14),
]
FILLER_WORDS = ["Chocolate", "Strawberry", "Vanilla", "Garlic", "Smoked", "Spicy", "Organic", "Mini", "Large", "Crunchy"]
FILLER_NOUNS = ["Muffins", "Crisps", "Soup", "Sausages", "Pizza", "Granola", "Cookies", "Noodles", "Hummus", "Pesto",
                "Ice Cream", "Fish Fingers", "Coleslaw", "Croissants", "Lasagne"]
FILLER_CATS = ["fresh_food", "food_cupboard", "frozen", "drinks", "household"]


def simulate_prices(base: float, retailer: str, branded: bool, n_days: int, wednesday_idx: np.ndarray,
                    rng: np.random.Generator, aldi_regular: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (price, regular_price) arrays of length n_days."""
    reg = np.empty(n_days)
    level = base * LEVEL[retailer] * rng.uniform(0.95, 1.05)
    lag = int(rng.integers(3, 11))
    for t in range(n_days):
        if aldi_regular is not None:                               # Aldi Price Match, with a lag
            level = aldi_regular[max(t - lag, 0)]
        elif rng.random() < 1 / 110:                                # occasional regular price change
            level *= rng.choice([0.95, 1.04, 1.06, 1.08], p=[0.15, 0.4, 0.3, 0.15])
        reg[t] = round(level, 2)
    price = reg.copy()
    rate = PROMO_RATE[retailer] * (1.0 if branded else 0.25) / 45
    rate *= rng.uniform(0.5, 1.5)
    t, cooldown_until = 0, 0
    while t < n_days:
        if t >= cooldown_until and wednesday_idx[t] and rng.random() < rate * 7:
            dur = int(rng.choice([7, 14, 21], p=[0.3, 0.5, 0.2]))
            depth = float(rng.choice([0.15, 0.25, 0.33, 0.5], p=[0.35, 0.35, 0.2, 0.1]))
            price[t:t + dur] = np.round(reg[t:t + dur] * (1 - depth), 2)
            cooldown_until = t + dur + int(rng.integers(14, 42))
            t += dur
        else:
            t += 1
    return price, reg


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic_raw")
    ap.add_argument("--staples-out", default="data/staples_synthetic.csv")
    ap.add_argument("--start", default="2024-01-09")
    ap.add_argument("--end", default="2025-06-30")
    ap.add_argument("--fillers", type=int, default=120, help="extra random products per retailer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    dates = pd.date_range(args.start, args.end, freq="D")
    n = len(dates)
    wed = (dates.dayofweek == 2)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    aldi_reg = {}
    retailers = ["Aldi", "ASDA", "Morrisons", "Sainsburys", "Tesco"]       # Aldi first: others match it
    matched_staples = set(rng.choice(13, 7, replace=False).tolist())     # own-brand lines in price match
    for retailer in retailers:
        rows = []
        retailer_down = rng.random(n) < 0.02                              # whole-site scrape failures
        products = []
        for j, (tmpl, cat, unit, qty, base, branded) in enumerate(STAPLES):
            if retailer == "Aldi" and branded and rng.random() < 0.5:
                continue                                                  # Aldi stocks fewer brands
            name = tmpl.format(own=OWN_PREFIX[retailer])
            match = (retailer in ("Tesco", "Sainsburys") and not branded and j in matched_staples)
            price, reg = simulate_prices(base, retailer, branded, n, wed, rng, aldi_reg.get(j) if match else None)
            if retailer == "Aldi":
                aldi_reg[j] = reg
            products.append((name, cat, unit, qty, not branded, price))
        for _ in range(args.fillers):
            name = f"{OWN_PREFIX[retailer]} {rng.choice(FILLER_WORDS)} {rng.choice(FILLER_NOUNS)} {int(rng.integers(2, 13))} Pack"
            branded = rng.random() < 0.4
            base = float(rng.uniform(0.8, 6.0))
            price, _ = simulate_prices(base, retailer, branded, n, wed, rng)
            products.append((name, str(rng.choice(FILLER_CATS)), "unit", 1.0, not branded, price))
        for name, cat, unit, qty, own, price in products:
            seen = ~retailer_down & (rng.random(n) > 0.01)
            for t in np.nonzero(seen)[0]:
                rows.append((FILE_LABEL[retailer], f"{price[t]:.2f}", f"{price[t] / qty:.4f}", unit, name,
                             int(dates[t].strftime("%Y%m%d")), cat, own))
        df = pd.DataFrame(rows, columns=["supermarket", "prices_(£)", "prices_unit_(£)", "unit", "names",
                                         "date", "category", "own_brand"]).drop_duplicates(["names", "date"])
        path = out / f"All_Data_{FILE_LABEL[retailer]}.csv"
        df.to_csv(path, index=False)
        print(f"wrote {path}: {len(df):,} rows")

    staples = pd.DataFrame([
        {"item": item, "query": q, "category": STAPLES[i][1], "unit": STAPLES[i][2], "qty": STAPLES[i][3],
         "shelf_days": shelf}
        for i, (item, q, shelf) in enumerate(STAPLE_ITEMS)
    ])
    staples_path = Path(args.staples_out)
    staples_path.parent.mkdir(parents=True, exist_ok=True)
    staples.to_csv(staples_path, index=False)
    print(f"wrote {staples_path}")


if __name__ == "__main__":
    main()
