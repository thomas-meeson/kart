"""Load raw supermarket price files into a clean daily panel.

Accepts the Kaggle "Time Series UK Supermarket Data" CSVs (one per retailer) or the
Hugging Face parquet mirror. Column names differ between the two, so headers are
normalised and mapped to one canonical schema:

    retailer, name, date, price, unit_price, unit, category, own_brand

Outputs (in --out):
    products.parquet  one row per product: pid, retailer, name, category, unit, own_brand
    panel.parquet     one row per product per day: pid, date, price, unit_price, observed

Usage:
    python -m basket.load_data --input data/raw/*.csv --list-categories
    python -m basket.load_data --input data/raw/*.csv --out data/processed \
        --categories fresh_food,food_cupboard,drinks --min-coverage 0.6
"""
from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .config import MAX_FFILL_DAYS

# Header after normalisation (lowercase, only a-z0-9 kept) -> canonical column name.
COLUMN_ALIASES = {
    "retailer": {"supermarket", "supermarketname", "retailer", "store"},
    "name": {"names", "name", "productname", "product"},
    "date": {"date", "capturedate", "day"},
    "price": {"prices", "price", "pricegbp", "prices"},
    "unit_price": {"pricesunit", "priceunit", "priceunitgbp", "unitprice", "pricesunitgbp"},
    "unit": {"unit", "units"},
    "category": {"category", "categoryname"},
    "own_brand": {"ownbrand", "isownbrand"},
}

RETAILER_PREFIXES = {
    "aldi": "Aldi",
    "asda": "ASDA",
    "morr": "Morrisons",
    "sain": "Sainsburys",
    "tesc": "Tesco",
}


def _norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(h).lower())


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the canonical schema. Fails loudly if a required column is missing."""
    rename = {}
    for col in df.columns:
        key = _norm_header(col)
        for canon, aliases in COLUMN_ALIASES.items():
            if key in aliases and canon not in rename.values():
                rename[col] = canon
                break
    df = df.rename(columns=rename)
    required = ["retailer", "name", "date", "price"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find columns {missing}. Columns in file: {list(df.columns)}. "
            "Add the header to COLUMN_ALIASES in load_data.py."
        )
    for opt in ["unit_price", "unit", "category", "own_brand"]:
        if opt not in df.columns:
            df[opt] = np.nan
    return df[["retailer", "name", "date", "price", "unit_price", "unit", "category", "own_brand"]]


def parse_dates(s: pd.Series) -> pd.Series:
    """Handles 20240109-style integers/strings and ordinary date strings/timestamps."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize(None).dt.normalize() if s.dt.tz is not None else s.dt.normalize()
    as_str = s.astype(str).str.strip()
    if as_str.str.fullmatch(r"\d{8}(\.0)?").all():
        return pd.to_datetime(as_str.str[:8], format="%Y%m%d")
    out = pd.to_datetime(as_str, errors="coerce", utc=True).dt.tz_localize(None)
    return out.dt.normalize()


def normalise_retailer(s: pd.Series) -> pd.Series:
    low = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    for prefix, canon in RETAILER_PREFIXES.items():
        out[low.str.startswith(prefix)] = canon
    unknown = low[out.isna()].unique()
    if len(unknown):
        raise ValueError(f"Unrecognised retailer names: {unknown[:10]}. Add them to RETAILER_PREFIXES.")
    return out


def _parse_bool(s: pd.Series) -> pd.Series:
    low = s.astype(str).str.strip().str.lower()
    return low.isin(["true", "1", "yes", "y", "t"])


def _clean_chunk(df: pd.DataFrame, categories: set[str] | None, name_regex: str | None) -> pd.DataFrame:
    df = normalise_columns(df)
    if categories is not None:
        df = df[df["category"].astype(str).isin(categories)]
    if name_regex:
        df = df[df["name"].astype(str).str.contains(name_regex, case=False, regex=True, na=False)]
    if df.empty:
        return df
    df = df.copy()
    df["retailer"] = normalise_retailer(df["retailer"])
    df["name"] = df["name"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    df["date"] = parse_dates(df["date"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["unit_price"] = pd.to_numeric(df["unit_price"], errors="coerce")
    df["unit"] = df["unit"].astype(str).str.strip().str.lower()
    df["own_brand"] = _parse_bool(df["own_brand"])
    df = df[df["price"].gt(0) & df["date"].notna()]
    return df


def read_raw(paths: list[str], categories: set[str] | None = None, name_regex: str | None = None,
             chunksize: int = 1_000_000) -> pd.DataFrame:
    parts = []
    for p in paths:
        if p.endswith(".parquet"):
            pf = pq.ParquetFile(p)
            for batch in pf.iter_batches(batch_size=chunksize):
                parts.append(_clean_chunk(batch.to_pandas(), categories, name_regex))
        else:
            for chunk in pd.read_csv(p, chunksize=chunksize, dtype=str, encoding_errors="replace"):
                parts.append(_clean_chunk(chunk, categories, name_regex))
        print(f"read {p}")
    parts = [x for x in parts if not x.empty]
    if not parts:
        raise ValueError("No rows left after filtering. Check --categories with --list-categories.")
    return pd.concat(parts, ignore_index=True)


def list_categories(paths: list[str]) -> pd.DataFrame:
    counts = {}
    for p in paths:
        it = (b.to_pandas() for b in pq.ParquetFile(p).iter_batches(batch_size=1_000_000)) \
            if p.endswith(".parquet") else pd.read_csv(p, chunksize=1_000_000, dtype=str)
        for chunk in it:
            chunk = normalise_columns(chunk)
            for (r, c), n in chunk.groupby(["retailer", "category"], dropna=False).size().items():
                counts[(r, c)] = counts.get((r, c), 0) + n
    out = pd.Series(counts, name="rows").rename_axis(["retailer", "category"]).reset_index()
    return out.sort_values("rows", ascending=False)


def build_panel(raw: pd.DataFrame, min_coverage: float = 0.6, min_days: int = 60,
                max_products: int | None = None, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn raw scraped rows into a daily panel.

    - Duplicate rows for the same product and day are collapsed to the median price.
    - Each product is reindexed to every day between its first and last observation.
    - Gaps of up to MAX_FFILL_DAYS are forward-filled; `observed` marks real observations.
    - Prices more than 5x away from the product's median are treated as scrape errors.
    - Products seen on fewer than `min_coverage` of the days in their span are dropped.
    """
    raw = (raw.groupby(["retailer", "name", "date"], observed=True, sort=False)
              .agg(price=("price", "median"), unit_price=("unit_price", "median"),
                   unit=("unit", "last"), category=("category", "last"), own_brand=("own_brand", "last"))
              .reset_index())

    products = (raw.sort_values("date").groupby(["retailer", "name"], sort=True)
                   .agg(category=("category", "last"), unit=("unit", "last"), own_brand=("own_brand", "last"),
                        first=("date", "min"), last=("date", "max"), n_obs=("date", "size"))
                   .reset_index())
    span = (products["last"] - products["first"]).dt.days + 1
    products["coverage"] = products["n_obs"] / span
    keep = (products["coverage"] >= min_coverage) & (span >= min_days)
    print(f"products: {len(products):,} total, {keep.sum():,} kept "
          f"(coverage >= {min_coverage}, span >= {min_days} days)")
    products = products[keep]
    if max_products and len(products) > max_products:
        products = products.sample(max_products, random_state=seed)
    products = products.reset_index(drop=True)
    products["pid"] = np.arange(len(products), dtype=np.int32)

    raw = raw.merge(products[["retailer", "name", "pid"]], on=["retailer", "name"], how="inner")
    dates = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")

    def wide(col: str) -> pd.DataFrame:
        w = raw.pivot(index="date", columns="pid", values=col).reindex(index=dates)
        return w.reindex(columns=products["pid"]).astype("float32")

    price_w = wide("price")
    unit_w = wide("unit_price")

    # Scrape errors: implausible jumps relative to the product's own median.
    med = np.nanmedian(price_w.to_numpy(), axis=0)
    bad = (price_w > 5 * med) | (price_w < med / 5)
    n_bad = int(bad.to_numpy().sum())
    if n_bad:
        print(f"masked {n_bad:,} implausible prices as missing")
    price_w = price_w.mask(bad)
    unit_w = unit_w.mask(bad)

    observed = price_w.notna()
    price_ff = price_w.ffill(limit=MAX_FFILL_DAYS)
    unit_ff = unit_w.ffill(limit=MAX_FFILL_DAYS)

    # Don't extend a product past its last real observation.
    last_obs = pd.Series(products["last"].to_numpy(), index=products["pid"])
    after_last = pd.DataFrame(dates.to_numpy()[:, None] > last_obs.to_numpy()[None, :],
                              index=dates, columns=price_ff.columns)
    price_ff = price_ff.mask(after_last)
    unit_ff = unit_ff.mask(after_last)

    arr = price_ff.to_numpy()
    di, pj = np.nonzero(~np.isnan(arr))
    panel = pd.DataFrame({
        "pid": price_ff.columns.to_numpy()[pj].astype(np.int32),
        "date": dates[di],
        "price": arr[di, pj],
        "unit_price": unit_ff.to_numpy()[di, pj],
        "observed": observed.to_numpy()[di, pj],
    }).sort_values(["pid", "date"]).reset_index(drop=True)

    products = products.drop(columns=["first", "last"])
    print(f"panel: {len(panel):,} rows, {panel['pid'].nunique():,} products, "
          f"{dates.min().date()} to {dates.max().date()}")
    return products, panel


def to_wide(panel: pd.DataFrame, col: str = "price") -> pd.DataFrame:
    """Dates x pids matrix (NaN where missing). Used by features, models and the backtest."""
    w = panel.pivot(index="date", columns="pid", values=col)
    full = pd.date_range(w.index.min(), w.index.max(), freq="D")
    return w.reindex(full)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", required=True, help="CSV or parquet files (globs allowed)")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--categories", default=None, help="comma-separated category names to keep")
    ap.add_argument("--name-regex", default=None, help="only keep products whose name matches")
    ap.add_argument("--min-coverage", type=float, default=0.6)
    ap.add_argument("--min-days", type=int, default=60)
    ap.add_argument("--max-products", type=int, default=None, help="random subsample for quick runs")
    ap.add_argument("--list-categories", action="store_true")
    args = ap.parse_args()

    paths = sorted({p for pattern in args.input for p in glob.glob(pattern)})
    if not paths:
        raise SystemExit(f"No files matched {args.input}")

    if args.list_categories:
        print(list_categories(paths).to_string(index=False))
        return

    cats = set(args.categories.split(",")) if args.categories else None
    raw = read_raw(paths, cats, args.name_regex)
    print(f"raw rows after filtering: {len(raw):,}")
    products, panel = build_panel(raw, args.min_coverage, args.min_days, args.max_products)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    products.to_parquet(out / "products.parquet", index=False)
    panel.to_parquet(out / "panel.parquet", index=False)
    print(f"wrote {out / 'products.parquet'} and {out / 'panel.parquet'}")


if __name__ == "__main__":
    main()
