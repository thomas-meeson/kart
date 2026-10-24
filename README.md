# Smart Basket — where and when to buy your shopping

**Pitch in one line:** give it your shopping list and the shops near you, and it tells you which shop to buy each item at and whether to buy now or wait for a predicted price drop, backed by a backtest on real UK supermarket prices.

**Why it fits HRT's theme:** we predict prices (a transformer over 5 retailers' daily price histories), turn predictions into decisions (a mixed-integer program re-solved daily, like an execution algorithm), and measure the decision honestly. Our purchases don't move supermarket prices, so the backtest is a true out-of-sample test, including "transaction costs" (travel).

---

## Quick start (synthetic data, ~5 minutes, no download needed)

```bash
pip install -r requirements.txt
python -m basket.make_synthetic --out data/synthetic_raw
python -m basket.load_data --input "data/synthetic_raw/*.csv" --out data/processed_syn
python -m basket.features --data data/processed_syn
python -m basket.matching propose --data data/processed_syn --staples data/staples_synthetic.csv
python -m basket.model_xgb --data data/processed_syn --refit-every 45 --n-estimators 200
python -m basket.model_transformer --data data/processed_syn --steps 1500
python -m basket.backtest --data data/processed_syn --staples data/staples_synthetic.csv \
    --preds xgb=preds_xgb.parquet --preds transformer=preds_transformer.parquet
python -m basket.recommend --data data/processed_syn --staples data/staples_synthetic.csv \
    --preds preds_xgb.parquet --items milk_semi_4pt,heinz_beans_415g --visit Tesco=1,Aldi=3 --home Tesco
python -m basket.analysis --data data/processed_syn --staples data/staples_synthetic.csv
```

Synthetic numbers mean nothing. They only prove the pipeline runs.

## Real data

1. Download **"Time Series UK Supermarket Data"** (Declan McAlinden) from Kaggle into `data/raw/`. Daily prices for Aldi, ASDA, Morrisons, Tesco and Sainsbury's from 9 Jan 2024. Check the last date in the files.
   Backup: the Hugging Face mirror `Rif-SQL/time-series-uk-retail-supermarket-price-data` (one parquet, Apache-2.0), but its preview only covers Jan–Apr 2024.
2. See what categories exist, then load a subset (the full dataset is ~100k rows per day; a laptop wants ~5–20k products):
   ```bash
   python -m basket.load_data --input "data/raw/*.csv" --list-categories
   python -m basket.load_data --input "data/raw/*.csv" --out data/processed --categories <a few categories> --min-coverage 0.7
   ```
   If the loader says it can't find a column, add the header to `COLUMN_ALIASES` in `load_data.py`.
3. `python -m basket.features --data data/processed` prints how often each retailer runs promotions. **Sanity-check this first.**
4. Copy `data/staples_example.csv` to `data/staples.csv`, run `matching propose`, then **hand-check `match_candidates.csv`** (set `selected` = 1 on the correct product per item and retailer). This is the most important manual step: wrong matches = wrong answers.
5. Models: `model_xgb` (use `--train-stride 3` if slow), then `model_transformer` (use a GPU/Colab for the real data; same command).
6. `backtest`, `analysis`, `recommend` as above.

## How it works

| Stage | File | What it does |
|---|---|---|
| Load | `load_data.py` | Normalises Kaggle/HF columns, cleans prices, builds a daily panel; gaps ≤ 3 days forward-filled, `observed` flags real data |
| Features | `features.py` | **Infers promotions** (no promo flag in the data): price ≥ 5% below a trailing 42-day reference. ~20 leak-free features + labels |
| Match | `matching.py` | Fuzzy-matches staples to products at each chain; compares on unit price where units line up |
| Baseline | `model_xgb.py` | Multi-output XGBoost for the 14-day price path + classifier for "≥5% cheaper within 14 days"; walk-forward refits |
| Transformer | `model_transformer.py` | Shared encoder over 56-day price sequences + retailer/category embeddings; same outputs and metrics |
| Decide | `optimise.py` | MILP (SciPy/HiGHS): item × store × day, visit costs, need-by days, impatience cost |
| Test | `backtest.py` | Random shoppers on held-out dates; policies scored on real prices that followed |
| Demo | `recommend.py` | JSON for the frontend |
| Insight | `analysis.py` | Lead-lag: do Tesco/Sainsbury's follow Aldi's price moves (Aldi Price Match)? |

### Honesty built in (say this in the pitch)
- **Naive baseline.** Prices are sticky, so "price stays the same" is hard to beat. Every forecast is reported as skill vs that.
- **Purging.** Labels look 14 days ahead, so training data stops 14 days before each test block.
- **Calibration table.** When the model says 30%, does it happen 30% of the time?
- **Oracle gap.** The backtest reports the *most* timing could ever add (perfect foresight vs best-of-today). That's the ceiling; we report what share of it each model captures. This directly answers "do prices change enough to matter?" with a number.
- **Policies:** home store (status quo), compare_today (what a comparison site does), our model, oracle.

### Known limitations
- Promotions are inferred; a permanent price cut looks like a promo until the 42-day window catches up.
- No Clubcard/Nectar prices or multibuys in the data, so loyalty deals are invisible.
- No store locations: "near you" is the set of chains you enter, with your own travel cost per trip.
- Transformer is trained once before the test period; XGBoost refits monthly (stricter on the transformer; use `--refit-every 100000` on XGBoost for a like-for-like comparison).

### If timing turns out small on real data
That's a real finding, not a failure. Two strong extensions:
1. **Stock up:** buy several units of non-perishables when a deep promo is on. That's an inventory problem under price uncertainty, and where promo prediction is worth most.
2. **Promo-ending alerts:** "this is on promo and likely ends within 3 days: buy now."

## Team plan (tonight)

| Who | First job | Then |
|---|---|---|
| ML lead | Run loader + features on real data, check promo summary | XGBoost, then transformer on GPU |
| Physics | Read `optimise.py` + `backtest.py`; pick realistic travel costs and need-by days | Stock-up extension; confidence intervals in the pitch |
| EIE | Data download and loading on the real files; fix any column issues | Wrap `recommend.py` in a small FastAPI endpoint |
| Computing (1st yr) | Hand-check `match_candidates.csv` for all ~30 staples | Lead-lag analysis + charts |
| Design eng | Build UI against `recommend.py` JSON (list in → plan out, price path chart per item) | Pitch deck: problem, model, backtest chart, lead-lag slide |

**Order:** real data loaded + promo summary by ~22:00 → matches checked + XGBoost backtest by ~01:00 → transformer overnight → UI wired up by late morning → freeze code 2h before judging.
