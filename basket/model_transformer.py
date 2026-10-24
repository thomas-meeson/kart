"""Transformer over each product's recent price history.

Input for product n on day t: the last SEQ_LEN days as a sequence of per-day vectors
    [log(p[d]/p[t]), observed?, on promo?, promo depth, retailer promo share, sin/cos day-of-week]
plus static embeddings (retailer, category, own brand, price level) added to a [CLS] token.
One model is shared across all products, so it can learn patterns like
"this retailer's promos last about three weeks" from thousands of series at once.

Outputs, in the same format as model_xgb so the backtest can use either:
    pred_lr_1..H   expected log(p[t+k]/p[t])     (trained with MSE, so it estimates the mean)
    pred_cheaper   P(price drops >= CHEAPER_DELTA within HORIZON days)

Training data stops HORIZON days before --test-start (purged), validation is the last part
of the training period, and the best checkpoint by validation loss is kept.

Usage:
    python -m basket.model_transformer --data data/processed --test-start 2025-07-01 --steps 4000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import CHEAPER_DELTA, HORIZON, SEQ_LEN
from .evaluate import calibration_table, classification_report, regression_report
from .features import compute_wide

N_CH = 7
TARGET_SCALE = 10.0   # log ratios are small; scale them up for the loss


class SeriesData:
    """Holds dates x products arrays and builds batches of (input window, labels) by gathering."""

    def __init__(self, wide: dict[str, pd.DataFrame], products: pd.DataFrame):
        P = wide["price"]
        self.dates = P.index
        self.pids = P.columns.to_numpy()
        pad = np.full((HORIZON, P.shape[1]), np.nan, dtype=np.float32)
        self.logp = np.vstack([np.log(P.to_numpy(dtype=np.float32)), pad])          # (T+H, N)
        self.promo = np.nan_to_num(wide["on_promo"].to_numpy(dtype=np.float32))
        self.depth = np.clip(np.nan_to_num(wide["depth"].to_numpy(dtype=np.float32)), -0.5, 0.9)
        self.rshare = np.nan_to_num(wide["retailer_promo_share"].to_numpy(dtype=np.float32))
        dow = self.dates.dayofweek.to_numpy()
        self.dow = np.stack([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)], 1).astype(np.float32)
        meta = products.set_index("pid").loc[self.pids]
        self.ret_codes, self.ret_names = pd.factorize(meta["retailer"].astype(str))
        self.cat_codes, self.cat_names = pd.factorize(meta["category"].astype(str))
        self.own = meta["own_brand"].astype(bool).to_numpy().astype(np.int64)
        self.T = len(self.dates)

    def index_pairs(self, t_from: int, t_to: int, need_label: bool) -> np.ndarray:
        """All (t, n) with t in [t_from, t_to), a full input window, and a price on day t."""
        t_from = max(t_from, SEQ_LEN - 1)
        t_idx, n_idx = np.nonzero(~np.isnan(self.logp[t_from:t_to]))
        t_idx = t_idx + t_from
        pairs = np.stack([t_idx, n_idx], 1)
        if need_label:
            has_next = ~np.isnan(self.logp[pairs[:, 0] + 1, pairs[:, 1]])
            pairs = pairs[has_next]
        return pairs

    def batch(self, pairs: np.ndarray):
        t, n = pairs[:, 0], pairs[:, 1]
        offs = np.arange(-SEQ_LEN + 1, HORIZON + 1)
        tt = t[:, None] + offs[None, :]                                      # (B, L+H)
        lp = self.logp[tt, n[:, None]]
        cur = lp[:, SEQ_LEN - 1:SEQ_LEN]
        hist = lp[:, :SEQ_LEN] - cur
        obs = ~np.isnan(hist)
        th = tt[:, :SEQ_LEN]
        x = np.stack([
            np.nan_to_num(hist), obs.astype(np.float32),
            self.promo[th, n[:, None]], self.depth[th, n[:, None]], self.rshare[th, n[:, None]],
            self.dow[th, 0], self.dow[th, 1],
        ], axis=-1).astype(np.float32)
        y_lr = lp[:, SEQ_LEN:] - cur                                          # (B, H), NaN if unknown
        fut = np.exp(lp[:, SEQ_LEN:])
        n_fut = np.sum(~np.isnan(fut), 1)
        fmin = np.nanmin(np.where(np.isnan(fut), np.inf, fut), 1)
        y_ch = (fmin <= np.exp(cur[:, 0]) * (1 - CHEAPER_DELTA)).astype(np.float32)
        y_ch[n_fut < HORIZON - 3] = np.nan
        return {
            "x": torch.from_numpy(x),
            "ret": torch.from_numpy(self.ret_codes[n].astype(np.int64)),
            "cat": torch.from_numpy(self.cat_codes[n].astype(np.int64)),
            "own": torch.from_numpy(self.own[n]),
            "level": torch.from_numpy(cur[:, 0].astype(np.float32)),
            "y_lr": torch.from_numpy(y_lr.astype(np.float32)),
            "y_ch": torch.from_numpy(y_ch),
        }


class PriceTransformer(nn.Module):
    def __init__(self, n_ret: int, n_cat: int, d: int = 64, heads: int = 4, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(N_CH, d)
        self.pos = nn.Parameter(torch.randn(1, SEQ_LEN + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.ret_emb = nn.Embedding(n_ret, d)
        self.cat_emb = nn.Embedding(n_cat, d)
        self.own_emb = nn.Embedding(2, d)
        self.level = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, HORIZON + 1))

    def forward(self, x, ret, cat, own, level):
        static = self.ret_emb(ret) + self.cat_emb(cat) + self.own_emb(own) + self.level(level[:, None])
        h = torch.cat([self.cls + static[:, None, :], self.inp(x)], dim=1) + self.pos
        out = self.head(self.enc(h)[:, 0])
        return out[:, :HORIZON], out[:, HORIZON]


def _loss(model, b, device):
    lr_hat, ch_logit = model(b["x"].to(device), b["ret"].to(device), b["cat"].to(device),
                             b["own"].to(device), b["level"].to(device))
    y_lr, y_ch = b["y_lr"].to(device), b["y_ch"].to(device)
    m = ~torch.isnan(y_lr)
    mse = ((lr_hat - torch.nan_to_num(y_lr) * TARGET_SCALE) ** 2)[m].mean()
    mc = ~torch.isnan(y_ch)
    bce = nn.functional.binary_cross_entropy_with_logits(ch_logit[mc], y_ch[mc]) if mc.any() else torch.zeros((), device=device)
    return mse + bce, mse.item(), bce.item()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(data: SeriesData, test_start: pd.Timestamp, steps: int = 4000, batch_size: int = 512,
          lr: float = 1e-3, val_frac: float = 0.1, seed: int = 0, device=None, log_every: int = 250):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or pick_device()
    t_test = int(np.searchsorted(data.dates, test_start))
    t_cut = t_test - HORIZON                       # purge: labels must not reach the test period
    t_val = int(t_cut - (t_cut - SEQ_LEN) * val_frac)
    train_pairs = data.index_pairs(0, t_val - HORIZON, need_label=True)
    val_pairs = data.index_pairs(t_val, t_cut, need_label=True)
    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise ValueError("Not enough history before --test-start for training and validation.")
    val_pairs = val_pairs[rng.permutation(len(val_pairs))[:20000]]
    print(f"device {device}; train pairs {len(train_pairs):,} (days < {data.dates[t_val - HORIZON].date()}), "
          f"val pairs {len(val_pairs):,}")

    model = PriceTransformer(len(data.ret_names), len(data.cat_names)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    best, best_state = np.inf, None
    for step in range(1, steps + 1):
        model.train()
        b = data.batch(train_pairs[rng.integers(0, len(train_pairs), batch_size)])
        loss, _, _ = _loss(model, b, device)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % log_every == 0 or step == steps:
            model.eval()
            with torch.no_grad():
                vl = [_loss(model, data.batch(val_pairs[i:i + 2048]), device) for i in range(0, len(val_pairs), 2048)]
            v = float(np.mean([x[0].item() for x in vl]))
            print(f"step {step}: train {loss.item():.4f}  val {v:.4f}  "
                  f"(mse {np.mean([x[1] for x in vl]):.4f}, bce {np.mean([x[2] for x in vl]):.4f})")
            if v < best:
                best, best_state = v, {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, data: SeriesData, pairs: np.ndarray, device=None, batch_size: int = 4096) -> pd.DataFrame:
    device = device or next(model.parameters()).device
    model.eval()
    lr_out, ch_out = [], []
    for i in range(0, len(pairs), batch_size):
        b = data.batch(pairs[i:i + batch_size])
        lr_hat, ch_logit = model(b["x"].to(device), b["ret"].to(device), b["cat"].to(device),
                                 b["own"].to(device), b["level"].to(device))
        lr_out.append((lr_hat / TARGET_SCALE).cpu().numpy())
        ch_out.append(torch.sigmoid(ch_logit).cpu().numpy())
    out = pd.DataFrame({"date": data.dates[pairs[:, 0]], "pid": data.pids[pairs[:, 1]].astype(np.int32)})
    out[[f"pred_lr_{k}" for k in range(1, HORIZON + 1)]] = np.concatenate(lr_out)
    out["pred_cheaper"] = np.concatenate(ch_out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--test-start", default=None, help="YYYY-MM-DD (default: last 25%% of dates)")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_dir = Path(args.data)
    products = pd.read_parquet(data_dir / "products.parquet")
    panel = pd.read_parquet(data_dir / "panel.parquet")
    wide = compute_wide(panel, products)
    data = SeriesData(wide, products)
    test_start = pd.Timestamp(args.test_start) if args.test_start else data.dates[int(data.T * 0.75)]
    print(f"test period starts {test_start.date()}")

    model = train(data, test_start, args.steps, args.batch_size, args.lr)
    torch.save(model.state_dict(), data_dir / "transformer.pt")

    t_test = int(np.searchsorted(data.dates, test_start))
    test_pairs = data.index_pairs(t_test, data.T, need_label=False)
    preds = predict(model, data, test_pairs)
    preds.to_parquet(data_dir / "preds_transformer.parquet", index=False)
    print(f"wrote {data_dir / 'preds_transformer.parquet'}: {len(preds):,} rows")

    # Score on exactly the same labels as XGBoost.
    table = pd.read_parquet(data_dir / "features.parquet",
                            columns=["date", "pid", "y_cheaper"] + [f"y_lr_{k}" for k in range(1, HORIZON + 1)])
    base_rate = float(table.loc[table["date"] < test_start - pd.Timedelta(days=HORIZON), "y_cheaper"].mean())
    merged = preds.merge(table, on=["date", "pid"])
    lr_cols = [f"y_lr_{k}" for k in range(1, HORIZON + 1)]
    pr_cols = [f"pred_lr_{k}" for k in range(1, HORIZON + 1)]
    reg = regression_report(merged[lr_cols].to_numpy(), merged[pr_cols].to_numpy())
    clf = classification_report(merged["y_cheaper"].to_numpy(), merged["pred_cheaper"].to_numpy(), base_rate)
    print("\nPrice path forecast (log ratio), skill vs 'no change':")
    print(reg.to_string(index=False))
    print("\nWill it be >= 5% cheaper within the horizon?")
    print(json.dumps(clf, indent=2))
    print("\nCalibration:")
    print(calibration_table(merged["y_cheaper"].to_numpy(), merged["pred_cheaper"].to_numpy()).to_string())
    with open(data_dir / "report_transformer.json", "w") as fh:
        json.dump({"classification": clf, "regression": reg.to_dict(orient="records"),
                   "test_start": str(test_start.date())}, fh, indent=2)


if __name__ == "__main__":
    main()
