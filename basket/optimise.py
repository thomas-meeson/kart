"""Decide where and when to buy each item on a shopping list.

Mixed-integer program over items i, stores s and days d (0 = today):

    minimise   sum_isd cost[i,s,d] * x[i,s,d]          expected price paid
             + sum_sd  visit[s] * v[s,d]               travel cost for each store visit
             + delay * sum_isd d * x[i,s,d]            small impatience cost per day of waiting
    subject to sum_sd x[i,s,d] = 1  for each item       buy every item exactly once
               d <= needby[i]                          perishables can't wait
               x[i,s,d] <= v[s,d]                      buying needs a visit
               x, v binary

Day 0 costs are today's real prices; later days use model forecasts. In the backtest the plan
is re-solved every day and only today's purchases are carried out (model predictive control),
so the plan adapts as new prices arrive.

Solved with SciPy's built-in HiGHS solver (scipy.optimize.milp), so no extra install is needed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


@dataclass
class Plan:
    choice: dict[int, tuple[int, int]]      # item -> (store, day)
    visits: set[tuple[int, int]]            # (store, day) pairs visited
    objective: float


def solve_plan(cost: np.ndarray, needby: np.ndarray, visit_cost: np.ndarray,
               delay: float = 0.02, max_stores_per_day: int | None = None) -> Plan:
    """cost: (items, stores, days), NaN where an item is unavailable or unknown."""
    n_i, n_s, n_d = cost.shape
    allowed = np.isfinite(cost) & (np.arange(n_d)[None, None, :] <= np.asarray(needby)[:, None, None])
    xi, xs, xd = np.nonzero(allowed)
    missing = sorted(set(range(n_i)) - set(xi.tolist()))
    if missing:
        raise ValueError(f"items {missing} have no available store on any allowed day")

    visit_keys = sorted(set(zip(xs.tolist(), xd.tolist())))
    v_index = {k: j for j, k in enumerate(visit_keys)}
    n_x, n_v = len(xi), len(visit_keys)

    c = np.concatenate([cost[xi, xs, xd] + delay * xd,
                        np.array([visit_cost[s] for s, _ in visit_keys], dtype=float)])

    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    # Each item bought exactly once.
    for i in range(n_i):
        ks = np.nonzero(xi == i)[0]
        rows += [r] * len(ks); cols += ks.tolist(); vals += [1.0] * len(ks)
        lo.append(1.0); hi.append(1.0); r += 1
    # x <= v (buying needs a visit).
    for k in range(n_x):
        rows += [r, r]; cols += [k, n_x + v_index[(xs[k], xd[k])]]; vals += [1.0, -1.0]
        lo.append(-np.inf); hi.append(0.0); r += 1
    # Optional cap on store visits per day.
    if max_stores_per_day:
        for d in range(n_d):
            js = [j for (s, dd), j in v_index.items() if dd == d]
            if js:
                rows += [r] * len(js); cols += [n_x + j for j in js]; vals += [1.0] * len(js)
                lo.append(-np.inf); hi.append(float(max_stores_per_day)); r += 1

    A = coo_matrix((vals, (rows, cols)), shape=(r, n_x + n_v)).tocsr()
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=np.ones(n_x + n_v),
               bounds=Bounds(0, 1))
    if res.status != 0 or res.x is None:
        raise RuntimeError(f"solver failed: {res.message}")
    sol = res.x > 0.5
    choice = {int(xi[k]): (int(xs[k]), int(xd[k])) for k in np.nonzero(sol[:n_x])[0]}
    visits = {visit_keys[j] for j in np.nonzero(sol[n_x:])[0]}
    return Plan(choice, visits, float(res.fun))


def expected_costs(today: np.ndarray, pred_lr: np.ndarray, n_days: int) -> np.ndarray:
    """Turn today's costs (items, stores) and predicted log ratios (items, stores, H) into
    expected costs (items, stores, n_days). Day 0 is today's actual cost."""
    n_i, n_s = today.shape
    out = np.full((n_i, n_s, n_days), np.nan)
    out[:, :, 0] = today
    h = min(n_days - 1, pred_lr.shape[2])
    if h > 0:
        # exp(E[log ratio]) slightly understates E[ratio]; fine for choosing between options.
        out[:, :, 1:h + 1] = today[:, :, None] * np.exp(pred_lr[:, :, :h])
    return out
