"""Backtest harness: the three tests of the handout, replayed on 2025 data at simulated decision dates.

Every simulation names a decision date; a model may only use holdings whose filing_date is on or before it
(the `info` frame handed to a model), plus dated company text. Outcomes come from later quarters.
Scores are computed from any 200 x 197 score matrix, so raw heuristics and cosine tables can be compared.
"""
import os
from itertools import combinations

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(HERE, "data")

comp = pd.read_csv(os.path.join(D, "companies.csv"), dtype=str)
mi = pd.read_csv(os.path.join(D, "manager_info.csv"), dtype={"manager_id": str, "sec_cik": str})
hold = pd.read_csv(os.path.join(D, "holdings.csv"), dtype={"manager_id": str, "stock_id": str})
fil = pd.read_csv(os.path.join(D, "filings.csv"), dtype=str)
SID = comp.stock_id.tolist()
MID = mi.manager_id.tolist()
si = {s: k for k, s in enumerate(SID)}
mi_ = {m: k for k, m in enumerate(MID)}
NS, NM = len(SID), len(MID)
QS = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def hold_matrix(q, h=hold):
    X = np.zeros((NM, NS))
    d = h[h.quarter_end == q]
    X[d.manager_id.map(mi_).values, d.stock_id.map(si).values] = 1
    return X


HQ = {q: hold_matrix(q) for q in QS}
FILERS = {q: np.isin(MID, fil[fil.quarter_end == q].manager_id) for q in QS}


def auc(scores, y):
    """AUC with ties counted one half."""
    y = np.asarray(y, bool)
    pos, neg = scores[y], scores[~y]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = pd.Series(np.concatenate([pos, neg])).rank().values
    return (r[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def first_qe(date):
    return next(q for q in QS if q >= date)


# ---------------------------------------------------------------- simulations
SIMS_IPO = {
    # name: (decision date, listing window) -> outcome at the first quarter-end on/after listing
    "ipoQ3": ("2025-07-01", "2025-07-01", "2025-09-30"),
    "ipoQ4": ("2025-11-17", "2025-10-01", "2025-12-31"),
}
SIMS_NEW = {
    # name: (decision date, base quarter, target quarter)
    "newB": ("2025-05-16", "2025-03-31", "2025-09-30"),
    "newA": ("2025-09-02", "2025-06-30", "2025-12-31"),
    "newC": ("2025-11-17", "2025-09-30", "2025-12-31"),
}


def info_at(date):
    """Holdings rows public on or before the decision date."""
    return hold[hold.filing_date <= date]


def ipo_cases(window_lo, window_hi):
    c = comp[(comp.status != "established") & (comp.listing_date >= window_lo) & (comp.listing_date <= window_hi)]
    out = []
    for s, ld in zip(c.stock_id, c.listing_date):
        q = first_qe(ld)
        f = FILERS[q]
        y = HQ[q][:, si[s]]
        if y[f].sum() >= 3:
            out.append((s, q, f, y))
    return out


def test1(score, cases):
    """Mean over listings of the AUC of investors (filers) ranked by score[:, stock]."""
    return np.nanmean([auc(score[f, si[s]], y[f]) for s, q, f, y in cases]) if cases else np.nan


def test2(score, cases):
    vals = []
    for (a, qa, fa, ya), (b, qb, fb, yb) in combinations(cases, 2):
        ok = fa & fb
        ia = np.where(ok & (ya > 0) & (yb == 0))[0]
        jb = np.where(ok & (yb > 0) & (ya == 0))[0]
        if len(ia) == 0 or len(jb) == 0:
            continue
        sa, sb = si[a], si[b]
        d = (score[ia, sa][:, None] + score[jb, sb][None, :]) - (score[ia, sb][:, None] + score[jb, sa][None, :])
        vals.append(np.mean((d > 0) + 0.5 * (d == 0)))
    return np.mean(vals) if vals else np.nan


def test3(score, base_q, target_q):
    base, tgt = HQ[base_q], HQ[target_q]
    trading = np.array([(st == "established") or (ld <= base_q)
                        for st, ld in zip(comp.status, comp.listing_date.fillna(""))])
    inv = np.where(FILERS[target_q] & FILERS[base_q])[0]
    new = (base == 0) & (tgt > 0)
    n_new = new[inv].sum(0)
    keep = trading & (n_new >= 3)
    pop = base.sum(0)
    ks = np.where(keep)[0]
    grp = np.full(NS, -1)
    grp[ks] = pd.qcut(pd.Series(pop[ks]).rank(method="first"), 4, labels=False).values
    res = []
    for i in inv:
        cand = keep & (base[i] == 0)
        g_auc = []
        for g in range(4):
            c = np.where(cand & (grp == g))[0]
            if len(c) < 3:
                continue
            y = tgt[i, c] > 0
            if y.all() or not y.any():
                continue
            g_auc.append(auc(score[i, c], y))
        if g_auc:
            res.append(np.mean(g_auc))
    return np.mean(res) if res else np.nan


def evaluate(build, verbose=True):
    """build(decision_date, info_holdings) -> 200 x 197 score matrix. Returns dict of backtest scores."""
    out = {}
    cache = {}

    def get(date):
        if date not in cache:
            cache[date] = build(date, info_at(date))
        return cache[date]

    for name, (dd, lo, hi) in SIMS_IPO.items():
        cases = ipo_cases(lo, hi)
        S = get(dd)
        out[f"T1_{name}"] = test1(S, cases)
        out[f"T2_{name}"] = test2(S, cases)
        out[f"n_{name}"] = len(cases)
    for name, (dd, bq, tq) in SIMS_NEW.items():
        out[f"T3_{name}"] = test3(get(dd), bq, tq)
    out["T1"] = np.nanmean([out[k] for k in out if k.startswith("T1_")])
    out["T2"] = np.nanmean([out[k] for k in out if k.startswith("T2_")])
    out["T3"] = np.nanmean([out[k] for k in out if k.startswith("T3_")])
    if verbose:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out


CUTOFF = "2026-03-02"


def cv_ipo(build, folds=5, seed=0, verbose=True):
    """Within-cutoff cross-validation over the 2025 listings: hide a fold's holdings, rebuild, score T1/T2.

    Uses only holdings public by 2 March 2026; each hidden listing is treated as a cold start, and its
    holders at the first quarter-end on/after listing are the outcome (as in Test 1).
    """
    cases = ipo_cases("2025-07-01", "2025-12-31")
    info = info_at(CUTOFF)
    seeds = [seed] if np.isscalar(seed) else list(seed)
    t1s, t2s = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        order = rng.permutation(len(cases))
        S_all = np.full((NM, NS), np.nan)
        t2 = []
        for f in range(folds):
            fold = [cases[k] for k in order[f::folds]]
            hide = [c[0] for c in fold]
            S = build(CUTOFF, info[~info.stock_id.isin(hide)])
            for s in hide:
                S_all[:, si[s]] = S[:, si[s]]
            if len(fold) >= 2:  # Test 2 compares listings scored by the same model, i.e. within a fold
                t2.append(test2(S, fold))
        t1s.append(test1(S_all, cases))
        t2s.append(np.nanmean(t2))
    out = {"T1_cv": float(np.mean(t1s)), "T2_cv": float(np.mean(t2s)), "n_cv": len(cases)}
    if verbose:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()})
    return out


def summary(build, cv_seeds=(0, 1, 2), cv_folds=3):
    """All backtests, plus the per-test averages used to choose the design."""
    r = evaluate(build, verbose=False)
    cv = cv_ipo(build, folds=cv_folds, seed=list(cv_seeds), verbose=False)
    r.update(cv)
    r["T1m"] = np.mean([r["T1_ipoQ3"], r["T1_ipoQ4"], r["T1_cv"]])
    r["T2m"] = np.mean([r["T2_ipoQ3"], r["T2_ipoQ4"], r["T2_cv"]])
    r["T3m"] = r["T3"]
    r["J"] = r["T1m"] + r["T2m"] + r["T3m"]
    keys = ["T1_ipoQ3", "T1_ipoQ4", "T1_cv", "T2_ipoQ3", "T2_ipoQ4", "T2_cv", "T3_newB", "T3_newA", "T3_newC",
            "T1m", "T2m", "T3m", "J"]
    return {k: round(float(r[k]), 4) for k in keys}


def cosine_scores(M, S):
    Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
    Sn = S / np.linalg.norm(S, axis=1, keepdims=True)
    return Mn @ Sn.T
