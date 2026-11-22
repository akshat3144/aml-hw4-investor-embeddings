"""Features from the investors' full 13F reports (public EDGAR filings, each row dated by its filing date).

- stock_cusip(): the CUSIP of each of the 197 stocks, found by matching holdings.csv rows to 13F lines
  (same investor, quarter-end and share count);
- panel(date): investor x stock holding matrices of the 197 stocks for 2023-12-31 .. 2025-12-31 quarter-ends,
  from reports filed on or before `date` (extends holdings.csv back to 2023);
- universe(date): investor x security matrix over every common stock in the reports (all CUSIPs);
- new_issue_appetite(date): how many securities each investor bought in the first report in which they appeared
  among all 200 investors (a proxy for buying new listings), from reports filed on or before `date`.
"""
import os
import re

import numpy as np
import pandas as pd

import evalkit as ek

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_F = None
# Title-of-class text is free-form ("COM", "CS", "EQUITY", "SPONSORED ADS", "SH BEN INT", ...), so lines are kept
# unless the title marks a fund, warrant, unit, right, debt, preferred share or option.
NON_EQUITY = re.compile(r"\b(ETF|ETFS|ETP|ETN|EMF|FUND|FD|CEF|NCEF|UNIT|UNITS|WARRANT|WARRANTS|WT|WTS|RIGHT|RIGHTS|"
                        r"RTS|NOTE|NOTES|BOND|BONDS|DEBT|DEB|DBCV|PFD|PREF|PREFERRED|CALL|PUT|OPTION|OPTIONS|CONV)\b"
                        r"|\*W|99/99", re.I)


def raw():
    global _F
    if _F is None:
        f = pd.read_parquet(os.path.join(HERE, "cache", "f13_full.parquet"))
        f = f[(f.put_call == "") & (f.sh_type == "SH") & (f.value > 0)]
        f = f[~f.title.fillna("").str.contains(NON_EQUITY)]
        _F = f.groupby(["manager_id", "report_date", "filing_date", "cusip"], as_index=False).agg(
            value=("value", "sum"), shares=("shares", "sum"), issuer=("issuer", "first"))
    return _F


def stock_cusip():
    f = raw()
    m = ek.hold.merge(f, left_on=["manager_id", "quarter_end", "shares"],
                      right_on=["manager_id", "report_date", "shares"], how="inner")
    cm = m.groupby(["stock_id", "cusip"]).size().reset_index(name="n").sort_values("n", ascending=False)
    return cm.groupby("stock_id").cusip.first().to_dict()


QUARTERS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31",
            "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31"]


def panel(date):
    """{quarter: 200 x 197 holdings of the 197 stocks} from full reports filed on or before date."""
    f = raw()
    f = f[f.filing_date <= date]
    cmap = stock_cusip()
    inv = {c: ek.si[s] for s, c in cmap.items()}
    out = {}
    for q in QUARTERS:
        d = f[(f.report_date == q) & f.cusip.isin(inv)]
        if d.empty:
            continue
        X = np.zeros((ek.NM, ek.NS))
        X[d.manager_id.map(ek.mi_).values, d.cusip.map(inv).values] = 1
        out[q] = X
    return out


def reporters(date):
    f = raw()
    f = f[f.filing_date <= date]
    return {q: np.isin(ek.MID, f[f.report_date == q].manager_id.unique()) for q in QUARTERS}


def universe(date, decay=0.5, last_n=4):
    """Recency-weighted investor x CUSIP holding matrix (last `last_n` quarters public by date)."""
    f = raw()
    f = f[f.filing_date <= date]
    qs = sorted(f.report_date.unique())[-last_n:]
    f = f[f.report_date.isin(qs)]
    cus = sorted(f.cusip.unique())
    ci = {c: k for k, c in enumerate(cus)}
    R = np.zeros((ek.NM, len(cus)))
    for t, q in enumerate(reversed(qs)):
        d = f[f.report_date == q]
        X = np.zeros_like(R)
        X[d.manager_id.map(ek.mi_).values, d.cusip.map(ci).values] = 1
        R = np.maximum(R, X * decay ** t)
    return R, cus


def new_issue_appetite(date, min_q="2024-03-31", exclude_cusips=()):
    """Number of 'new securities' each investor held in the quarter they first appeared in any report.

    A security is new in quarter q if no investor reported it in any earlier quarter and no earlier
    security had the same issuer name (that catches CUSIP changes). Only quarters >= min_q count, so there
    is at least one earlier quarter to compare with.
    """
    f = raw()
    f = f[f.filing_date <= date]
    first_q = f.groupby("cusip").report_date.min()
    norm = f.issuer.str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.split().str[:2].str.join(" ")
    f = f.assign(iss=norm)
    first_iss = f.groupby("iss").report_date.min()
    f = f.assign(fq=f.cusip.map(first_q), fi=f.iss.map(first_iss))
    new = f[(f.report_date == f.fq) & (f.fq >= min_q) & (f.fi == f.fq) & ~f.cusip.isin(exclude_cusips)]
    # a spin-off or merger shows up in many portfolios at once; a listing in a few
    width = new.groupby("cusip").manager_id.nunique()
    new = new[new.cusip.map(width) <= 25]
    cnt = new.groupby("manager_id").cusip.nunique()
    n_rep = f.groupby("manager_id").report_date.nunique()
    out = np.zeros(ek.NM)
    for m, v in cnt.items():
        out[ek.mi_[m]] = v / max(n_rep.get(m, 1), 1)
    return out
