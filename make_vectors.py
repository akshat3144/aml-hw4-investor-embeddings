"""make_vectors.py -- rebuild stocks.csv and managers.csv offline from data/ and external.parquet.

Usage:  python make_vectors.py            (writes stocks.csv and managers.csv next to this file)

Inputs
  data/holdings.csv, data/filings.csv, data/companies.csv, data/manager_info.csv   (supplied)
  external.parquet                                                                  (brought in; one long table)
    kind="f13"  : the investors' full 13F reports (common-stock lines; 199 of the 200 file them) for quarter-ends
                  2023-12-31..2025-12-31, all filed before 2026-03-02 (EDGAR);
    kind="doc"  : for each stock, the dated SEC filing its business description comes from and that filing's SIC
                  code (established: last annual report filed before 2025-05-16; listing: last registration
                  statement filed before its listing date);
    kind="emb"  : embeddings of those descriptions from approved models at the pinned revisions, run locally;
    kind="usic" : SIC codes of other securities in the 13F reports, from SEC Financial Statement Data Sets
                  2025q1-q2 (filings before 2025-07-01), matched by issuer name.

Only holdings public by 2 March 2026 are used; a 2026 listing contributes only its pre-IPO prospectus text and SIC.
Everything is closed-form (SVD, weighted means, standardisation) and SVD signs are fixed, so the output is
reproducible to rounding error.

The 16 numbers (see design.md)
  d1   investor padding   so every investor vector has length 1 (0 for stocks)
  d2   stock padding      so every stock vector has length 1 (0 for investors)
  d3   new-listing axis   investor: standardised appetite for new listings; stock: 1 for a listing without holders,
                          0.5 for a 2025 listing that already has holders, 0 for an established stock
  d4   popularity axis    investor: constant; stock: standardised log number of holders at 31 Dec 2025
  d5-d16  12 co-holding factors: truncated SVD of investors x (197 stocks + every other common stock in their 13F
          reports); a stock without holders gets text-neighbour and same-industry averages of others' factors,
          kept on the first 8 factors only.
With both paddings every vector has length 1, so cos(m_i, s_j) = m_i . s_j exactly.
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CUTOFF = "2026-03-02"            # holdings must be public by this date
K = 12                           # co-holding factors
DECAY = 0.5                      # weight of a holding one quarter older
ROW_POW = 0.4                    # rows of the holdings matrix are divided by (number of positions) ** ROW_POW
UNIV_QUARTERS = 4                # quarter-ends of full 13F reports used for the extra columns
UNIV_WEIGHT = 1.0                # weight of those columns relative to the 197 stocks
TEXT_MODELS = ["bge"]            # embeddings averaged for text similarity
TAU_LISTINGS, TAU_ALL, MIX_ALL = 20.0, 20.0, 0.3   # text neighbourhoods: among listings / among all stocks
INDUSTRY_W, INDUSTRY_MIN = 0.4, 3                   # weight of the same-industry average, minimum pool size
COLD_K = 8                       # a stock without holders keeps only its first COLD_K factors
NEW_ISSUE_W = 0.03               # tie-breaker weight of new-issue buying seen in older 13F reports
APPETITE_W = 0.35                # weight of the new-listing axis relative to the co-holding factors
POP_W = 0.5                      # weight of the popularity axis
LISTED_FLAG = 0.5                # new-listing flag of a 2025 listing that already has holders
OLD_QUARTERS = ["2023-12-31", "2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31"]


def z(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def main():
    d = os.path.join(HERE, "data")
    comp = pd.read_csv(os.path.join(d, "companies.csv"), dtype=str)
    mi = pd.read_csv(os.path.join(d, "manager_info.csv"), dtype={"manager_id": str, "sec_cik": str})
    hold = pd.read_csv(os.path.join(d, "holdings.csv"), dtype={"manager_id": str, "stock_id": str})
    ext = pd.read_parquet(os.path.join(HERE, "external.parquet"))

    SID, MID = comp.stock_id.tolist(), mi.manager_id.tolist()
    si = {s: k for k, s in enumerate(SID)}
    mix = {m: k for k, m in enumerate(MID)}
    NS, NM = len(SID), len(MID)
    IPO = (comp.status != "established").values

    # ---- holdings public by the cutoff ---------------------------------------------------------------------------
    hold = hold[hold.filing_date <= CUTOFF]
    f13 = ext[(ext.kind == "f13") & (ext.filing_date < CUTOFF)]
    f13 = f13[["manager_id", "report_date", "filing_date", "cusip", "issuer", "value", "shares"]]

    def indicator(df, col, index, ncol):
        X = np.zeros((NM, ncol))
        X[df.manager_id.map(mix).values, df[col].map(index).values] = 1
        return X

    qs = sorted(hold.quarter_end.unique())
    Xs = {q: indicator(hold[hold.quarter_end == q], "stock_id", si, NS) for q in qs}

    # CUSIP of each stock: the 13F line with the same investor, quarter-end and share count
    m = hold.merge(f13, left_on=["manager_id", "quarter_end", "shares"],
                   right_on=["manager_id", "report_date", "shares"], how="inner")
    cm = m.groupby(["stock_id", "cusip"]).size().reset_index(name="n").sort_values(
        ["n", "stock_id", "cusip"], ascending=[False, True, True])
    cusip_of = cm.groupby("stock_id").cusip.first().to_dict()
    stock_of = {c: si[s] for s, c in cusip_of.items()}

    # recency-weighted holdings of the 197 stocks: 2025 from holdings.csv, 2023-24 quarter-ends from 13F reports
    hist = [Xs[q] for q in reversed(qs)]
    for q in sorted([q for q in OLD_QUARTERS if q < qs[0]], reverse=True):
        dq = f13[(f13.report_date == q) & f13.cusip.isin(stock_of)]
        X = np.zeros((NM, NS))
        X[dq.manager_id.map(mix).values, dq.cusip.map(stock_of).values] = 1
        hist.append(X)
    R = np.zeros((NM, NS))
    for t, X in enumerate(hist):
        R = np.maximum(R, X * DECAY ** t)
    seen = R.sum(0) > 0

    # every other common stock in the investors' last UNIV_QUARTERS reports, held by at least two of them
    uq = sorted(f13.report_date.unique())[-UNIV_QUARTERS:]
    fu = f13[f13.report_date.isin(uq) & ~f13.cusip.isin(set(cusip_of.values()))]
    cus = sorted(fu.cusip.unique())
    ci = {c: k for k, c in enumerate(cus)}
    Ru = np.zeros((NM, len(cus)))
    for t, q in enumerate(reversed(uq)):
        Ru = np.maximum(Ru, indicator(fu[fu.report_date == q], "cusip", ci, len(cus)) * DECAY ** t)
    wide = (Ru > 0).sum(0) >= 2
    Ru, cus = Ru[:, wide], [c for c, w in zip(cus, wide) if w]

    # ---- co-holding factors: truncated SVD ---------------------------------------------------------------------
    Rall = np.hstack([R, Ru * UNIV_WEIGHT])
    Rn = Rall / np.maximum(np.hstack([R, Ru]).sum(1), 1)[:, None] ** ROW_POW
    U, s, Vt = np.linalg.svd(Rn, full_matrices=False)
    U, s, V = U[:, :K], s[:K], Vt[:K].T
    sign = np.sign(V[np.abs(V).argmax(0), np.arange(K)])      # fix the sign of each factor
    U, V = U * sign * np.sqrt(s), V * sign * np.sqrt(s)
    V = V / np.concatenate([np.ones(NS), np.full(Ru.shape[1], UNIV_WEIGHT)])[:, None]
    V, Vu = V[:NS].copy(), V[NS:]
    V[~seen] = 0.0

    # ---- stocks without holders: text neighbours and same-industry securities --------------------------------------
    E = np.zeros((NS, 0))
    for name in TEXT_MODELS:
        e = ext[(ext.kind == "emb") & (ext.model == name)].set_index("stock_id").loc[SID]
        Ei = np.vstack(e.vec.values).astype(float)
        E = np.hstack([E, Ei / np.linalg.norm(Ei, axis=1, keepdims=True)])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    sim = E @ E.T
    sic = ext[ext.kind == "doc"].set_index("stock_id").loc[SID].sic.astype(str).values
    us = ext[ext.kind == "usic"].set_index("cusip").sic.to_dict()
    usic = np.array([str(us.get(c, "")) for c in cus])
    listings = seen & IPO
    for j in np.where(~seen)[0]:
        v = np.zeros(K)
        for pool, wt, tau in ((listings, 1 - MIX_ALL, TAU_LISTINGS), (seen, MIX_ALL, TAU_ALL)):
            w = np.exp(tau * (sim[j, pool] - sim[j, pool].max()))
            v += wt * (w @ V[pool]) / w.sum()
        for L in (4, 3, 2):                                    # narrowest SIC level with enough securities
            pool = np.array([u[:L] == sic[j][:L] for u in usic]) if len(sic[j]) >= L else np.zeros(len(cus), bool)
            if pool.sum() >= INDUSTRY_MIN:
                vs = Vu[pool].mean(0)
                v = (1 - INDUSTRY_W) * v + INDUSTRY_W * vs * np.linalg.norm(v) / np.linalg.norm(vs)
                break
        v[COLD_K:] = 0.0                                       # a prospectus only pins down the leading factors
        V[j] = v

    # ---- appetite for new listings ------------------------------------------------------------------------------
    n_listings = (R[:, IPO] > 0).sum(1).astype(float)
    # new-issue buying in older reports: securities seen for the first time (by CUSIP and by issuer name) in a
    # quarter, held by at most 25 investors then; the 197 stocks without holders here are excluded
    hidden = [cusip_of[x] for x in SID if x in cusip_of and not seen[si[x]]]
    first_q = f13.groupby("cusip").report_date.min()
    iss = f13.issuer.str.upper().str.replace(r"[^A-Z0-9 ]", "", regex=True).str.split().str[:2].str.join(" ")
    g = f13.assign(iss=iss)
    first_iss = g.groupby("iss").report_date.min()
    g = g.assign(fq=g.cusip.map(first_q), fi=g.iss.map(first_iss))
    new = g[(g.report_date == g.fq) & (g.fq >= "2024-03-31") & (g.fi == g.fq) & ~g.cusip.isin(hidden)]
    width = new.groupby("cusip").manager_id.nunique()
    new = new[new.cusip.map(width) <= 25]
    cnt = new.groupby("manager_id").cusip.nunique()
    n_rep = g.groupby("manager_id").report_date.nunique()
    new_issue = np.zeros(NM)
    for mgr, v in cnt.items():
        new_issue[mix[mgr]] = v / max(n_rep.get(mgr, 1), 1)
    app = z(z(np.log1p(n_listings)) + NEW_ISSUE_W * z(np.log1p(4 * new_issue)))

    # ---- popularity axis ------------------------------------------------------------------------------------------
    b = z(np.log1p((Xs[qs[-1]] > 0).sum(0).astype(float)))

    # ---- assemble and pad to unit length ------------------------------------------------------------------------------
    scale = np.sqrt(np.sqrt((U ** 2).sum(1).mean() * (V[seen] ** 2).sum(1).mean()))
    flag = np.where(IPO & seen, LISTED_FLAG, IPO.astype(float))
    core_m = np.hstack([scale * APPETITE_W * app[:, None], scale * np.ones((NM, 1)), U])
    core_s = np.hstack([scale * flag[:, None], scale * POP_W * b[:, None], V])
    r = max(np.sqrt((core_m ** 2).sum(1)).max(), np.sqrt((core_s ** 2).sum(1)).max()) * 1.0001
    cm_, cs_ = core_m / r, core_s / r
    M = np.hstack([np.sqrt(np.clip(1 - (cm_ ** 2).sum(1), 0, None))[:, None], np.zeros((NM, 1)), cm_])
    S = np.hstack([np.zeros((NS, 1)), np.sqrt(np.clip(1 - (cs_ ** 2).sum(1), 0, None))[:, None], cs_])

    cols = [f"d{j}" for j in range(1, 17)]
    pd.DataFrame(S, columns=cols).assign(stock_id=SID)[["stock_id"] + cols].to_csv(
        os.path.join(HERE, "stocks.csv"), index=False, float_format="%.10f")
    pd.DataFrame(M, columns=cols).assign(manager_id=MID)[["manager_id"] + cols].to_csv(
        os.path.join(HERE, "managers.csv"), index=False, float_format="%.10f")
    print(f"wrote stocks.csv ({NS} x 16) and managers.csv ({NM} x 16)")


if __name__ == "__main__":
    main()
