"""Model C: explicit construction, cosine made equal to a dot product by two padding coordinates.

Layout of the 16 numbers
  d1      investor padding (0 for stocks)
  d2      stock padding    (0 for investors)
  d3      listing axis:     investor appetite for new listings  x  stock 'is a listing' flag
  d4      stock bias axis:  constant for investors              x  stock bias (Test 3 prior)
  d5-d16  12 co-holding factors from a truncated SVD of the (recency-weighted, normalised) holdings matrix;
          a stock without holdings gets a text-similarity-weighted mean of the factors of similar stocks.
score(i, j) = a*app_i*flag_j + c*bias_j + U_i . V_j, and cos(m_i, s_j) = score(i, j) / R^2 exactly.
"""
import numpy as np

import evalkit as ek

IPO = np.array([s != "established" for s in ek.comp.status])


def svd_factors(R, k, row_pow, col_pow, col_w, alpha):
    npos = R.sum(1)
    pop = R.sum(0)
    Rn = R / np.maximum(npos, 1)[:, None] ** row_pow / np.maximum(pop, 1)[None, :] ** col_pow * col_w[None, :]
    U, s, Vt = np.linalg.svd(Rn, full_matrices=False)
    U, s, V = U[:, :k], s[:k], Vt[:k].T
    return U * s ** alpha, V * s ** (1 - alpha) / np.where(col_w > 0, col_w, 1)[:, None]


def cold_features(Tn, seen, n_pc):
    """Text PCA (fitted on stocks with holdings), a listing flag and a constant."""
    mu = Tn[seen].mean(0)
    _, sv, Vt = np.linalg.svd(Tn[seen] - mu, full_matrices=False)
    P = (Tn - mu) @ Vt[:n_pc].T / sv[:n_pc] * np.sqrt(seen.sum())
    return np.hstack([P, IPO[:, None].astype(float), np.ones((len(Tn), 1))])


def fit_cold_map(date, info, R, core_m, text, seen, n_pc=32, lam=1.0, w_est=0.1):
    """Convex softmax regression: a listing's text -> the stock vector that best ranks its actual holders.

    For each training stock j (listings with known first-quarter holders, weight 1; established stocks with
    their current holders, weight w_est) and each holder i among the investors that reported that quarter:
        loss_j = -mean_{i in holders_j} log softmax_{i' in reporters_j}( m_i' . W phi_j )
    minimised over W with a ridge penalty lam * |W|^2; investor vectors m are fixed.
    """
    import torch

    Tn = text / np.linalg.norm(text, axis=1, keepdims=True)
    phi = torch.tensor(cold_features(Tn, seen, n_pc), dtype=torch.float64)
    Mt = torch.tensor(core_m, dtype=torch.float64)
    qs = sorted(info.quarter_end.unique())
    fl = ek.fil[ek.fil.filing_date <= date]
    rep = {q: np.isin(ek.MID, fl[fl.quarter_end == q].manager_id) for q in qs}
    Xq = {q: ek.hold_matrix(q, info) for q in qs}
    items = []
    for j in np.where(seen)[0]:
        if IPO[j]:
            q = ek.first_qe(ek.comp.listing_date[j])
            if q not in Xq:
                continue
            y, r, w = Xq[q][:, j] > 0, rep[q], 1.0
        else:
            y, r, w = Xq[qs[-1]][:, j] > 0, rep[qs[-1]], w_est
        if w > 0 and y[r].sum() >= 1:
            items.append((j, torch.tensor(r), torch.tensor(y[r]), w))
    W = torch.zeros(phi.shape[1], core_m.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")
    tot = sum(w for *_, w in items)

    def closure():
        opt.zero_grad()
        loss = lam * (W ** 2).sum()
        for j, r, y, w in items:
            sc = Mt[r] @ (phi[j] @ W)
            loss = loss - w * torch.log_softmax(sc, 0)[y].mean() / tot
        loss.backward()
        return loss

    opt.step(closure)
    return W.detach().numpy()


def pad16(core_m, core_s):
    """core vectors (14 numbers each) -> 16 numbers with constant norm on both sides."""
    r = max(np.sqrt((core_m ** 2).sum(1)).max(), np.sqrt((core_s ** 2).sum(1)).max()) * 1.0001
    m, s = core_m / r, core_s / r
    pm = np.sqrt(np.clip(1 - (m ** 2).sum(1), 0, None))
    ps = np.sqrt(np.clip(1 - (s ** 2).sum(1), 0, None))
    M = np.hstack([pm[:, None], np.zeros((len(m), 1)), m])
    S = np.hstack([np.zeros((len(s), 1)), ps[:, None], s])
    return M, S


def build(date, info, text=None, k=12, decay=0.5, row_pow=0.5, col_pow=0.0, ipo_col_w=1.0, alpha=0.5,
          a=1.0, c=0.0, bias="none", tau=20.0, donors="ipo", app_form="log", text_cf=1.0, return_vecs=False,
          ext_app=None, ext_w=0.0, tau_all=10.0, univ_w=0.0, univ_q=4, long_hist=False, weight_value=0.0,
          layout="pad2", sic=None, sic_w=0.0, ridge_w=0.0, ridge_lam=10.0, ridge_ipo_w=3.0, n_pc=32,
          cold_fit=0.0, cold_lam=1.0, cold_w_est=0.1, seen_flag=1.0, usic_w=0.0, usic_min=5, usic_pow=0.0,
          usic_norm=True, usic_new=1.0, thin_w=0.0, thin_n=3, cold_k=None):
    qs = sorted(info.quarter_end.unique())
    Xs = [ek.hold_matrix(q, info) for q in qs]
    R = np.zeros((ek.NM, ek.NS))
    hist = list(reversed(Xs))
    if long_hist:
        # older quarter-ends of the same 197 stocks, from the full 13F reports public by the decision date
        import f13
        P = f13.panel(date)
        hist += [P[q] for q in sorted(P, reverse=True) if q < qs[0]]
    for t, X in enumerate(hist):
        R = np.maximum(R, X * decay ** t)
    if weight_value:
        d = info[info.quarter_end == qs[-1]]
        Wv = np.zeros((ek.NM, ek.NS))
        Wv[d.manager_id.map(ek.mi_).values, d.stock_id.map(ek.si).values] = d.value_usd.clip(lower=1).values
        share = Wv / np.maximum(Wv.sum(1, keepdims=True), 1)
        rank_w = 1 + weight_value * np.log1p(share * 100)  # >1 for large positions
        R = np.where(Wv > 0, R * rank_w, R)
    seen = R.sum(0) > 0
    col_w = np.where(IPO, ipo_col_w, 1.0)
    if univ_w > 0:
        # add every other security in the investors' full reports as extra columns (the 197 stocks' own
        # CUSIPs are dropped there: they enter only through holdings.csv rows public by the decision date)
        import f13
        Ru, cus = f13.universe(date, decay=decay, last_n=univ_q)
        mapped = set(f13.stock_cusip().values())
        keep = np.array([c not in mapped for c in cus])
        Ru = Ru[:, keep]
        cus = [c for c, kp in zip(cus, keep) if kp]
        wide = (Ru > 0).sum(0) >= 2
        Ru = Ru[:, wide]
        cus = [c for c, kp in zip(cus, wide) if kp]
        Rall = np.hstack([R, Ru])
        cw = np.concatenate([col_w, np.full(Ru.shape[1], univ_w)])
        U, Vall = svd_factors(Rall, k, row_pow, col_pow, cw, alpha)
        V = Vall[: ek.NS].copy()
        Vu = Vall[ek.NS:]
        nu = (Ru > 0).sum(0)
        if usic_w > 0:
            import pandas as pd
            us = pd.read_parquet("cache/univ_sic.parquet").set_index("cusip").sic.to_dict()
            usic = np.array([str(us.get(c, "")) for c in cus])
            fr = f13.raw()
            first_q = fr[fr.filing_date <= date].groupby("cusip").report_date.min()
            is_new = np.array([first_q.get(c, "") >= "2024-03-31" for c in cus])
    else:
        U, V = svd_factors(R, k, row_pow, col_pow, col_w, alpha)
    npos = (R > 0).sum(1)
    if (npos == 0).any():
        U[npos == 0] = U[npos > 0].mean(0)
    # cold start from text
    cold = ~seen
    if text is not None and cold.any():
        Tn = text / np.linalg.norm(text, axis=1, keepdims=True)
        sim = Tn @ Tn.T
        if sic is not None and sic_w > 0:
            s2 = np.array([str(x)[:2] for x in sic])
            s3 = np.array([str(x)[:3] for x in sic])
            sim = sim + sic_w * (s2[:, None] == s2[None, :]) + sic_w * (s3[:, None] == s3[None, :])
        pool_ipo0 = seen & IPO if (seen & IPO).sum() >= 3 else seen
        mix = {"ipo": 0.0, "all": 1.0}.get(donors, donors)  # share of the all-stock neighbourhood
        holders = (R > 0).sum(0)
        thin = seen & IPO & (holders < thin_n) if thin_w > 0 else np.zeros(ek.NS, bool)
        V0 = V.copy()
        est = {}
        for j in np.where(cold | thin)[0]:
            notj = np.arange(ek.NS) != j
            v = np.zeros(V.shape[1])
            for pool, wt, tt in ((pool_ipo0 & notj, 1 - mix, tau), (seen & notj, mix, tau_all)):
                if wt == 0:
                    continue
                w = np.exp(tt * (sim[j, pool] - sim[j, pool].max()))
                v += wt * (w @ V0[pool]) / w.sum()
            est[j] = text_cf * v
        for j in np.where(cold | thin)[0]:
            if thin[j]:
                e = est[j] * np.linalg.norm(V0[j]) / max(np.linalg.norm(est[j]), 1e-12)
                V[j] = (1 - thin_w) * V0[j] + thin_w * e
                continue
            V[j] = est[j]
            if usic_w > 0 and univ_w > 0 and sic is not None:
                # the same-industry securities in the investors' full reports (SIC from SEC financial data sets)
                code = str(sic[j])
                vs = None
                for L in (4, 3, 2):
                    pool = np.array([u[:L] == code[:L] for u in usic]) if len(code) >= L else np.zeros(len(usic), bool)
                    if pool.sum() >= usic_min:
                        wts = nu[pool] ** usic_pow * np.where(is_new[pool], usic_new, 1.0)
                        vs = (wts @ Vu[pool]) / wts.sum()
                        break
                if vs is not None:
                    nv, ns_ = np.linalg.norm(V[j]), np.linalg.norm(vs)
                    V[j] = (1 - usic_w) * V[j] + usic_w * vs * (nv / ns_ if (usic_norm and ns_ > 0) else 1.0)
            if cold_k is not None and cold_k < V.shape[1]:
                V[j, cold_k:] = 0.0  # a prospectus can only pin down the leading factors
        if ridge_w > 0:
            # linear map from text (PCA fitted on stocks with holdings) to co-holding factors
            mu = Tn[seen].mean(0)
            _, sv, Vt = np.linalg.svd(Tn[seen] - mu, full_matrices=False)
            P = (Tn - mu) @ Vt[:n_pc].T / sv[:n_pc] * np.sqrt(seen.sum())
            X1 = np.hstack([P, np.ones((ek.NS, 1))])
            wts = np.where(IPO[seen], ridge_ipo_w, 1.0)
            A = X1[seen].T @ (X1[seen] * wts[:, None]) + ridge_lam * np.eye(X1.shape[1])
            B = np.linalg.solve(A, X1[seen].T @ (V[seen] * wts[:, None]))
            V[cold] = (1 - ridge_w) * V[cold] + ridge_w * (X1[cold] @ B)
    elif cold.any():
        V[cold] = 0.0
    # appetite for new listings
    n_ipo = (R[:, IPO] > 0).sum(1).astype(float)
    if app_form == "log":
        app = np.log1p(n_ipo)
    elif app_form == "rate":
        app = n_ipo / np.sqrt(np.maximum(npos, 1))
    else:
        app = n_ipo
    if ext_w > 0:
        import f13
        cmap = f13.stock_cusip()
        hidden = [cmap[s] for s in ek.SID if s in cmap and not seen[ek.si[s]]]
        ea = f13.new_issue_appetite(date, exclude_cusips=hidden)
        ea = np.log1p(ea * 4)
        app = (app - app.mean()) / (app.std() + 1e-12) + ext_w * (ea - ea.mean()) / (ea.std() + 1e-12)
    app = (app - app.mean()) / (app.std() + 1e-12)
    # stock bias for Test 3
    pop = (Xs[-1] > 0).sum(0).astype(float)
    if bias == "logpop":
        b = np.log1p(pop)
    elif bias == "trend" and len(Xs) >= 2:
        b = ((Xs[-1] > 0) & (Xs[-2] == 0)).sum(0) - ((Xs[-1] == 0) & (Xs[-2] > 0)).sum(0)
        b = b / np.sqrt(pop + 5)
    else:
        b = np.zeros(ek.NS)
    b = (b - b.mean()) / (b.std() + 1e-12)
    scale = np.sqrt((U ** 2).sum(1).mean() * (V[seen] ** 2).sum(1).mean())  # typical size of U.V terms
    cols_m = [np.sqrt(scale) * a * app[:, None]]
    flag = np.where(IPO & seen, seen_flag, IPO.astype(float))
    cols_s = [np.sqrt(scale) * flag[:, None]]
    if c != 0:
        cols_m.append(np.sqrt(scale) * np.ones((ek.NM, 1)))
        cols_s.append(np.sqrt(scale) * c * b[:, None])
    core_m = np.hstack(cols_m + [U])
    core_s = np.hstack(cols_s + [V])
    if cold_fit > 0 and text is not None:
        Wc = fit_cold_map(date, info, R, core_m, text, seen, n_pc=n_pc, lam=cold_lam, w_est=cold_w_est)
        Tn = text / np.linalg.norm(text, axis=1, keepdims=True)
        phi = cold_features(Tn, seen, n_pc)
        pred = phi @ Wc
        # match the size of known listing vectors, then blend with the neighbour estimate
        ref = np.sqrt((core_s[seen & IPO] ** 2).sum(1)).mean() if (seen & IPO).any() else np.sqrt((core_s[seen] ** 2).sum(1)).mean()
        pred = pred / np.sqrt((pred[cold] ** 2).sum(1)).mean() * ref
        core_s[cold] = (1 - cold_fit) * core_s[cold] + cold_fit * pred[cold]
    if layout == "pad2":
        M, S = pad16(core_m, core_s)
    else:  # one padding coordinate, investors only: cosine is then m.s / (R |s|)
        r = np.sqrt((core_m ** 2).sum(1)).max() * 1.0001
        m = core_m / r
        M = np.hstack([np.sqrt(1 - (m ** 2).sum(1))[:, None], m])
        S = np.hstack([np.zeros((ek.NS, 1)), core_s])
    if M.shape[1] < 16:  # fewer factors: zero columns leave every cosine unchanged
        M = np.hstack([M, np.zeros((len(M), 16 - M.shape[1]))])
        S = np.hstack([S, np.zeros((len(S), 16 - S.shape[1]))])
    assert M.shape[1] == 16 and S.shape[1] == 16, (M.shape, S.shape)
    if return_vecs:
        return M, S
    return ek.cosine_scores(M, S)
