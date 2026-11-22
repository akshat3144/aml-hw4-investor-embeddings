"""Model A: truncated SVD of a weighted holdings matrix + an explicit 'new listing' axis + text kNN for cold start."""
import numpy as np

import evalkit as ek

IPO = np.array([s != "established" for s in ek.comp.status])


def holdings_tensor(info):
    """Matrices per quarter from the info rows, latest quarter last."""
    qs = sorted(info.quarter_end.unique())
    return qs, [ek.hold_matrix(q, info) for q in qs]


def build(date, info, k=15, decay=0.5, col_pow=0.5, row_pow=0.5, alpha=0.5, app_w=1.0, flag=1.0,
          app_prior=1.0, text=None, tau=20.0, knn_ipo_only=False, text_mix=1.0, return_vecs=False):
    qs, Xs = holdings_tensor(info)
    # recency-weighted holdings; any holding of a listing counts at full weight (they are rare)
    R = np.zeros((ek.NM, ek.NS))
    for t, X in enumerate(reversed(Xs)):
        R = np.maximum(R, X * (decay ** t))
    seen = R.sum(0) > 0
    pop = R.sum(0)
    npos = R.sum(1)
    Rn = R / np.maximum(npos, 1)[:, None] ** row_pow / np.maximum(pop, 1)[None, :] ** col_pow
    U, s, Vt = np.linalg.svd(Rn, full_matrices=False)
    U, s, V = U[:, :k], s[:k], Vt[:k].T
    M = U * s ** alpha
    S = V * s ** (1 - alpha)
    if (npos == 0).any():  # investors with no public report yet get the average investor
        M[npos == 0] = M[npos > 0].mean(0)
    # cold start: stocks with no holdings get a text-similarity-weighted mean of known stocks' vectors
    cold = ~seen
    if text is not None:
        T = text / np.linalg.norm(text, axis=1, keepdims=True)
        sim = T @ T.T
        donors = seen & IPO if (knn_ipo_only and (seen & IPO).sum() >= 3) else seen
        ipo_mean = S[seen & IPO].mean(0) if (seen & IPO).any() else S[seen].mean(0)
        for j in np.where(cold)[0]:
            w = np.exp(tau * (sim[j, donors] - sim[j, donors].max()))
            S[j] = text_mix * (w @ S[donors]) / w.sum() + (1 - text_mix) * ipo_mean
    else:
        S[cold] = S[seen & IPO].mean(0) if (seen & IPO).any() else S[seen].mean(0)
    # 'new listing' axis: investor appetite for listings seen so far, stock flag for listings
    n_ipo = (R[:, IPO] > 0).sum(1)
    app = (n_ipo + app_prior * n_ipo.mean()) / (np.log1p(npos) + 1)
    app = app / app.std() if app.std() > 0 else app
    scale = np.sqrt((M ** 2).sum(1).mean())
    Mx = np.hstack([M, app_w * scale * app[:, None]])
    Sx = np.hstack([S, flag * np.sqrt((S ** 2).sum(1).mean()) * IPO[:, None].astype(float)])
    if return_vecs:
        return Mx, Sx
    return ek.cosine_scores(Mx, Sx)
