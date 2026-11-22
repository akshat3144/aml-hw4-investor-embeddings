"""Model B: vectors trained directly for the three uses, with two padding coordinates so cosine == scaled dot product.

score(i, j) = m_i . s_j with m_i, s_j in R^14.
  stock vector  s_j = W phi_j + e_j      (phi_j: dated text embedding PCA + SIC + listing flag; e_j only for
                                         stocks with holdings, so a new listing is a pure content vector)
  investor vec  m_i free (all 200 investors report holdings)
Loss (all public by the decision date):
  L_ipo : for every listing whose first-quarter holders are known, softmax over that quarter's filers of
          m_i . (W phi_j)  -> who buys a new listing, from its content alone (Tests 1 and 2)
  L_new : for each investor and each quarter-to-quarter transition, softmax over not-held stocks of
          m_i . s_j for the stocks newly bought (Test 3)
  L_hold: logistic loss on current holdings (co-holding structure)
  + ridge penalties.
Padding: append p_i = sqrt(R^2 - |m_i|^2) to investors (0 for stocks) and q_j = sqrt(R^2 - |s_j|^2) to stocks
(0 for investors); every vector then has norm R and cos(m_i, s_j) = m_i . s_j / R^2, which preserves every
ranking the tests use.
"""
import numpy as np
import torch

import evalkit as ek

IPO = np.array([s != "established" for s in ek.comp.status])
LISTING = ek.comp.listing_date.fillna("").values
torch.set_default_dtype(torch.float64)


def content_features(text, sic, known, n_pc=24):
    """phi_j: PCA of text embeddings (fitted on known stocks only), SIC one-hots, listing flag, constant."""
    T = text / np.linalg.norm(text, axis=1, keepdims=True)
    mu = T[known].mean(0)
    _, sv, Vt = np.linalg.svd(T[known] - mu, full_matrices=False)
    P = (T - mu) @ Vt[:n_pc].T / sv[:n_pc] * np.sqrt(known.sum())
    feats = [P]
    for digits in (1, 2):
        codes = np.array([s[:digits] if isinstance(s, str) and s else "?" for s in sic])
        u = sorted(set(codes))
        feats.append((codes[:, None] == np.array(u)[None, :]).astype(float))
    feats += [IPO[:, None].astype(float), np.ones((len(T), 1))]
    return np.hstack(feats)


def events(date, info):
    qs = sorted(info.quarter_end.unique())
    Xs = {q: ek.hold_matrix(q, info) for q in qs}
    fl = ek.fil[ek.fil.filing_date <= date]
    filers = {q: np.isin(ek.MID, fl[fl.quarter_end == q].manager_id) for q in qs}
    # listings with known first-quarter holders
    ipo_ev = []
    for j in np.where(IPO)[0]:
        if not LISTING[j] or LISTING[j] > qs[-1]:
            continue
        q = ek.first_qe(LISTING[j])
        if q not in Xs:
            continue
        y = Xs[q][:, j]
        if y[filers[q]].sum() >= 1:
            ipo_ev.append((j, filers[q], y))
    # transitions for new holdings (1 and 2 quarters ahead)
    new_ev = []
    for a in range(len(qs)):
        for b in (a + 1, a + 2):
            if b >= len(qs):
                continue
            qa, qb = qs[a], qs[b]
            trading = np.array([(not IPO[j]) or (LISTING[j] <= qa) for j in range(ek.NS)])
            inv = filers[qa] & filers[qb]
            new = (Xs[qa] == 0) & (Xs[qb] > 0) & trading[None, :] & inv[:, None]
            cand = (Xs[qa] == 0) & trading[None, :] & inv[:, None]
            new_ev.append((new, cand, 1.0 if b == a + 1 else 0.7))
    return qs, Xs, ipo_ev, new_ev


def build(date, info, text=None, sic=None, k=14, n_pc=24, w_ipo=1.0, w_new=1.0, w_hold=0.3, lam_m=1e-3,
          lam_e=1e-2, lam_w=1e-2, steps=600, lr=0.05, seed=0, return_vecs=False, decay=0.5):
    qs, Xs, ipo_ev, new_ev = events(date, info)
    R = np.zeros((ek.NM, ek.NS))
    for t, q in enumerate(reversed(qs)):
        R = np.maximum(R, Xs[q] * decay ** t)
    known = R.sum(0) > 0
    phi = torch.tensor(content_features(text, sic, known, n_pc))
    g = torch.Generator().manual_seed(seed)
    M = (0.1 * torch.randn(ek.NM, k, generator=g)).requires_grad_()
    W = (0.1 * torch.randn(phi.shape[1], k, generator=g)).requires_grad_()
    E = (0.1 * torch.randn(ek.NS, k, generator=g)).requires_grad_()
    b0 = torch.zeros(1, requires_grad=True)
    kn = torch.tensor(known, dtype=torch.float64)[:, None]
    Rt = torch.tensor(R)
    latest = torch.tensor(Xs[qs[-1]])
    has_rep = torch.tensor(Xs[qs[-1]].sum(1) > 0)
    ipo_idx = [(j, torch.tensor(f), torch.tensor(y[f] > 0)) for j, f, y in ipo_ev]
    new_t = [(torch.tensor(n), torch.tensor(c), w) for n, c, w in new_ev]
    opt = torch.optim.Adam([M, W, E, b0], lr=lr)
    for it in range(steps):
        opt.zero_grad()
        C = phi @ W
        S = C + E * kn
        loss = lam_m * (M ** 2).sum() + lam_e * (E ** 2).sum() + lam_w * (W ** 2).sum()
        if ipo_idx and w_ipo > 0:
            li = 0.0
            for j, f, y in ipo_idx:
                sc = M[f] @ C[j]
                li = li - (torch.log_softmax(sc, 0)[y]).mean()
            loss = loss + w_ipo * li / len(ipo_idx) * 10
        if new_t and w_new > 0:
            Z = M @ S.T
            ln, tot = 0.0, 0.0
            for n, c, w in new_t:
                Zc = Z.masked_fill(~c, -1e9)
                lsm = torch.log_softmax(Zc, 1)
                rows = n.any(1)
                ln = ln - w * ((lsm * n).sum(1)[rows] / n.sum(1)[rows]).sum()
                tot += w * rows.sum().item()
            loss = loss + w_new * ln / max(tot, 1) * 10
        if w_hold > 0:
            Z = M @ S.T + b0
            lh = torch.nn.functional.binary_cross_entropy_with_logits(Z[has_rep][:, known], latest[has_rep][:, known],
                                                                       pos_weight=torch.tensor(5.0))
            loss = loss + w_hold * lh * 10
        loss.backward()
        opt.step()
    with torch.no_grad():
        C = phi @ W
        S = (C + E * kn).numpy()
        Mn = M.numpy().copy()
    if return_vecs:
        return Mn, S
    return Mn @ S.T


def pad(M, S):
    """Append two coordinates so that every vector has the same norm; cosine then equals scaled dot product."""
    r_m = np.sqrt((M ** 2).sum(1)).max() * 1.0001
    r_s = np.sqrt((S ** 2).sum(1)).max() * 1.0001
    # rescale so both sides share one radius
    M2, S2 = M / r_m, S / r_s
    pm = np.sqrt(1 - (M2 ** 2).sum(1))
    ps = np.sqrt(1 - (S2 ** 2).sum(1))
    Mp = np.hstack([M2, pm[:, None], np.zeros((len(M2), 1))])
    Sp = np.hstack([S2, np.zeros((len(S2), 1)), ps[:, None]])
    return Mp, Sp
