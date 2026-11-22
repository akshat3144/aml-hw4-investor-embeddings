"""Probe: how much does sector affinity (text-similar stocks' holders) add to listing appetite for Tests 1 and 2?
Leave-one-listing-out on the 2025 listings, data public by 2 March 2026."""
import sys

import numpy as np
import pandas as pd

import evalkit as ek

IPO = np.array([s != "established" for s in ek.comp.status])


def run(T, taus=(10, 20, 40), gammas=(0, 0.25, 0.5, 1, 2), pop_pow=0.5, donors="all"):
    cases = ek.ipo_cases("2025-07-01", "2025-12-31")
    X = np.maximum(ek.HQ["2025-09-30"], ek.HQ["2025-12-31"])
    Tn = T / np.linalg.norm(T, axis=1, keepdims=True)
    sim = Tn @ Tn.T
    res = []
    for tau in taus:
        for gm in gammas:
            S = np.zeros((ek.NM, ek.NS))
            for s, q, f, y in cases:
                j = ek.si[s]
                Xo = X.copy()
                Xo[:, j] = 0
                app = (Xo[:, IPO] > 0).sum(1)
                known = Xo.sum(0) > 0
                if donors == "ipo":
                    known &= IPO
                if donors == "est":
                    known &= ~IPO
                pop = Xo.sum(0)
                w = np.where(known, np.exp(tau * (sim[j] - sim[j, known].max())), 0)
                H = Xo / np.maximum(pop, 1) ** pop_pow
                aff = H @ w / w.sum()
                aff = (aff - aff.mean()) / (aff.std() + 1e-12)
                a = np.log1p(app)
                a = (a - a.mean()) / (a.std() + 1e-12)
                S[:, j] = a + gm * aff
            res.append(dict(tau=tau, gamma=gm, T1=ek.test1(S, cases), T2=ek.test2(S, cases)))
    return pd.DataFrame(res)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        T = np.load(f"cache/emb_{name}.npy")
        for d in ("all", "ipo", "est"):
            r = run(T, donors=d)
            print(name, d)
            print(r.round(4).to_string())
