"""Compare text models (and averages of them) for the cold start, on the Test 1/2 backtests and CV."""
import functools
import sys

import numpy as np

import evalkit as ek
import model_c as C

BASE = dict(a=1, donors=0.3, ext_w=0.03, univ_w=1.0, long_hist=True, k=12, c=0.5, bias="logpop", seen_flag=0.5)


def combo(names):
    E = np.zeros((ek.NS, 0))
    for n in names:
        e = np.load(f"cache/emb_{n}.npy")
        E = np.hstack([E, e / np.linalg.norm(e, axis=1, keepdims=True)])
    return E / np.linalg.norm(E, axis=1, keepdims=True)


def t12(f, seeds=(0, 1, 2)):
    r = {}
    for name, (dd, lo, hi) in ek.SIMS_IPO.items():
        cs = ek.ipo_cases(lo, hi)
        S = f(dd, ek.info_at(dd))
        r["T1_" + name] = ek.test1(S, cs)
        r["T2_" + name] = ek.test2(S, cs)
    cv = ek.cv_ipo(f, folds=3, seed=list(seeds), verbose=False)
    r["T1_cv"], r["T2_cv"] = cv["T1_cv"], cv["T2_cv"]
    cv5 = ek.cv_ipo(f, folds=5, seed=list(seeds), verbose=False)
    r["T1_cv5"], r["T2_cv5"] = cv5["T1_cv"], cv5["T2_cv"]
    r["T1m"] = np.mean([r["T1_ipoQ3"], r["T1_ipoQ4"], r["T1_cv"], r["T1_cv5"]])
    r["T2m"] = np.mean([r["T2_ipoQ3"], r["T2_ipoQ4"], r["T2_cv"], r["T2_cv5"]])
    return {k: round(float(v), 4) for k, v in r.items()}


if __name__ == "__main__":
    sets = [s.split("+") for s in sys.argv[1:]]
    for names in sets:
        f = functools.partial(C.build, text=combo(names), **BASE)
        print("+".join(names), t12(f), flush=True)
