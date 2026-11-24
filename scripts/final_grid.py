"""Final comparison of the leading configurations on every backtest (Tests 1-3), for the design choice."""
import functools
import json
import sys

import numpy as np
import pandas as pd

import evalkit as ek
import model_c as C
from sweep_text import BASE, combo, t12

SIC = pd.read_parquet("cache/company_texts.parquet").set_index("stock_id").loc[ek.SID].sic.astype(str).values
FINAL = dict(BASE, usic_w=0.4, usic_min=3, tau_all=20.0, row_pow=0.4, cold_k=8, a=0.35)


def t3(f):
    out = {}
    for name, (dd, bq, tq) in ek.SIMS_NEW.items():
        out["T3_" + name] = round(float(ek.test3(f(dd, ek.info_at(dd)), bq, tq)), 4)
    out["T3m"] = round(float(np.mean(list(out.values()))), 4)
    return out


GRID = {
    # the submitted model and its ablations
    "final": (["bge"], dict()),
    "final_no_industry": (["bge"], dict(usic_w=0.0)),
    "final_no_universe": (["bge"], dict(univ_w=0.0, usic_w=0.0)),
    "final_no_appetite": (["bge"], dict(a=0.0)),
    "final_one_pad": (["bge"], dict(layout="pad1", k=13)),
    "final_no_history": (["bge"], dict(long_hist=False, ext_w=0.0)),
    "final_all_factors_cold": (["bge"], dict(cold_k=None)),
    # earlier comparisons
    "bge_r50": (["bge"], dict(row_pow=0.5)),
    "bge_r40": (["bge"], dict(row_pow=0.4)),
    "bge_r30": (["bge"], dict(row_pow=0.3)),
    "ens4_r40": (["bge", "gte", "mpnet", "e5"], dict(row_pow=0.4)),
    "e5_r40": (["e5"], dict(row_pow=0.4)),
    "nousic_r40": (["bge"], dict(row_pow=0.4, usic_w=0.0)),
    # ablations of the final model (bge_r40)
    "abl_no_universe": (["bge"], dict(row_pow=0.4, univ_w=0.0, usic_w=0.0)),
    "abl_no_appetite": (["bge"], dict(row_pow=0.4, a=0.0)),
    "abl_one_pad": (["bge"], dict(row_pow=0.4, layout="pad1", k=13)),
    "abl_no_history": (["bge"], dict(row_pow=0.4, long_hist=False, ext_w=0.0)),
}

if __name__ == "__main__":
    names = sys.argv[1:] or list(GRID)
    for name in names:
        models, kw = GRID[name]
        f = functools.partial(C.build, text=combo(models), sic=SIC, **{**FINAL, **kw})
        r = t12(f)
        r.update(t3(f))
        r["J"] = round(r["T1m"] + r["T2m"] + r["T3m"], 4)
        print(name, json.dumps(r), flush=True)
