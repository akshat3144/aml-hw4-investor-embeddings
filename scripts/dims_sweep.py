"""Numbers quoted in design.md sections 5-6: number of factors (Test 3) and factors kept by new listings (Test 2)."""
import functools
import json

import evalkit as ek
import model_c as C
from final_grid import FINAL, SIC, t3
from sweep_text import combo, t12

T = combo(["bge"])
if __name__ == "__main__":
    for kw in [dict(k=8), dict(k=10), dict(k=13, c=0.0)]:
        f = functools.partial(C.build, text=T, sic=SIC, **dict(FINAL, **kw))
        print("factors", json.dumps(kw), json.dumps(t3(f)), flush=True)
    for ck in [4, 6, 10]:
        f = functools.partial(C.build, text=T, sic=SIC, **dict(FINAL, cold_k=ck))
        r = t12(f)
        print("cold_k", ck, json.dumps({k: r[k] for k in ("T1m", "T2m")}), flush=True)
