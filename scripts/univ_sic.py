"""SIC codes for the other securities in the investors' 13F reports.

Source: SEC Financial Statement Data Sets 2025q1 and 2025q2 (sub.txt: one row per XBRL filing, with the filer's
name, SIC code and filing date). Only filings dated before 2025-07-01 are used, i.e. public before the earliest
listing backtest as well as before the 2 March 2026 cutoff. A 13F line is matched to a filer by normalised
issuer name: an exact match first; otherwise filers sharing the first word whose later words abbreviate-match the
issuer's in order (accepted only if all such filers share one SIC code). Fund and ETF lines are left unmatched.
Writes cache/univ_sic.parquet (cusip, issuer, sic, matched_name, how).
"""
import os
import re

import pandas as pd

import f13

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DROP = {"INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY", "LTD", "LIMITED", "PLC", "LLC", "LP", "L P",
        "THE", "SA", "NV", "N V", "AG", "SE", "AB", "ASA", "AS", "SPA", "BV", "HLDGS", "HOLDINGS", "HOLDING", "GROUP",
        "GRP", "COM", "CL", "A", "B", "C", "NEW", "DEL", "DE", "ADR", "SPONSORED", "SPON", "ADS", "ORD", "SHS", "NV",
        "TR", "TRUST", "INTL", "INTERNATIONAL", "CAP", "STK", "COMMON", "STOCK", "CLASS", "REIT", "MD", "NY", "PA"}
ABBR = {"LABS": "LABORATORIES", "PHARMACEUTICALS": "PHARMACEUTICALS", "PHARMA": "PHARMACEUTICALS",
        "TECH": "TECHNOLOGIES", "TECHNOLOGY": "TECHNOLOGIES", "SYS": "SYSTEMS", "SVCS": "SERVICES", "MGMT": "MANAGEMENT",
        "FINL": "FINANCIAL", "BANCORPORATION": "BANCORP", "COMMUNICATIONS": "COMMUNICATIONS", "COMM": "COMMUNICATIONS",
        "INDS": "INDUSTRIES", "MTRS": "MOTORS", "ENTMT": "ENTERTAINMENT", "PPTYS": "PROPERTIES", "RES": "RESOURCES",
        "THERAPEUTICS": "THERAPEUTICS", "THERAPEUTIC": "THERAPEUTICS", "SOLUTIONS": "SOLUTIONS", "BK": "BANK",
        "AMER": "AMERICA", "AMERN": "AMERICAN", "NATL": "NATIONAL", "ELEC": "ELECTRIC", "MFG": "MANUFACTURING"}


def norm(name):
    s = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper().replace("&", " AND "))
    toks = [ABBR.get(t, t) for t in s.split()]
    toks = [t for t in toks if t not in DROP]
    return " ".join(toks)


def main():
    subs = []
    for q in ("2025q1", "2025q2"):
        d = pd.read_csv(os.path.join(HERE, "cache", "fsds", f"sub_{q}.txt"), sep="\t", dtype=str,
                        usecols=["cik", "name", "sic", "filed", "former"])
        subs.append(d)
    sub = pd.concat(subs)
    sub = sub[(sub.filed < "20250701") & sub.sic.notna()]
    sub = sub.sort_values("filed").drop_duplicates("cik", keep="last")
    rows = []
    for _, r in sub.iterrows():
        rows.append((norm(r["name"]), r["sic"], r["name"]))
        if isinstance(r["former"], str):
            rows.append((norm(r["former"]), r["sic"], r["former"]))
    names = pd.DataFrame(rows, columns=["key", "sic", "name"]).drop_duplicates("key")
    names = names[names.key.str.len() > 0]
    exact = dict(zip(names.key, zip(names.sic, names.name)))
    by_first = {}
    for key, sic, nm in zip(names.key, names.sic, names.name):
        by_first.setdefault(key.split()[0], []).append((key.split(), sic, nm))

    def abbrev_match(toks, cand):
        """Every issuer word after the first is a prefix of (or prefixed by) a later candidate word, in order,
        and the candidate has no extra leading words that the issuer lacks."""
        pos = 1
        for t in toks[1:]:
            while pos < len(cand) and not (cand[pos].startswith(t) or t.startswith(cand[pos])):
                pos += 1
            if pos >= len(cand):
                return False
            pos += 1
        return len(cand) - len(toks) <= 1

    fund = re.compile(r"\b(ETF|FD|FUND|ISHARES|SPDR|VANGUARD|INVESCO|PROSHARES|DIREXION|ISHARE|INDEX|TR UNIT)\b")
    f = f13.raw()
    iss = f.groupby("cusip").issuer.agg(lambda s: s.value_counts().index[0]).reset_index()
    out = []
    for c, name in zip(iss.cusip, iss.issuer):
        key = norm(name)
        toks = key.split()
        hit, how = None, ""
        if fund.search(str(name).upper()) or not toks:
            pass
        elif key in exact:
            hit, how = exact[key], "exact"
        elif len(toks) >= 2 or len(toks[0]) >= 6:
            cands = [(sic, nm) for ct, sic, nm in by_first.get(toks[0], []) if abbrev_match(toks, ct)]
            if cands and len({s for s, _ in cands}) == 1:
                hit, how = cands[0], "abbrev"
        out.append(dict(cusip=c, issuer=name, sic=hit[0] if hit else "", matched_name=hit[1] if hit else "", how=how))
    out = pd.DataFrame(out)
    out.to_parquet(os.path.join(HERE, "cache", "univ_sic.parquet"), index=False)
    print(out.how.value_counts().to_dict(), "of", len(out))


if __name__ == "__main__":
    main()
