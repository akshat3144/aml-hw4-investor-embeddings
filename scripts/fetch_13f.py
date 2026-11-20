"""Step 3 of data collection: the full 13F holdings (all securities, not only the 197 stocks) of the 200 investors.

Original 13F-HR reports for quarter-ends 2023-12-31 .. 2025-12-31 that were filed on or before 2026-03-01,
so everything is public before the 2 March 2026 cutoff. Each row keeps its filing date, so a backtest can
use only the reports that were public at its own simulated decision date.
Writes cache/f13_full.parquet.
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgar  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIRST, LAST, CUTOFF = "2023-12-31", "2025-12-31", "2026-03-02"


def strip_ns(tag):
    return tag.split("}")[-1]


def parse_table(xml_text):
    root = ET.fromstring(xml_text)
    out = []
    for it in root.iter():
        if strip_ns(it.tag) != "infoTable":
            continue
        d = {}
        for ch in it.iter():
            t = strip_ns(ch.tag)
            if t in ("nameOfIssuer", "titleOfClass", "cusip", "value", "putCall", "sshPrnamt", "sshPrnamtType"):
                d[t] = (ch.text or "").strip()
        out.append(d)
    return out


def main():
    from concurrent.futures import ThreadPoolExecutor

    mi = pd.read_csv(os.path.join(HERE, "data", "manager_info.csv"), dtype=str)
    with ThreadPoolExecutor(max_workers=8) as ex:  # requests are still spaced globally by edgar.get
        parts = list(ex.map(one_manager, [m for _, m in mi.iterrows()]))
    df = pd.DataFrame([r for p in parts for r in p])
    df.to_parquet(os.path.join(HERE, "cache", "f13_full.parquet"), index=False)
    print(df.shape)


def one_manager(m):
    rows = []
    if True:
        _, subs = edgar.submissions(m.sec_cik)
        fil = [r for r in subs if r["form"] == "13F-HR" and FIRST <= r.get("reportDate", "") <= LAST
               and r["filingDate"] < CUTOFF]
        if len(fil) < 9:  # older reports may sit on extra pages
            _, subs = edgar.submissions(m.sec_cik, older=True)
            fil = [r for r in subs if r["form"] == "13F-HR" and FIRST <= r.get("reportDate", "") <= LAST
                   and r["filingDate"] < CUTOFF]
        # one original report per quarter-end: the first one filed
        fil.sort(key=lambda r: (r["reportDate"], r["filingDate"], r["accessionNumber"]))
        seen = set()
        for r in fil:
            if r["reportDate"] in seen:
                continue
            seen.add(r["reportDate"])
            idx = edgar.get(edgar.archive_url(m.sec_cik, r["accessionNumber"], "index.json"))
            if not idx:
                continue
            import json

            items = json.loads(idx)["directory"]["item"]
            xmls = [it["name"] for it in items if it["name"].lower().endswith(".xml")
                    and it["name"].lower() != "primary_doc.xml"]
            recs = []
            for name in xmls:
                body = edgar.get(edgar.archive_url(m.sec_cik, r["accessionNumber"], name))
                if body and re.search(r"infoTable", body[:5000] + body[-2000:], flags=re.I):
                    try:
                        recs = parse_table(body.encode("utf-8"))
                    except ET.ParseError:
                        recs = []
                    if recs:
                        break
            for d in recs:
                rows.append(dict(manager_id=m.manager_id, report_date=r["reportDate"], filing_date=r["filingDate"],
                                 accession=r["accessionNumber"], cusip=d.get("cusip", "").upper(),
                                 issuer=d.get("nameOfIssuer", ""), title=d.get("titleOfClass", ""),
                                 value=float(d.get("value") or 0), shares=float(d.get("sshPrnamt") or 0),
                                 sh_type=d.get("sshPrnamtType", ""), put_call=d.get("putCall", "")))
        print(m.manager_id, len(seen), "reports", flush=True)
    return rows


if __name__ == "__main__":
    main()
