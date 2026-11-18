"""Step 1 of data collection: pick, for every stock, one dated SEC document describing its business.

- established stocks: the latest annual report (10-K, 20-F or 40-F) filed before 2026-03-02;
- 2025 and 2026 listings: the latest registration statement (S-1, F-1 or an amendment) filed
  strictly before the listing date, i.e. information public before the IPO.
The SIC code is read from the header of that same filing, so it is dated too.
Writes cache/company_docs.csv.
"""
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgar  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF = "2026-03-02"
ANNUAL = ("10-K", "20-F", "40-F", "10-K405", "10-KT")
REG = ("S-1", "S-1/A", "F-1", "F-1/A", "S-11", "S-11/A", "424B4", "424B1", "424B3", "10-12B", "10-12B/A", "20-FR12B")


def header_info(cik, acc):
    """SIC code and conformed name from the filing's own SGML header."""
    txt = edgar.get(edgar.archive_url(cik, acc, acc + "-index-headers.html")) or ""
    m = re.search(r"STANDARD INDUSTRIAL CLASSIFICATION:\s*([^\[\n<]*)\[(\d{4})\]", txt)
    sic, sicdesc = (m.group(2), m.group(1).strip()) if m else ("", "")
    return sic, sicdesc


def main():
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype=str)
    out = []
    for _, c in comp.iterrows():
        j, rows = edgar.submissions(c.sec_cik)
        if c.status == "established":
            limit, forms = CUTOFF, ANNUAL
        else:
            limit, forms = c.listing_date, REG
        cands = [r for r in rows if r["form"] in forms and r["filingDate"] < limit]
        if not cands:  # older filings live on extra pages
            j, rows = edgar.submissions(c.sec_cik, older=True)
            cands = [r for r in rows if r["form"] in forms and r["filingDate"] < limit]
        cands.sort(key=lambda r: r["filingDate"], reverse=True)
        if c.status != "established":
            # prefer a full registration statement: amendments that are only exhibits are small
            full = [r for r in cands if r.get("size", 0) > 300_000 and r["form"] in REG]
            cands = full or cands
        if not cands:
            print("NO DOC", c.stock_id, c.ticker, c.company_name, c.status, file=sys.stderr)
            out.append(dict(stock_id=c.stock_id, form="", filing_date="", accession="", doc="", sic=j.get("sic", ""),
                            sic_desc=j.get("sicDescription", ""), sic_source="submissions"))
            continue
        r = cands[0]
        sic, sicdesc = header_info(c.sec_cik, r["accessionNumber"])
        src = "filing header"
        if not sic:
            sic, sicdesc, src = j.get("sic", ""), j.get("sicDescription", ""), "submissions"
        out.append(dict(stock_id=c.stock_id, form=r["form"], filing_date=r["filingDate"], accession=r["accessionNumber"],
                        doc=r["primaryDocument"], sic=sic, sic_desc=sicdesc, sic_source=src))
        print(c.stock_id, c.ticker, c.status, c.listing_date, "->", r["form"], r["filingDate"], sic, sicdesc, flush=True)
    pd.DataFrame(out).to_csv(os.path.join(HERE, "cache", "company_docs.csv"), index=False)


if __name__ == "__main__":
    main()
