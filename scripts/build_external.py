"""Pack everything brought in from outside into external.parquet (one long table, column `kind` says which part).

kind = "doc"   one row per stock: the dated SEC filing used for its business description (form, filing date,
               accession, URL), the SIC code from that filing's header, and the extracted text
kind = "emb"   one row per stock and model: the text embedding (list of floats) from an approved model run locally
kind = "f13"   one row per investor, quarter-end and security: the investors' full 13F reports (common stock lines,
               options excluded) for quarter-ends 2023-12-31 .. 2025-12-31, all filed before 2026-03-02
kind = "usic"  one row per security in those reports that could be matched by issuer name to an XBRL filer in the
               SEC Financial Statement Data Sets 2025q1-q2 (filings before 2025-07-01): its SIC code
All sources are public EDGAR filings; every row carries the date it became public.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import f13  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = {"bge": "BAAI/bge-base-en-v1.5@70e8c52057794ed37381f7da00a03097927500a7",
          "gte": "thenlper/gte-base@7cb59c7765dcf58720e0e63bdbbf8c10cbba8668",
          "mpnet": "sentence-transformers/all-mpnet-base-v2@5681fe04da6e48e851d5dd1af673670cdb299753",
          "minilm": "sentence-transformers/all-MiniLM-L6-v2@7dbbc90392e2f80f3d3c277d6e90027e55de9125",
          "e5": "intfloat/e5-large-v2@b322e09026e4ea05f42beadf4d661fb4e101d311"}


def main():
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype=str)
    tx = pd.read_parquet(os.path.join(HERE, "cache", "company_texts.parquet")).set_index("stock_id").loc[comp.stock_id]
    doc = pd.DataFrame({"kind": "doc", "stock_id": comp.stock_id.values, "form": tx.form.values,
                        "filing_date": tx.filing_date.values, "accession": tx.accession.values,
                        "source": tx.source_url.values, "sic": tx.sic.astype(str).values,
                        "sic_desc": tx.sic_desc.values, "text": tx.text.values})
    embs = []
    for name, ref in MODELS.items():
        p = os.path.join(HERE, "cache", f"emb_{name}.npy")
        if not os.path.exists(p):
            print("missing embedding", name)
            continue
        E = np.load(p)
        embs.append(pd.DataFrame({"kind": "emb", "stock_id": comp.stock_id.values, "model": name, "source": ref,
                                  "vec": [list(map(float, row)) for row in E]}))
    f = f13.raw().copy()
    f13rows = pd.DataFrame({"kind": "f13", "manager_id": f.manager_id.values, "report_date": f.report_date.values,
                            "filing_date": f.filing_date.values, "cusip": f.cusip.values, "issuer": f.issuer.values,
                            "value": f.value.values, "shares": f.shares.values})
    us = pd.read_parquet(os.path.join(HERE, "cache", "univ_sic.parquet"))
    us = us[us.sic != ""]
    usic = pd.DataFrame({"kind": "usic", "cusip": us.cusip.values, "issuer": us.issuer.values, "sic": us.sic.values,
                         "sic_desc": us.matched_name.values,
                         "source": "SEC Financial Statement Data Sets 2025q1-2025q2 (sub.txt), filings before 2025-07-01"})
    out = pd.concat([doc] + embs + [f13rows, usic], ignore_index=True)
    path = os.path.join(HERE, "external.parquet")
    out.to_parquet(path, index=False, compression="zstd")
    print(out.kind.value_counts().to_dict(), round(os.path.getsize(path) / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
