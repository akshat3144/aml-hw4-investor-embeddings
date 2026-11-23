"""Step 5 of data collection: SEC Financial Statement Data Sets 2025q1 and 2025q2 (filer list with SIC codes).

Only sub.txt is kept (one row per XBRL filing: filer CIK, name, SIC code, filing date). These two quarters cover
filings made January-June 2025, so they were public before every simulated decision date and before the cutoff.
Writes cache/fsds/sub_2025q1.txt and cache/fsds/sub_2025q2.txt; univ_sic.py turns them into SIC codes for the
securities in the investors' 13F reports.
"""
import io
import os
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgar  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    out = os.path.join(HERE, "cache", "fsds")
    os.makedirs(out, exist_ok=True)
    for q in ("2025q1", "2025q2"):
        b = edgar.get(f"https://www.sec.gov/files/dera/data/financial-statement-data-sets/{q}.zip", binary=True)
        with zipfile.ZipFile(io.BytesIO(b)) as z, open(os.path.join(out, f"sub_{q}.txt"), "wb") as f:
            f.write(z.read("sub.txt"))
        print(q, "ok")


if __name__ == "__main__":
    main()
