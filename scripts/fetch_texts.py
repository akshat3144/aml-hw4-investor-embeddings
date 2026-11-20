"""Step 2 of data collection: download each stock's chosen SEC document and extract a business description.

For registration statements the description starts at the prospectus summary / overview; for annual
reports at "Item 1. Business" (10-K) or "Item 4. Information on the Company" (20-F). If the newest
eligible filing has no usable description (an exhibit-only amendment, say), older eligible filings are tried.
Writes cache/company_texts.parquet with the filing used and up to ~2,500 words of text.
"""
import html
import os
import re
import sys

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edgar  # noqa: E402
from fetch_companies import ANNUAL, CUTOFF, REG, header_info  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_WORDS = 2500
# Established stocks: the last annual report filed before 16 May 2025, the earliest simulated decision date
# of the backtests, so the same text is legitimate in every backtest and in the final tables.
EST_LIMIT = "2025-05-16"
DASH = "[-–—:.]?"


def h(*words):
    """Heading regex tolerant of small-caps styling that splits a word's first letter ("B USINESS")."""
    return r"\b" + r"\s+".join(w[0] + r"\s?" + w[1:] for w in words) + r"\b"


HEADS_REG = [h("prospectus", "summary"), h("overview"), h("our", "company"), h("our", "business"),
             h("business", "overview"), h("summary")]
HEADS_10K = [r"item\s*1\s*\.?\s*" + DASH + r"\s*" + h("business"), h("description", "of", "business"),
             h("overview"), h("general"), h("business")]
HEADS_20F = [r"item\s*4\s*\.?\s*" + DASH + r"\s*" + h("information", "on", "the", "company"), h("business", "overview"),
             h("overview")]
# a table of contents mentions many other items / section names right after the heading
TOC_RE = r"\bitem\s*\d+[a-c]?\b|risk\s+factors|use\s+of\s+proceeds|dilution|capitalization|properties|legal\s+proceedings"
BUSINESS_RE = (r"\b(customers?|clients?|products?|platform|services|solutions|patients|markets?|revenue|operat\w+|"
               r"provid\w+|develop\w+|leading|industry|technolog\w+|global)\b")
LEGAL_RE = (r"\b(indemnif\w+|material\s+weakness|articles\s+of\s+association|adversely|forward-looking|"
            r"could\s+harm|shareholders?\s+(may|will)|pro\s+forma|footnote|note\s+\d+|underwriters?|"
            r"cash\s+equivalents|net\s+loss|in\s+millions)\b")
DESCRIBE_RE = r"(\bwe\s+are\s+(a|an|the|one)\b|\bis\s+(a|an|the|one\s+of)\s+(leading|global|largest|clinical|[a-z\-]+\s+(company|provider|platform)))"


def html_to_text(raw):
    soup = BeautifulSoup(raw, "lxml")
    for t in soup(["script", "style", "ix:header"]):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        t.decompose()
    txt = soup.get_text("\n")
    txt = html.unescape(txt).replace("\xa0", " ").replace("​", "")
    txt = txt.translate(str.maketrans({"\x91": "'", "\x92": "'", "\x93": '"', "\x94": '"', "\x96": "-", "\x97": "-"}))
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n\s*\n+", "\n", txt)
    return txt


def prose_density(window):
    """Share of characters that sit in long lines, i.e. running prose rather than a table of contents."""
    lines = window.split("\n")
    tot = sum(len(line) for line in lines) or 1
    return sum(len(line) for line in lines if len(line) > 200) / tot


def good_window(win):
    """Running prose that describes a business, not a contents table, a legal clause or a risk factor."""
    head = win[:1500]
    if len(re.findall(TOC_RE, head, flags=re.I)) >= 3:
        return False
    junk = sum(ch.isdigit() or ch in "._$%|" for ch in head) / max(len(head), 1)
    if junk > 0.08:  # dotted leaders, page numbers, financial tables
        return False
    biz = len(re.findall(BUSINESS_RE, win[:3000], flags=re.I))
    legal = len(re.findall(LEGAL_RE, win[:3000], flags=re.I))
    return biz >= 10 and legal <= 3


def extract(txt, heads):
    """Text starting at the first heading occurrence that is followed by a business description."""
    n = len(txt)
    for pat in heads:
        for m in re.finditer(pat, txt, flags=re.I):
            if m.start() < 0.005 * n:
                continue
            if good_window(txt[m.end(): m.end() + 6000]):
                words = txt[m.end():].split()
                return " ".join(words[:N_WORDS]), "head:" + pat[:20]
    # fallback: the first paragraph that describes the company ("We are a ...", "X is a leading ...")
    for m in re.finditer(DESCRIBE_RE, txt, flags=re.I):
        start = txt.rfind("\n", 0, m.start()) + 1
        if start < 0.005 * n:
            continue
        if good_window(txt[start: start + 6000]):
            words = txt[start:].split()
            return " ".join(words[:N_WORDS]), "describe"
    words = txt.split()
    return " ".join(words[:N_WORDS]), "fallback:start"


def main():
    comp = pd.read_csv(os.path.join(HERE, "data", "companies.csv"), dtype=str)
    rows = []
    for _, c in comp.iterrows():
        _, subs = edgar.submissions(c.sec_cik)
        limit, forms = (EST_LIMIT, ANNUAL) if c.status == "established" else (c.listing_date, REG)
        cands = [r for r in subs if r["form"] in forms and r["filingDate"] < limit]
        if not cands:
            _, subs = edgar.submissions(c.sec_cik, older=True)
            cands = [r for r in subs if r["form"] in forms and r["filingDate"] < limit]
        if c.status == "established":  # the last annual report before the earliest backtest date
            cands = [r for r in cands if r["filingDate"] >= "2024-01-01"]
        cands.sort(key=lambda r: (r["filingDate"], r["accessionNumber"]), reverse=True)
        chosen = None
        for r in cands[:8]:
            raw = edgar.get(edgar.archive_url(c.sec_cik, r["accessionNumber"], r["primaryDocument"]))
            if not raw:
                continue
            txt = html_to_text(raw)
            if r["form"] in ("20-F", "40-F"):
                heads = HEADS_20F
            elif r["form"] in ANNUAL:
                heads = HEADS_10K
            else:
                heads = HEADS_REG
            body, how = extract(txt, heads)
            if len(body.split()) >= 1200 and not how.startswith("fallback"):
                chosen = (r, body, how, len(txt))
                break
            if chosen is None:
                chosen = (r, body, how, len(txt))
        r, body, how, ntxt = chosen
        sic, sicdesc = header_info(c.sec_cik, r["accessionNumber"])
        rows.append(dict(stock_id=c.stock_id, form=r["form"], filing_date=r["filingDate"],
                         accession=r["accessionNumber"], document=r["primaryDocument"],
                         source_url=edgar.archive_url(c.sec_cik, r["accessionNumber"], r["primaryDocument"]),
                         sic=sic, sic_desc=html.unescape(sicdesc), section=how[:12], doc_chars=ntxt, text=body))
        print(c.stock_id, c.ticker, r["form"], r["filingDate"], how[:12], len(body.split()), "|",
              body[:110].replace("\n", " "), flush=True)
    pd.DataFrame(rows).to_parquet(os.path.join(HERE, "cache", "company_texts.parquet"), index=False)


if __name__ == "__main__":
    main()
