"""Small cached, rate-limited client for SEC EDGAR (used only by the data-collection scripts).

SEC asks automated clients to identify themselves and to stay below 10 requests/second.
All responses are cached under cache/edgar so every download happens at most once.
"""
import hashlib
import os
import threading
import time

import requests

# SEC requires a User-Agent with your own name and email: set SEC_USER_AGENT="Your Name you@example.com"
UA = {"User-Agent": os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com"),
      "Accept-Encoding": "gzip, deflate"}
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache", "edgar")
os.makedirs(ROOT, exist_ok=True)
_last = [0.0]
_lock = threading.Lock()
MIN_GAP = 0.13  # about 7.5 requests per second, under SEC's limit of 10


def _path(url):
    h = hashlib.sha1(url.encode()).hexdigest()
    ext = os.path.splitext(url.split("?")[0])[1][:6] or ".bin"
    return os.path.join(ROOT, h[:2], h + ext)


def get(url, binary=False, retries=5):
    """Return the body of url (str, or bytes if binary), downloading it once and caching it."""
    p = _path(url)
    if os.path.exists(p):
        with open(p, "rb") as f:
            data = f.read()
        return data if binary else data.decode("utf-8", errors="replace")
    for attempt in range(retries):
        with _lock:  # global spacing of request starts, shared by all threads
            wait = MIN_GAP - (time.time() - _last[0])
            if wait > 0:
                time.sleep(wait)
            _last[0] = time.time()
        try:
            r = requests.get(url, headers=UA, timeout=60)
        except requests.RequestException:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 200:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            tmp = p + f".{threading.get_ident()}.tmp"
            with open(tmp, "wb") as f:
                f.write(r.content)
            os.replace(tmp, p)
            return r.content if binary else r.content.decode("utf-8", errors="replace")
        if r.status_code == 404:
            return None
        time.sleep(2 * (attempt + 1))  # 403/429/5xx: back off
    raise RuntimeError(f"failed to fetch {url}")


def submissions(cik, older=False):
    """Company/filer submissions JSON: the 'recent' block, plus older pages if older=True."""
    import json

    j = json.loads(get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"))
    rec = j["filings"]["recent"]
    rows = [dict(zip(rec.keys(), vals)) for vals in zip(*rec.values())]
    for extra in (j["filings"].get("files", []) if older else []):
        e = json.loads(get("https://data.sec.gov/submissions/" + extra["name"]))
        rows += [dict(zip(e.keys(), vals)) for vals in zip(*e.values())]
    return j, rows


def archive_url(cik, accession, doc=""):
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{doc}"
