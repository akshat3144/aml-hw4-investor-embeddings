# HW 4: Who Should Be in the Room

16-number vectors for 197 stocks and 200 investors in one shared space, scored by cosine similarity.

## Reproduce the two tables

Put the supplied `data/` folder (the four CSV files) next to `make_vectors.py`, then run:

```
python make_vectors.py                       # writes stocks.csv and managers.csv (needs numpy, pandas, pyarrow)
python check_tables.py stocks.csv managers.csv
```

`make_vectors.py` runs offline from `data/` and `external.parquet` only. It is closed-form (SVD with fixed signs,
weighted means), so the output repeats exactly; it took a few seconds on a laptop.

## Files

| file | what it is |
|---|---|
| `stocks.csv`, `managers.csv` | the submitted vectors |
| `make_vectors.py` | rebuilds both tables from `data/` and `external.parquet` |
| `external.parquet` | everything brought in from outside, one long table with a `kind` column (see below) |
| `design.md` | the design, at most 800 words |
| `answers.md` | answers to the six questions |
| `scripts/` | how `external.parquet` was collected, plus the backtests used to choose the design |

`external.parquet`, by `kind`:
- `f13`: the investors' full 13F-HR reports from EDGAR (common-stock lines, excluding funds, warrants, units, debt, preferreds and options), quarter-ends 2023-12-31 to 2025-12-31, all filed before 2026-03-02. 199 of the 200 investors have such reports.
- `doc`: for each stock, the dated SEC filing its description comes from, with form, filing date, accession number and URL, the SIC code from that filing's header, and the extracted business text. For established stocks this is the last 10-K/20-F filed before 2025-05-16; for listings, the last S-1/F-1 (or amendment) filed before the listing date.
- `emb`: embeddings of that text from approved models at the pinned revisions, run locally. The final tables use `bge`.
- `usic`: SIC codes for other securities in the 13F reports, from the SEC Financial Statement Data Sets 2025q1–q2 (filings before 2025-07-01), matched by issuer name.

## Rebuilding external.parquet (needs internet; not needed to reproduce the tables)

Set `SEC_USER_AGENT="Your Name you@example.com"`, as SEC requires. Then run these from the repository root, in order:

1. `python scripts/fetch_companies.py`: chooses one dated filing per stock.
2. `python scripts/fetch_texts.py`: extracts the business descriptions.
3. `python scripts/fetch_13f.py`: downloads the full 13F reports.
4. `python scripts/embed_texts.py bge gte mpnet minilm e5`: runs the approved models locally. They must first be downloaded to `cache/hf` at the revisions in the handout.
5. `python scripts/fetch_fsds.py`, then `python scripts/univ_sic.py`: industry codes for other securities.
6. `python scripts/build_external.py`

Downloads are cached in `cache/` and rate-limited below SEC's 10 requests per second.

## Backtests

The backtests live in `scripts/evalkit.py` (the three tests replayed on 2025 data at simulated decision dates, plus cross-validation over the 2025 listings). `scripts/model_c.py` is the research version of the final model, `scripts/final_grid.py` holds the final comparison and ablations, and `scripts/dims_sweep.py` the dimension sweeps quoted in `design.md`. Run them with `PYTHONPATH=scripts`. `model_a.py`, `model_b.py` and `probe_t1.py` are earlier alternatives, kept for the record.
