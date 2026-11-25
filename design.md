# Design: one shared 16-number space

## 1. Data and cutoffs
- **Holdings:** `holdings.csv` (2025 quarter-ends, all filed by 27 Feb 2026). No 2026 holdings are used.
- **Full 13F reports** (EDGAR, Dec 2023–Dec 2025, filed before 2 March 2026): 262,861 common-stock lines in 5,322 securities. Share-count matching gives each stock's CUSIP, hence 2023–24 history.
- **Company text:** listings use their last S-1/F-1 before the listing date (nothing post-IPO). Established stocks use their last 10-K/20-F before 16 May 2025, the earliest backtest date, so one text serves all backtests and the final model. About 2,500 words of business overview; SIC from the filing header.
- **Other securities' SIC codes:** SEC Financial Statement Data Sets (filings January–June 2025), matched by issuer name.
- **Language models:** approved only, at pinned revisions, run locally. Final: the mean `bge-base-en-v1.5` embedding of up to eight 300-word chunks (gte, mpnet, MiniLM and e5 also tried). No model was asked about any company. EDGAR requests carried a named User-Agent, under 10 per second.

Everything is in `external.parquet` (8.6 MB). `make_vectors.py` is closed-form with fixed SVD signs, so reruns are identical.

## 2. Cosine as an exact dot product
Coordinate 1 pads investors, and coordinate 2 pads stocks, to length 1, each being 0 on the other side. So cos(i,j) = mᵢ·sⱼ: the other coordinates add ordinary dot-product terms, and lengths cannot distort rankings. Weights are relative to the typical factor size.
- **3:** investor appetite (×0.35) × stock listing flag (1 without holders, 0.5 with holders, 0 established).
- **4:** a constant × standardised log holder count (×0.5).
- **5–16:** co-holding factors, U√Σ for investors and V√Σ for stocks.

## 3. How investors get their vectors
R is investors × (197 stocks + 2,907 other securities held by ≥2 investors). Rᵢₗ = max over quarters of 0.5^age if i held ℓ, else 0, and rows are divided by (row total)^0.4. A truncated SVD R ≈ UΣVᵀ keeps 12 factors. The other securities place specialists (biotech, bank investors) far better.

**Appetite** is the standardised log count of 2025 listings held, with 2024–25 new-issue buying breaking ties. 136 of the 200 investors held none, so it carries Test 1; in Test 2 common axes cancel.

## 4. How established stocks get their vectors
SVD factors V√Σ plus popularity (Test 3 compares within popularity groups). 2025 listings with holders are treated the same, with flag 0.5.

## 5. How new listings get their vectors
A listing without holders (all 44 of the 2026 listings) gets its factors from:
- 0.42 × a text-similarity softmax mean (τ=20) over 2025 listings with holders;
- 0.18 × the same mean over all stocks with holders;
- 0.40 × the mean factors of same-SIC other securities (4, 3, then 2 digits; ≥3 securities), rescaled to the text estimate's length.

Only the first 8 factors are kept: a prospectus locates broad investor groups, not fine structure. Test 2 scores 0.79 with 8 factors and 0.78 with 12. Flag 1 lets appetite rank IPO-active investors first.

## 6. How I reached 16 dimensions
16 = 2 padding + 2 axes + 12 factors.
- **Padding:** investor padding alone lets stock lengths re-enter the cosine, and Test 3 drops by 0.036.
- **Factors:** 12 suited Test 3 best (0.668 with 8, 0.685 with 12); a 13th instead of the popularity axis gave 0.683.

## 7. Testing on 2025 data
Backtests use only reports filed by their decision date and text dated before each listing. Industry codes, which barely change, affect only listings without holders.
- **Tests 1–2:** decisions on 1 Jul 2025 (July–September listings, judged 30 Sep; 16 with ≥3 holders) and 17 Nov (October–December; 6). Cross-validation within the cutoff hides the 2025 listings' holders and rebuilds everything.
- **Test 3:** Mar→Sep (decision 16 May), Jun→Dec (2 Sep) and Sep→Dec 2025 (17 Nov), with the handout's popularity groups.

| AUC (0.5 = chance) | Test 1 | Test 2 | Test 3 |
|---|---|---|---|
| **final model** | **0.875** | **0.794** | **0.685** |
| no industry averages | 0.860 | 0.736 | 0.685 |
| 197 stocks only | 0.865 | 0.696 | 0.649 |
| no appetite axis | 0.678 | 0.794 | 0.683 |
| no older reports | 0.808 | 0.794 | 0.680 |

Test 1 estimates span 0.81–0.91 (lowest on 1 July, before any 2025 listing had holders); Test 2 spans 0.72–0.90. Weights were tuned on few listings, so expect some optimism. Rejected: learned text maps, two-tower networks and position-size weights. Vectors fitted to a listing's own holders would reach 0.90 on Test 2: the gap is prospectus versus holders.
