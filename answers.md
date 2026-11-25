# HW4 answers

### 1. AUC and ranks

**Why only ranks:** AUC = P(a random holder's cosine > a random non-holder's) + ½·P(tie). Only these pairwise comparisons enter, so any strictly increasing transformation leaves it unchanged.

**Non-rank statistic:** Cohen's d: holders' mean cosine minus non-holders', over the pooled standard deviation.

**What it adds:** how far apart the groups are: how clearly a cut-off separates whom to call.

**When it misleads:** with few holders (often 3–10 of ~200) or heavy-tailed scores, one outlier shifts mean and SD, so d swings while AUC barely moves.

### 2. Objective and representation

**Objective:** Rᵢₗ = max over t of 0.5ᵗ·1[i held ℓ t quarters before Dec 2025] (197 stocks + other 13F stocks); D = diag(row totals). Minimise ‖D^(−0.4)R − UVᵀ‖²_F over rank-12 U, V (truncated SVD). Listings without holders: 0.6 × text-neighbour mean of V + 0.4 × same-SIC mean, first 8 factors. Two axes and two paddings are then appended.

**Missing holding:** a confident zero; only a past holding leaves a decayed 0.5ᵗ trace.

**Lost:** 12 free factors (8 for listings) miss niche tastes, position sizes, stock histories.

### 3. One vector for every stock

**Test 1: yes, it could do well.** Investors are ranked separately for each listing, so one shared ranking by IPO appetite still scores well: ranking by the number of other 2025 listings held gave AUC 0.87 (leave-one-out).

**Test 2: exactly 0.5.** With s_A = s_B, cos(i,A)+cos(j,B) = cos(i,B)+cos(j,A); every comparison ties and earns 0.5.

**Interpretation:** a high Test 1 score may only show who buys IPOs at all. Company–investor fit shows in Test 2.

### 4. Two other business decisions

**(a) Follow-on offerings and block trades.**

*Why it could help:* investors near a stock's holders who don't yet own it are natural buyers.

*Evidence needed:* on past placements, did these rankings beat the bank's call lists in allocations actually taken?

**(b) Peer groups for valuation or benchmarking.**

*Why it could help:* a high stock–stock cosine means a shared investor base, a similarity that complements industry codes.

*Evidence needed:* do these peers co-move in returns and valuation multiples, and do the groups stay stable across quarters?

### 5. Reported positions versus interest

**Reason 1:** 13F reports are quarter-end snapshots of long positions. A listing bought and flipped within the quarter is invisible, while hedges, merger arbitrage or index sleeves look like conviction.

**Reason 2:** the bank sets IPO allocations. A keen investor may get none; a token allocation may be held without interest.

**Additional test:** count as interest only new positions kept for two consecutive quarters and above a minimum portfolio weight. Better still, compare against the bank's book-building orders, which record demand directly.

### 6. If every bank used such tools

**Still an advantage?** Mostly no: everyone ranks the same public 13F data, so the tool becomes table stakes; the edge moves to private data (order books, meeting notes).

**Banks:** all call the same IPO-active funds, whose attention becomes scarce; they compete on access instead.

**Companies:** call lists no longer distinguish banks, so they choose on access, pricing and allocation quality.

**Investors:** may request confidential treatment, trade around quarter-ends, or use predictability to demand allocations. Models trained on allocations they steered would narrow shareholder bases and overlook newcomers.
