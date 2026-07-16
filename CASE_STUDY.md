# Case Study — Tracking Domestic Institutional Smart Money on the NSE

*A portfolio write-up. Adapt freely for LinkedIn/blog once you've re-run the pipeline on live NSE data and swapped in the real numbers.*

---

Recently, I developed a **smart-money analytics pipeline for the Indian stock market (NSE)**, adapting the Indonesian *bandarmology* methodology to India's disclosure regime — tracking institutional footprints through **bulk/block-deal disclosures, FII/DII flows and delivery data** to identify potential accumulation by domestic institutional investors (DIIs).

**Why India is a better lab for this than Indonesia:** IDX analysts infer intent from anonymous broker codes. NSE publishes the **actual client name** on every bulk deal (>0.5 % of equity) and block deal. You don't have to guess who "Broker II" is — the disclosure tells you which mutual fund, insurer, AIF or prop desk traded. The methodological question becomes sharper: *whose repeated buying is actually followed by returns, and whose is noise?*

## Case study: BEL

One domestic fund kept appearing on Bharat Electronics' bulk-deal tape. Running the validation layer:

- **13 net-buy events**
- **+7.2 % average 10-day forward return**
- **77 % win rate**
- **p-value = 0.0098** (one-sided)
- Granger causality: the fund's deal flow **leads** the stock's returns at lag 5 (p = 0.048)

The event-study curve shows the classic fingerprint — flat drift before the fund's buy days, steady climb after.

## The part most write-ups skip

The same pipeline produced three sobering results:

1. A second accumulator with 9 repeated net-buys **failed validation** (33 % win rate, p = 0.58). Persistence ≠ predictiveness.
2. Only **6 of 69** participant-stock pairs passed the 5 % significance bar — with 69 tests, ~3.5 would pass **by chance**. Multiple testing is the trap.
3. Pooled across the whole NIFTY 50 panel, smart-money features predict **nothing** (OLS R² ≈ 0.0005; ML models ROC-AUC ≈ 0.47). Edge is participant-specific and stock-specific, or it doesn't exist.

## What the project combines

🛠️ **Data Engineering** — ETL pipeline (yfinance + NSE bulk/block deals + FII/DII flows + delivery bhavcopies), SQLite warehouse, live ingestion with deterministic offline fallback

📊 **Analytics** — 7-tab Streamlit dashboard: accumulation screener, "who keeps buying?" cumulative flow curves, delivery-spike detection, participant validation tables

🔬 **Data Science** — event studies with one-sided t-tests, Granger causality, OLS with Newey–West errors, logistic regression, random forest — and explicit multiple-testing accounting

A fun project that brought data engineering, analytics and quantitative research together in one workflow — and a reminder that in markets, the most persistent buyer is not always the most predictive one.

---

*Numbers above are from the bundled sample dataset (fictional participants) used to build and validate the pipeline; regenerate with `python -m nse_smartmoney.pipeline --live` before publishing. Education and research only — not investment advice.*
