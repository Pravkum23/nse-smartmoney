# 📈 NSE Smart-Money — Institutional Flow Tracker for Indian Stocks

An end-to-end data pipeline for testing a simple question:

> **Does institutional ("smart money") accumulation — visible through bulk/block-deal disclosures, delivery percentages and FII/DII flows — actually precede stronger returns in Indian stocks, or is it folklore?**

This is the Indian adaptation of the *bandarmology* methodology used on the Indonesian Stock Exchange. India has no public broker-level flow data, but NSE discloses something arguably better: **named counterparties**. Every bulk deal (>0.5 % of equity) and block deal publishes the actual client name — so instead of profiling anonymous broker codes, this pipeline profiles **Foreign Institutional Investors (FIIs), Domestic Institutional Investors (DIIs — the primary focus), prop desks and HNIs by name**, then statistically validates whose repeated buying actually predicts forward returns.

Built around a notebook-first workflow, with a Streamlit dashboard for interactive exploration.

---

## What this project demonstrates

One project exercising the full data lifecycle — practical **Data Engineering**, **Data Analysis** and **Data Science** in one place.

| Role | What's built here |
| ---- | ----------------- |
| 🛠️ **Data Engineer** | End-to-end **ETL pipeline** ingesting yfinance OHLCV, NSE FII/DII provisional flows, historical bulk/block deals and security-wise delivery bhavcopies into a **SQLite analytics warehouse**; graceful **fallback to a bundled deterministic sample dataset** when live sources are unreachable; a modular, reusable package (`config` · `nse_api` · `prices` · `storage` · `pipeline` · `features`). |
| 📊 **Data Analyst** | A **7-tab interactive Streamlit dashboard** (KPI cards, price + deal-marker charts, FII/DII flow history, cumulative "who keeps buying?" curves, delivery-spike analysis, validation tables, cross-universe screener); business framing that turns raw disclosures into *"which institution is actually accumulating?"* |
| 🔬 **Data Scientist** | **Feature engineering** (forward/backward returns, delivery & volume z-scores, deal-flow aggregates, composite accumulation score); **event studies** with one-sided t-tests and multiple-testing awareness; **Granger causality** for lead/lag; **OLS with HAC/Newey–West errors**; **logistic regression & random forest** with chronological splits, scored by precision/recall/ROC-AUC. |

**Tech stack:** Python · pandas · NumPy · statsmodels · scikit-learn · SciPy · SQLite · Streamlit · Plotly · yfinance · Jupyter

---

## Why India needs a different playbook (and why it's better)

Indonesian *bandarmology* tracks anonymous **broker codes** — you infer who is behind "BK" or "II". Indian exchanges disclose more:

| Signal | Source | What it reveals |
| ------ | ------ | --------------- |
| **Bulk deals** (>0.5 % of equity) | NSE daily disclosure | **Named** buyer/seller, quantity, price |
| **Block deals** (₹10 cr+ window trades) | NSE daily disclosure | Institution-to-institution crosses, named |
| **FII/DII daily flows** | NSE provisional data | Aggregate foreign vs domestic conviction (₹ crore) |
| **Delivery percentage** | NSE bhavcopy | Share of volume actually taken to demat — conviction vs intraday churn |
| **Shareholding patterns** (roadmap) | Quarterly filings | FII/DII/promoter stake changes per company |

The pipeline classifies every deal participant into behavioural profiles:

| Profile | Represents |
| ------- | ---------- |
| 🔵 **DII — Mutual Fund** | SBI MF, Nippon, HDFC AMC … patient domestic money |
| 🩵 **DII — Insurance** | LIC & friends — the slowest, stickiest capital |
| 🟢 **DII — AIF / PMS / NBFC** | Domestic alternatives — often the first movers |
| 🟣 **FII / FPI** | Foreign institutions routing via SG/Mauritius/Lux entities |
| 🟠 **Prop / HFT Desk** | Two-way churn — only the *net* position matters |
| 🔴 **HNI / Family Office** | Known super-investors and family trusts |
| ⚪ **Other / Retail-HNI** | Everyone else |

> Profiles are keyword heuristics over disclosed names — they describe how an account *tends* to behave, not the identity of any end client. **This project focuses on the DII profiles**: domestic institutional accumulation has historically been the counter-cyclical, price-insensitive bid in Indian markets.

---

## Architecture

```
   yfinance OHLCV ─┐
 NSE FII/DII flows ─┤   pipeline.py    ┌─> SQLite warehouse (5 tables)
NSE bulk/block CSV ─┼──> clean/land ──>│      data/db/smartmoney.sqlite
 NSE delivery bhav ─┘   (sample-data   └─> features + validation tables
                         fallback)            │
                                              ├─> notebooks/01_…ipynb  (analysis & models)
                                              └─> dashboard/app.py    (Streamlit, 7 tabs)
```

## Repository structure

```
nse-smartmoney/
├── requirements.txt
├── notebooks/
│   └── 01_smartmoney_end_to_end.ipynb
├── dashboard/
│   └── app.py
├── src/nse_smartmoney/
│   ├── config.py        # universe, profiles, parameters
│   ├── nse_api.py       # live NSE fetchers (cookies, retries)
│   ├── prices.py        # yfinance OHLCV
│   ├── sample_data.py   # deterministic offline dataset
│   ├── storage.py       # SQLite warehouse
│   ├── pipeline.py      # ETL orchestration + CLI
│   ├── features.py      # participant classification + features
│   ├── analysis.py      # event studies, Granger, delivery studies
│   └── modeling.py      # OLS (HAC), logistic regression, random forest
└── data/
    ├── raw/  processed/  db/smartmoney.sqlite
```

## Setup

```bash
git clone <your-repo-url>
cd nse-smartmoney
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run the pipeline (tries live sources, falls back to sample data)
python -m nse_smartmoney.pipeline           # auto
python -m nse_smartmoney.pipeline --live    # live only, fail loudly
python -m nse_smartmoney.pipeline --sample  # bundled sample data

# explore
jupyter notebook notebooks/01_smartmoney_end_to_end.ipynb
streamlit run dashboard/app.py
```

**Note on live NSE access:** NSE rate-limits and blocks datacenter IPs. The fetchers warm up session cookies with browser headers, but run them from a residential connection. FII/DII provisional data covers the latest day — schedule the pipeline daily (Task Scheduler / cron) to build a history, exactly like the IDX broker-snapshot backfill.

---

## Results — sample-run case study

> ⚠️ **Read this first:** live NSE endpoints are unreachable from the sandbox this repo was built in, so the numbers below come from the **bundled deterministic sample dataset** (`--sample`, fictional participant names, seed=42). They demonstrate *what the pipeline surfaces and how to read it* — regenerate them on real data with `--live`. The methodology, statistics and dashboard are identical either way.

### The screener surfaces an accumulator — BEL

Scanning all 69 participant–stock pairs with ≥5 net-buy events, one domestic fund dominates the validation table for **BEL (Bharat Electronics)**:

| Participant | Profile | Events | Mean 10-day fwd return | Win rate | p-value | Significant? |
| ----------- | ------- | ------ | ---------------------- | -------- | ------- | ------------ |
| **MERU GROWTH FUND - SERIES II** | DII — AIF/PMS | **13** | **+7.22 %** | **76.9 %** | **0.0098** | ✅ |

The event-study curve shows the classic smart-money fingerprint: flat-to-negative drift *before* the fund's net-buy days, then a steady climb to **+7.2 % by day 10** after them. Granger causality agrees — smart-money deal flow *leads* BEL returns at lag 5 (**p = 0.048**), while delivery z-score alone does not.

### The honest counterweights

The same pipeline also produces the results a skeptic would demand:

1. **Most participants have no edge.** Only 6 of 69 pairs clear the one-sided 5 % test — and with 69 tests, ~3.5 false positives are expected by chance. A second planted accumulator (SILVER OAK MUTUAL FUND on TATAMOTORS: 9 events, 33 % win rate, p = 0.58) **failed validation** — repeated buying alone proves nothing.
2. **Pooled predictability is ~zero.** OLS across the full panel: R² ≈ 0.0005, no significant coefficients (HAC errors). Logistic regression and random forest score **ROC-AUC ≈ 0.46–0.47** out-of-sample — coin flips.
3. **Delivery spikes alone don't pay.** Across the universe, volume-confirmed delivery spikes show no reliable positive edge at 5 days.

**The verdict the data supports:** smart-money edge — when it exists — is *participant-specific and stock-specific*, not a universe-wide factor. Generic "institutions bought today" features are noise; the value is in the event-study layer that isolates *who* has a repeatable, statistically significant footprint. That's exactly what the IDX project found with broker GA vs the big-volume brokers, replicated here with named Indian participants.

---

## Methodology

- **Forward/backward returns** — `fwd_ret_5d/10d` are labels; every feature is computable at the close of the signal day (no look-ahead).
- **Participant validation** (`participant_alpha_scan`) — for each (stock, participant) with ≥5 net-buy days: mean/median forward return, win rate, one-sided t-test (H₁: μ > 0), flagged significant at α = 5 % only if the mean is positive.
- **Event-study curves** — average cumulative return path from day −5 to +15 around net-buy events.
- **Granger causality** — do lagged flows (aggregate FII/DII, per-stock smart-money deal flow, delivery z) improve return prediction beyond price history? (statsmodels, min-p across lags 1–5.)
- **Delivery-spike study** — Welch t-test of forward returns on spike days vs all other days.
- **OLS with HAC (Newey–West) errors** — panel-level signal check robust to autocorrelation.
- **Classification** — logistic regression + random forest, chronological 70/30 split, precision/recall/ROC-AUC vs base rate.
- **Multiple-testing awareness** — every validation table reports how many false positives chance alone would produce.

## Roadmap

- [ ] Quarterly shareholding-pattern ingestion (FII/DII/promoter stake deltas per company)
- [ ] F&O participant-wise open interest (FII index/stock futures positioning)
- [ ] Walk-forward backtest of validated-participant-follow strategies
- [ ] Daily scheduled runs + Telegram/email alerts on new validated accumulation events
- [ ] Benjamini–Hochberg FDR correction in the validation table

## Disclaimer

Education and personal research only — **not investment advice**. Bulk/block-deal disclosures identify executing counterparties, not intent; a validated historical pattern is a research lead, not a trading signal. Respect NSE's terms of use when fetching data. Sample-mode participant names are fictional; any resemblance to real entities is coincidental.
