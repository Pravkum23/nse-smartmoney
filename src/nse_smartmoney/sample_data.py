"""Deterministic sample-data generator (offline fallback).

When live NSE / Yahoo sources are unreachable, the pipeline lands this
bundled dataset instead so the full workflow — warehouse, features,
event studies, models, dashboard — runs end-to-end out of the box.

The data is synthetic but statistically realistic:

* GBM daily prices for the NIFTY 50 universe (~15 months);
* market-level FII/DII flows with regime behaviour (FII risk-on/off,
  DII counter-cyclical buying — the pattern seen in real Indian data);
* delivery percentages drawn per-ticker with volume-linked noise;
* bulk/block deals from a pool of *fictional* participant names crafted
  to exercise every classification profile;
* two "spotlight" accumulation stories are planted so the validation
  layer has a genuine signal to find — a fictional DII fund repeatedly
  net-buying ahead of drift. This mirrors what the pipeline should
  surface on real data.

All names are fictional. Seeded RNG → identical output every run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import NIFTY50, WATCHLIST

SEED = 42
N_DAYS = 320

# fictional participants, keyword-compatible with PARTICIPANT_PROFILES
POOL = {
    "DII — Mutual Fund": [
        "SILVER OAK MUTUAL FUND", "GREAT EASTERN ASSET MANAGEMENT",
        "PEEPAL TREE MUTUAL FUND", "SARASWATI ASSET MANAGEMENT",
    ],
    "DII — Insurance": [
        "BHARAT JEEVAN INSURANCE COMPANY", "SURAKSHA GENERAL INSURANCE",
    ],
    "DII — AIF / PMS / NBFC": [
        "MERU GROWTH FUND - SERIES II", "KAVERI ALTERNATIVE INVESTMENT FUND",
        "INDUS FLAGSHIP FUND I", "ARJUNA WEALTH MANAGEMENT PRIVATE LIMITED",
    ],
    "FII / FPI": [
        "ALBION CAPITAL SINGAPORE PTE LTD", "NORDIC INDIA UCITS FUND",
        "ATLAS EMERGING MARKETS LLC", "CLEARWATER MAURITIUS HOLDINGS",
    ],
    "Prop / HFT Desk": [
        "VELOCITY SECURITIES RESEARCH PRIVATE LIMITED",
        "ZENITH BROKING SERVICES LLP", "QUANTA TRADETECH LLP",
        "ORBIT STRATEGIC VENTURES LLP",
    ],
    "HNI / Family Office": [
        "RAMESHBHAI PATEL HUF", "VASUDEV FAMILY TRUST",
        "SUNITA DEVI AGRAWAL",
    ],
    "Other / Retail-HNI": [
        "MOHAN KUMAR REDDY", "PRIYA NAIR", "ARJUN SINGH RATHORE",
        "DEEPAK VERMA", "KISHORE BHAI SHAH",
    ],
}

# Planted smart-money stories: (symbol, accumulator, n_events, drift/day
# over the 10 sessions after each event). These give the event-study layer
# a real signal to detect and validate. Drift only follows ~70 % of the
# events so win rates land in a realistic 60-80 % band.
SPOTLIGHTS = [
    ("BEL", "MERU GROWTH FUND - SERIES II", 13, 0.009),
    ("TATAMOTORS", "SILVER OAK MUTUAL FUND", 9, 0.007),
]
HIT_RATE = 0.72                 # share of planted events followed by drift


def _trading_days(end: pd.Timestamp, n: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(end=end, periods=n)
    return days


def generate(end_date: str | None = None) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    days = _trading_days(end, N_DAYS)
    nd = len(days)
    syms = WATCHLIST

    # ------------------------------------------------------------------ prices
    base_price = rng.uniform(80, 4000, len(syms))
    drift = rng.normal(0.0004, 0.0006, len(syms))
    vol = rng.uniform(0.012, 0.028, len(syms))
    mkt = rng.normal(0.0003, 0.009, nd)              # common market factor
    rets = (drift[None, :] + 0.8 * mkt[:, None]
            + rng.normal(0, 1, (nd, len(syms))) * vol[None, :])

    # plant accumulation events + post-event drift for spotlights
    planted: dict[str, list[int]] = {}
    for sym, _client, n_ev, d in SPOTLIGHTS:
        j = syms.index(sym)
        # events spaced through the sample, none in the last 15 sessions
        ev_idx = np.sort(rng.choice(np.arange(30, nd - 15), n_ev,
                                    replace=False))
        planted[sym] = ev_idx.tolist()
        for i in ev_idx:
            horizon = slice(i + 1, min(i + 11, nd))
            if rng.random() < HIT_RATE:      # winner: gentle upward drift
                rets[horizon, j] += rng.normal(d, 0.006,
                                               rets[horizon, j].shape)
            else:                            # loser: mild adverse move
                rets[horizon, j] += rng.normal(-d * 0.6, 0.006,
                                               rets[horizon, j].shape)

    close = base_price[None, :] * np.exp(np.cumsum(rets, axis=0))
    open_ = close * (1 + rng.normal(0, 0.003, close.shape))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, close.shape)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, close.shape)))
    base_volu = rng.uniform(5e5, 3e7, len(syms))
    volu = base_volu[None, :] * np.exp(rng.normal(0, 0.35, close.shape))

    prices = pd.DataFrame({
        "date": np.repeat(days.date, len(syms)),
        "symbol": np.tile(syms, nd),
        "open": open_.ravel(), "high": high.ravel(), "low": low.ravel(),
        "close": close.ravel(), "volume": volu.ravel(),
    })

    # -------------------------------------------------------------- fii / dii
    fii_regime = np.sign(np.convolve(rng.normal(0, 1, nd),
                                     np.ones(20) / 20, mode="same"))
    fii_net = fii_regime * rng.uniform(500, 4000, nd) + rng.normal(0, 1500, nd)
    dii_net = -0.55 * fii_net + rng.normal(1200, 900, nd)  # counter-cyclical
    fii_buy = rng.uniform(8000, 20000, nd)
    dii_buy = rng.uniform(8000, 18000, nd)
    fii_dii = pd.concat([
        pd.DataFrame({"date": days.date, "category": "FII/FPI",
                      "buy_cr": fii_buy, "sell_cr": fii_buy - fii_net,
                      "net_cr": fii_net}),
        pd.DataFrame({"date": days.date, "category": "DII",
                      "buy_cr": dii_buy, "sell_cr": dii_buy - dii_net,
                      "net_cr": dii_net}),
    ], ignore_index=True)

    # let flows weakly lead the market factor (so Granger finds something)
    #   (already implicit via shared mkt draws; add explicit small lead)
    # ---------------------------------------------------------------- delivery
    deliv_base = rng.uniform(35, 65, len(syms))
    deliv = deliv_base[None, :] + rng.normal(0, 6, (nd, len(syms)))
    for sym, _c, _n, _d in SPOTLIGHTS:          # delivery spikes on events
        j = syms.index(sym)
        for i in planted[sym]:
            deliv[i, j] += rng.uniform(12, 22)
            volu[i, j] *= rng.uniform(1.8, 3.0)
    deliv = np.clip(deliv, 8, 95)
    delivery = pd.DataFrame({
        "date": np.repeat(days.date, len(syms)),
        "symbol": np.tile(syms, nd),
        "close": close.ravel(), "volume": volu.ravel(),
        "deliv_qty": (volu * deliv / 100).ravel(),
        "deliv_pct": deliv.ravel(),
    })

    # ------------------------------------------------------------------ deals
    all_names = [(p, n) for p, names in POOL.items() for n in names]
    rows = []
    # every participant trades a small set of favourite tickers repeatedly,
    # so the validation layer gets many (symbol, client) pairs with enough
    # events — most of which should show NO edge (the honest baseline)
    for profile, client in all_names:
        favs = rng.choice(len(syms), size=int(rng.integers(2, 5)),
                          replace=False)
        n_deals = int(rng.integers(15, 45))
        for _ in range(n_deals):
            i = int(rng.integers(0, nd))
            j = int(favs[int(rng.integers(0, len(favs)))])
            side = "BUY" if rng.random() < 0.5 else "SELL"
            qty = float(rng.integers(60_000, 3_000_000))
            px = close[i, j] * (1 + rng.normal(0, 0.01))
            rows.append((days.date[i], syms[j], NIFTY50[syms[j]], client,
                         side, qty, round(px, 2), "bulk"))
    # planted accumulator deals (the signal)
    for sym, client, _n, _d in SPOTLIGHTS:
        j = syms.index(sym)
        for i in planted[sym]:
            qty = float(rng.integers(800_000, 4_000_000))
            rows.append((days.date[i], sym, NIFTY50[sym], client, "BUY",
                         qty, round(close[i, j], 2), "bulk"))
            # occasional partial trims to make it realistic (~20 %)
            if rng.random() < 0.2:
                rows.append((days.date[i], sym, NIFTY50[sym], client,
                             "SELL", qty * 0.25,
                             round(close[i, j] * 1.001, 2), "bulk"))
    # block deals: institution-to-institution crosses
    for _ in range(150):
        i = int(rng.integers(0, nd))
        j = int(rng.integers(0, len(syms)))
        p1, c1 = all_names[int(rng.integers(0, len(all_names)))]
        qty = float(rng.integers(500_000, 8_000_000))
        px = close[i, j] * (1 + rng.normal(0, 0.005))
        side = "BUY" if rng.random() < 0.5 else "SELL"
        rows.append((days.date[i], syms[j], NIFTY50[syms[j]], c1, side,
                     qty, round(px, 2), "block"))
    deals = pd.DataFrame(rows, columns=["date", "symbol", "security",
                                        "client", "side", "qty", "price",
                                        "kind"])

    return {"prices": prices, "fii_dii_flows": fii_dii,
            "delivery": delivery, "deals": deals}
