"""Statistical validation layer.

- participant_alpha_scan : do a participant's repeated net-buys precede
  positive forward returns? (event study + one-sided t-test)
- delivery_spike_study   : do delivery/volume accumulation days predict
  forward returns?
- granger_flows          : do FII/DII flows *lead* market returns?
- event_study_curve      : average cumulative return path around events
"""
from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from scipy import stats


def _granger(data, maxlag):
    """Run statsmodels grangercausalitytests without console spam."""
    from statsmodels.tsa.stattools import grangercausalitytests
    with contextlib.redirect_stdout(io.StringIO()):
        return grangercausalitytests(data, maxlag=maxlag)

from .config import ALPHA, MIN_EVENTS
from .features import _signed_qty, classify_deals


# ---------------------------------------------------------------------------
def participant_alpha_scan(deals: pd.DataFrame, features: pd.DataFrame,
                           horizon: int = 10,
                           min_events: int = MIN_EVENTS) -> pd.DataFrame:
    """For every (symbol, client) with >= min_events net-buy days, test
    whether the mean forward return after those days is > 0 (one-sided).

    This is the Indian analog of broker-specific validation: bulk/block
    deal disclosures give us *named* participants instead of broker codes.
    """
    col = f"fwd_ret_{horizon}d"
    dl = classify_deals(deals)
    dl["date"] = pd.to_datetime(dl["date"])
    dl["signed_qty"] = _signed_qty(dl)

    daily = (dl.groupby(["symbol", "client", "profile", "date"])
             ["signed_qty"].sum().reset_index())
    buys = daily[daily["signed_qty"] > 0]

    ft = features.copy()
    ft["date"] = pd.to_datetime(ft["date"])
    merged = buys.merge(ft[["date", "symbol", col]],
                        on=["date", "symbol"], how="left").dropna(subset=[col])

    rows = []
    for (sym, client, profile), grp in merged.groupby(
            ["symbol", "client", "profile"]):
        n = len(grp)
        if n < min_events:
            continue
        r = grp[col].values
        mean, med = r.mean(), np.median(r)
        win = float((r > 0).mean())
        if n > 1 and r.std(ddof=1) > 0:
            t, p_two = stats.ttest_1samp(r, 0.0)
            p = p_two / 2 if t > 0 else 1 - p_two / 2   # one-sided H1: mu>0
        else:
            t, p = np.nan, np.nan
        rows.append({"symbol": sym, "client": client, "profile": profile,
                     "horizon": horizon, "n_events": n,
                     "mean_fwd_ret": mean, "median_fwd_ret": med,
                     "win_rate": win, "t_stat": t, "p_value": p,
                     "significant": int(bool(p is not None
                                             and not np.isnan(p)
                                             and p < ALPHA and mean > 0)),
                     "net_qty": grp["signed_qty"].sum()})
    out = pd.DataFrame(rows)
    return (out.sort_values("p_value").reset_index(drop=True)
            if len(out) else out)


# ---------------------------------------------------------------------------
def delivery_spike_study(features: pd.DataFrame,
                         horizon: int = 5) -> pd.DataFrame:
    """Compare forward returns on delivery-spike days vs all other days,
    per symbol and pooled."""
    col = f"fwd_ret_{horizon}d"
    ft = features.dropna(subset=[col]).copy()
    rows = []
    for sym, grp in ft.groupby("symbol"):
        spike = grp.loc[grp["deliv_spike"] == 1, col]
        base = grp.loc[grp["deliv_spike"] == 0, col]
        if len(spike) < 3:
            continue
        t, p = stats.ttest_ind(spike, base, equal_var=False)
        rows.append({"symbol": sym, "n_spikes": len(spike),
                     "spike_mean": spike.mean(), "base_mean": base.mean(),
                     "edge": spike.mean() - base.mean(),
                     "t_stat": t, "p_value": p})
    return pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
def granger_flows(features: pd.DataFrame, fii_dii: pd.DataFrame,
                  maxlag: int = 5) -> pd.DataFrame:
    """Does aggregate FII / DII net flow Granger-cause equal-weight
    market returns? Returns min-p across lags for each flow series."""
    mkt = (features.assign(date=pd.to_datetime(features["date"]))
           .groupby("date")["ret_1d"].mean().rename("mkt_ret"))
    fl = fii_dii.copy()
    fl["date"] = pd.to_datetime(fl["date"])
    piv = fl.pivot_table(index="date", columns="category", values="net_cr",
                         aggfunc="sum")
    df = pd.concat([mkt, piv], axis=1).dropna()

    rows = []
    for col in piv.columns:
        data = df[["mkt_ret", col]].dropna()
        try:
            res = _granger(data, maxlag)
            best_lag, best_p = min(
                ((lag, r[0]["ssr_ftest"][1]) for lag, r in res.items()),
                key=lambda x: x[1])
            rows.append({"flow": col, "direction": f"{col} -> market",
                         "best_lag": best_lag, "p_value": best_p,
                         "significant": int(best_p < ALPHA)})
        except Exception as exc:                     # noqa: BLE001
            rows.append({"flow": col, "direction": f"{col} -> market",
                         "best_lag": None, "p_value": np.nan,
                         "significant": 0, "note": str(exc)[:80]})
    return pd.DataFrame(rows)


def granger_stock_flow(features: pd.DataFrame, symbol: str,
                       maxlag: int = 5) -> pd.DataFrame:
    """Per-stock: do smart-money deal flows / delivery z lead the stock's
    own returns?"""
    ft = features[features["symbol"] == symbol].copy()
    ft["date"] = pd.to_datetime(ft["date"])
    ft = ft.set_index("date").sort_index()
    rows = []
    for col in ("smart_deal_net_qty", "dii_deal_net_qty", "deliv_z"):
        data = ft[["ret_1d", col]].dropna()
        if len(data) < maxlag * 4 or data[col].std() == 0:
            continue
        try:
            res = _granger(data, maxlag)
            best_lag, best_p = min(
                ((lag, r[0]["ssr_ftest"][1]) for lag, r in res.items()),
                key=lambda x: x[1])
            rows.append({"signal": col, "direction": f"{col} -> {symbol}",
                         "best_lag": best_lag, "p_value": best_p,
                         "significant": int(best_p < ALPHA)})
        except Exception:                            # noqa: BLE001
            continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def event_study_curve(prices: pd.DataFrame, events: pd.DataFrame,
                      pre: int = 5, post: int = 15) -> pd.DataFrame:
    """Average cumulative return path around event days.
    `events` needs columns: date, symbol."""
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    curves = []
    for _, ev in events.iterrows():
        s = px[px["symbol"] == ev["symbol"]].sort_values("date") \
            .reset_index(drop=True)
        idx = s.index[s["date"] == pd.to_datetime(ev["date"])]
        if len(idx) == 0:
            continue
        i = idx[0]
        lo, hi = i - pre, i + post
        if lo < 0 or hi >= len(s):
            continue
        window = s.loc[lo:hi, "close"].values
        curves.append(window / window[pre] - 1)
    if not curves:
        return pd.DataFrame()
    arr = np.vstack(curves)
    rel = np.arange(-pre, post + 1)
    return pd.DataFrame({"rel_day": rel, "mean_cumret": arr.mean(0),
                         "median_cumret": np.median(arr, 0),
                         "n_events": len(curves)})
