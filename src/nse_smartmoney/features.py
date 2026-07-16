"""Participant classification + feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (BACK_HORIZON, DEFAULT_PROFILE, DELIVERY_SPIKE_Z,
                     DELIVERY_Z_WINDOW, DII_PROFILES, PARTICIPANT_PROFILES,
                     SMART_MONEY_PROFILES, VOLUME_SPIKE_Z)


# ---------------------------------------------------------------------------
# Participant classification — the Indian analog of broker profiling
# ---------------------------------------------------------------------------
def classify_participant(name: str) -> str:
    """Map a bulk/block-deal client name to a behavioural profile."""
    if not isinstance(name, str) or not name.strip():
        return DEFAULT_PROFILE
    up = f" {name.upper()} "
    for profile, keywords in PARTICIPANT_PROFILES.items():
        for kw in keywords:
            if kw in up:
                return profile
    return DEFAULT_PROFILE


def classify_deals(deals: pd.DataFrame) -> pd.DataFrame:
    deals = deals.copy()
    deals["profile"] = deals["client"].map(classify_participant)
    return deals


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _signed_qty(deals: pd.DataFrame) -> pd.Series:
    return np.where(deals["side"].str.upper().eq("BUY"),
                    deals["qty"], -deals["qty"])


def build_features(prices: pd.DataFrame, delivery: pd.DataFrame,
                   deals: pd.DataFrame,
                   fii_dii: pd.DataFrame) -> pd.DataFrame:
    """Per (date, symbol) feature matrix used by analysis, models and the
    dashboard. Forward returns are the labels; everything else must be
    known at the close of the signal day (no look-ahead)."""
    px = prices.sort_values(["symbol", "date"]).copy()
    px["date"] = pd.to_datetime(px["date"])
    g = px.groupby("symbol", group_keys=False)

    px["ret_1d"] = g["close"].pct_change()
    px["back_ret_5d"] = g["close"].pct_change(BACK_HORIZON)
    px["fwd_ret_5d"] = g["close"].transform(
        lambda s: s.shift(-5) / s - 1)
    px["fwd_ret_10d"] = g["close"].transform(
        lambda s: s.shift(-10) / s - 1)
    px["volume_z"] = g["volume"].transform(
        lambda s: (s - s.rolling(DELIVERY_Z_WINDOW, min_periods=20).mean())
        / s.rolling(DELIVERY_Z_WINDOW, min_periods=20).std())

    # delivery z-score (accumulation proxy: high delivery = positions kept)
    dv = delivery.copy()
    dv["date"] = pd.to_datetime(dv["date"])
    dv = dv.sort_values(["symbol", "date"])
    dv["deliv_z"] = dv.groupby("symbol", group_keys=False)["deliv_pct"].apply(
        lambda s: (s - s.rolling(DELIVERY_Z_WINDOW, min_periods=20).mean())
        / s.rolling(DELIVERY_Z_WINDOW, min_periods=20).std())
    feat = px.merge(dv[["date", "symbol", "deliv_pct", "deliv_z"]],
                    on=["date", "symbol"], how="left")
    feat["deliv_spike"] = ((feat["deliv_z"] >= DELIVERY_SPIKE_Z)
                           & (feat["volume_z"] >= VOLUME_SPIKE_Z)).astype(int)

    # market-level FII / DII flows broadcast to every symbol
    fl = fii_dii.copy()
    fl["date"] = pd.to_datetime(fl["date"])
    piv = fl.pivot_table(index="date", columns="category",
                         values="net_cr", aggfunc="sum")
    piv = piv.rename(columns={"FII/FPI": "fii_net_cr", "DII": "dii_net_cr"})
    feat = feat.merge(piv.reset_index(), on="date", how="left")

    # deal-based net quantities per symbol-date
    dl = classify_deals(deals)
    dl["date"] = pd.to_datetime(dl["date"])
    dl["signed_qty"] = _signed_qty(dl)
    total = dl.groupby(["date", "symbol"])["signed_qty"].sum() \
              .rename("deal_net_qty")
    dii = dl[dl["profile"].isin(DII_PROFILES)] \
        .groupby(["date", "symbol"])["signed_qty"].sum() \
        .rename("dii_deal_net_qty")
    smart = dl[dl["profile"].isin(SMART_MONEY_PROFILES)] \
        .groupby(["date", "symbol"])["signed_qty"].sum() \
        .rename("smart_deal_net_qty")
    for s in (total, dii, smart):
        feat = feat.merge(s.reset_index(), on=["date", "symbol"], how="left")
    for c in ("deal_net_qty", "dii_deal_net_qty", "smart_deal_net_qty"):
        feat[c] = feat[c].fillna(0.0)

    # composite accumulation score (0-100): delivery spike + smart deals +
    # DII tape support, volume-confirmed
    z = feat["deliv_z"].clip(-3, 3).fillna(0) / 3
    vz = feat["volume_z"].clip(-3, 3).fillna(0) / 3
    deal_sig = np.sign(feat["smart_deal_net_qty"])
    dii_sig = np.sign(feat["dii_net_cr"].fillna(0))
    feat["accum_score"] = (50 + 20 * z + 10 * vz + 12 * deal_sig
                           + 8 * dii_sig).clip(0, 100)

    cols = ["date", "symbol", "ret_1d", "back_ret_5d", "fwd_ret_5d",
            "fwd_ret_10d", "volume_z", "deliv_z", "deliv_spike",
            "dii_net_cr", "fii_net_cr", "deal_net_qty", "dii_deal_net_qty",
            "smart_deal_net_qty", "accum_score"]
    out = feat[cols].copy()
    out["date"] = out["date"].dt.date
    return out
