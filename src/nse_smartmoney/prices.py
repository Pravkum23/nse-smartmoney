"""OHLCV price ingestion via yfinance for NSE (.NS) tickers."""
from __future__ import annotations

import logging

import pandas as pd

from .config import WATCHLIST, yahoo_symbol

log = logging.getLogger(__name__)


def fetch_prices(symbols: list[str] | None = None,
                 period: str = "2y") -> pd.DataFrame:
    """Download daily OHLCV for the watchlist. Long format:
    date, symbol, open, high, low, close, volume."""
    import yfinance as yf  # imported lazily — optional at runtime

    symbols = symbols or WATCHLIST
    tickers = {yahoo_symbol(s): s for s in symbols}
    raw = yf.download(list(tickers), period=period, interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False,
                      threads=True)
    frames = []
    for yt, sym in tickers.items():
        try:
            df = raw[yt].dropna(how="all").reset_index()
        except KeyError:
            log.warning("no data for %s", yt)
            continue
        df.columns = [str(c).lower() for c in df.columns]
        df["symbol"] = sym
        df["date"] = pd.to_datetime(df["date"]).dt.date
        frames.append(df[["date", "symbol", "open", "high", "low",
                          "close", "volume"]])
    if not frames:
        raise RuntimeError("yfinance returned no data for any symbol")
    return pd.concat(frames, ignore_index=True)
