"""Live NSE data fetchers.

NSE blocks naive scripted access, so every call goes through a session that
first warms up cookies on the NSE homepage with browser-like headers.
All fetchers raise ``DataSourceError`` on failure so the pipeline can fall
back to bundled sample data instead of crashing.

Sources
-------
- FII/DII daily provisional flows : /api/fiidiiTradeReact
- Historical bulk deals           : /api/historical/bulk-deals
- Historical block deals          : /api/historical/block-deals
- Security-wise delivery bhavcopy : archives sec_bhavdata_full_DDMMYYYY.csv
"""
from __future__ import annotations

import io
import logging
import time
from datetime import date, timedelta

import pandas as pd
import requests

from .config import NSE_ARCHIVES, NSE_BASE, REQUEST_TIMEOUT

log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/125.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{NSE_BASE}/",
}


class DataSourceError(RuntimeError):
    """Raised when a live source is unreachable or returns junk."""


def _warmup(s: requests.Session) -> None:
    """NSE APIs 401/serve HTML without cookies from the real pages."""
    s.get(NSE_BASE, timeout=REQUEST_TIMEOUT)
    time.sleep(0.6)
    # the bulk/block API checks for cookies set by its report page
    s.get(f"{NSE_BASE}/report-detail/display-bulk-and-block-deals",
          timeout=REQUEST_TIMEOUT)
    time.sleep(0.6)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        _warmup(s)
    except requests.RequestException as exc:
        raise DataSourceError(f"cannot reach NSE: {exc}") from exc
    return s


def _get_json(sess: requests.Session, url: str, retries: int = 2, **kw):
    """GET a JSON endpoint; on HTML/error responses re-warm cookies and
    retry (NSE intermittently serves block pages to scripted clients)."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT, **kw)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            log.warning("attempt %s failed for %s (%s) — re-warming "
                        "cookies", attempt + 1, url, type(exc).__name__)
            time.sleep(2 * (attempt + 1))
            try:
                _warmup(sess)
            except requests.RequestException:
                pass
    raise DataSourceError(f"{url}: {last}") from last


# ---------------------------------------------------------------------------
# FII / DII aggregate flows (₹ crore, provisional, cash market)
# ---------------------------------------------------------------------------
def fetch_fii_dii_daily(sess: requests.Session | None = None) -> pd.DataFrame:
    """Latest day's FII/FPI and DII buy/sell/net values (₹ crore)."""
    sess = sess or _session()
    data = _get_json(sess, f"{NSE_BASE}/api/fiidiiTradeReact")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y").dt.date
    for c in ("buyValue", "sellValue", "netValue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.rename(columns={"buyValue": "buy_cr", "sellValue": "sell_cr",
                              "netValue": "net_cr"})


# ---------------------------------------------------------------------------
# Bulk & block deals (client names are disclosed — the smart-money footprint)
# ---------------------------------------------------------------------------
def _fetch_deals(kind: str, start: date, end: date,
                 sess: requests.Session | None = None) -> pd.DataFrame:
    """kind: 'bulk-deals' or 'block-deals'. NSE limits ranges to ~1 year."""
    sess = sess or _session()
    frames = []
    chunk_start = start
    while chunk_start <= end:  # request in <=90-day chunks to be polite
        chunk_end = min(chunk_start + timedelta(days=90), end)
        url = (f"{NSE_BASE}/api/historical/{kind}"
               f"?from={chunk_start:%d-%m-%Y}&to={chunk_end:%d-%m-%Y}")
        payload = _get_json(sess, url)
        rows = payload.get("data", [])
        if rows:
            frames.append(pd.DataFrame(rows))
        chunk_start = chunk_end + timedelta(days=1)
        time.sleep(1.0)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    colmap = {"BD_DT_DATE": "date", "BD_SYMBOL": "symbol",
              "BD_SCRIP_NAME": "security", "BD_CLIENT_NAME": "client",
              "BD_BUY_SELL": "side", "BD_QTY_TRD": "qty",
              "BD_TP_WATP": "price", "BD_REMARKS": "remarks"}
    df = df.rename(columns={k: v for k, v in colmap.items() if k in df})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["side"] = df["side"].str.upper().str.strip()
    return df[["date", "symbol", "security", "client", "side", "qty", "price"]]


def fetch_bulk_deals(start: date, end: date, **kw) -> pd.DataFrame:
    return _fetch_deals("bulk-deals", start, end, **kw)


def fetch_block_deals(start: date, end: date, **kw) -> pd.DataFrame:
    return _fetch_deals("block-deals", start, end, **kw)


def fetch_latest_bulk_csv(kind: str = "bulk") -> pd.DataFrame:
    """Latest-day bulk/block deals from the NSE archives CSV
    (no cookies needed — reliable fallback when the historical API
    serves HTML block pages)."""
    url = f"{NSE_ARCHIVES}/content/equities/{kind}.csv"
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceError(f"{url}: {exc}") from exc
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Date": "date", "Symbol": "symbol", "Security Name": "security",
        "Client Name": "client", "Buy/Sell": "side",
        "Quantity Traded": "qty",
        "Trade Price / Wght. Avg. Price": "price"})
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y").dt.date
    return df[["date", "symbol", "security", "client", "side", "qty", "price"]]


# ---------------------------------------------------------------------------
# Delivery data (security-wise bhavcopy with DELIV_PER)
# ---------------------------------------------------------------------------
def fetch_delivery_bhavcopy(day: date) -> pd.DataFrame:
    """Full bhavcopy incl. delivery % for one trading day."""
    url = (f"{NSE_ARCHIVES}/products/content/"
           f"sec_bhavdata_full_{day:%d%m%Y}.csv")
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except requests.RequestException as exc:
        raise DataSourceError(f"{url}: {exc}") from exc
    df = pd.read_csv(io.StringIO(r.text))
    df.columns = [c.strip() for c in df.columns]
    df = df[df["SERIES"].str.strip() == "EQ"].copy()
    out = pd.DataFrame({
        "date": pd.to_datetime(df["DATE1"].str.strip(),
                               format="%d-%b-%Y").dt.date,
        "symbol": df["SYMBOL"].str.strip(),
        "close": pd.to_numeric(df["CLOSE_PRICE"], errors="coerce"),
        "volume": pd.to_numeric(df["TTL_TRD_QNTY"], errors="coerce"),
        "turnover": pd.to_numeric(df["TURNOVER_LACS"], errors="coerce"),
        "deliv_qty": pd.to_numeric(df["DELIV_QTY"], errors="coerce"),
        "deliv_pct": pd.to_numeric(df["DELIV_PER"], errors="coerce"),
    })
    return out
