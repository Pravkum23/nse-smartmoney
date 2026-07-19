"""ETL orchestration.

    python -m nse_smartmoney.pipeline            # live, sample fallback
    python -m nse_smartmoney.pipeline --sample   # force bundled sample data
    python -m nse_smartmoney.pipeline --live     # live only, fail loudly

Live mode pulls yfinance prices, NSE FII/DII flows, bulk/block deals and
delivery bhavcopies; each source that fails falls back to sample data
(unless --live). Everything lands in the SQLite warehouse, then features
and validation tables are rebuilt.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

import pandas as pd

from . import storage
from .analysis import participant_alpha_scan
from .config import FWD_HORIZONS, RAW_DIR, WATCHLIST
from .features import build_features, classify_deals

log = logging.getLogger("pipeline")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
def _land_raw(df: pd.DataFrame, name: str) -> None:
    path = RAW_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    log.info("landed %s rows -> %s", len(df), path.name)


DEAL_COLS = ["date", "symbol", "security", "client", "side", "qty",
             "price", "kind"]


def ingest_live(lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    """Pull every live source, degrading gracefully: prices are required,
    each NSE source that fails is logged and skipped so one block page
    never aborts the whole run."""
    from . import nse_api, prices as price_mod

    end = date.today()
    start = end - timedelta(days=lookback_days)
    out: dict[str, pd.DataFrame] = {}

    log.info("fetching prices via yfinance ...")
    out["prices"] = price_mod.fetch_prices(WATCHLIST, period="2y")

    sess = nse_api._session()

    log.info("fetching FII/DII flows ...")
    try:
        fii = nse_api.fetch_fii_dii_daily(sess)
        out["fii_dii_flows"] = fii[["date", "category", "buy_cr",
                                    "sell_cr", "net_cr"]]
    except nse_api.DataSourceError as exc:
        log.warning("FII/DII flows unavailable: %s", exc)
        out["fii_dii_flows"] = pd.DataFrame(
            columns=["date", "category", "buy_cr", "sell_cr", "net_cr"])

    deal_frames = []
    try:
        log.info("fetching bulk deals %s..%s ...", start, end)
        bulk = nse_api.fetch_bulk_deals(start, end, sess=sess)
        bulk["kind"] = "bulk"
        deal_frames.append(bulk)
        log.info("fetching block deals ...")
        block = nse_api.fetch_block_deals(start, end, sess=sess)
        block["kind"] = "block"
        deal_frames.append(block)
    except nse_api.DataSourceError as exc:
        log.warning("historical deals API unavailable (%s) — falling "
                    "back to latest-day archive CSVs", exc)
        for kind in ("bulk", "block"):
            try:
                df = nse_api.fetch_latest_bulk_csv(kind)
                df["kind"] = kind
                deal_frames.append(df)
                log.info("archive %s.csv: %s deals", kind, len(df))
            except Exception as exc2:               # noqa: BLE001
                log.warning("archive %s.csv unavailable: %s", kind, exc2)
    out["deals"] = (pd.concat(deal_frames, ignore_index=True)
                    if deal_frames else pd.DataFrame(columns=DEAL_COLS))

    log.info("fetching delivery bhavcopies (last 120 trading days) ...")
    frames = []
    d = end
    fetched = 0
    while fetched < 120 and d > start:
        if d.weekday() < 5:
            try:
                frames.append(nse_api.fetch_delivery_bhavcopy(d))
                fetched += 1
            except nse_api.DataSourceError:
                pass  # holiday / missing file / block page
        d -= timedelta(days=1)
    if not frames:
        log.warning("no delivery bhavcopies retrieved")
    out["delivery"] = (pd.concat(frames, ignore_index=True)
                       if frames else pd.DataFrame(
                           columns=["date", "symbol", "close", "volume",
                                    "deliv_qty", "deliv_pct"]))
    return out


def ingest_sample() -> dict[str, pd.DataFrame]:
    from .sample_data import generate
    log.info("generating bundled sample dataset (deterministic, seed=42)")
    return generate()


def ingest_manual_deals() -> pd.DataFrame:
    """Read user-downloaded NSE bulk/block-deal CSVs from data/raw/manual/.

    NSE's historical deals API blocks scripts, but the report page at
    nseindia.com/report-detail/display-bulk-and-block-deals lets a human
    download up to one year per CSV. Drop those files here:

        data/raw/manual/bulk*.csv   (bulk deals)
        data/raw/manual/block*.csv  (block deals)

    Filenames decide the kind; any other name defaults to bulk.
    """
    manual_dir = RAW_DIR / "manual"
    if not manual_dir.exists():
        return pd.DataFrame(columns=DEAL_COLS)
    frames = []
    for p in sorted(manual_dir.glob("*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception as exc:                     # noqa: BLE001
            log.warning("manual file %s unreadable: %s", p.name, exc)
            continue
        df.columns = [c.strip() for c in df.columns]
        ren = {}
        for c in df.columns:
            k = c.upper().replace(" ", "")
            if k == "DATE":
                ren[c] = "date"
            elif k == "SYMBOL":
                ren[c] = "symbol"
            elif "SECURITY" in k:
                ren[c] = "security"
            elif "BUY/SELL" in k or k == "BUYSELL":
                ren[c] = "side"
            elif "CLIENT" in k:
                ren[c] = "client"
            elif "QUANTITY" in k or k == "QTYTRADED":
                ren[c] = "qty"
            elif "PRICE" in k:
                ren[c] = "price"
        df = df.rename(columns=ren)
        need = {"date", "symbol", "client", "side", "qty"}
        if not need.issubset(df.columns):
            log.warning("manual file %s: unrecognised columns %s — "
                        "skipped", p.name, list(df.columns)[:8])
            continue
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y",
                                    errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        df["qty"] = pd.to_numeric(
            df["qty"].astype(str).str.replace(",", ""), errors="coerce")
        if "price" in df.columns:
            df["price"] = pd.to_numeric(
                df["price"].astype(str).str.replace(",", ""),
                errors="coerce")
        else:
            df["price"] = pd.NA
        if "security" not in df.columns:
            df["security"] = df["symbol"]
        df["side"] = df["side"].astype(str).str.upper().str.strip()
        df["kind"] = "block" if "block" in p.name.lower() else "bulk"
        frames.append(df[DEAL_COLS])
        log.info("manual %s: %s deals", p.name, len(df))
    if not frames:
        return pd.DataFrame(columns=DEAL_COLS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
def run(mode: str = "auto") -> None:
    storage.init_db()

    data: dict[str, pd.DataFrame] | None = None
    source = mode
    if mode in ("auto", "live"):
        try:
            data = ingest_live()
            source = "live"
        except Exception as exc:                     # noqa: BLE001
            if mode == "live":
                raise
            log.warning("live ingestion failed (%s) — falling back to "
                        "sample data", exc)
    if data is None:
        data = ingest_sample()
        source = "sample"

    # merge any manually downloaded deal history (data/raw/manual/*.csv);
    # the warehouse primary key dedupes overlapping rows
    if source != "sample":
        manual = ingest_manual_deals()
        if len(manual):
            data["deals"] = pd.concat([data["deals"], manual],
                                      ignore_index=True)
            log.info("merged %s manually downloaded deals", len(manual))

    # land raw + load warehouse
    for name, df in data.items():
        _land_raw(df, name)
    deals = classify_deals(data["deals"])
    n = 0
    n += storage.upsert(data["prices"], "prices")
    n += storage.upsert(data["fii_dii_flows"], "fii_dii_flows")
    n += storage.upsert(deals[["date", "symbol", "security", "client",
                               "side", "qty", "price", "kind", "profile"]],
                        "deals")
    n += storage.upsert(data["delivery"], "delivery")
    log.info("warehouse loaded: %s rows (%s)", n, source)

    # features
    feats = build_features(data["prices"], data["delivery"],
                           data["deals"], data["fii_dii_flows"])
    storage.upsert(feats, "features")
    log.info("features rebuilt: %s rows", len(feats))

    # participant validation for each horizon
    frames = []
    for h in FWD_HORIZONS:
        v = participant_alpha_scan(data["deals"], feats, horizon=h)
        if len(v):
            frames.append(v)
    if frames:
        val = pd.concat(frames, ignore_index=True)
        storage.upsert(val[["symbol", "client", "profile", "horizon",
                            "n_events", "mean_fwd_ret", "median_fwd_ret",
                            "win_rate", "t_stat", "p_value", "significant",
                            "net_qty"]], "validation")
        log.info("validation rebuilt: %s participant-symbol rows", len(val))

    storage.set_meta("last_run", str(pd.Timestamp.now()))
    storage.set_meta("source", source)
    log.info("pipeline complete (source=%s)", source)


def main() -> None:
    ap = argparse.ArgumentParser(description="nse-smartmoney ETL")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--live", action="store_true",
                   help="live sources only; fail if unreachable")
    g.add_argument("--sample", action="store_true",
                   help="force bundled sample data")
    args = ap.parse_args()
    mode = "live" if args.live else "sample" if args.sample else "auto"
    run(mode)


if __name__ == "__main__":
    sys.exit(main())
