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


def ingest_live(lookback_days: int = 400) -> dict[str, pd.DataFrame]:
    from . import nse_api, prices as price_mod

    end = date.today()
    start = end - timedelta(days=lookback_days)
    out: dict[str, pd.DataFrame] = {}

    log.info("fetching prices via yfinance ...")
    out["prices"] = price_mod.fetch_prices(WATCHLIST, period="2y")

    log.info("fetching FII/DII flows ...")
    sess = nse_api._session()
    fii = nse_api.fetch_fii_dii_daily(sess)
    out["fii_dii_flows"] = fii[["date", "category", "buy_cr",
                                "sell_cr", "net_cr"]]

    log.info("fetching bulk deals %s..%s ...", start, end)
    bulk = nse_api.fetch_bulk_deals(start, end, sess=sess)
    bulk["kind"] = "bulk"
    log.info("fetching block deals ...")
    block = nse_api.fetch_block_deals(start, end, sess=sess)
    block["kind"] = "block"
    out["deals"] = pd.concat([bulk, block], ignore_index=True)

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
                pass  # holiday / missing file
        d -= timedelta(days=1)
    out["delivery"] = (pd.concat(frames, ignore_index=True)
                       if frames else pd.DataFrame())
    return out


def ingest_sample() -> dict[str, pd.DataFrame]:
    from .sample_data import generate
    log.info("generating bundled sample dataset (deterministic, seed=42)")
    return generate()


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
