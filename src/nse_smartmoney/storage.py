"""SQLite analytics warehouse.

One tidy table per source, plus derived feature/validation tables.
Everything is idempotent: re-running the pipeline upserts by natural key.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd

from .config import DB_PATH

SCHEMA = {
    "prices": """
        CREATE TABLE IF NOT EXISTS prices (
            date TEXT, symbol TEXT, open REAL, high REAL, low REAL,
            close REAL, volume REAL,
            PRIMARY KEY (date, symbol))""",
    "fii_dii_flows": """
        CREATE TABLE IF NOT EXISTS fii_dii_flows (
            date TEXT, category TEXT, buy_cr REAL, sell_cr REAL, net_cr REAL,
            PRIMARY KEY (date, category))""",
    "deals": """
        CREATE TABLE IF NOT EXISTS deals (
            date TEXT, symbol TEXT, security TEXT, client TEXT,
            side TEXT, qty REAL, price REAL, kind TEXT, profile TEXT,
            PRIMARY KEY (date, symbol, client, side, qty, kind))""",
    "delivery": """
        CREATE TABLE IF NOT EXISTS delivery (
            date TEXT, symbol TEXT, close REAL, volume REAL,
            deliv_qty REAL, deliv_pct REAL,
            PRIMARY KEY (date, symbol))""",
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            date TEXT, symbol TEXT,
            ret_1d REAL, back_ret_5d REAL, fwd_ret_5d REAL, fwd_ret_10d REAL,
            volume_z REAL, deliv_z REAL, deliv_spike INTEGER,
            dii_net_cr REAL, fii_net_cr REAL,
            deal_net_qty REAL, dii_deal_net_qty REAL, smart_deal_net_qty REAL,
            accum_score REAL,
            PRIMARY KEY (date, symbol))""",
    "validation": """
        CREATE TABLE IF NOT EXISTS validation (
            symbol TEXT, client TEXT, profile TEXT, horizon INTEGER,
            n_events INTEGER, mean_fwd_ret REAL, median_fwd_ret REAL,
            win_rate REAL, t_stat REAL, p_value REAL, significant INTEGER,
            net_qty REAL,
            PRIMARY KEY (symbol, client, horizon))""",
    "meta": """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY, value TEXT)""",
}


@contextmanager
def connect(db_path=DB_PATH):
    con = sqlite3.connect(db_path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db(db_path=DB_PATH) -> None:
    with connect(db_path) as con:
        for ddl in SCHEMA.values():
            con.execute(ddl)


def upsert(df: pd.DataFrame, table: str, db_path=DB_PATH) -> int:
    """INSERT OR REPLACE a dataframe into `table`. Returns row count."""
    if df is None or df.empty:
        return 0
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    with connect(db_path) as con:
        cols = ",".join(df.columns)
        ph = ",".join("?" * len(df.columns))
        con.executemany(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})",
            df.itertuples(index=False, name=None))
    return len(df)


def read(query: str, db_path=DB_PATH, parse_dates=("date",)) -> pd.DataFrame:
    with connect(db_path) as con:
        df = pd.read_sql(query, con)
    for c in parse_dates:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


def set_meta(key: str, value: str, db_path=DB_PATH) -> None:
    with connect(db_path) as con:
        con.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))


def get_meta(key: str, db_path=DB_PATH) -> str | None:
    with connect(db_path) as con:
        row = con.execute("SELECT value FROM meta WHERE key=?",
                          (key,)).fetchone()
    return row[0] if row else None
