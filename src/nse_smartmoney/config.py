"""Central configuration for nse-smartmoney.

Paths, universe definitions, participant-classification rules and
analysis parameters live here so every module shares one source of truth.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DB_DIR = DATA_DIR / "db"
# Override with NSE_SM_DB env var (useful on network-mounted filesystems
# where SQLite locking fails).
DB_PATH = Path(os.environ.get("NSE_SM_DB", DB_DIR / "smartmoney.sqlite"))

for _d in (RAW_DIR, PROCESSED_DIR, DB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Universe — NIFTY 50 (symbol -> company name)
# ---------------------------------------------------------------------------
NIFTY50: dict[str, str] = {
    "ADANIENT": "Adani Enterprises", "ADANIPORTS": "Adani Ports & SEZ",
    "APOLLOHOSP": "Apollo Hospitals", "ASIANPAINT": "Asian Paints",
    "AXISBANK": "Axis Bank", "BAJAJ-AUTO": "Bajaj Auto",
    "BAJFINANCE": "Bajaj Finance", "BAJAJFINSV": "Bajaj Finserv",
    "BEL": "Bharat Electronics", "BHARTIARTL": "Bharti Airtel",
    "CIPLA": "Cipla", "COALINDIA": "Coal India",
    "DRREDDY": "Dr. Reddy's Labs", "EICHERMOT": "Eicher Motors",
    "ETERNAL": "Eternal (Zomato)", "GRASIM": "Grasim Industries",
    "HCLTECH": "HCL Technologies", "HDFCBANK": "HDFC Bank",
    "HDFCLIFE": "HDFC Life", "HEROMOTOCO": "Hero MotoCorp",
    "HINDALCO": "Hindalco", "HINDUNILVR": "Hindustan Unilever",
    "ICICIBANK": "ICICI Bank", "INDUSINDBK": "IndusInd Bank",
    "INFY": "Infosys", "ITC": "ITC",
    "JIOFIN": "Jio Financial Services", "JSWSTEEL": "JSW Steel",
    "KOTAKBANK": "Kotak Mahindra Bank", "LT": "Larsen & Toubro",
    "M&M": "Mahindra & Mahindra", "MARUTI": "Maruti Suzuki",
    "NESTLEIND": "Nestle India", "NTPC": "NTPC",
    "ONGC": "ONGC", "POWERGRID": "Power Grid",
    "RELIANCE": "Reliance Industries", "SBILIFE": "SBI Life",
    "SBIN": "State Bank of India", "SHRIRAMFIN": "Shriram Finance",
    "SUNPHARMA": "Sun Pharma", "TATACONSUM": "Tata Consumer",
    "TATAMOTORS": "Tata Motors", "TATASTEEL": "Tata Steel",
    "TCS": "TCS", "TECHM": "Tech Mahindra",
    "TITAN": "Titan", "TRENT": "Trent",
    "ULTRACEMCO": "UltraTech Cement", "WIPRO": "Wipro",
}

WATCHLIST: list[str] = list(NIFTY50)


def yahoo_symbol(nse_symbol: str) -> str:
    """Map an NSE symbol to its Yahoo Finance ticker."""
    return f"{nse_symbol}.NS"


# ---------------------------------------------------------------------------
# Participant behavioural profiles (Indian analog of broker profiles)
#
# Bulk/block-deal client names are matched (case-insensitive, substring)
# against these keyword lists, first hit wins, top-to-bottom priority.
# ---------------------------------------------------------------------------
PARTICIPANT_PROFILES: dict[str, list[str]] = {
    # Domestic institutions — the "smart money" this project focuses on
    "DII — Mutual Fund": [
        "MUTUAL FUND", "ASSET MANAGEMENT", "AMC", " MF ", "SBI FUNDS",
        "NIPPON INDIA", "HDFC AMC", "ICICI PRUDENTIAL AMC", "AXIS AMC",
        "KOTAK MAHINDRA ASSET", "UTI ", "DSP ", "MIRAE ASSET",
        "FRANKLIN TEMPLETON", "ADITYA BIRLA SUN LIFE", "TATA ASSET",
        "QUANT MONEY", "PPFAS", "MOTILAL OSWAL ASSET", "EDELWEISS ASSET",
    ],
    "DII — Insurance": [
        "LIFE INSURANCE CORPORATION", "LIC OF INDIA", "INSURANCE COMPANY",
        "SBI LIFE", "HDFC LIFE", "ICICI PRUDENTIAL LIFE", "MAX LIFE",
        "BAJAJ ALLIANZ", "TATA AIA", "GENERAL INSURANCE",
    ],
    "DII — AIF / PMS / NBFC": [
        "ALTERNATIVE INVESTMENT FUND", " AIF", "PORTFOLIO MANAGEMENT",
        "GROWTH FUND", "FLAGSHIP FUND", "YIELD FUND", "INDIA FUND",
        "VENTURE", "CAPITAL SERVICES", "FINANCE COMPANY", "NBFC",
        "WEALTH MANAGEMENT", "INVESTMENT COMPANY", "INVESTMENT MANAGERS",
    ],
    # Foreign institutions
    "FII / FPI": [
        "GOLDMAN SACHS", "MORGAN STANLEY", "BOFA SECURITIES", "CITIGROUP",
        "JP MORGAN", "J.P. MORGAN", "MERRILL LYNCH", "UBS ", "BNP PARIBAS",
        "SOCIETE GENERALE", "NOMURA", "MACQUARIE", "CLSA", "JEFFERIES",
        "GOVERNMENT PENSION FUND", "VANGUARD", "BLACKROCK", "ISHARES",
        "FTSE", "MSCI", "UCITS", "PTE LTD", "PTE. LTD",
        "SINGAPORE", "MAURITIUS", "LUXEMBOURG", "EUROPE SA", "LLC",
        "COPTHALL", "GRAVITON RESEARCH CAPITAL LLP",
    ],
    # Prop / HFT / arb desks — high two-way volume, net position matters
    "Prop / HFT Desk": [
        "SECURITIES RESEARCH", "BROKING SERVICES LLP", "QE SECURITIES",
        "ALPHAGREP", "IRAGE", "BLITZQUANT", "JUMP TRADING", "TOWER RESEARCH",
        "QUADEYE", "APT PORTFOLIO", "DOLAT", "MICROCURVES", "JUNOMONETA",
        "HRTI", "MUSIGMA", "SILVERLEAF", "PATRONUS", "NK SECURITIES",
        "TRADETECH", "HITECH PRIVATE", "STRATEGIC VENTURES LLP",
    ],
    # Known HNI investors / family offices (extend as needed)
    "HNI / Family Office": [
        "JHUNJHUNWALA", "DAMANI", "KEDIA", "KACHOLIA",
        "FAMILY TRUST", "FAMILY OFFICE", "HUF",
    ],
}
DEFAULT_PROFILE = "Other / Retail-HNI"

# Profiles counted as "smart money" in aggregations
SMART_MONEY_PROFILES = ["DII — Mutual Fund", "DII — Insurance",
                        "DII — AIF / PMS / NBFC", "FII / FPI"]
# The project's primary focus
DII_PROFILES = ["DII — Mutual Fund", "DII — Insurance",
                "DII — AIF / PMS / NBFC"]

# ---------------------------------------------------------------------------
# Analysis parameters
# ---------------------------------------------------------------------------
FWD_HORIZONS = [5, 10]          # forward-return horizons (trading days)
BACK_HORIZON = 5                # momentum lookback
MIN_EVENTS = 5                  # min net-buy events for participant validation
ALPHA = 0.05                    # one-sided significance level
DELIVERY_Z_WINDOW = 60          # rolling window for delivery-% z-score
DELIVERY_SPIKE_Z = 1.5          # z-score defining an "accumulation day"
VOLUME_SPIKE_Z = 1.0            # volume confirmation threshold

# Live-source settings
NSE_BASE = "https://www.nseindia.com"
NSE_ARCHIVES = "https://archives.nseindia.com"
REQUEST_TIMEOUT = int(os.environ.get("NSE_TIMEOUT", "15"))
