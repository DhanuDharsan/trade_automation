"""
NSE Option Chain Fetcher → PostgreSQL Updater  v3.0
════════════════════════════════════════════════════
Collects full option chain + computes ALL Greeks + India VIX +
IV Rank/Percentile + Bid-Ask spread + DTE → stores everything
needed for a 90 %+ accuracy options trading system.

New in v3.0
───────────
  • Black-Scholes Greeks: Delta, Gamma, Theta, Vega, Rho
  • IV Rank & IV Percentile (rolling 52-week high/low)
  • India VIX fetched from NSE
  • Bid-Ask spread % (liquidity filter)
  • Days-to-Expiry (DTE)
  • Moneyness: ITM / ATM / OTM
  • Intrinsic & Time value
  • market_ticks populated on every fetch
  • signals table auto-created

Requirements:
    pip install pnsea psycopg2-binary scipy numpy requests

Usage:
    python collector.py --symbol NIFTY --loop 60
    python collector.py --symbol BANKNIFTY --loop 60
    python collector.py --symbol SBIN --equity --loop 60
"""

import math
import time
import argparse
import logging
from datetime import datetime, timezone, date
from typing import Optional

import numpy as np
import psycopg2
import requests
from psycopg2.extras import execute_values
from scipy.stats import norm
from pnsea import NSE

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DB Config
# ─────────────────────────────────────────────
DB_CONFIG = {
    "host":     "ep-morning-lake-aoiahvcv-pooler.c-2.ap-southeast-1.aws.neon.tech",
    "port":      5432,
    "dbname":   "Trade",
    "user":     "neondb_owner",
    "password": "npg_YCL6POljqg9i",
}

RISK_FREE_RATE = 0.065   # India 10-yr bond yield approx

# ─────────────────────────────────────────────
# DDL — All tables
# ─────────────────────────────────────────────
CREATE_TABLES_SQL = """
-- ── Spot price feed ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_ticks (
    id          BIGSERIAL   PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    price       NUMERIC     NOT NULL,
    volume      BIGINT,
    india_vix   NUMERIC,
    timestamp   TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
    ON market_ticks (symbol, timestamp DESC);

-- ── Option chain metadata (per fetch summary) ─────────────────
CREATE TABLE IF NOT EXISTS nse_option_chain_metadata (
    id                  SERIAL      PRIMARY KEY,
    symbol              TEXT        NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL,
    underlying_value    NUMERIC,
    india_vix           NUMERIC,
    total_call_oi       BIGINT,
    total_put_oi        BIGINT,
    total_call_vol      BIGINT,
    total_put_vol       BIGINT,
    pcr_oi              NUMERIC,
    pcr_vol             NUMERIC,
    atm_strike          NUMERIC,
    max_pain            NUMERIC,
    iv_rank             NUMERIC,
    iv_percentile       NUMERIC
);

-- ── Full option chain with Greeks ────────────────────────────
CREATE TABLE IF NOT EXISTS nse_option_chain (
    id                  BIGSERIAL   PRIMARY KEY,
    symbol              TEXT        NOT NULL,
    fetched_at          TIMESTAMPTZ NOT NULL,
    expiry_date         DATE,
    strike_price        NUMERIC,
    option_type         CHAR(2)     NOT NULL,

    -- Market data
    open_interest       BIGINT,
    change_in_oi        BIGINT,
    pct_change_oi       NUMERIC,
    total_traded_volume BIGINT,
    implied_volatility  NUMERIC,
    last_price          NUMERIC,
    change              NUMERIC,
    pct_change          NUMERIC,
    bid_qty             BIGINT,
    bid_price           NUMERIC,
    ask_qty             BIGINT,
    ask_price           NUMERIC,
    total_buy_qty       BIGINT,
    total_sell_qty      BIGINT,
    open                NUMERIC,
    high                NUMERIC,
    low                 NUMERIC,
    close               NUMERIC,
    prev_close          NUMERIC,
    underlying_value    NUMERIC,

    -- Derived: time & moneyness
    dte                 INTEGER,        -- days to expiry
    moneyness           TEXT,           -- ITM / ATM / OTM
    intrinsic_value     NUMERIC,
    time_value          NUMERIC,

    -- Greeks (Black-Scholes)
    delta               NUMERIC,
    gamma               NUMERIC,
    theta               NUMERIC,        -- per day
    vega                NUMERIC,        -- per 1% IV move
    rho                 NUMERIC,

    -- Liquidity
    bid_ask_spread      NUMERIC,        -- ask - bid
    bid_ask_spread_pct  NUMERIC,        -- spread / mid * 100

    -- IV analytics
    iv_rank             NUMERIC,        -- 0-100
    iv_percentile       NUMERIC,        -- 0-100

    UNIQUE (symbol, fetched_at, expiry_date, strike_price, option_type)
);
CREATE INDEX IF NOT EXISTS idx_oc_symbol_fetched ON nse_option_chain (symbol, fetched_at DESC);
CREATE INDEX IF NOT EXISTS idx_oc_expiry         ON nse_option_chain (expiry_date);
CREATE INDEX IF NOT EXISTS idx_oc_strike         ON nse_option_chain (strike_price);
CREATE INDEX IF NOT EXISTS idx_oc_moneyness      ON nse_option_chain (moneyness);

-- ── Signals (written by strategy.py) ─────────────────────────
CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL    PRIMARY KEY,
    symbol          TEXT         NOT NULL,
    signal          TEXT         NOT NULL,
    price           NUMERIC      NOT NULL,
    confidence      NUMERIC,
    ema_fast        NUMERIC,
    ema_slow        NUMERIC,
    rsi             NUMERIC,
    macd_histogram  NUMERIC,
    supertrend_bull BOOLEAN,
    vwap            NUMERIC,
    pcr             NUMERIC,
    india_vix       NUMERIC,
    votes           JSONB,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
    ON signals (symbol, timestamp DESC);

-- ── IV history (for IV Rank / Percentile) ────────────────────
CREATE TABLE IF NOT EXISTS iv_history (
    id          BIGSERIAL   PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    recorded_at DATE        NOT NULL,
    atm_iv      NUMERIC     NOT NULL,
    UNIQUE (symbol, recorded_at)
);
"""

# ─────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────
def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(CREATE_TABLES_SQL)
        # Safely relax NOT NULL added by older versions
        for col in ["expiry_date"]:
            cur.execute(f"""
                ALTER TABLE nse_option_chain
                ALTER COLUMN {col} DROP NOT NULL;
            """)
    conn.commit()
    log.info("Schema ready.")

# ─────────────────────────────────────────────
# India VIX
# ─────────────────────────────────────────────
def fetch_india_vix() -> Optional[float]:
    """Fetch India VIX from NSE website."""
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0",
            "Referer":    "https://www.nseindia.com",
        })
        session.get("https://www.nseindia.com", timeout=8)
        r = session.get(
            "https://www.nseindia.com/api/allIndices",
            timeout=8
        )
        data = r.json()
        for item in data.get("data", []):
            if item.get("index") == "INDIA VIX":
                vix = float(item.get("last", 0))
                log.info("  India VIX: %.2f", vix)
                return vix
    except Exception as e:
        log.warning("  Could not fetch India VIX: %s", e)
    return None

# ─────────────────────────────────────────────
# Black-Scholes Greeks
# ─────────────────────────────────────────────
def black_scholes_greeks(
    S: float,        # underlying price
    K: float,        # strike price
    T: float,        # time to expiry in years
    r: float,        # risk-free rate
    sigma: float,    # implied volatility (decimal)
    option_type: str # "CE" or "PE"
) -> dict:
    """
    Compute full Black-Scholes Greeks.
    Returns dict with delta, gamma, theta, vega, rho.
    All values are per-contract (1 unit of underlying).
    Theta is expressed per calendar day.
    Vega is expressed per 1% move in IV.
    """
    greeks = {"delta": None, "gamma": None,
              "theta": None, "vega":  None, "rho": None}

    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return greeks

    try:
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1  = norm.cdf(d1)
        nd2  = norm.cdf(d2)
        nd1n = norm.cdf(-d1)
        nd2n = norm.cdf(-d2)
        npd1 = norm.pdf(d1)   # standard normal PDF at d1

        gamma = npd1 / (S * sigma * sqrt_T)
        vega  = S * npd1 * sqrt_T / 100   # per 1% IV move

        if option_type == "CE":
            delta = nd1
            theta = (-(S * npd1 * sigma) / (2 * sqrt_T)
                     - r * K * math.exp(-r * T) * nd2) / 365
            rho   = K * T * math.exp(-r * T) * nd2 / 100
        else:  # PE
            delta = nd1 - 1
            theta = (-(S * npd1 * sigma) / (2 * sqrt_T)
                     + r * K * math.exp(-r * T) * nd2n) / 365
            rho   = -K * T * math.exp(-r * T) * nd2n / 100

        greeks = {
            "delta": float(round(delta, 4)),
            "gamma": float(round(gamma, 6)),
            "theta": float(round(theta, 4)),
            "vega":  float(round(vega,  4)),
            "rho":   float(round(rho,   4)),
        }
    except Exception as e:
        log.debug("Greeks calc failed for S=%.1f K=%.1f T=%.4f σ=%.4f: %s",
                  S, K, T, sigma, e)

    return greeks


# ─────────────────────────────────────────────
# IV Rank & Percentile
# ─────────────────────────────────────────────
def get_iv_rank_percentile(conn, symbol: str, current_iv: float) -> tuple:
    """
    IV Rank   = (current_iv - 52w_low) / (52w_high - 52w_low) * 100
    IV %ile   = % of days in past year where IV was BELOW current_iv
    Returns (iv_rank, iv_percentile) — both None if < 5 history rows.
    """
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT atm_iv FROM iv_history
                WHERE  symbol = %s
                  AND  recorded_at >= CURRENT_DATE - INTERVAL '365 days'
                ORDER  BY recorded_at
            """, (symbol,))
            rows = [r[0] for r in cur.fetchall()]

        if len(rows) < 5:
            return None, None

        iv_arr     = np.array(rows, dtype=float)
        iv_high    = iv_arr.max()
        iv_low     = iv_arr.min()
        iv_rank    = round((current_iv - iv_low) / (iv_high - iv_low) * 100, 1) \
                     if iv_high != iv_low else 50.0
        iv_pct     = round(float(np.mean(iv_arr < current_iv)) * 100, 1)
        return iv_rank, iv_pct

    except Exception as e:
        log.debug("IV rank failed: %s", e)
        return None, None


def save_iv_history(conn, symbol: str, atm_iv: float):
    """Store today's ATM IV for future IV Rank calculation."""
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO iv_history (symbol, recorded_at, atm_iv)
                VALUES (%s, CURRENT_DATE, %s)
                ON CONFLICT (symbol, recorded_at)
                DO UPDATE SET atm_iv = EXCLUDED.atm_iv;
            """, (symbol, atm_iv))
        conn.commit()
    except Exception as e:
        log.debug("iv_history save failed: %s", e)


# ─────────────────────────────────────────────
# Max Pain
# ─────────────────────────────────────────────
def calculate_max_pain(rows: list) -> Optional[float]:
    """
    Max Pain = strike at which total option buyer loss is maximised
    (i.e., where market makers profit most at expiry).
    Uses only the nearest expiry rows.
    """
    try:
        # Group OI by strike for nearest expiry
        from collections import defaultdict
        nearest = min({r[2] for r in rows if r[2] is not None}, default=None)
        if not nearest:
            return None

        ce_oi: dict = defaultdict(float)
        pe_oi: dict = defaultdict(float)
        for r in rows:
            if r[2] != nearest:
                continue
            strike = r[3]
            oi     = r[5] or 0
            if r[4] == "CE":
                ce_oi[strike] += oi
            else:
                pe_oi[strike] += oi

        strikes = sorted(set(ce_oi) | set(pe_oi))
        if not strikes:
            return None

        min_pain   = float("inf")
        max_pain_k = strikes[0]

        for k in strikes:
            # Total loss to CE buyers if market settles at k
            ce_loss = sum(max(0, k - s) * oi for s, oi in ce_oi.items())
            # Total loss to PE buyers if market settles at k
            pe_loss = sum(max(0, s - k) * oi for s, oi in pe_oi.items())
            total   = ce_loss + pe_loss
            if total < min_pain:
                min_pain   = total
                max_pain_k = k

        return float(max_pain_k)
    except Exception as e:
        log.debug("Max pain calc failed: %s", e)
        return None


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _date(val):
    if val is None:
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(val).strip(), fmt).date()
        except:
            pass
    return None

def _int(row, *keys):
    col_map = {c.lower(): c for c in row.index}
    for k in keys:
        c = col_map.get(k.lower())
        if c and row[c] is not None and str(row[c]) not in ("", "-", "nan", "NaN"):
            try:
                return int(float(row[c]))
            except:
                pass
    return None

def _float(row, *keys):
    col_map = {c.lower(): c for c in row.index}
    for k in keys:
        c = col_map.get(k.lower())
        if c and row[c] is not None and str(row[c]) not in ("", "-", "nan", "NaN"):
            try:
                return float(row[c])
            except:
                pass
    return None

def _dte(expiry_date) -> Optional[int]:
    """Days to expiry from today."""
    if expiry_date is None:
        return None
    today = date.today()
    if isinstance(expiry_date, str):
        expiry_date = _date(expiry_date)
    try:
        return max(0, (expiry_date - today).days)
    except:
        return None

def _moneyness(strike, underlying, option_type, threshold=0.005) -> str:
    """Classify ITM / ATM / OTM."""
    if underlying is None or strike is None:
        return "OTM"
    ratio = (strike - underlying) / underlying
    if abs(ratio) <= threshold:
        return "ATM"
    if option_type == "CE":
        return "ITM" if ratio < 0 else "OTM"
    else:  # PE
        return "ITM" if ratio > 0 else "OTM"

def _intrinsic(strike, underlying, option_type) -> float:
    if underlying is None or strike is None:
        return 0.0
    if option_type == "CE":
        return max(0.0, underlying - strike)
    return max(0.0, strike - underlying)

# ─────────────────────────────────────────────
# Parse one expiry DataFrame — with all enrichments
# ─────────────────────────────────────────────
def parse_df(df, symbol, fetched_at, expiry_date, underlying, conn):
    import pandas as pd
    rows = []

    dte = _dte(expiry_date)
    T   = dte / 365.0 if dte and dte > 0 else 0.0001   # time in years

    for _, row in df.iterrows():
        row    = row.where(pd.notna(row), None)
        strike = _float(row, "strikePrice", "strike_price", "strike")

        for opt in ("CE", "PE"):
            p   = opt + "_"
            oi  = _int(row,   f"{p}openInterest",         f"{p}oi")
            ltp = _float(row, f"{p}lastPrice",            f"{p}ltp")
            vol = _int(row,   f"{p}totalTradedVolume",    f"{p}volume")
            iv  = _float(row, f"{p}impliedVolatility",    f"{p}iv")
            coi = _int(row,   f"{p}changeinOpenInterest", f"{p}change_in_oi")
            bq  = _int(row,   f"{p}bidQty",               f"{p}bid_qty")
            bp  = _float(row, f"{p}bidprice",             f"{p}bid_price", f"{p}bidPrice")
            aq  = _int(row,   f"{p}askQty",               f"{p}ask_qty")
            ap  = _float(row, f"{p}askPrice",             f"{p}ask_price")

            # ── Derived fields ────────────────────────────────
            moneyness  = _moneyness(strike, underlying, opt)
            intrinsic  = _intrinsic(strike, underlying, opt)
            time_val   = max(0.0, (ltp or 0) - intrinsic)

            # Bid-Ask spread
            spread     = None
            spread_pct = None
            if bp is not None and ap is not None and ap > 0:
                spread     = round(ap - bp, 2)
                mid        = (ap + bp) / 2
                spread_pct = round(spread / mid * 100, 2) if mid > 0 else None

            # ── Greeks ────────────────────────────────────────
            sigma  = (iv or 0) / 100.0     # convert % to decimal
            greeks = black_scholes_greeks(
                S=underlying or 0,
                K=strike     or 0,
                T=T,
                r=RISK_FREE_RATE,
                sigma=sigma,
                option_type=opt,
            )

            # ── IV Rank / Percentile ──────────────────────────
            iv_rank, iv_pct = get_iv_rank_percentile(conn, symbol, iv or 0) \
                              if iv else (None, None)

            def _n(v):
                """Convert any numpy scalar to Python native type."""
                if v is None: return None
                try: return float(v)
                except: return None

            rows.append((
                # Core
                symbol, fetched_at, expiry_date,
                _n(strike), opt,
                oi, coi, None, vol, _n(iv), _n(ltp),
                None, None,       # change, pct_change
                bq, _n(bp), aq, _n(ap),
                None, None,       # total_buy/sell_qty
                None, None, None, None, None,   # OHLC
                _n(underlying),
                # Derived
                dte, moneyness,
                _n(round(intrinsic, 2)), _n(round(time_val, 2)),
                # Greeks
                greeks["delta"], greeks["gamma"],
                greeks["theta"], greeks["vega"], greeks["rho"],
                # Liquidity
                _n(spread), _n(spread_pct),
                # IV analytics
                _n(iv_rank), _n(iv_pct),
            ))

    return rows


# ─────────────────────────────────────────────
# Fetch ALL expiries
# ─────────────────────────────────────────────
def fetch_all_expiries(nse, symbol: str, equity: bool,
                       fetched_at: datetime, conn):
    sym        = symbol.upper()
    all_rows   = []
    underlying = 0.0

    if equity:
        expiries = nse.equityOptions.expiry_dates(sym)
    else:
        expiries = nse.options.expiry_dates(sym)

    log.info("Found %d expiries for %s", len(expiries), sym)

    for expiry in expiries:
        try:
            if equity:
                df, _, ul = nse.equityOptions.option_chain(sym, expiry_date=expiry)
            else:
                df, _, ul = nse.options.option_chain(sym, expiry_date=expiry)

            if ul:
                underlying = float(ul)
            expiry_date = _date(expiry)
            rows = parse_df(df, symbol, fetched_at, expiry_date, underlying, conn)
            all_rows.extend(rows)
            log.info("  %s → %d rows", expiry, len(rows))

        except Exception as e:
            log.warning("  Skipping expiry %s: %s", expiry, e)

    log.info("Total rows: %d", len(all_rows))
    return all_rows, underlying


# ─────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────
def build_metadata(rows, symbol, fetched_at, underlying, india_vix, conn):
    ce = [r for r in rows if r[4] == "CE"]
    pe = [r for r in rows if r[4] == "PE"]

    call_oi  = sum(r[5]  or 0 for r in ce) or None
    put_oi   = sum(r[5]  or 0 for r in pe) or None
    call_vol = sum(r[8]  or 0 for r in ce) or None
    put_vol  = sum(r[8]  or 0 for r in pe) or None
    pcr_oi   = round(put_oi  / call_oi,  4) if call_oi  else None
    pcr_vol  = round(put_vol / call_vol, 4) if call_vol else None

    strikes  = sorted({r[3] for r in rows if r[3] is not None})
    atm      = min(strikes, key=lambda s: abs(s - underlying)) \
               if (underlying and strikes) else None

    max_pain = calculate_max_pain(rows)

    # ATM IV for IV rank (nearest expiry ATM CE IV)
    atm_iv_rows = [r for r in rows
                   if r[3] == atm and r[4] == "CE" and r[9] is not None]
    atm_iv = float(atm_iv_rows[0][9]) if atm_iv_rows else None

    iv_rank, iv_pct = get_iv_rank_percentile(conn, symbol, atm_iv) \
                      if atm_iv else (None, None)

    if atm_iv:
        save_iv_history(conn, symbol, atm_iv)

    return (symbol, fetched_at, underlying, india_vix,
            call_oi, put_oi, call_vol, put_vol,
            pcr_oi, pcr_vol, atm, max_pain, iv_rank, iv_pct)


# ─────────────────────────────────────────────
# DB write
# ─────────────────────────────────────────────
OC_COLS = """
    symbol, fetched_at, expiry_date, strike_price, option_type,
    open_interest, change_in_oi, pct_change_oi, total_traded_volume,
    implied_volatility, last_price, change, pct_change,
    bid_qty, bid_price, ask_qty, ask_price,
    total_buy_qty, total_sell_qty,
    open, high, low, close, prev_close, underlying_value,
    dte, moneyness, intrinsic_value, time_value,
    delta, gamma, theta, vega, rho,
    bid_ask_spread, bid_ask_spread_pct,
    iv_rank, iv_percentile
"""

UPSERT_OC_SQL = f"""
    INSERT INTO nse_option_chain ({OC_COLS})
    VALUES %s
    ON CONFLICT (symbol, fetched_at, expiry_date, strike_price, option_type)
    DO UPDATE SET
        open_interest       = EXCLUDED.open_interest,
        change_in_oi        = EXCLUDED.change_in_oi,
        total_traded_volume = EXCLUDED.total_traded_volume,
        implied_volatility  = EXCLUDED.implied_volatility,
        last_price          = EXCLUDED.last_price,
        bid_qty             = EXCLUDED.bid_qty,
        bid_price           = EXCLUDED.bid_price,
        ask_qty             = EXCLUDED.ask_qty,
        ask_price           = EXCLUDED.ask_price,
        underlying_value    = EXCLUDED.underlying_value,
        dte                 = EXCLUDED.dte,
        moneyness           = EXCLUDED.moneyness,
        intrinsic_value     = EXCLUDED.intrinsic_value,
        time_value          = EXCLUDED.time_value,
        delta               = EXCLUDED.delta,
        gamma               = EXCLUDED.gamma,
        theta               = EXCLUDED.theta,
        vega                = EXCLUDED.vega,
        rho                 = EXCLUDED.rho,
        bid_ask_spread      = EXCLUDED.bid_ask_spread,
        bid_ask_spread_pct  = EXCLUDED.bid_ask_spread_pct,
        iv_rank             = EXCLUDED.iv_rank,
        iv_percentile       = EXCLUDED.iv_percentile;
"""

INSERT_META_SQL = """
    INSERT INTO nse_option_chain_metadata
        (symbol, fetched_at, underlying_value, india_vix,
         total_call_oi, total_put_oi, total_call_vol, total_put_vol,
         pcr_oi, pcr_vol, atm_strike, max_pain, iv_rank, iv_percentile)
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s);
"""

INSERT_TICK_SQL = """
    INSERT INTO market_ticks (symbol, price, india_vix, timestamp)
    VALUES (%s, %s, %s, %s);
"""

def upsert_to_db(conn, rows, meta, symbol, underlying, india_vix, fetched_at):
    with conn.cursor() as cur:
        if rows:
            execute_values(cur, UPSERT_OC_SQL, rows, page_size=500)
            log.info("  → upserted %d option rows", len(rows))
        cur.execute(INSERT_META_SQL, meta)
        log.info("  → metadata: underlying=%.2f  ATM=%s  PCR=%.4f  VIX=%s  MaxPain=%s",
                 meta[2] or 0, meta[10], meta[8] or 0, meta[3], meta[11])
        if underlying:
            cur.execute(INSERT_TICK_SQL, (symbol, underlying, india_vix, fetched_at))
            log.info("  → market_tick: %s @ %.2f", symbol, underlying)
    conn.commit()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run(symbol: str, equity: bool = False):
    conn = get_connection()
    ensure_schema(conn)
    nse        = NSE()
    fetched_at = datetime.now(timezone.utc)

    india_vix               = fetch_india_vix()
    rows, underlying        = fetch_all_expiries(nse, symbol, equity, fetched_at, conn)
    meta                    = build_metadata(rows, symbol, fetched_at,
                                             underlying, india_vix, conn)
    upsert_to_db(conn, rows, meta, symbol, underlying, india_vix, fetched_at)
    conn.close()
    log.info("Done. Rows: %d  Spot: %.2f  VIX: %s",
             len(rows), underlying, india_vix)


def _ist_now():
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)

def _is_market_hours() -> bool:
    """True between 9:00 AM and 3:35 PM IST Mon-Fri."""
    ist = _ist_now()
    if ist.weekday() >= 5:
        return False
    return (ist.hour == 9 and ist.minute >= 0) or            (10 <= ist.hour <= 14) or            (ist.hour == 15 and ist.minute <= 35)

def run_loop(symbol: str, interval: int, equity: bool = False):
    log.info("Loop: %s every %d sec. Ctrl+C to stop.", symbol, interval)
    conn = get_connection()
    ensure_schema(conn)
    conn.close()
    nse = NSE()

    consecutive_errors = 0

    while True:
        if not _is_market_hours():
            log.info("Outside market hours — exiting collector.")
            break
        try:
            conn       = get_connection()
            fetched_at = datetime.now(timezone.utc)

            india_vix        = fetch_india_vix()
            rows, underlying = fetch_all_expiries(nse, symbol, equity, fetched_at, conn)
            meta             = build_metadata(rows, symbol, fetched_at,
                                              underlying, india_vix, conn)
            upsert_to_db(conn, rows, meta, symbol, underlying, india_vix, fetched_at)
            conn.close()
            consecutive_errors = 0   # reset on success

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break

        except Exception as e:
            consecutive_errors += 1
            log.warning("Error #%d: %s", consecutive_errors, e)

            # After 3 consecutive errors, refresh NSE session
            if consecutive_errors >= 3:
                log.warning("3 consecutive errors — refreshing NSE session…")
                try:
                    nse = NSE()
                    consecutive_errors = 0
                    log.info("NSE session refreshed.")
                except Exception as se:
                    log.error("Session refresh failed: %s", se)

        log.info("Sleeping %d sec…\n", interval)
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE Option Chain Collector v3.0")
    parser.add_argument("--symbol", default="NIFTY")
    parser.add_argument("--loop",   type=int, default=0,
                        help="Refresh every N seconds (0 = once)")
    parser.add_argument("--equity", action="store_true",
                        help="Stock options instead of index")
    args = parser.parse_args()

    if args.loop > 0:
        run_loop(args.symbol, args.loop, args.equity)
    else:
        run(args.symbol, args.equity)