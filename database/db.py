"""
database/db.py  —  Single source of truth for all DB operations
════════════════════════════════════════════════════════════════
- All table DDL lives here (not in collector.py or strategy.py)
- Reusable connection context manager
- Common insert / query helpers used by collector, strategy, paper_trader

Usage:
    from database.db import db, ensure_schema, insert_tick, ...

Requirements:
    pip install psycopg2-binary
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import execute_values

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# Config  —  reads from env vars, falls back to dev defaults
# ══════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host":     os.getenv("DB_HOST",     "127.0.0.1"),
    "port":     int(os.getenv("DB_PORT", "5433")),
    "dbname":   os.getenv("DB_NAME",     "trading_db"),
    "user":     os.getenv("DB_USER",     "postgres"),
    "password": os.getenv("DB_PASSWORD", "2000"),       # override via env in prod
}


# ══════════════════════════════════════════════════════════════
# Connection
# ══════════════════════════════════════════════════════════════
@contextmanager
def db() -> Generator[PgConnection, None, None]:
    """
    Context manager that yields a live psycopg2 connection
    and closes it cleanly on exit.

    Usage:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
    """
    conn = None
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        yield conn
    except psycopg2.OperationalError as e:
        log.error("DB connection failed: %s", e)
        raise
    finally:
        if conn and not conn.closed:
            conn.close()


# ══════════════════════════════════════════════════════════════
# Schema  —  ALL table DDL in one place
# ══════════════════════════════════════════════════════════════
_SCHEMA_SQL = """
-- ── Spot price feed ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS market_ticks (
    id          BIGSERIAL    PRIMARY KEY,
    symbol      TEXT         NOT NULL,
    price       NUMERIC      NOT NULL,
    volume      BIGINT,                      -- populated when available
    india_vix   NUMERIC,
    timestamp   TIMESTAMPTZ  NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ticks_symbol_ts
    ON market_ticks (symbol, timestamp DESC);

-- ── Option chain metadata (per-fetch summary) ─────────────────
CREATE TABLE IF NOT EXISTS nse_option_chain_metadata (
    id                  SERIAL       PRIMARY KEY,
    symbol              TEXT         NOT NULL,
    fetched_at          TIMESTAMPTZ  NOT NULL,
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
    id                  BIGSERIAL    PRIMARY KEY,
    symbol              TEXT         NOT NULL,
    fetched_at          TIMESTAMPTZ  NOT NULL,
    expiry_date         DATE,
    strike_price        NUMERIC,
    option_type         CHAR(2)      NOT NULL,

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

    -- Derived
    dte                 INTEGER,
    moneyness           TEXT,
    intrinsic_value     NUMERIC,
    time_value          NUMERIC,

    -- Greeks (Black-Scholes)
    delta               NUMERIC,
    gamma               NUMERIC,
    theta               NUMERIC,
    vega                NUMERIC,
    rho                 NUMERIC,

    -- Liquidity
    bid_ask_spread      NUMERIC,
    bid_ask_spread_pct  NUMERIC,

    -- IV analytics
    iv_rank             NUMERIC,
    iv_percentile       NUMERIC,

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
    signal          TEXT         NOT NULL,   -- BUY_CE / BUY_PE / HOLD
    action          TEXT,                    -- same as signal (kept for compat)
    price           NUMERIC      NOT NULL,
    confidence      NUMERIC,
    -- Indicators
    ema_fast        NUMERIC,
    ema_slow        NUMERIC,
    ema_medium      NUMERIC,
    ema_long        NUMERIC,
    rsi             NUMERIC,
    macd_histogram  NUMERIC,
    supertrend_bull BOOLEAN,
    vwap            NUMERIC,
    pcr             NUMERIC,
    india_vix       NUMERIC,
    max_pain        NUMERIC,
    atm_strike      NUMERIC,
    -- Option contract selected
    option_type     TEXT,
    strike          NUMERIC,
    expiry          TEXT,
    dte             INTEGER,
    option_price    NUMERIC,
    delta           NUMERIC,
    theta           NUMERIC,
    vega            NUMERIC,
    iv              NUMERIC,
    iv_rank         NUMERIC,
    -- Meta
    votes           JSONB,
    reason          TEXT,
    timestamp       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_ts
    ON signals (symbol, timestamp DESC);

-- ── Trades (written by paper_trader.py) ──────────────────────
CREATE TABLE IF NOT EXISTS trades (
    id              BIGSERIAL    PRIMARY KEY,
    symbol          TEXT         NOT NULL,
    option_type     TEXT         NOT NULL,   -- CE / PE
    strike          NUMERIC,
    expiry          TEXT,
    direction       TEXT         NOT NULL,   -- BUY
    quantity        INTEGER      NOT NULL DEFAULT 1,
    entry_price     NUMERIC      NOT NULL,
    exit_price      NUMERIC,
    stop_loss       NUMERIC,
    target          NUMERIC,
    status          TEXT         NOT NULL DEFAULT 'OPEN',  -- OPEN / CLOSED
    pnl             NUMERIC,
    signal_id       BIGINT       REFERENCES signals(id),
    entry_time      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    exit_time       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_trades_symbol_status
    ON trades (symbol, status);

-- ── IV history (for IV Rank / Percentile) ────────────────────
CREATE TABLE IF NOT EXISTS iv_history (
    id          BIGSERIAL   PRIMARY KEY,
    symbol      TEXT        NOT NULL,
    recorded_at DATE        NOT NULL,
    atm_iv      NUMERIC     NOT NULL,
    UNIQUE (symbol, recorded_at)
);
"""


def ensure_schema(conn: Optional[PgConnection] = None) -> None:
    """
    Create all tables and indexes if they don't exist.
    Accepts an existing connection or opens its own.
    Safe to call on every startup — fully idempotent.
    """
    def _run(c):
        with c.cursor() as cur:
            cur.execute(_SCHEMA_SQL)
        c.commit()
        log.info("Schema ready.")

    if conn:
        _run(conn)
    else:
        with db() as c:
            _run(c)


# ══════════════════════════════════════════════════════════════
# market_ticks helpers
# ══════════════════════════════════════════════════════════════
def insert_tick(conn: PgConnection,
                symbol: str,
                price: float,
                india_vix: Optional[float],
                timestamp: datetime,
                volume: Optional[int] = None) -> None:
    """Insert one spot-price tick."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO market_ticks (symbol, price, volume, india_vix, timestamp)
            VALUES (%s, %s, %s, %s, %s)
        """, (symbol, price, volume, india_vix, timestamp))


def get_market_ticks(conn: PgConnection,
                     symbol: str,
                     limit: int = 200):
    """
    Return the latest `limit` ticks for a symbol as a list of dicts,
    oldest-first (ready for indicator calculation).
    """
    import pandas as pd
    df = pd.read_sql_query("""
        SELECT id, symbol, price::FLOAT,
               volume::FLOAT,
               india_vix::FLOAT,
               timestamp
        FROM   market_ticks
        WHERE  symbol = %s
        ORDER  BY timestamp DESC
        LIMIT  %s
    """, conn, params=(symbol, limit))

    if df.empty:
        return df

    return df.iloc[::-1].reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# signals helpers
# ══════════════════════════════════════════════════════════════
def insert_signal(conn: PgConnection, data: dict) -> int:
    """
    Insert a signal row. `data` is a flat dict matching the signals columns.
    Returns the new signal id.
    """
    sql = """
        INSERT INTO signals (
            symbol, signal, action, price, confidence,
            ema_fast, ema_slow, ema_medium, ema_long,
            rsi, macd_histogram, supertrend_bull, vwap,
            pcr, india_vix, max_pain, atm_strike,
            option_type, strike, expiry, dte, option_price,
            delta, theta, vega, iv, iv_rank,
            votes, reason, timestamp
        ) VALUES (
            %(symbol)s, %(signal)s, %(signal)s, %(price)s, %(confidence)s,
            %(ema_fast)s, %(ema_slow)s, %(ema_medium)s, %(ema_long)s,
            %(rsi)s, %(macd_histogram)s, %(supertrend_bull)s, %(vwap)s,
            %(pcr)s, %(india_vix)s, %(max_pain)s, %(atm_strike)s,
            %(option_type)s, %(strike)s, %(expiry)s, %(dte)s, %(option_price)s,
            %(delta)s, %(theta)s, %(vega)s, %(iv)s, %(iv_rank)s,
            %(votes)s, %(reason)s, %(timestamp)s
        )
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {**data, "votes": json.dumps(data.get("votes", {}))})
        return cur.fetchone()[0]


# ══════════════════════════════════════════════════════════════
# trades helpers
# ══════════════════════════════════════════════════════════════
def open_trade(conn: PgConnection, data: dict) -> int:
    """Insert an OPEN trade. Returns the new trade id."""
    sql = """
        INSERT INTO trades (
            symbol, option_type, strike, expiry,
            direction, quantity, entry_price,
            stop_loss, target, signal_id, entry_time
        ) VALUES (
            %(symbol)s, %(option_type)s, %(strike)s, %(expiry)s,
            %(direction)s, %(quantity)s, %(entry_price)s,
            %(stop_loss)s, %(target)s, %(signal_id)s, %(entry_time)s
        )
        RETURNING id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, data)
        return cur.fetchone()[0]


def close_trade(conn: PgConnection,
                trade_id: int,
                exit_price: float,
                pnl: float,
                exit_time: datetime) -> None:
    """Mark a trade as CLOSED with exit price and PnL."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE trades
            SET exit_price = %s,
                pnl        = %s,
                status     = 'CLOSED',
                exit_time  = %s
            WHERE id = %s
        """, (exit_price, pnl, exit_time, trade_id))


def get_open_trades(conn: PgConnection, symbol: Optional[str] = None):
    """Return all open trades, optionally filtered by symbol."""
    import pandas as pd
    query = "SELECT * FROM trades WHERE status = 'OPEN'"
    params = ()
    if symbol:
        query += " AND symbol = %s"
        params = (symbol,)
    return pd.read_sql_query(query, conn, params=params)


# ══════════════════════════════════════════════════════════════
# Quick connection test  (run this file directly to verify)
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    print("Testing DB connection...")
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print("PostgreSQL:", cur.fetchone()[0])
        ensure_schema(conn)
        print("All tables created/verified.")

        import pandas as pd
        df = pd.read_sql_query(
            "SELECT symbol, COUNT(*) AS ticks FROM market_ticks GROUP BY symbol",
            conn
        )
        if df.empty:
            print("market_ticks: empty (collector not run yet)")
        else:
            print("\nmarket_ticks summary:")
            print(df.to_string(index=False))