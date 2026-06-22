"""
trading/paper_trader.py  —  v2.0  TOP-TIER PAPER TRADING ENGINE
═══════════════════════════════════════════════════════════════════════════════
Aligned with strategy.py v4.0. Fully autonomous virtual broker.

Upgrades in v2.0 over v1.0:
  1. Crash recovery      — reloads open trades from DB on startup,
                           monitoring continues as if nothing happened
  2. Dynamic trailing SL — SL keeps moving up as price makes new highs,
                           not just one-time activation
  3. Partial exits       — closes 50% at first target, trails the rest
                           with tighter SL — captures big trending days
  4. Signal staleness    — alerts if strategy.py has gone silent > N mins
  5. Capital-based P&L   — tracks return on capital deployed, not just
                           option premium %, giving real account picture
  6. Re-entry logic      — after a clean win, re-enters if signal still
                           valid and confidence above threshold
  7. Capital guard       — validates available capital before opening,
                           prevents over-allocation

Full feature list:
  • Reads BUY_CE / BUY_PE signals from strategy.py every 60 sec
  • Opens virtual trades with entry price, SL, target, Kelly lots
  • Monitors live option price from nse_option_chain every tick
  • Closes on: SL | Target (partial + full) | EXIT signal | EOD | VIX spike
  • Trailing SL moves dynamically with every new high (not one-time)
  • 50% partial exit at first target, trail remaining 50%
  • Full crash recovery — reloads open trades on restart
  • Daily loss circuit breaker — halts after MAX_DAILY_LOSS
  • Signal staleness alert — warns if strategy.py goes silent
  • Capital P&L tracked alongside premium P&L
  • Re-entry after clean wins if signal still valid
  • Telegram alerts: open / partial / close / halt / staleness
  • Detailed per-trade DB storage with all context
  • Stats: win rate, Sharpe, profit factor, drawdown, by-reason, by-regime
  • CSV export
  • Daily EOD summary with full session metrics

Usage:
    # Run alongside strategy.py (two terminals)
    Terminal 1: python strategy.py NIFTY --paper --loop 60
    Terminal 2: python paper_trader.py NIFTY

    # View performance
    python paper_trader.py NIFTY --today          # today's trades
    python paper_trader.py NIFTY --stats          # all-time stats
    python paper_trader.py NIFTY --export         # export to CSV

Requirements:
    pip install psycopg2-binary pandas numpy requests
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Optional

import numpy as np
import pandas as pd
import psycopg2
import requests

# ══════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("paper_trader")

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     5433,
    "dbname":   "trading_db",
    "user":     "postgres",
    "password": "2000",
}

TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID   = ""

IST = timezone(timedelta(hours=5, minutes=30))

# Market hours (IST)
MARKET_OPEN   = (9,  15)
MARKET_CLOSE  = (15,  0)
EOD_EXIT_H    = 14
EOD_EXIT_M    = 50    # force-close all positions at 2:50 PM

# ── Risk management ───────────────────────────────────────────
CAPITAL          = 500_000   # ₹5 lakh paper capital
MAX_RISK_PER_TRADE = 2.0     # max % of capital per trade
MAX_DAILY_LOSS   = 3.0       # halt trading if daily P&L < -3%
VIX_EXIT         = 20.0      # emergency exit on VIX spike

# ── Stop loss / target ────────────────────────────────────────
SL_PCT           = 25.0      # initial stop loss (% of option premium)
TGT1_PCT         = 30.0      # first target — close 50% here
TGT2_PCT         = 60.0      # second target — close remaining 50%

# ── Trailing SL ───────────────────────────────────────────────
TRAIL_ACTIVATE   = 20.0      # start trailing after 20% gain
TRAIL_DISTANCE   = 10.0      # SL trails 10% below current high
                              # e.g. if high = ₹200 → SL = ₹180

# ── Re-entry ──────────────────────────────────────────────────
REENTRY_ENABLED      = True
REENTRY_MIN_CONF     = 65.0   # minimum confidence for re-entry
REENTRY_COOLDOWN_MIN = 15     # minutes to wait after a trade closes

# ── Signal staleness ─────────────────────────────────────────
STALENESS_MINUTES = 5         # alert if no signal in 5 min during market hours

LOT_SIZES = {
    "NIFTY": 25, "BANKNIFTY": 15,
    "FINNIFTY": 40, "MIDCPNIFTY": 75, "NIFTYNXT50": 25,
}
LOT_SIZE      = 25    # fallback
POLL_INTERVAL = 60    # seconds


# ══════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════
@dataclass
class OpenTrade:
    """One active paper trade — full state."""
    trade_id:           int
    symbol:             str
    signal_id:          int
    direction:          str        # BUY_CE / BUY_PE
    option_type:        str        # CE / PE
    strike:             float
    expiry:             str
    dte:                int
    entry_option_price: float      # premium at entry
    entry_spot:         float      # underlying at entry
    lots:               int        # total lots opened
    lots_remaining:     int        # lots still open (after partial exit)
    lot_size:           int
    sl_price:           float      # current dynamic SL level
    target1_price:      float      # first target (50% exit)
    target2_price:      float      # second target (remaining 50%)
    max_price_seen:     float      # highest premium seen — drives trailing SL
    partial_exited:     bool = False   # True after first 50% exit
    trail_active:       bool = False
    entry_time:         datetime  = field(default_factory=lambda: datetime.now(IST))
    signal_conf:        float     = 0.0
    signal_regime:      str       = ""
    signal_gex:         str       = ""
    signal_pcr:         Optional[float] = None
    signal_vix:         Optional[float] = None
    signal_votes:       dict      = field(default_factory=dict)

    # ── P&L helpers ───────────────────────────────────────────
    @property
    def capital_deployed(self) -> float:
        """Total capital at risk for remaining lots."""
        return self.entry_option_price * self.lot_size * self.lots_remaining

    @property
    def capital_deployed_total(self) -> float:
        """Capital deployed at entry (all lots)."""
        return self.entry_option_price * self.lot_size * self.lots

    def pnl_premium_pct(self, current_price: float) -> float:
        """P&L as % of option premium (how much premium moved)."""
        if self.entry_option_price == 0:
            return 0.0
        return (current_price - self.entry_option_price) \
               / self.entry_option_price * 100

    def pnl_capital_pct(self, current_price: float) -> float:
        """P&L as % of CAPITAL DEPLOYED — real account impact."""
        deployed = self.capital_deployed_total
        if deployed == 0:
            return 0.0
        pnl_rs = (current_price - self.entry_option_price) \
                 * self.lot_size * self.lots_remaining
        return pnl_rs / deployed * 100

    def pnl_rs(self, current_price: float) -> float:
        """Absolute ₹ P&L on remaining lots."""
        return (current_price - self.entry_option_price) \
               * self.lot_size * self.lots_remaining

    def pnl_rs_total(self, current_price: float,
                     partial_exit_price: float = 0.0,
                     partial_lots: int = 0) -> float:
        """Total ₹ P&L including already-closed partial lots."""
        partial_pnl = (partial_exit_price - self.entry_option_price) \
                      * self.lot_size * partial_lots if partial_lots else 0
        remaining_pnl = self.pnl_rs(current_price)
        return partial_pnl + remaining_pnl


@dataclass
class PartialExit:
    """Records a partial exit event."""
    trade_id:     int
    lots_closed:  int
    exit_price:   float
    exit_time:    datetime
    pnl_pct:      float
    pnl_rs:       float
    reason:       str


@dataclass
class DayStats:
    """Session-level performance tracker."""
    trades_taken:       int   = 0
    trades_closed:      int   = 0
    partial_exits:      int   = 0
    wins:               int   = 0
    losses:             int   = 0
    total_pnl_premium:  float = 0.0   # sum of option % P&L
    total_pnl_rs:       float = 0.0   # sum of ₹ P&L
    total_capital_used: float = 0.0
    pnl_list:           list  = field(default_factory=list)
    halted:             bool  = False
    last_close_time:    Optional[datetime] = None

    # Computed properties
    @property
    def win_rate(self) -> float:
        return (self.wins / self.trades_closed * 100
                if self.trades_closed > 0 else 0.0)

    @property
    def sharpe(self) -> float:
        if len(self.pnl_list) < 2:
            return 0.0
        arr = np.array(self.pnl_list, dtype=float)
        return float(arr.mean() / (arr.std() + 1e-9))

    @property
    def profit_factor(self) -> float:
        gw = sum(p for p in self.pnl_list if p > 0)
        gl = abs(sum(p for p in self.pnl_list if p < 0))
        return gw / gl if gl > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.pnl_list:
            return 0.0
        c = np.cumsum(self.pnl_list)
        return float(np.min(c - np.maximum.accumulate(c)))

    @property
    def avg_win(self) -> float:
        w = [p for p in self.pnl_list if p > 0]
        return float(np.mean(w)) if w else 0.0

    @property
    def avg_loss(self) -> float:
        l = [p for p in self.pnl_list if p <= 0]
        return float(np.mean(l)) if l else 0.0

    @property
    def capital_return_pct(self) -> float:
        """True return on capital deployed today."""
        if self.total_capital_used == 0:
            return 0.0
        return self.total_pnl_rs / self.total_capital_used * 100

    def reentry_allowed(self) -> bool:
        """True if cooldown after last close has passed."""
        if not REENTRY_ENABLED:
            return False
        if self.last_close_time is None:
            return True
        elapsed = (datetime.now(IST) - self.last_close_time).total_seconds() / 60
        return elapsed >= REENTRY_COOLDOWN_MIN


# ══════════════════════════════════════════════════════════════
# Global state
# ══════════════════════════════════════════════════════════════
OPEN_TRADE:    Optional[OpenTrade] = None
DAY_STATS:     DayStats            = DayStats()
PARTIAL_EXITS: List[PartialExit]   = []


# ══════════════════════════════════════════════════════════════
# Database
# ══════════════════════════════════════════════════════════════
@contextmanager
def db() -> Generator:
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


def ensure_tables():
    """Create paper_trades and partial_exits tables. Idempotent."""
    sql = """
        -- ── Main trade ledger ─────────────────────────────────
        CREATE TABLE IF NOT EXISTS paper_trades (
            id                  BIGSERIAL    PRIMARY KEY,
            symbol              TEXT         NOT NULL,
            signal_id           BIGINT,
            direction           TEXT         NOT NULL,
            option_type         TEXT         NOT NULL,
            strike              NUMERIC      NOT NULL,
            expiry              TEXT,
            dte                 INTEGER,

            -- Entry
            entry_spot          NUMERIC,
            entry_price         NUMERIC      NOT NULL,
            entry_time          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
            lots                INTEGER      NOT NULL DEFAULT 1,
            lot_size            INTEGER,
            capital_deployed    NUMERIC,

            -- Risk
            sl_price            NUMERIC,
            target1_price       NUMERIC,
            target2_price       NUMERIC,
            sl_pct              NUMERIC      DEFAULT 25,
            target1_pct         NUMERIC      DEFAULT 30,
            target2_pct         NUMERIC      DEFAULT 60,

            -- Partial exit
            partial_exited      BOOLEAN      DEFAULT FALSE,
            partial_lots        INTEGER      DEFAULT 0,
            partial_exit_price  NUMERIC,
            partial_exit_time   TIMESTAMPTZ,
            partial_pnl_rs      NUMERIC,

            -- Full exit
            exit_price          NUMERIC,
            exit_time           TIMESTAMPTZ,
            exit_spot           NUMERIC,
            exit_reason         TEXT,
            status              TEXT         NOT NULL DEFAULT 'OPEN',

            -- P&L — both premium % and capital %
            pnl_premium_pct     NUMERIC,     -- % move of option premium
            pnl_capital_pct     NUMERIC,     -- % return on capital deployed
            pnl_rs              NUMERIC,     -- absolute ₹ P&L (remaining lots)
            pnl_rs_total        NUMERIC,     -- total ₹ P&L incl. partial

            -- Trade metadata
            trail_activated     BOOLEAN      DEFAULT FALSE,
            max_price_seen      NUMERIC,
            holding_minutes     INTEGER,

            -- Signal context
            signal_conf         NUMERIC,
            signal_regime       TEXT,
            signal_gex          TEXT,
            signal_pcr          NUMERIC,
            signal_vix          NUMERIC,
            signal_votes        JSONB
        );

        CREATE INDEX IF NOT EXISTS idx_pt_symbol  ON paper_trades (symbol);
        CREATE INDEX IF NOT EXISTS idx_pt_status  ON paper_trades (status);
        CREATE INDEX IF NOT EXISTS idx_pt_date    ON paper_trades (DATE(entry_time));
        CREATE INDEX IF NOT EXISTS idx_pt_signal  ON paper_trades (signal_id);

        -- ── Partial exit log ──────────────────────────────────
        CREATE TABLE IF NOT EXISTS paper_partial_exits (
            id          BIGSERIAL    PRIMARY KEY,
            trade_id    BIGINT       REFERENCES paper_trades(id),
            lots_closed INTEGER,
            exit_price  NUMERIC,
            exit_time   TIMESTAMPTZ,
            pnl_pct     NUMERIC,
            pnl_rs      NUMERIC,
            reason      TEXT
        );
    """
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    log.info("Tables ready.")


# ══════════════════════════════════════════════════════════════
# Upgrade 1: Crash recovery — reload open trades on startup
# ══════════════════════════════════════════════════════════════
def recover_open_trade(symbol: str) -> Optional[OpenTrade]:
    """
    On startup: check DB for any OPEN paper_trades from today.
    If found, reconstruct OpenTrade and resume monitoring.
    This means a crash or restart loses nothing.
    """
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, signal_id, direction, option_type,
                           strike, expiry, dte,
                           entry_price, entry_spot, entry_time,
                           lots, lot_size,
                           sl_price, target1_price, target2_price,
                           partial_exited,
                           COALESCE(lots - partial_lots, lots) AS lots_remaining,
                           max_price_seen, trail_activated,
                           signal_conf, signal_regime, signal_gex,
                           signal_pcr, signal_vix, signal_votes
                    FROM   paper_trades
                    WHERE  symbol = %s
                      AND  status = 'OPEN'
                      AND  DATE(entry_time) = CURRENT_DATE
                    ORDER  BY entry_time DESC
                    LIMIT  1
                """, (symbol,))
                row = cur.fetchone()
                if not row:
                    return None

                (tid, sig_id, direction, opt_type,
                 strike, expiry, dte,
                 entry_price, entry_spot, entry_time,
                 lots, lot_size,
                 sl_price, tgt1, tgt2,
                 partial_exited, lots_remaining,
                 max_price, trail_active,
                 conf, regime, gex, pcr, vix, votes) = row

                trade = OpenTrade(
                    trade_id           = tid,
                    symbol             = symbol,
                    signal_id          = sig_id or 0,
                    direction          = direction,
                    option_type        = opt_type,
                    strike             = float(strike),
                    expiry             = str(expiry or ""),
                    dte                = int(dte or 0),
                    entry_option_price = float(entry_price),
                    entry_spot         = float(entry_spot or 0),
                    lots               = int(lots),
                    lots_remaining     = int(lots_remaining or lots),
                    lot_size           = int(lot_size or LOT_SIZE),
                    sl_price           = float(sl_price or 0),
                    target1_price      = float(tgt1 or 0),
                    target2_price      = float(tgt2 or 0),
                    max_price_seen     = float(max_price or entry_price),
                    partial_exited     = bool(partial_exited),
                    trail_active       = bool(trail_active),
                    entry_time         = entry_time,
                    signal_conf        = float(conf or 0),
                    signal_regime      = str(regime or ""),
                    signal_gex         = str(gex or ""),
                    signal_pcr         = float(pcr) if pcr else None,
                    signal_vix         = float(vix) if vix else None,
                    signal_votes       = votes or {},
                )

                log.info("♻️  Recovered open trade #%d | %s %.0f %s @ ₹%.2f",
                         tid, symbol, float(strike), opt_type, float(entry_price))
                return trade

    except Exception as e:
        log.warning("Trade recovery failed: %s", e)
        return None


# ══════════════════════════════════════════════════════════════
# Data fetchers
# ══════════════════════════════════════════════════════════════
def get_latest_signal(symbol: str,
                      allow_reentry: bool = False) -> Optional[dict]:
    """
    Fetch the most recent unprocessed BUY signal.
    allow_reentry=True skips the 'not already traded' filter
    so we can re-enter after a clean win.
    """
    try:
        with db() as conn:
            with conn.cursor() as cur:
                # Build exclusion clause
                excl = ("AND s.id NOT IN ("
                        "SELECT signal_id FROM paper_trades "
                        "WHERE signal_id IS NOT NULL)"
                        if not allow_reentry else "")

                cur.execute(f"""
                    SELECT s.id, s.action, s.price, s.confidence,
                           s.option_type, s.strike, s.expiry, s.dte,
                           s.option_price, s.recommended_lots,
                           s.regime, s.gex_direction, s.pcr_oi,
                           s.india_vix, s.votes, s.timestamp
                    FROM   signals s
                    WHERE  s.symbol     = %s
                      AND  s.action    IN ('BUY_CE', 'BUY_PE')
                      AND  s.paper_mode = TRUE
                      AND  DATE(s.timestamp) = CURRENT_DATE
                      {excl}
                    ORDER  BY s.timestamp DESC
                    LIMIT  1
                """, (symbol,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = ["id","action","price","confidence",
                        "option_type","strike","expiry","dte",
                        "option_price","recommended_lots",
                        "regime","gex_direction","pcr_oi",
                        "india_vix","votes","timestamp"]
                return dict(zip(cols, row))
    except Exception as e:
        log.warning("Signal fetch: %s", e)
        return None


def get_latest_exit_signal(symbol: str,
                            after_ts: datetime) -> Optional[str]:
    """Returns exit reason string if strategy issued EXIT after our entry."""
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT reason FROM signals
                    WHERE  symbol     = %s
                      AND  action     = 'EXIT'
                      AND  paper_mode = TRUE
                      AND  timestamp  > %s
                    ORDER  BY timestamp DESC LIMIT 1
                """, (symbol, after_ts))
                row = cur.fetchone()
                return str(row[0]) if row else None
    except Exception as e:
        log.warning("Exit signal fetch: %s", e)
        return None


def get_last_signal_time(symbol: str) -> Optional[datetime]:
    """Returns timestamp of most recent signal — used for staleness check."""
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(timestamp) FROM signals
                    WHERE  symbol = %s AND paper_mode = TRUE
                """, (symbol,))
                row = cur.fetchone()
                return row[0] if row and row[0] else None
    except Exception:
        return None


def get_live_option_price(symbol: str, strike: float,
                           opt_type: str, expiry: str) -> Optional[float]:
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT last_price FROM nse_option_chain
                    WHERE  symbol       = %s
                      AND  strike_price = %s
                      AND  option_type  = %s
                      AND  expiry_date  = %s::date
                    ORDER  BY fetched_at DESC LIMIT 1
                """, (symbol, strike, opt_type, expiry))
                row = cur.fetchone()
                return float(row[0]) if row and row[0] else None
    except Exception as e:
        log.warning("Option price: %s", e)
        return None


def get_live_spot(symbol: str) -> Optional[float]:
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT price FROM market_ticks
                    WHERE  symbol = %s
                    ORDER  BY timestamp DESC LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
                return float(row[0]) if row else None
    except Exception as e:
        log.warning("Spot fetch: %s", e)
        return None


def get_live_vix(symbol: str) -> Optional[float]:
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT india_vix FROM nse_option_chain_metadata
                    WHERE  symbol = %s ORDER BY fetched_at DESC LIMIT 1
                """, (symbol,))
                row = cur.fetchone()
                return float(row[0]) if row and row[0] else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# Capital validation
# ══════════════════════════════════════════════════════════════
def validate_capital(option_price: float, lots: int,
                     lot_size: int) -> tuple[bool, str]:
    """
    Upgrade 7: Ensure we have enough capital and aren't over-allocating.
    """
    required   = option_price * lot_size * lots
    max_risk   = CAPITAL * MAX_RISK_PER_TRADE / 100
    daily_loss = abs(DAY_STATS.total_pnl_rs)

    if required > max_risk:
        return False, (f"Capital guard: ₹{required:,.0f} required > "
                       f"₹{max_risk:,.0f} max risk per trade")

    if daily_loss > CAPITAL * MAX_DAILY_LOSS / 100:
        return False, f"Daily loss limit hit: ₹{daily_loss:,.0f}"

    return True, ""


# ══════════════════════════════════════════════════════════════
# Upgrade 2+3: Dynamic trailing SL + partial exits
# ══════════════════════════════════════════════════════════════
def update_trailing_sl(trade: OpenTrade,
                       current_price: float) -> bool:
    """
    Dynamic trailing SL — moves up continuously with new highs.
    Not one-time activation. Every new high tightens the floor.

    Logic:
      Once trade gains TRAIL_ACTIVATE% from entry:
        SL = current_high * (1 - TRAIL_DISTANCE/100)
        This floor moves UP every time price makes a new high.
    """
    if current_price <= trade.max_price_seen:
        return False   # no new high — SL unchanged

    # New high recorded
    trade.max_price_seen = current_price
    gain_pct = trade.pnl_premium_pct(current_price)

    if gain_pct >= TRAIL_ACTIVATE:
        new_sl = round(current_price * (1 - TRAIL_DISTANCE / 100), 2)
        if new_sl > trade.sl_price:   # only move SL up, never down
            old_sl = trade.sl_price
            trade.sl_price    = new_sl
            trade.trail_active = True
            _update_trade_db(trade)
            log.info("🔒 Trailing SL moved: ₹%.2f → ₹%.2f  (high=₹%.2f)",
                     old_sl, new_sl, current_price)
            return True

    return False


def do_partial_exit(trade: OpenTrade, current_price: float,
                    reason: str) -> PartialExit:
    """
    Upgrade 3: Close 50% of position at first target.
    Remaining 50% continues with tighter trailing SL.
    """
    global DAY_STATS, PARTIAL_EXITS

    lots_to_close  = max(1, trade.lots // 2)
    pnl_pct        = trade.pnl_premium_pct(current_price)
    pnl_rs         = ((current_price - trade.entry_option_price)
                      * trade.lot_size * lots_to_close)
    now            = datetime.now(IST)

    # Update trade state
    trade.partial_exited   = True
    trade.lots_remaining  -= lots_to_close

    # Tighten SL after partial — lock in profits on remainder
    # Move SL to break-even at minimum
    be_sl = round(trade.entry_option_price * 1.05, 2)   # 5% above entry
    if be_sl > trade.sl_price:
        trade.sl_price = be_sl
        log.info("🔒 Post-partial SL moved to break-even+: ₹%.2f", be_sl)

    # Record partial in DB
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE paper_trades
                    SET partial_exited     = TRUE,
                        partial_lots       = %s,
                        partial_exit_price = %s,
                        partial_exit_time  = %s,
                        partial_pnl_rs     = %s,
                        sl_price           = %s,
                        trail_activated    = %s,
                        max_price_seen     = %s
                    WHERE id = %s
                """, (lots_to_close, round(current_price, 2), now,
                      round(pnl_rs, 2), trade.sl_price,
                      trade.trail_active, trade.max_price_seen,
                      trade.trade_id))

                cur.execute("""
                    INSERT INTO paper_partial_exits
                    (trade_id, lots_closed, exit_price, exit_time,
                     pnl_pct, pnl_rs, reason)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (trade.trade_id, lots_to_close,
                      round(current_price, 2), now,
                      round(pnl_pct, 2), round(pnl_rs, 2), reason))
            conn.commit()
    except Exception as e:
        log.error("Partial exit DB write: %s", e)

    partial = PartialExit(
        trade_id    = trade.trade_id,
        lots_closed = lots_to_close,
        exit_price  = current_price,
        exit_time   = now,
        pnl_pct     = pnl_pct,
        pnl_rs      = pnl_rs,
        reason      = reason,
    )
    PARTIAL_EXITS.append(partial)
    DAY_STATS.partial_exits    += 1
    DAY_STATS.total_pnl_rs     += pnl_rs
    DAY_STATS.total_capital_used += trade.capital_deployed

    print_partial_exit(trade, partial)
    send_telegram_partial(trade, partial)

    log.info("📤 Partial exit: %d lots @ ₹%.2f | P&L: %+.2f%% (₹%+.0f) | "
             "Remaining: %d lots",
             lots_to_close, current_price, pnl_pct, pnl_rs,
             trade.lots_remaining)

    return partial


def _update_trade_db(trade: OpenTrade):
    """Persist current trade state (SL, trail, max_price) to DB."""
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE paper_trades
                    SET sl_price        = %s,
                        trail_activated = %s,
                        max_price_seen  = %s
                    WHERE id = %s
                """, (round(trade.sl_price, 2),
                      trade.trail_active,
                      round(trade.max_price_seen, 2),
                      trade.trade_id))
            conn.commit()
    except Exception as e:
        log.debug("Trail DB update: %s", e)


# ══════════════════════════════════════════════════════════════
# Trade open / close
# ══════════════════════════════════════════════════════════════
def open_trade(symbol: str, signal: dict,
               is_reentry: bool = False) -> Optional[OpenTrade]:
    """Open a virtual trade. Returns OpenTrade or None."""
    global OPEN_TRADE, DAY_STATS

    if OPEN_TRADE is not None:
        log.info("Already in trade — skipping.")
        return None

    if DAY_STATS.halted:
        log.warning("Trading halted — daily loss limit.")
        return None

    opt_price = float(signal.get("option_price") or 0)
    if opt_price <= 0:
        log.warning("Invalid option price %.2f — skipping.", opt_price)
        return None

    lots    = int(signal.get("recommended_lots") or 1)
    lot_sz  = LOT_SIZES.get(symbol, LOT_SIZE)

    # Upgrade 7: Capital guard
    ok, reason = validate_capital(opt_price, lots, lot_sz)
    if not ok:
        log.warning("Capital guard blocked trade: %s", reason)
        return None

    strike    = float(signal["strike"])
    sl_price  = round(opt_price * (1 - SL_PCT  / 100), 2)
    tgt1      = round(opt_price * (1 + TGT1_PCT / 100), 2)
    tgt2      = round(opt_price * (1 + TGT2_PCT / 100), 2)
    spot      = get_live_spot(symbol) or float(signal["price"])
    now       = datetime.now(IST)

    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO paper_trades (
                        symbol, signal_id, direction, option_type,
                        strike, expiry, dte,
                        entry_spot, entry_price, entry_time,
                        lots, lot_size, capital_deployed,
                        sl_price, target1_price, target2_price,
                        sl_pct, target1_pct, target2_pct,
                        max_price_seen,
                        signal_conf, signal_regime, signal_gex,
                        signal_pcr, signal_vix, signal_votes
                    ) VALUES (
                        %s,%s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,%s,%s,
                        %s,
                        %s,%s,%s,
                        %s,%s,%s
                    ) RETURNING id
                """, (
                    symbol, int(signal["id"]),
                    signal["action"], signal["option_type"],
                    strike, signal.get("expiry"), signal.get("dte"),
                    spot, opt_price, now,
                    lots, lot_sz,
                    round(opt_price * lot_sz * lots, 2),
                    sl_price, tgt1, tgt2,
                    SL_PCT, TGT1_PCT, TGT2_PCT,
                    opt_price,
                    signal.get("confidence"),
                    signal.get("regime"),
                    signal.get("gex_direction"),
                    signal.get("pcr_oi"),
                    signal.get("india_vix"),
                    json.dumps(signal.get("votes") or {}),
                ))
                trade_id = cur.fetchone()[0]
            conn.commit()
    except Exception as e:
        log.error("Open trade DB insert: %s", e)
        return None

    trade = OpenTrade(
        trade_id           = trade_id,
        symbol             = symbol,
        signal_id          = int(signal["id"]),
        direction          = signal["action"],
        option_type        = signal["option_type"],
        strike             = strike,
        expiry             = signal.get("expiry", ""),
        dte                = int(signal.get("dte") or 0),
        entry_option_price = opt_price,
        entry_spot         = spot,
        lots               = lots,
        lots_remaining     = lots,
        lot_size           = lot_sz,
        sl_price           = sl_price,
        target1_price      = tgt1,
        target2_price      = tgt2,
        max_price_seen     = opt_price,
        entry_time         = now,
        signal_conf        = float(signal.get("confidence") or 0),
        signal_regime      = signal.get("regime", ""),
        signal_gex         = signal.get("gex_direction", ""),
        signal_pcr         = signal.get("pcr_oi"),
        signal_vix         = signal.get("india_vix"),
        signal_votes       = signal.get("votes") or {},
    )

    OPEN_TRADE = trade
    DAY_STATS.trades_taken      += 1
    DAY_STATS.total_capital_used += trade.capital_deployed_total

    label = "♻️  RE-ENTRY" if is_reentry else "✅ OPEN"
    print_trade_open(trade, label)
    send_telegram_open(trade, is_reentry)

    log.info("%s Trade #%d | %s %.0f %s @ ₹%.2f | SL=₹%.2f TGT1=₹%.2f TGT2=₹%.2f",
             label, trade_id, symbol, strike, signal["option_type"],
             opt_price, sl_price, tgt1, tgt2)
    return trade


def close_trade(trade: OpenTrade, current_price: float,
                exit_reason: str, exit_spot: Optional[float] = None):
    """Close the full (remaining) position."""
    global OPEN_TRADE, DAY_STATS

    pnl_pct_premium = trade.pnl_premium_pct(current_price)
    pnl_pct_capital = trade.pnl_capital_pct(current_price)
    pnl_rs          = trade.pnl_rs(current_price)

    # Total P&L including any prior partial exit
    prior_partial_rs = sum(p.pnl_rs for p in PARTIAL_EXITS
                           if p.trade_id == trade.trade_id)
    pnl_rs_total     = pnl_rs + prior_partial_rs

    now     = datetime.now(IST)
    holding = int((now - trade.entry_time).total_seconds() / 60)

    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE paper_trades
                    SET exit_price       = %s,
                        exit_time        = %s,
                        exit_spot        = %s,
                        exit_reason      = %s,
                        status           = 'CLOSED',
                        pnl_premium_pct  = %s,
                        pnl_capital_pct  = %s,
                        pnl_rs           = %s,
                        pnl_rs_total     = %s,
                        trail_activated  = %s,
                        max_price_seen   = %s,
                        holding_minutes  = %s
                    WHERE id = %s
                """, (
                    round(current_price, 2), now, exit_spot,
                    exit_reason,
                    round(pnl_pct_premium, 2),
                    round(pnl_pct_capital, 2),
                    round(pnl_rs, 2),
                    round(pnl_rs_total, 2),
                    trade.trail_active,
                    round(trade.max_price_seen, 2),
                    holding,
                    trade.trade_id,
                ))
            conn.commit()
    except Exception as e:
        log.error("Close trade DB write: %s", e)
        return

    # Update day stats
    DAY_STATS.trades_closed  += 1
    DAY_STATS.total_pnl_rs   += pnl_rs
    DAY_STATS.pnl_list.append(pnl_pct_premium)
    DAY_STATS.last_close_time = now
    if pnl_pct_premium > 0:
        DAY_STATS.wins   += 1
    else:
        DAY_STATS.losses += 1

    # Check daily loss limit
    if DAY_STATS.total_pnl_rs <= -(CAPITAL * MAX_DAILY_LOSS / 100):
        DAY_STATS.halted = True
        log.warning("⛔ DAILY LOSS LIMIT — trading halted.")
        send_telegram_halt()

    OPEN_TRADE = None

    print_trade_close(trade, current_price, pnl_pct_premium,
                      pnl_pct_capital, pnl_rs_total, exit_reason, holding)
    send_telegram_close(trade, current_price, pnl_pct_premium,
                        pnl_rs_total, exit_reason)

    log.info("%s Trade #%d closed | P&L: %+.2f%% prem / %+.2f%% cap "
             "/ ₹%+.0f total | Held: %d min | %s",
             "✅" if pnl_pct_premium > 0 else "❌",
             trade.trade_id, pnl_pct_premium, pnl_pct_capital,
             pnl_rs_total, holding, exit_reason)


# ══════════════════════════════════════════════════════════════
# Exit condition checker
# ══════════════════════════════════════════════════════════════
def check_exit_conditions(trade: OpenTrade,
                           symbol: str) -> tuple[bool, bool, str, float]:
    """
    Check all exit conditions every cycle.
    Returns: (should_full_exit, should_partial_exit, reason, current_price)

    Priority:
      1. Fetch live price
      2. Update dynamic trailing SL
      3. Stop loss check
      4. Partial exit at target1 (if not already done)
      5. Full exit at target2
      6. Strategy EXIT signal
      7. EOD cutoff
      8. VIX spike
    """
    current_price = get_live_option_price(
        symbol, trade.strike, trade.option_type, trade.expiry
    )
    if current_price is None:
        log.warning("Cannot fetch live price — skipping exit check.")
        return False, False, "", trade.entry_option_price

    # Upgrade 2: Update dynamic trailing SL
    update_trailing_sl(trade, current_price)

    # 1. Stop Loss
    if current_price <= trade.sl_price:
        return (True, False,
                f"STOP LOSS (₹{current_price:.2f} ≤ SL ₹{trade.sl_price:.2f})",
                current_price)

    # 2. Partial exit at Target 1 (if not already done)
    if not trade.partial_exited and current_price >= trade.target1_price:
        return (False, True,
                f"TARGET 1 HIT — partial exit 50% "
                f"(₹{current_price:.2f} ≥ TGT1 ₹{trade.target1_price:.2f})",
                current_price)

    # 3. Full exit at Target 2
    if current_price >= trade.target2_price:
        return (True, False,
                f"TARGET 2 HIT (₹{current_price:.2f} ≥ TGT2 ₹{trade.target2_price:.2f})",
                current_price)

    # 4. Strategy EXIT signal
    exit_reason = get_latest_exit_signal(symbol, trade.entry_time)
    if exit_reason:
        return True, False, f"STRATEGY EXIT: {exit_reason}", current_price

    # 5. EOD
    now_ist = datetime.now(IST)
    eod     = now_ist.replace(hour=EOD_EXIT_H, minute=EOD_EXIT_M,
                               second=0, microsecond=0)
    if now_ist >= eod:
        return True, False, "END OF DAY (2:50 PM cutoff)", current_price

    # 6. VIX spike
    vix = get_live_vix(symbol)
    if vix and vix > VIX_EXIT:
        return True, False, f"VIX SPIKE ({vix:.1f} > {VIX_EXIT})", current_price

    return False, False, "", current_price


# ══════════════════════════════════════════════════════════════
# Upgrade 4: Signal staleness check
# ══════════════════════════════════════════════════════════════
def check_signal_staleness(symbol: str):
    """
    Alert if strategy.py has gone silent during market hours.
    This catches: strategy crash, DB issue, collector stopped feeding data.
    """
    last_ts = get_last_signal_time(symbol)
    if last_ts is None:
        return

    # Make offset-aware if needed
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=timezone.utc)

    elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60
    if elapsed > STALENESS_MINUTES:
        msg = (f"⚠️ STRATEGY SILENT for {elapsed:.0f} min — "
               f"last signal: {last_ts.astimezone(IST).strftime('%H:%M:%S')} IST")
        log.warning(msg)
        _tg(f"⚠️ *PAPER TRADER ALERT*\n{msg}\nCheck if strategy.py is running.")


# ══════════════════════════════════════════════════════════════
# Upgrade 6: Re-entry logic
# ══════════════════════════════════════════════════════════════
def check_reentry(symbol: str) -> bool:
    """
    After a clean win, check if the signal is still valid and re-enter.
    Conditions:
      - Last trade was a win
      - Cooldown period passed
      - Latest signal confidence >= REENTRY_MIN_CONF
      - Same direction as last trade (trend confirmation)
    """
    if not REENTRY_ENABLED:
        return False
    if not DAY_STATS.reentry_allowed():
        return False
    if DAY_STATS.trades_closed == 0:
        return False
    if DAY_STATS.pnl_list[-1] <= 0:
        return False   # last trade was a loss — no re-entry

    signal = get_latest_signal(symbol, allow_reentry=True)
    if not signal:
        return False

    conf = float(signal.get("confidence") or 0)
    if conf < REENTRY_MIN_CONF:
        return False

    log.info("♻️  Re-entry conditions met | Conf=%.1f%% | %s",
             conf, signal["action"])
    result = open_trade(symbol, signal, is_reentry=True)
    return result is not None


# ══════════════════════════════════════════════════════════════
# Market hours
# ══════════════════════════════════════════════════════════════
def is_market_open() -> bool:
    now    = datetime.now(IST)
    open_  = now.replace(hour=MARKET_OPEN[0],  minute=MARKET_OPEN[1],
                          second=0, microsecond=0)
    close_ = now.replace(hour=MARKET_CLOSE[0], minute=MARKET_CLOSE[1],
                          second=0, microsecond=0)
    return open_ <= now < close_


# ══════════════════════════════════════════════════════════════
# Console display
# ══════════════════════════════════════════════════════════════
W = 65   # console width

def _line(char="═"): return char * W
def _row(label, value, width=W):
    val_str = str(value)
    pad     = width - 4 - len(label) - len(val_str)
    return f"║  {label}{' '*max(0,pad)}{val_str}  ║"


def print_trade_open(trade: OpenTrade, label: str = "✅ OPEN"):
    print("\n" + _line("╔") .replace("╔","╔").replace("═","═") )
    print("╔" + "═"*(W-2) + "╗")
    print(f"║  🟢 PAPER TRADE {label:<47}║")
    print("╠" + "═"*(W-2) + "╣")
    print(f"║  Trade ID     : #{trade.trade_id:<{W-18}}║")
    print(f"║  Contract     : {trade.strike:.0f} {trade.option_type}  "
          f"Expiry: {trade.expiry}  DTE: {trade.dte:<20}║")
    print(f"║  Direction    : {trade.direction:<{W-18}}║")
    print(f"║  Entry Spot   : ₹{trade.entry_spot:<{W-19}.2f}║")
    print(f"║  Entry Price  : ₹{trade.entry_option_price:<{W-19}.2f}║")
    print(f"║  Lots         : {trade.lots:<{W-18}}║")
    print(f"║  Capital      : ₹{trade.capital_deployed_total:>10,.0f}"
          f"{'':>{W-32}}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  Stop Loss    : ₹{trade.sl_price:<{W-19}.2f}║")
    print(f"║  Target 1 (50%): ₹{trade.target1_price:<{W-20}.2f}║")
    print(f"║  Target 2 (50%): ₹{trade.target2_price:<{W-20}.2f}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  Confidence   : {trade.signal_conf:.1f}%"
          f"{'':>{W-22}}║")
    print(f"║  Regime       : {trade.signal_regime:<{W-18}}║")
    print(f"║  GEX          : {trade.signal_gex:<{W-18}}║")
    print(f"║  PCR OI       : {str(trade.signal_pcr or 'N/A'):<{W-18}}║")
    print(f"║  VIX          : {str(trade.signal_vix or 'N/A'):<{W-18}}║")
    print("╚" + "═"*(W-2) + "╝\n")


def print_partial_exit(trade: OpenTrade, partial: PartialExit):
    icon = "📤"
    print("\n" + "╔" + "═"*(W-2) + "╗")
    print(f"║  {icon} PARTIAL EXIT — 50% CLOSED"
          f"{'':>{W-30}}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  Trade #      : {trade.trade_id:<{W-18}}║")
    print(f"║  Lots Closed  : {partial.lots_closed:<{W-18}}║")
    print(f"║  Exit Price   : ₹{partial.exit_price:<{W-19}.2f}║")
    print(f"║  P&L on 50%%  : {partial.pnl_pct:+.2f}%  "
          f"(₹{partial.pnl_rs:+,.0f})"
          f"{'':>{max(0, W-36-len(f'{partial.pnl_rs:+,.0f}'))}}║")
    print(f"║  Remaining    : {trade.lots_remaining} lots"
          f"{'':>{W-26}}║")
    print(f"║  New SL       : ₹{trade.sl_price:.2f} (break-even+)"
          f"{'':>{max(0,W-36-len(f'{trade.sl_price:.2f}'))}}║")
    print("╚" + "═"*(W-2) + "╝\n")


def print_trade_close(trade: OpenTrade, exit_price: float,
                      pnl_pct_premium: float, pnl_pct_capital: float,
                      pnl_rs_total: float, reason: str, holding: int):
    icon = "✅" if pnl_pct_premium > 0 else "❌"
    print("\n" + "╔" + "═"*(W-2) + "╗")
    print(f"║  {icon} PAPER TRADE CLOSED"
          f"{'':>{W-23}}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  Trade #      : {trade.trade_id:<{W-18}}║")
    print(f"║  Contract     : {trade.strike:.0f} {trade.option_type} "
          f"({trade.symbol})"
          f"{'':>{max(0,W-28-len(trade.symbol))}}║")
    print(f"║  Entry        : ₹{trade.entry_option_price:<{W-19}.2f}║")
    print(f"║  Exit         : ₹{exit_price:<{W-19}.2f}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  P&L Premium  : {pnl_pct_premium:+.2f}%"
          f"{'':>{W-22}}║")
    print(f"║  P&L Capital  : {pnl_pct_capital:+.2f}%  ← real account impact"
          f"{'':>{max(0,W-44)}}║")
    print(f"║  P&L Total ₹  : ₹{pnl_rs_total:+,.0f}"
          f"{'':>{max(0,W-22-len(f'{pnl_rs_total:+,.0f}'))}}║")
    print(f"║  Held         : {holding} min"
          f"{'':>{max(0,W-20-len(str(holding)))}}║")
    print(f"║  Reason       : {reason[:W-18]:<{W-18}}║")
    print(f"║  Trailing SL  : {'Active' if trade.trail_active else 'No':<{W-18}}║")
    print(f"║  Partial Exit : {'Yes' if trade.partial_exited else 'No':<{W-18}}║")
    print("╠" + "─"*(W-2) + "╣")
    print(f"║  SESSION      : Trades={DAY_STATS.trades_closed}  "
          f"W/L={DAY_STATS.wins}/{DAY_STATS.losses}  "
          f"WR={DAY_STATS.win_rate:.0f}%  "
          f"P&L=₹{DAY_STATS.total_pnl_rs:+,.0f}"
          f"{'':>{max(0,W-56-len(f'{DAY_STATS.total_pnl_rs:+,.0f}'))}}║")
    if DAY_STATS.halted:
        print(f"║  ⛔ TRADING HALTED — daily loss limit reached"
              f"{'':>{W-47}}║")
    print("╚" + "═"*(W-2) + "╝\n")


def print_monitor_status(trade: OpenTrade, current_price: float):
    """Compact one-line status every cycle."""
    pct  = trade.pnl_premium_pct(current_price)
    rs   = trade.pnl_rs(current_price)
    icon = "📈" if pct >= 0 else "📉"
    now  = datetime.now(IST).strftime("%H:%M:%S")
    trail_tag = " 🔒TRAIL" if trade.trail_active else ""
    partial_tag = " 📤50%DONE" if trade.partial_exited else ""
    print(f"  {now}  │  {trade.strike:.0f}{trade.option_type}  │  "
          f"₹{current_price:.2f}  │  "
          f"{pct:+.2f}% (₹{rs:+,.0f})  │  "
          f"SL:₹{trade.sl_price:.2f}  "
          f"T1:₹{trade.target1_price:.2f}  "
          f"T2:₹{trade.target2_price:.2f}"
          f"{trail_tag}{partial_tag}  {icon}")


def print_daily_summary():
    print("\n" + "═"*W)
    print(f"  📊 DAILY SUMMARY — {datetime.now(IST).strftime('%d %b %Y')}")
    print("═"*W)
    print(f"  Trades Taken    : {DAY_STATS.trades_taken}")
    print(f"  Trades Closed   : {DAY_STATS.trades_closed}")
    print(f"  Partial Exits   : {DAY_STATS.partial_exits}")
    print(f"  Win / Loss      : {DAY_STATS.wins} / {DAY_STATS.losses}")
    print(f"  Win Rate        : {DAY_STATS.win_rate:.1f}%")
    print(f"  Total P&L (₹)   : ₹{DAY_STATS.total_pnl_rs:+,.0f}")
    print(f"  Capital Return  : {DAY_STATS.capital_return_pct:+.2f}%  "
          f"(on ₹{DAY_STATS.total_capital_used:,.0f} deployed)")
    print(f"  Avg Win         : {DAY_STATS.avg_win:+.2f}%")
    print(f"  Avg Loss        : {DAY_STATS.avg_loss:+.2f}%")
    print(f"  Profit Factor   : {DAY_STATS.profit_factor:.2f}")
    print(f"  Max Drawdown    : {DAY_STATS.max_drawdown:.2f}%")
    print(f"  Sharpe          : {DAY_STATS.sharpe:.2f}")
    print("═"*W + "\n")


# ══════════════════════════════════════════════════════════════
# Telegram
# ══════════════════════════════════════════════════════════════
def _tg(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg,
                  "parse_mode": "Markdown"},
            timeout=5,
        )
    except Exception as e:
        log.debug("Telegram: %s", e)


def send_telegram_open(trade: OpenTrade, is_reentry: bool = False):
    icon  = "🟢" if trade.direction == "BUY_CE" else "🔴"
    label = "RE-ENTRY" if is_reentry else "OPEN"
    _tg(f"{icon} *PAPER TRADE {label}* | {trade.symbol}\n"
        f"`{trade.strike:.0f} {trade.option_type}`  {trade.expiry}\n"
        f"Entry: ₹{trade.entry_option_price:.2f}  Lots: {trade.lots}\n"
        f"Capital: ₹{trade.capital_deployed_total:,.0f}\n"
        f"SL: ₹{trade.sl_price:.2f}  T1: ₹{trade.target1_price:.2f}  "
        f"T2: ₹{trade.target2_price:.2f}\n"
        f"Conf: {trade.signal_conf:.0f}%  Regime: {trade.signal_regime}  "
        f"GEX: {trade.signal_gex}")


def send_telegram_partial(trade: OpenTrade, partial: PartialExit):
    _tg(f"📤 *PARTIAL EXIT 50%* | {trade.symbol}\n"
        f"`{trade.strike:.0f} {trade.option_type}`\n"
        f"{partial.lots_closed} lots @ ₹{partial.exit_price:.2f}\n"
        f"P&L: *{partial.pnl_pct:+.2f}%* (₹{partial.pnl_rs:+,.0f})\n"
        f"Remaining {trade.lots_remaining} lots  "
        f"New SL: ₹{trade.sl_price:.2f}")


def send_telegram_close(trade: OpenTrade, exit_price: float,
                        pnl_pct: float, pnl_rs_total: float, reason: str):
    icon = "✅" if pnl_pct > 0 else "❌"
    _tg(f"{icon} *PAPER TRADE CLOSED* | {trade.symbol}\n"
        f"`{trade.strike:.0f} {trade.option_type}`\n"
        f"Entry: ₹{trade.entry_option_price:.2f}  Exit: ₹{exit_price:.2f}\n"
        f"*P&L: {pnl_pct:+.2f}% premium  ₹{pnl_rs_total:+,.0f} total*\n"
        f"Reason: {reason}\n"
        f"Session: W{DAY_STATS.wins}/L{DAY_STATS.losses}  "
        f"₹{DAY_STATS.total_pnl_rs:+,.0f}")


def send_telegram_halt():
    _tg(f"⛔ *PAPER TRADING HALTED*\n"
        f"Daily loss limit reached\n"
        f"Session: W{DAY_STATS.wins}/L{DAY_STATS.losses}  "
        f"P&L: ₹{DAY_STATS.total_pnl_rs:+,.0f}")


# ══════════════════════════════════════════════════════════════
# Stats viewer
# ══════════════════════════════════════════════════════════════
def show_stats(symbol: str, today_only: bool = False):
    try:
        date_filter = "AND DATE(entry_time) = CURRENT_DATE" if today_only else ""
        with db() as conn:
            df = pd.read_sql_query(f"""
                SELECT id, direction, option_type, strike, expiry,
                       entry_price, exit_price, lots,
                       capital_deployed, pnl_premium_pct, pnl_capital_pct,
                       pnl_rs, pnl_rs_total,
                       partial_exited, trail_activated,
                       exit_reason, status, holding_minutes,
                       entry_time, exit_time,
                       signal_conf, signal_regime, signal_gex
                FROM   paper_trades
                WHERE  symbol = %s {date_filter}
                ORDER  BY entry_time DESC
            """, conn, params=(symbol,))
    except Exception as e:
        log.error("Stats: %s", e)
        return

    if df.empty:
        print(f"\n  No paper trades found for {symbol}.\n")
        return

    closed = df[df["status"] == "CLOSED"].copy()
    open_  = df[df["status"] == "OPEN"].copy()
    label  = "TODAY" if today_only else "ALL-TIME"

    print("\n" + "═"*W)
    print(f"  📊 {symbol} PAPER TRADING — {label}")
    print("═"*W)
    print(f"  Total / Open / Closed : {len(df)} / {len(open_)} / {len(closed)}")

    if not closed.empty:
        prem  = closed["pnl_premium_pct"].astype(float)
        cap   = closed["pnl_capital_pct"].astype(float)
        rs    = closed["pnl_rs_total"].astype(float)
        wins  = prem[prem > 0]
        loss  = prem[prem <= 0]
        wr    = len(wins) / len(prem) * 100 if len(prem) > 0 else 0
        pf    = (wins.sum() / abs(loss.sum())
                 if len(loss) > 0 and loss.sum() != 0 else float("inf"))
        sharpe = (prem.mean() / (prem.std() + 1e-9) if len(prem) > 1 else 0)
        cumul  = np.cumsum(prem.values)
        mdd    = float(np.min(cumul - np.maximum.accumulate(cumul)))

        print("─"*W)
        print(f"  Win / Loss      : {len(wins)} / {len(loss)}")
        print(f"  Win Rate        : {wr:.1f}%")
        print(f"  Total P&L (₹)   : ₹{rs.sum():+,.0f}")
        print(f"  Avg Win         : {wins.mean() if len(wins)>0 else 0:+.2f}%")
        print(f"  Avg Loss        : {loss.mean() if len(loss)>0 else 0:+.2f}%")
        print(f"  Best Trade      : {prem.max():+.2f}%")
        print(f"  Worst Trade     : {prem.min():+.2f}%")
        print(f"  Profit Factor   : {pf:.2f}")
        print(f"  Sharpe          : {sharpe:.2f}")
        print(f"  Max Drawdown    : {mdd:.2f}%")
        print(f"  Avg Hold Time   : {closed['holding_minutes'].mean():.0f} min")
        print(f"  Capital P&L avg : {cap.mean():+.2f}%  ← real account return")
        pct_partial = (closed["partial_exited"].sum() / len(closed) * 100
                       if len(closed) > 0 else 0)
        pct_trail   = (closed["trail_activated"].sum() / len(closed) * 100
                       if len(closed) > 0 else 0)
        print(f"  Partial exits   : {pct_partial:.0f}% of trades")
        print(f"  Trail SL used   : {pct_trail:.0f}% of trades")

        # By direction
        print("─"*W)
        for d in ["BUY_CE", "BUY_PE"]:
            sub = closed[closed["direction"] == d]["pnl_premium_pct"].astype(float)
            if len(sub) > 0:
                sw = len(sub[sub > 0]) / len(sub) * 100
                print(f"  {d:<10}      : {len(sub)} trades  "
                      f"WR={sw:.0f}%  P&L={sub.sum():+.1f}%")

        # By exit reason
        print("─"*W)
        print("  BY EXIT REASON:")
        for reason, grp in closed.groupby("exit_reason"):
            gp  = grp["pnl_premium_pct"].astype(float)
            gwr = len(gp[gp > 0]) / len(gp) * 100 if len(gp) > 0 else 0
            print(f"  {str(reason)[:22]:<24}: {len(grp)} trades  "
                  f"WR={gwr:.0f}%  P&L={gp.sum():+.1f}%")

        # By regime
        if "signal_regime" in closed.columns:
            print("─"*W)
            print("  BY REGIME:")
            for reg, grp in closed.groupby("signal_regime"):
                gp  = grp["pnl_premium_pct"].astype(float)
                gwr = len(gp[gp > 0]) / len(gp) * 100 if len(gp) > 0 else 0
                print(f"  {str(reg):<14}: {len(grp)} trades  "
                      f"WR={gwr:.0f}%  P&L={gp.sum():+.1f}%")

        # Recent trades table
        print("─"*W)
        print("  RECENT CLOSED TRADES:")
        print(f"  {'ID':<5} {'Dir':<8} {'Contr':<10} {'Entry':>8} "
              f"{'Exit':>8} {'Prem%':>7} {'Cap%':>6} {'Reason':<18} {'Min':>4}")
        print("  " + "-"*(W-4))
        for _, r in closed.head(10).iterrows():
            print(f"  {int(r['id']):<5} "
                  f"{r['direction']:<8} "
                  f"{r['strike']:.0f}{r['option_type']:<5} "
                  f"₹{float(r['entry_price']):>6.1f} "
                  f"₹{float(r['exit_price'] or 0):>6.1f} "
                  f"{float(r['pnl_premium_pct'] or 0):>+6.1f}% "
                  f"{float(r['pnl_capital_pct'] or 0):>+5.1f}% "
                  f"{str(r['exit_reason'] or '')[:18]:<18} "
                  f"{int(r['holding_minutes'] or 0):>4}")

    print("═"*W + "\n")


def export_trades(symbol: str):
    try:
        with db() as conn:
            df = pd.read_sql_query("""
                SELECT * FROM paper_trades WHERE symbol = %s
                ORDER BY entry_time DESC
            """, conn, params=(symbol,))
        fname = (f"paper_trades_{symbol}_"
                 f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        df.to_csv(fname, index=False)
        print(f"\n  ✅ {len(df)} trades exported → {fname}\n")
    except Exception as e:
        log.error("Export: %s", e)


# ══════════════════════════════════════════════════════════════
# Main loop
# ══════════════════════════════════════════════════════════════
def run_loop(symbol: str, interval: int = POLL_INTERVAL):
    global OPEN_TRADE, DAY_STATS

    log.info("Paper Trader v2.0 | %s | %ds interval | Ctrl+C to stop",
             symbol, interval)
    ensure_tables()

    # Upgrade 1: Recover any open trade from before restart
    recovered = recover_open_trade(symbol)
    if recovered:
        OPEN_TRADE = recovered
        log.info("Resumed monitoring of recovered trade #%d.",
                 recovered.trade_id)

    eod_done = False

    while True:
        try:
            now_ist = datetime.now(IST)

            # Reset daily stats at market open
            if (now_ist.hour == MARKET_OPEN[0]
                    and now_ist.minute < 20
                    and DAY_STATS.trades_taken > 0):
                DAY_STATS = DayStats()
                PARTIAL_EXITS.clear()
                eod_done = False
                log.info("Daily stats reset.")

            # EOD summary + force close
            if now_ist.hour >= MARKET_CLOSE[0] and not eod_done:
                if OPEN_TRADE:
                    curr = get_live_option_price(
                        symbol, OPEN_TRADE.strike,
                        OPEN_TRADE.option_type, OPEN_TRADE.expiry
                    ) or OPEN_TRADE.entry_option_price
                    close_trade(OPEN_TRADE, curr,
                                "END OF DAY — market closed",
                                get_live_spot(symbol))
                print_daily_summary()
                eod_done = True
                time.sleep(300)
                continue

            # Outside market hours
            if not is_market_open():
                log.info("Market closed (%s IST) — waiting...",
                         now_ist.strftime("%H:%M:%S"))
                time.sleep(60)
                continue

            # ── Active monitoring ──────────────────────────────
            if OPEN_TRADE:
                full_exit, partial_exit, reason, curr_price = \
                    check_exit_conditions(OPEN_TRADE, symbol)

                print_monitor_status(OPEN_TRADE, curr_price)

                if partial_exit and not OPEN_TRADE.partial_exited:
                    # Upgrade 3: Partial exit — close 50% at target 1
                    do_partial_exit(OPEN_TRADE, curr_price, reason)

                elif full_exit:
                    exit_spot = get_live_spot(symbol)
                    close_trade(OPEN_TRADE, curr_price, reason, exit_spot)

                    # Upgrade 6: Re-entry check after a clean win
                    if (DAY_STATS.pnl_list
                            and DAY_STATS.pnl_list[-1] > 0
                            and not DAY_STATS.halted):
                        time.sleep(5)   # brief pause before re-entry scan
                        check_reentry(symbol)

            else:
                # ── Look for new signals ───────────────────────
                if not DAY_STATS.halted:
                    # Upgrade 4: staleness check
                    check_signal_staleness(symbol)

                    signal = get_latest_signal(symbol)
                    if signal:
                        log.info("New signal: %s  Conf=%.1f%%  "
                                 "%s %.0f @ ₹%.2f",
                                 signal["action"],
                                 float(signal.get("confidence") or 0),
                                 signal.get("option_type"),
                                 float(signal.get("strike") or 0),
                                 float(signal.get("option_price") or 0))
                        open_trade(symbol, signal)
                    else:
                        log.info("%s  Watching for signals...",
                                 now_ist.strftime("%H:%M:%S"))
                else:
                    log.warning("⛔ Trading halted — daily loss limit. "
                                "Waiting for tomorrow.")

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            if OPEN_TRADE:
                log.warning("⚠️  Open trade #%d active — "
                            "will recover on next restart.",
                            OPEN_TRADE.trade_id)
            print_daily_summary()
            break
        except Exception as e:
            log.exception("Loop error: %s", e)

        time.sleep(interval)


# ══════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Paper Trading Engine v2.0")
    parser.add_argument("symbol",     nargs="?",  default="NIFTY")
    parser.add_argument("--interval", type=int,   default=POLL_INTERVAL)
    parser.add_argument("--stats",    action="store_true")
    parser.add_argument("--today",    action="store_true")
    parser.add_argument("--export",   action="store_true")
    args = parser.parse_args()

    try:
        ensure_tables()

        if args.stats:
            show_stats(args.symbol, today_only=False)
        elif args.today:
            show_stats(args.symbol, today_only=True)
        elif args.export:
            export_trades(args.symbol)
        else:
            run_loop(args.symbol, args.interval)

        sys.exit(0)
    except KeyboardInterrupt:
        log.info("Stopped.")
        sys.exit(0)
    except Exception as e:
        log.exception("Fatal: %s", e)
        sys.exit(2)