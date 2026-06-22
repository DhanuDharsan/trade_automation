"""
strategy.py  v4.0  INSTITUTIONAL-GRADE OPTIONS STRATEGY ENGINE
══════════════════════════════════════════════════════════════
15 Confluence Factors | GEX | FII/DII | Regime Detection |
Kelly Sizing | Drawdown Protection | Backtesting | Telegram

Data Sources:
  market_ticks              → EMA/RSI/MACD/SuperTrend/BB/VWAP
  nse_option_chain          → Greeks/GEX/OI walls/contract
  nse_option_chain_metadata → PCR OI+Vol/Total OI/VIX/MaxPain

Usage:
  python strategy.py NIFTY --loop 60         # live
  python strategy.py NIFTY --paper --loop 60 # paper trade
  python strategy.py NIFTY --backtest        # backtest
  python strategy.py NIFTY --stats           # performance
"""
from __future__ import annotations
import json, logging, sys, time, argparse, math, requests
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Generator, Optional
import numpy as np
import pandas as pd
import psycopg2

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("strategy")

# ── Config ────────────────────────────────────────────────────
DB_CONFIG = {
    "host":     "ep-morning-lake-aoiahvcv-pooler.c-2.ap-southeast-1.aws.neon.tech",
    "port":      5432,
    "dbname":   "Trade",
    "user":     "neondb_owner",
    "password": "npg_YCL6POljqg9i",
}
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID   = ""

LOOKBACK       = 200
EMA_FAST       = 9
EMA_SLOW       = 21
EMA_MEDIUM     = 50
EMA_LONG       = 200
ST_ATR         = 10
ST_MULT        = 3.0
RSI_PERIOD     = 14
RSI_OB         = 70.0
RSI_OS         = 30.0
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIG       = 9
BB_PERIOD      = 20
BB_STD         = 2.0
PCR_BULL       = 0.7
PCR_BEAR       = 1.3
VIX_LOW        = 12.0
VIX_MED        = 15.0
VIX_HIGH       = 20.0
BASE_BUY       = 0.58
BASE_SELL      = -0.58
DELTA_MIN      = 0.30
DELTA_MAX      = 0.70
THETA_MAX      = -15.0
IV_RANK_MAX    = 70.0
SPREAD_MAX     = 2.0
DTE_MIN        = 3
SL_PCT         = 15.0
TGT_PCT        = 30.0
MAX_LOSSES     = 3
RISK_PCT       = 2.0
CAPITAL        = 500000
LOT_SIZE       = 75

class Regime:
    BULL="BULL"; BEAR="BEAR"; SIDEWAYS="SIDEWAYS"; VOLATILE="VOLATILE"

@dataclass
class Votes:
    ema_trend:int=0; supertrend:int=0; vwap:int=0
    macd:int=0; rsi:int=0; bb:int=0
    pcr_oi:int=0; pcr_vol:int=0; oi_ratio:int=0
    oi_wall:int=0; vix:int=0; max_pain:int=0
    gex:int=0; regime:int=0; fii_dii:int=0

    WEIGHTS:dict = field(default_factory=lambda:{
        "ema_trend":2,"supertrend":2,"vwap":2,
        "macd":1,"rsi":1,"bb":1,
        "pcr_oi":2,"pcr_vol":1,"oi_ratio":1,
        "oi_wall":2,"vix":2,"max_pain":1,
        "gex":2,"regime":2,"fii_dii":2,
    })
    def score(self)->float:
        v=vars(self); w=self.WEIGHTS
        return sum(v[k]*w[k] for k in w)/sum(w.values())
    def to_dict(self)->dict:
        v=vars(self); return {k:v[k] for k in self.WEIGHTS}

@dataclass
class GEXResult:
    total_gex:float; gex_flip:Optional[float]
    key_levels:list; direction:str

@dataclass
class OptionTrade:
    symbol:str; option_type:str; strike:float; expiry:str
    dte:int; last_price:float; delta:float; theta:float
    vega:float; iv:float; iv_rank:Optional[float]
    moneyness:str; recommended_lots:int=1

@dataclass
class TradeState:
    in_trade:bool=False; direction:str=""
    entry_price:float=0.0; option_strike:float=0.0
    option_type:str=""; option_expiry:str=""
    option_entry:float=0.0; entry_time:str=""; lots:int=1

@dataclass
class PerfStats:
    total:int=0; buys:int=0; holds:int=0; exits:int=0
    wins:int=0; losses:int=0; total_pnl:float=0.0
    avg_win:float=0.0; avg_loss:float=0.0
    win_rate:float=0.0; sharpe:float=0.0
    consec_losses:int=0; halted:bool=False

@dataclass
class SignalResult:
    symbol:str; action:str; confidence:float; price:float
    regime:str; implied_move:Optional[float]; gex:Optional[GEXResult]
    ema_fast:float; ema_slow:float; ema_medium:float; ema_long:float
    rsi:float; macd_hist:float; supertrend_bull:bool; vwap:Optional[float]
    pcr_oi:Optional[float]; pcr_vol:Optional[float]
    total_call_oi:Optional[float]; total_put_oi:Optional[float]
    total_call_vol:Optional[float]; total_put_vol:Optional[float]
    india_vix:Optional[float]; max_pain:Optional[float]; atm_strike:Optional[float]
    fii_buy:Optional[float]; fii_sell:Optional[float]
    dii_buy:Optional[float]; dii_sell:Optional[float]
    votes:dict; trade:Optional[OptionTrade]
    timestamp:datetime; reason:str
    exit_reason:Optional[str]=None; pnl_pct:Optional[float]=None
    paper_mode:bool=False

CURRENT_TRADE = TradeState()

# ── DB ────────────────────────────────────────────────────────
@contextmanager
def db()->Generator:
    conn=None
    try:
        conn=psycopg2.connect(**DB_CONFIG); yield conn
    except psycopg2.OperationalError as e:
        log.error("DB failed: %s",e); raise
    finally:
        if conn and not conn.closed: conn.close()

def ensure_signals_table():
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS signals(
                id BIGSERIAL PRIMARY KEY,
                symbol TEXT NOT NULL, action TEXT NOT NULL,
                confidence NUMERIC, price NUMERIC NOT NULL,
                regime TEXT, implied_move NUMERIC, paper_mode BOOLEAN DEFAULT FALSE,
                ema_fast NUMERIC, ema_slow NUMERIC, ema_medium NUMERIC, ema_long NUMERIC,
                rsi NUMERIC, macd_histogram NUMERIC, supertrend_bull BOOLEAN, vwap NUMERIC,
                pcr_oi NUMERIC, pcr_vol NUMERIC,
                total_call_oi BIGINT, total_put_oi BIGINT,
                total_call_vol BIGINT, total_put_vol BIGINT,
                india_vix NUMERIC, max_pain NUMERIC, atm_strike NUMERIC,
                gex_total NUMERIC, gex_flip NUMERIC, gex_direction TEXT,
                fii_buy NUMERIC, fii_sell NUMERIC, dii_buy NUMERIC, dii_sell NUMERIC,
                option_type TEXT, strike NUMERIC, expiry TEXT, dte INTEGER,
                option_price NUMERIC, delta NUMERIC, theta NUMERIC,
                vega NUMERIC, iv NUMERIC, iv_rank NUMERIC,
                moneyness TEXT, recommended_lots INTEGER,
                exit_reason TEXT, pnl_pct NUMERIC,
                votes JSONB, reason TEXT,
                timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_sig_sym_ts ON signals(symbol,timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_sig_action ON signals(action);
            """)
        conn.commit()

# ── Data fetch ────────────────────────────────────────────────
def get_market_data(symbol:str)->pd.DataFrame:
    sql="""SELECT id,symbol,price::FLOAT,
           COALESCE(volume::FLOAT,NULL) AS volume,
           COALESCE(india_vix::FLOAT,NULL) AS india_vix,
           timestamp
           FROM market_ticks WHERE symbol=%s
           ORDER BY timestamp DESC LIMIT %s"""
    with db() as conn:
        df=pd.read_sql_query(sql,conn,params=(symbol,LOOKBACK))
    if df.empty: raise ValueError(f"No ticks for '{symbol}'")
    if len(df)<50: raise ValueError(f"Only {len(df)} ticks — need 50+")
    df=df.iloc[::-1].reset_index(drop=True)
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    log.info("Ticks=%d  price=%.2f",len(df),df["price"].iloc[-1])
    return df

def get_options_context(symbol:str)->dict:
    ctx={k:None for k in ["pcr_oi","pcr_vol","total_call_oi","total_put_oi",
         "total_call_vol","total_put_vol","india_vix","max_pain","atm_strike",
         "max_call_oi_strike","max_put_oi_strike"]}
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT pcr_oi,pcr_vol,total_call_oi,total_put_oi,
                    total_call_vol,total_put_vol,india_vix,max_pain,atm_strike
                    FROM nse_option_chain_metadata WHERE symbol=%s
                    ORDER BY fetched_at DESC LIMIT 1""",(symbol,))
                row=cur.fetchone()
                if row:
                    for i,k in enumerate(["pcr_oi","pcr_vol","total_call_oi",
                        "total_put_oi","total_call_vol","total_put_vol",
                        "india_vix","max_pain","atm_strike"]):
                        ctx[k]=float(row[i]) if row[i] is not None else None
            with conn.cursor() as cur:
                cur.execute("""SELECT option_type,strike_price,SUM(open_interest) AS oi
                    FROM nse_option_chain
                    WHERE symbol=%s AND fetched_at=(
                        SELECT MAX(fetched_at) FROM nse_option_chain WHERE symbol=%s)
                    AND dte BETWEEN 3 AND 30
                    GROUP BY option_type,strike_price ORDER BY oi DESC""",(symbol,symbol))
                for ot,strike,_ in cur.fetchall():
                    if ot=="CE" and not ctx["max_call_oi_strike"]:
                        ctx["max_call_oi_strike"]=float(strike)
                    if ot=="PE" and not ctx["max_put_oi_strike"]:
                        ctx["max_put_oi_strike"]=float(strike)
                    if ctx["max_call_oi_strike"] and ctx["max_put_oi_strike"]: break
    except Exception as e: log.warning("Options ctx: %s",e)
    return ctx

def get_best_option(symbol:str,direction:str,underlying:float)->Optional[OptionTrade]:
    opt_type="CE" if direction=="BUY_CE" else "PE"
    dmin=DELTA_MIN if opt_type=="CE" else -DELTA_MAX
    dmax=DELTA_MAX if opt_type=="CE" else -DELTA_MIN
    sql="""SELECT strike_price,expiry_date,dte,last_price,delta,theta,vega,
           implied_volatility,iv_rank,bid_ask_spread_pct,moneyness
           FROM nse_option_chain
           WHERE symbol=%s AND option_type=%s
           AND fetched_at=(SELECT MAX(fetched_at) FROM nse_option_chain WHERE symbol=%s)
           AND dte>=%s AND delta BETWEEN %s AND %s
           AND last_price>0 AND implied_volatility>0
           ORDER BY ABS(delta-0.5) ASC,dte ASC LIMIT 10"""
    try:
        with db() as conn:
            df=pd.read_sql_query(sql,conn,
               params=(symbol,opt_type,symbol,DTE_MIN,dmin,dmax))
        if df.empty: return None
        for _,row in df.iterrows():
            theta=float(row["theta"] or 0)
            ivr=float(row["iv_rank"]) if row["iv_rank"] else None
            sp=float(row["bid_ask_spread_pct"]) if row["bid_ask_spread_pct"] else None
            if theta<THETA_MAX: continue
            if ivr and ivr>IV_RANK_MAX: continue
            if sp and sp>SPREAD_MAX: continue
            return OptionTrade(symbol=symbol,option_type=opt_type,
                strike=float(row["strike_price"]),expiry=str(row["expiry_date"]),
                dte=int(row["dte"]),last_price=float(row["last_price"]),
                delta=float(row["delta"] or 0),theta=theta,
                vega=float(row["vega"] or 0),iv=float(row["implied_volatility"] or 0),
                iv_rank=ivr,moneyness=str(row["moneyness"] or ""))
    except Exception as e: log.warning("Option select: %s",e)
    return None

def get_current_option_price(symbol:str,strike:float,
                              opt_type:str,expiry:str)->Optional[float]:
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT last_price FROM nse_option_chain
                    WHERE symbol=%s AND strike_price=%s
                    AND option_type=%s AND expiry_date=%s::date
                    ORDER BY fetched_at DESC LIMIT 1""",
                    (symbol,strike,opt_type,expiry))
                row=cur.fetchone()
                return float(row[0]) if row and row[0] else None
    except: return None

# ── GEX ───────────────────────────────────────────────────────
def calculate_gex(symbol:str,underlying:float)->Optional[GEXResult]:
    try:
        with db() as conn:
            df=pd.read_sql_query("""
                SELECT strike_price,option_type,gamma,open_interest
                FROM nse_option_chain
                WHERE symbol=%s AND fetched_at=(
                    SELECT MAX(fetched_at) FROM nse_option_chain WHERE symbol=%s)
                AND dte BETWEEN 1 AND 30
                AND gamma IS NOT NULL AND open_interest>0
                ORDER BY strike_price""",conn,params=(symbol,symbol))
        if df.empty: return None
        gex_by_strike={}
        for _,row in df.iterrows():
            strike=float(row["strike_price"])
            gamma=float(row["gamma"] or 0)
            oi=float(row["open_interest"] or 0)
            sign=1 if row["option_type"]=="CE" else -1
            gex=sign*gamma*oi*(underlying**2)*0.01
            gex_by_strike[strike]=gex_by_strike.get(strike,0)+gex
        if not gex_by_strike: return None
        strikes=sorted(gex_by_strike.keys())
        gex_vals=[gex_by_strike[s] for s in strikes]
        total_gex=sum(gex_vals)
        gex_flip=None
        for i in range(len(strikes)-1):
            if gex_vals[i]*gex_vals[i+1]<0:
                gex_flip=strikes[i]+(strikes[i+1]-strikes[i])*\
                    abs(gex_vals[i])/(abs(gex_vals[i])+abs(gex_vals[i+1]))
                break
        sorted_abs=sorted(gex_by_strike.items(),key=lambda x:abs(x[1]),reverse=True)
        key_levels=[s for s,_ in sorted_abs[:3]]
        direction="BULLISH" if total_gex>0 else ("BEARISH" if total_gex<0 else "NEUTRAL")
        log.info("GEX: %.0f  flip=%.0f  %s",total_gex,gex_flip or 0,direction)
        return GEXResult(total_gex=round(total_gex,0),
            gex_flip=round(gex_flip,0) if gex_flip else None,
            key_levels=key_levels,direction=direction)
    except Exception as e:
        log.warning("GEX: %s",e); return None

# ── Regime ────────────────────────────────────────────────────
def detect_regime(df:pd.DataFrame,vix:Optional[float])->str:
    price=float(df["price"].iloc[-1])
    e50=float(df["price"].ewm(span=50,adjust=False).mean().iloc[-1])
    e200=float(df["price"].ewm(span=200,adjust=False).mean().iloc[-1])
    ret=df["price"].pct_change().dropna()
    dvol=float(ret.std()*math.sqrt(252)*100)
    if vix and vix>VIX_HIGH: return Regime.VOLATILE
    if dvol>25: return Regime.VOLATILE
    if price>e50>e200: return Regime.BULL
    if price<e50<e200: return Regime.BEAR
    return Regime.SIDEWAYS

def get_dynamic_threshold(regime:str,vix:Optional[float])->tuple:
    if regime==Regime.VOLATILE: return BASE_BUY+0.10,BASE_SELL-0.10
    if regime==Regime.BULL:     return BASE_BUY-0.05,BASE_SELL-0.05
    if regime==Regime.BEAR:     return BASE_BUY+0.05,BASE_SELL+0.05
    return BASE_BUY,BASE_SELL

# ── FII/DII ───────────────────────────────────────────────────
def fetch_fii_dii()->dict:
    result={"fii_buy":None,"fii_sell":None,"dii_buy":None,"dii_sell":None,
            "fii_net":None,"dii_net":None}
    try:
        s=requests.Session()
        s.headers.update({"User-Agent":"Mozilla/5.0 Chrome/124.0.0.0",
                           "Referer":"https://www.nseindia.com"})
        s.get("https://www.nseindia.com",timeout=5)
        r=s.get("https://www.nseindia.com/api/fiidiiTradeReact",timeout=8)
        for item in r.json():
            cat=item.get("category","").upper()
            if "FII" in cat or "FPI" in cat:
                result["fii_buy"]=float(item.get("buyValue",0) or 0)
                result["fii_sell"]=float(item.get("sellValue",0) or 0)
                result["fii_net"]=result["fii_buy"]-result["fii_sell"]
            elif "DII" in cat:
                result["dii_buy"]=float(item.get("buyValue",0) or 0)
                result["dii_sell"]=float(item.get("sellValue",0) or 0)
                result["dii_net"]=result["dii_buy"]-result["dii_sell"]
        if result["fii_net"] is not None:
            log.info("FII Net: %.0f Cr  DII Net: %.0f Cr",
                     result["fii_net"] or 0,result["dii_net"] or 0)
    except Exception as e: log.debug("FII/DII: %s",e)
    return result

# ── Implied move + Kelly ──────────────────────────────────────
def implied_move(atm_iv:Optional[float],dte:int=1)->Optional[float]:
    if not atm_iv or atm_iv<=0: return None
    return round(atm_iv/100*math.sqrt(dte/365),4)

def kelly_lots(win_rate:float,avg_win:float,avg_loss:float,
               capital:float,option_price:float)->int:
    try:
        if avg_loss==0 or win_rate<=0: return 1
        lr=1-win_rate
        r=abs(avg_win)/abs(avg_loss)
        k=max(0,(win_rate*r-lr)/r)*0.5
        max_risk=capital*(RISK_PCT/100)
        cost=option_price*LOT_SIZE
        if cost<=0: return 1
        return max(1,min(int(max_risk/cost*k),5))
    except: return 1

# ── Indicators ────────────────────────────────────────────────
def _ema(s:pd.Series,n:int)->pd.Series:
    return s.ewm(span=n,adjust=False).mean()

def calc_ema(df:pd.DataFrame)->dict:
    p=df["price"]
    e9=_ema(p,EMA_FAST); e21=_ema(p,EMA_SLOW)
    e50=_ema(p,EMA_MEDIUM); e200=_ema(p,EMA_LONG)
    df["ema9"]=e9; df["ema21"]=e21
    df["ema50"]=e50; df["ema200"]=e200
    return {"fast":round(float(e9.iloc[-1]),2),
            "slow":round(float(e21.iloc[-1]),2),
            "medium":round(float(e50.iloc[-1]),2),
            "long":round(float(e200.iloc[-1]),2)}

def calc_supertrend(df:pd.DataFrame)->dict:
    p=df["price"]; h=p*1.005; l=p*0.995; pv=p.shift(1)
    tr=pd.concat([h-l,(h-pv).abs(),(l-pv).abs()],axis=1).max(axis=1)
    atr=tr.ewm(span=ST_ATR,adjust=False).mean()
    ub=(h+l)/2+ST_MULT*atr; lb=(h+l)/2-ST_MULT*atr
    d=pd.Series(1,index=df.index)
    for i in range(1,len(df)):
        ub.iloc[i]=min(ub.iloc[i],ub.iloc[i-1]) if p.iloc[i-1]<=ub.iloc[i-1] else ub.iloc[i]
        lb.iloc[i]=max(lb.iloc[i],lb.iloc[i-1]) if p.iloc[i-1]>=lb.iloc[i-1] else lb.iloc[i]
        if d.iloc[i-1]==-1 and p.iloc[i]>ub.iloc[i]: d.iloc[i]=1
        elif d.iloc[i-1]==1 and p.iloc[i]<lb.iloc[i]: d.iloc[i]=-1
        else: d.iloc[i]=d.iloc[i-1]
    bull=bool(d.iloc[-1]==1)
    return {"bull":bull,"value":round(float(lb.iloc[-1] if bull else ub.iloc[-1]),2)}

def calc_rsi(df:pd.DataFrame)->float:
    delta=df["price"].diff()
    ag=delta.clip(lower=0).ewm(alpha=1/RSI_PERIOD,adjust=False).mean()
    al=(-delta).clip(lower=0).ewm(alpha=1/RSI_PERIOD,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return round(float((100-(100/(1+rs))).iloc[-1]),2)

def calc_macd(df:pd.DataFrame)->dict:
    fast=_ema(df["price"],MACD_FAST); slow=_ema(df["price"],MACD_SLOW)
    macd=fast-slow; sig=_ema(macd,MACD_SIG); hist=macd-sig
    return {"hist":round(float(hist.iloc[-1]),2),"prev":round(float(hist.iloc[-2]),2)}

def calc_bb(df:pd.DataFrame)->dict:
    sma=df["price"].rolling(BB_PERIOD).mean()
    std=df["price"].rolling(BB_PERIOD).std()
    upper=sma+BB_STD*std; lower=sma-BB_STD*std
    price=float(df["price"].iloc[-1])
    return {"above_mid":price>float(sma.iloc[-1]),
            "near_upper":price>=float(upper.iloc[-1])*0.995,
            "near_lower":price<=float(lower.iloc[-1])*1.005,
            "bandwidth":round(float((upper-lower).iloc[-1]/sma.iloc[-1]*100),2)}

def calc_vwap(df:pd.DataFrame)->Optional[float]:
    if "volume" not in df.columns or df["volume"].isna().all(): return None
    d=df.copy(); d["date"]=d["timestamp"].dt.date
    d["cpv"]=(d["price"]*d["volume"]).groupby(d["date"]).cumsum()
    d["cv"]=d["volume"].groupby(d["date"]).cumsum()
    return round(float((d["cpv"]/d["cv"]).iloc[-1]),2)

# ── Performance stats ─────────────────────────────────────────
def load_stats(symbol:str)->PerfStats:
    stats=PerfStats()
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT action,pnl_pct FROM signals
                    WHERE symbol=%s 
                    AND timestamp >= CURRENT_DATE::timestamptz
                    AND timestamp <  (CURRENT_DATE + INTERVAL '1 day')::timestamptz
                    ORDER BY timestamp""",(symbol,))
                rows=cur.fetchall()
        pnl_list=[]
        for action,pnl in rows:
            stats.total+=1
            if action in("BUY_CE","BUY_PE"): stats.buys+=1
            elif action=="HOLD":              stats.holds+=1
            elif action=="EXIT":
                stats.exits+=1
                if pnl is not None:
                    pnl_list.append(float(pnl))
                    if float(pnl)>0: stats.wins+=1
                    else:            stats.losses+=1
        if pnl_list:
            w=[p for p in pnl_list if p>0]; l=[p for p in pnl_list if p<=0]
            stats.total_pnl=round(sum(pnl_list),2)
            stats.avg_win=round(np.mean(w),2) if w else 0
            stats.avg_loss=round(np.mean(l),2) if l else 0
            t=stats.wins+stats.losses
            stats.win_rate=round(stats.wins/t*100,1) if t>0 else 0
            if len(pnl_list)>1:
                arr=np.array(pnl_list)
                stats.sharpe=round(float(arr.mean()/(arr.std()+1e-9)),2)
        consec=0
        for action,pnl in reversed(rows):
            if action=="EXIT" and pnl is not None:
                if float(pnl)<0: consec+=1
                else: break
            elif action=="EXIT": break
        stats.consec_losses=consec
        stats.halted=consec>=MAX_LOSSES
    except Exception as e: log.warning("Stats: %s",e)
    return stats

# ── Vote casting ──────────────────────────────────────────────
def cast_votes(df,ema,st,rsi_val,macd,bb,vwap_val,
               ctx,gex_result,regime,fii_dii)->Votes:
    v=Votes(); price=float(df["price"].iloc[-1])

    # EMA STACK
    if ema["fast"]>ema["slow"]>ema["medium"]:   v.ema_trend=1
    elif ema["fast"]<ema["slow"]<ema["medium"]:  v.ema_trend=-1
    else:                                         v.ema_trend=0

    # SUPERTREND
    v.supertrend=1 if st["bull"] else -1

    # VWAP
    if vwap_val:
        v.vwap=1 if price>vwap_val*1.001 else(-1 if price<vwap_val*0.999 else 0)
    else:
        v.vwap=1 if price>ema["long"] else -1

    # MACD
    v.macd=1 if macd["hist"]>0 else(-1 if macd["hist"]<0 else 0)

    # RSI
    if rsi_val < RSI_OS:      v.rsi = 1   # oversold = bounce likely
    elif rsi_val > RSI_OB:    v.rsi = -1  # overbought = pullback likely
    elif rsi_val >= 55:       v.rsi = 1   # bullish momentum zone
    elif rsi_val <= 45:       v.rsi = -1  # bearish momentum zone
    else:                     v.rsi = 0   # neutral 45-55

    # BOLLINGER BANDS
    if bb["near_lower"] and ema["fast"]>ema["slow"]:   v.bb=1
    elif bb["near_upper"] and ema["fast"]<ema["slow"]: v.bb=-1
    else:                                               v.bb=1 if bb["above_mid"] else -1

    # PCR OI
    pcr=ctx.get("pcr_oi")
    if pcr: v.pcr_oi=1 if pcr<PCR_BULL else(-1 if pcr>PCR_BEAR else(1 if pcr<1 else -1))

    # PCR VOL
    pv=ctx.get("pcr_vol")
    if pv: v.pcr_vol=1 if pv<PCR_BULL else(-1 if pv>PCR_BEAR else(1 if pv<1 else -1))

    # OI RATIO
    coi=ctx.get("total_call_oi"); poi=ctx.get("total_put_oi")
    if coi and poi and coi>0:
        r=poi/coi; v.oi_ratio=1 if r>1.2 else(-1 if r<0.8 else 0)

    # OI WALLS
    cw=ctx.get("max_call_oi_strike"); pw=ctx.get("max_put_oi_strike")
    if cw and pw:
        gu=cw-price; gd=price-pw
        v.oi_wall=1 if gu>gd*1.2 else(-1 if gd>gu*1.2 else 0)

    # VIX
    vix=ctx.get("india_vix")
    if vix:
        if vix<VIX_LOW:   v.vix=1
        elif vix>VIX_HIGH: v.vix=-1
        else:              v.vix=1 if vix<VIX_MED else 0

    # MAX PAIN
    mp=ctx.get("max_pain")
    if mp:
        if mp>price*1.003:    v.max_pain=1
        elif mp<price*0.997:  v.max_pain=-1

    # GEX
    if gex_result:
        v.gex=1 if gex_result.direction=="BULLISH" else \
             (-1 if gex_result.direction=="BEARISH" else 0)
        if gex_result.gex_flip:
            dist=abs(price-gex_result.gex_flip)/price
            if dist<0.002:
                v.gex=1 if price>gex_result.gex_flip else -1

    # REGIME
    v.regime=1 if regime==Regime.BULL else \
            (-1 if regime==Regime.BEAR else 0)

    # FII/DII
    fn=fii_dii.get("fii_net"); dn=fii_dii.get("dii_net")
    if fn is not None:
        if fn>500:    v.fii_dii=1
        elif fn<-500: v.fii_dii=-1
        elif dn and dn>500: v.fii_dii=1

    return v

# ── Exit logic ────────────────────────────────────────────────
def check_exit(symbol:str,votes:Votes,ctx:dict)->tuple:
    global CURRENT_TRADE
    if not CURRENT_TRADE.in_trade: return False,"",None
    curr=get_current_option_price(symbol,CURRENT_TRADE.option_strike,
                                  CURRENT_TRADE.option_type,CURRENT_TRADE.option_expiry)
    pnl=None
    if curr and CURRENT_TRADE.option_entry>0:
        pnl=round((curr-CURRENT_TRADE.option_entry)/CURRENT_TRADE.option_entry*100,2)
        log.info("P&L: %+.2f%%  entry=%.2f  curr=%.2f",pnl,CURRENT_TRADE.option_entry,curr)
        if pnl<=-SL_PCT:  return True,f"STOP LOSS ({pnl:.1f}%)",pnl
        if pnl>=TGT_PCT:  return True,f"TARGET HIT ({pnl:.1f}%)",pnl
    if CURRENT_TRADE.direction=="BUY_CE" and votes.ema_trend==-1 and votes.supertrend==-1:
        return True,"Trend flipped BEARISH — exit CE",pnl
    if CURRENT_TRADE.direction=="BUY_PE" and votes.ema_trend==1 and votes.supertrend==1:
        return True,"Trend flipped BULLISH — exit PE",pnl
    vix=ctx.get("india_vix")
    if vix and vix>VIX_HIGH: return True,f"VIX spike {vix:.1f}",pnl
    from datetime import timedelta
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if ist_now.hour >= 15 and ist_now.minute >= 15:
        return True, "End of day — 3:15 PM IST", pnl
    return False,"",pnl

# ── Signal generation ─────────────────────────────────────────
def generate_signal(votes:Votes,ctx:dict,regime:str,
                    in_trade:bool,stats:PerfStats)->tuple:
    score=votes.score(); conf=abs(score)*100
    vix=ctx.get("india_vix")
    if stats.halted:
        return "HOLD",conf,f"HALTED — {MAX_LOSSES} consecutive losses"
    if vix and vix>VIX_HIGH:
        return "HOLD",conf,f"VIX={vix:.1f} too high"
    if in_trade:
        return "HOLD",conf,"Already in trade"
    bt,st=get_dynamic_threshold(regime,vix)
    if score>=bt:
        return "BUY_CE",conf,(
            f"Confluence {conf:.1f}% BULLISH | Regime={regime} "
            f"GEX={votes.gex:+d} FII={votes.fii_dii:+d} "
            f"EMA={votes.ema_trend:+d} ST={votes.supertrend:+d} "
            f"PCR={votes.pcr_oi:+d} VIX={votes.vix:+d}")
    if score<=st:
        return "BUY_PE",conf,(
            f"Confluence {conf:.1f}% BEARISH | Regime={regime} "
            f"GEX={votes.gex:+d} FII={votes.fii_dii:+d} "
            f"EMA={votes.ema_trend:+d} ST={votes.supertrend:+d} "
            f"PCR={votes.pcr_oi:+d} VIX={votes.vix:+d}")
    return "HOLD",conf,f"Confluence {conf:.1f}% — threshold {bt*100:.0f}%"

# ── Save signal ───────────────────────────────────────────────
def save_signal(r:SignalResult):
    sql="""INSERT INTO signals(
        symbol,action,confidence,price,regime,implied_move,paper_mode,
        ema_fast,ema_slow,ema_medium,ema_long,
        rsi,macd_histogram,supertrend_bull,vwap,
        pcr_oi,pcr_vol,total_call_oi,total_put_oi,
        total_call_vol,total_put_vol,
        india_vix,max_pain,atm_strike,
        gex_total,gex_flip,gex_direction,
        fii_buy,fii_sell,dii_buy,dii_sell,
        option_type,strike,expiry,dte,option_price,
        delta,theta,vega,iv,iv_rank,moneyness,recommended_lots,
        exit_reason,pnl_pct,votes,reason,timestamp
    ) VALUES(%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s)"""
    t=r.trade; g=r.gex
    try:
        with db() as conn:
            with conn.cursor() as cur:
                cur.execute(sql,(
                    r.symbol,r.action,r.confidence,r.price,
                    r.regime,r.implied_move,r.paper_mode,
                    r.ema_fast,r.ema_slow,r.ema_medium,r.ema_long,
                    r.rsi,r.macd_hist,r.supertrend_bull,r.vwap,
                    r.pcr_oi,r.pcr_vol,r.total_call_oi,r.total_put_oi,
                    r.total_call_vol,r.total_put_vol,
                    r.india_vix,r.max_pain,r.atm_strike,
                    g.total_gex if g else None,
                    g.gex_flip  if g else None,
                    g.direction if g else None,
                    r.fii_buy,r.fii_sell,r.dii_buy,r.dii_sell,
                    t.option_type if t else None,
                    t.strike      if t else None,
                    t.expiry      if t else None,
                    t.dte         if t else None,
                    t.last_price  if t else None,
                    t.delta       if t else None,
                    t.theta       if t else None,
                    t.vega        if t else None,
                    t.iv          if t else None,
                    t.iv_rank     if t else None,
                    t.moneyness   if t else None,
                    t.recommended_lots if t else None,
                    r.exit_reason,r.pnl_pct,
                    json.dumps(r.votes),r.reason,r.timestamp,
                ))
            conn.commit()
        log.info("Signal saved to DB ✅ | action=%s | paper=%s",
                 r.action, r.paper_mode)
    except Exception as e:
        log.error("save_signal FAILED ❌: %s", e)
        log.error("action=%s paper=%s price=%s", r.action, r.paper_mode, r.price)

# ── Telegram ──────────────────────────────────────────────────
def send_telegram(r:SignalResult):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    if r.action=="HOLD": return
    try:
        icons={"BUY_CE":"🟢","BUY_PE":"🔴","EXIT":"🔶"}
        t=r.trade
        msg=(f"{icons.get(r.action,'⚪')} *{r.action}* | {r.symbol}\n"
             f"₹{r.price:.2f} | Conf: {r.confidence:.1f}% | {r.regime}\n"
             f"VIX: {r.india_vix or 'N/A'} | PCR: {r.pcr_oi or 'N/A'}\n")
        if t:
            msg+=(f"Contract: {t.strike:.0f} {t.option_type} @ ₹{t.last_price:.2f}\n"
                  f"Expiry: {t.expiry} | DTE: {t.dte}\n"
                  f"SL: ₹{t.last_price*(1-SL_PCT/100):.2f} "
                  f"TGT: ₹{t.last_price*(1+TGT_PCT/100):.2f}")
        if r.action=="EXIT":
            msg+=f"\n{r.exit_reason}"
            if r.pnl_pct: msg+=f"\nP&L: {r.pnl_pct:+.2f}%"
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id":TELEGRAM_CHAT_ID,"text":msg,"parse_mode":"Markdown"},
            timeout=5)
    except Exception as e: log.debug("Telegram: %s",e)

# ── Print signal (v4.0 + Gemini GEX display merged) ──────────
def print_signal(r:SignalResult, stats:PerfStats):
    bv = lambda v: "▲" if v==1 else ("▼" if v==-1 else "◆")
    icons = {
        "BUY_CE": "🟢 BUY CALL (BULLISH)",
        "BUY_PE": "🔴 BUY PUT  (BEARISH)",
        "HOLD":   "⚪ HOLD     (WAIT)",
        "EXIT":   "🔶 EXIT     (CLOSE NOW)",
    }
    paper = " [PAPER]" if r.paper_mode else ""

    print("\n" + "═"*65)
    print(f"  {r.symbol}  ₹{r.price:.2f}  "
          f"{r.timestamp.strftime('%H:%M:%S')} IST  "
          f"│  Regime: {r.regime}{paper}")

    # ── GEX Section (Gemini style display) ───────────────────
    print("─"*65)
    print(f"  🧠 DEALER GAMMA EXPOSURE (GEX)")
    if r.gex:
        g = r.gex
        print(f"  Zero Gamma Flip : {g.gex_flip or 'N/A'}        "
              f"{bv(r.votes.get('gex',0))}")
        print(f"  GEX Direction   : {g.direction}  "
              f"Net Flow: {g.total_gex:+,.0f}")
        print(f"  GEX Resistance  : {g.key_levels[0] if g.key_levels else 'N/A'}")
        print(f"  GEX Support     : {g.key_levels[-1] if len(g.key_levels)>1 else 'N/A'}")
        print(f"  Key Magnets     : {g.key_levels}")
    else:
        print(f"  GEX             : Building... (need gamma data)")

    # ── Technical Indicators ─────────────────────────────────
    print("─"*65)
    print(f"  EMA 9/21/50/200 : {r.ema_fast:.0f} / {r.ema_slow:.0f} / "
          f"{r.ema_medium:.0f} / {r.ema_long:.0f}  "
          f"{bv(r.votes.get('ema_trend',0))}")
    print(f"  SuperTrend      : {'Bullish' if r.supertrend_bull else 'Bearish'}  "
          f"{bv(r.votes.get('supertrend',0))}  "
          f"│  RSI-14: {r.rsi:.1f}  {bv(r.votes.get('rsi',0))}")
    print(f"  MACD Hist       : {r.macd_hist:.2f}  {bv(r.votes.get('macd',0))}  "
          f"│  VWAP: {r.vwap or 'N/A (no vol)'}")

    # ── Options Sentiment ─────────────────────────────────────
    print("─"*65)
    print(f"  PCR OI / Vol    : {r.pcr_oi or 'N/A'} / {r.pcr_vol or 'N/A'}  "
          f"{bv(r.votes.get('pcr_oi',0))} / {bv(r.votes.get('pcr_vol',0))}")
    print(f"  Total OI C/P    : {r.total_call_oi or 'N/A'} / "
          f"{r.total_put_oi or 'N/A'}  {bv(r.votes.get('oi_ratio',0))}")
    print(f"  VIX / MaxPain   : {r.india_vix or 'N/A'} / {r.max_pain or 'N/A'}  "
          f"{bv(r.votes.get('vix',0))} / {bv(r.votes.get('max_pain',0))}")

    # ── FII/DII ───────────────────────────────────────────────
    if r.fii_buy is not None:
        fn = (r.fii_buy or 0) - (r.fii_sell or 0)
        dn = (r.dii_buy or 0) - (r.dii_sell or 0)
        print(f"  FII / DII Net   : {fn:+.0f} Cr / {dn:+.0f} Cr  "
              f"{bv(r.votes.get('fii_dii',0))}")

    if r.implied_move:
        print(f"  Implied Move    : ±{r.implied_move*100:.2f}% today")

    # ── Signal ────────────────────────────────────────────────
    print("─"*65)
    print(f"  CONFLUENCE      : {r.confidence:.1f}%  │  "
          f"Threshold: {60 if r.regime == 'SIDEWAYS' else 55 if r.regime == 'BULL' else 65}%")
    print(f"  ▶  SIGNAL       : {icons.get(r.action, r.action)}")

    if r.action == "EXIT":
        print(f"  EXIT REASON     : {r.exit_reason}")
        if r.pnl_pct is not None:
            e = "✅" if r.pnl_pct > 0 else "❌"
            print(f"  P&L             : {e} {r.pnl_pct:+.2f}%")

    elif r.trade:
        t = r.trade
        print("─"*65)
        print(f"  CONTRACT        : {r.symbol} {t.strike:.0f} {t.option_type}  "
              f"│  Expiry: {t.expiry}  │  DTE: {t.dte}")
        print(f"  PREMIUM         : ₹{t.last_price:.2f}  "
              f"│  {t.moneyness}  │  IV: {t.iv:.1f}%  "
              f"│  IVR: {t.iv_rank or 'Building...'}")
        print(f"  GREEKS          : Δ={t.delta:.3f}  "
              f"θ={t.theta:.2f}/day  ν={t.vega:.3f}")
        print(f"  LOTS            : {t.recommended_lots}  "
              f"(Capital: ₹{t.last_price*LOT_SIZE*t.recommended_lots:,.0f})")
        print(f"  STOP LOSS  ❌   : ₹{t.last_price*(1-SL_PCT/100):.2f}  "
              f"(-{SL_PCT:.0f}%)")
        print(f"  TARGET     ✅   : ₹{t.last_price*(1+TGT_PCT/100):.2f}  "
              f"(+{TGT_PCT:.0f}%)")
    else:
        print(f"  REASON          : {r.reason}")

    # ── Open trade tracker ────────────────────────────────────
    if CURRENT_TRADE.in_trade:
        print("─"*65)
        print(f"  📌 OPEN TRADE   : {CURRENT_TRADE.option_type} "
              f"{CURRENT_TRADE.option_strike:.0f}  "
              f"Entry ₹{CURRENT_TRADE.option_entry:.2f}  "
              f"@ {CURRENT_TRADE.entry_time}  "
              f"Lots: {CURRENT_TRADE.lots}")

    # ── Performance footer ────────────────────────────────────
    print("─"*65)
    print(f"  📊 TODAY        : Signals={stats.total}  "
          f"Buys={stats.buys}  Exits={stats.exits}  "
          f"W/L={stats.wins}/{stats.losses}  "
          f"WinRate={stats.win_rate:.0f}%  "
          f"P&L={stats.total_pnl:+.1f}%  "
          f"Sharpe={stats.sharpe:.2f}")
    if stats.halted:
        print(f"  ⛔ TRADING HALTED — {MAX_LOSSES} consecutive losses reached")
    print("═"*65 + "\n")

# ── Backtest ──────────────────────────────────────────────────
def run_backtest(symbol:str):
    print("\n"+"═"*65)
    print(f"  BACKTEST — {symbol}")
    print("═"*65)
    try:
        with db() as conn:
            df=pd.read_sql_query("""SELECT timestamp,action,confidence,price,
                option_price,pnl_pct,regime,india_vix,pcr_oi
                FROM signals WHERE symbol=%s ORDER BY timestamp""",
                conn,params=(symbol,))
        if df.empty: print("  No signals. Run live first."); return
        exits=df[df["action"]=="EXIT"].dropna(subset=["pnl_pct"])
        pnl_list=exits["pnl_pct"].astype(float).tolist()
        wins=[p for p in pnl_list if p>0]
        losses=[p for p in pnl_list if p<=0]
        wr=len(wins)/len(pnl_list)*100 if pnl_list else 0
        sharpe=(np.mean(pnl_list)/(np.std(pnl_list)+1e-9)
                if len(pnl_list)>1 else 0)
        print(f"  Period     : {df['timestamp'].min()} → {df['timestamp'].max()}")
        print(f"  Signals    : {len(df)}  Trades: {len(exits)}")
        print("─"*65)
        print(f"  Win Rate   : {wr:.1f}%")
        print(f"  Avg Win    : +{np.mean(wins) if wins else 0:.2f}%")
        print(f"  Avg Loss   : {np.mean(losses) if losses else 0:.2f}%")
        print(f"  Total P&L  : {sum(pnl_list):+.2f}%")
        print(f"  Sharpe     : {sharpe:.2f}")
        print(f"  W / L      : {len(wins)} / {len(losses)}")
        if "regime" in df.columns:
            print("─"*65+"  By Regime:")
            for reg in df["regime"].dropna().unique():
                rx=exits[exits["regime"]==reg]["pnl_pct"]
                if len(rx)>0:
                    rwr=len(rx[rx>0])/len(rx)*100
                    print(f"    {reg:<10}: {len(rx)} trades  "
                          f"WR={rwr:.0f}%  P&L={rx.sum():+.1f}%")
        print("═"*65+"\n")
    except Exception as e: log.exception("Backtest: %s",e)

# ── Orchestrator ──────────────────────────────────────────────
def is_market_open() -> bool:
    """Returns True only during NSE market hours 9:15 AM - 3:30 PM IST."""
    from datetime import timedelta
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if ist.weekday() >= 5:          # Saturday=5, Sunday=6
        return False
    market_open  = ist.replace(hour=9,  minute=15, second=0, microsecond=0)
    market_close = ist.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= ist <= market_close


def run_strategy(symbol:str,paper:bool=False)->Optional[SignalResult]:
    global CURRENT_TRADE

    # Hard stop outside market hours — prevents post-close BUY/EXIT loop
    from datetime import timedelta
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if not is_market_open():
        log.info("Market closed (%s IST) — skipping cycle", 
                 ist_now.strftime("%H:%M"))
        return None

    stats=load_stats(symbol)
    df=get_market_data(symbol)
    ctx=get_options_context(symbol)
    price=float(df["price"].iloc[-1])
    fii_dii=fetch_fii_dii()
    if ctx["india_vix"] is None and "india_vix" in df.columns:
        vx=df["india_vix"].dropna()
        if not vx.empty: ctx["india_vix"]=float(vx.iloc[-1])
    ema=calc_ema(df); st=calc_supertrend(df)
    rsi=calc_rsi(df); macd=calc_macd(df)
    bb=calc_bb(df); vwap=calc_vwap(df)
    regime=detect_regime(df,ctx.get("india_vix"))
    gex=calculate_gex(symbol,price)
    impl=implied_move(ctx.get("india_vix"),1)
    votes=cast_votes(df,ema,st,rsi,macd,bb,vwap,ctx,gex,regime,fii_dii)
    should_exit,exit_reason,pnl_pct=check_exit(symbol,votes,ctx)
    ts=datetime.now(timezone.utc)

    if should_exit:
        result=SignalResult(
            symbol=symbol,action="EXIT",
            confidence=abs(votes.score())*100,price=price,
            regime=regime,implied_move=impl,gex=gex,
            ema_fast=ema["fast"],ema_slow=ema["slow"],
            ema_medium=ema["medium"],ema_long=ema["long"],
            rsi=rsi,macd_hist=macd["hist"],
            supertrend_bull=st["bull"],vwap=vwap,
            pcr_oi=ctx.get("pcr_oi"),pcr_vol=ctx.get("pcr_vol"),
            total_call_oi=ctx.get("total_call_oi"),
            total_put_oi=ctx.get("total_put_oi"),
            total_call_vol=ctx.get("total_call_vol"),
            total_put_vol=ctx.get("total_put_vol"),
            india_vix=ctx.get("india_vix"),
            max_pain=ctx.get("max_pain"),
            atm_strike=ctx.get("atm_strike"),
            fii_buy=fii_dii.get("fii_buy"),
            fii_sell=fii_dii.get("fii_sell"),
            dii_buy=fii_dii.get("dii_buy"),
            dii_sell=fii_dii.get("dii_sell"),
            votes=votes.to_dict(),trade=None,
            timestamp=ts,reason=exit_reason,
            exit_reason=exit_reason,pnl_pct=pnl_pct,
            paper_mode=paper)
        CURRENT_TRADE=TradeState()
        print_signal(result,stats)
        save_signal(result)
        send_telegram(result)
        return result

    action,conf,reason=generate_signal(votes,ctx,regime,
                                        CURRENT_TRADE.in_trade,stats)
    trade=None
    if action in("BUY_CE","BUY_PE"):
        trade=get_best_option(symbol,action,price)
        if trade is None:
            action="HOLD"; reason+=" | No contract found"
        else:
            trade.recommended_lots=kelly_lots(
                stats.win_rate/100 if stats.win_rate>0 else 0.5,
                stats.avg_win,stats.avg_loss,CAPITAL,trade.last_price)
            CURRENT_TRADE=TradeState(
                    in_trade=True,direction=action,entry_price=price,
                    option_strike=trade.strike,option_type=trade.option_type,
                    option_expiry=trade.expiry,option_entry=trade.last_price,
                    entry_time=ts.strftime("%H:%M:%S"),
                    lots=trade.recommended_lots)

    result=SignalResult(
        symbol=symbol,action=action,
        confidence=round(conf,1),price=price,
        regime=regime,implied_move=impl,gex=gex,
        ema_fast=ema["fast"],ema_slow=ema["slow"],
        ema_medium=ema["medium"],ema_long=ema["long"],
        rsi=rsi,macd_hist=macd["hist"],
        supertrend_bull=st["bull"],vwap=vwap,
        pcr_oi=ctx.get("pcr_oi"),pcr_vol=ctx.get("pcr_vol"),
        total_call_oi=ctx.get("total_call_oi"),
        total_put_oi=ctx.get("total_put_oi"),
        total_call_vol=ctx.get("total_call_vol"),
        total_put_vol=ctx.get("total_put_vol"),
        india_vix=ctx.get("india_vix"),
        max_pain=ctx.get("max_pain"),
        atm_strike=ctx.get("atm_strike"),
        fii_buy=fii_dii.get("fii_buy"),
        fii_sell=fii_dii.get("fii_sell"),
        dii_buy=fii_dii.get("dii_buy"),
        dii_sell=fii_dii.get("dii_sell"),
        votes=votes.to_dict(),trade=trade,
        timestamp=ts,reason=reason,paper_mode=paper)

    print_signal(result,stats)
    save_signal(result)
    send_telegram(result)
    return result

# ── Entry point ───────────────────────────────────────────────
def run_loop(symbol:str,interval:int,paper:bool=False):
    from datetime import timedelta
    mode="PAPER" if paper else "LIVE"
    log.info("v4.0 | %s | %s | every %ds",symbol,mode,interval)
    ensure_signals_table()
    while True:
        try:
            ist_now=datetime.now(timezone.utc)+timedelta(hours=5,minutes=30)
            if not is_market_open():
                log.info("Market closed (%s IST) — waiting…",
                         ist_now.strftime("%H:%M"))
                time.sleep(interval)
                continue
            run_strategy(symbol,paper=paper)
        except KeyboardInterrupt:
            raise
        except ValueError as e:
            log.warning("Skip: %s",e)
        except Exception as e:
            log.exception("Error: %s",e)
        log.info("Next in %ds", interval)
        time.sleep(interval)

if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Strategy Engine v4.0")
    parser.add_argument("symbol",nargs="?",default="NIFTY")
    parser.add_argument("--loop",type=int,default=0)
    parser.add_argument("--paper",action="store_true")
    parser.add_argument("--backtest",action="store_true")
    parser.add_argument("--stats",action="store_true")
    args=parser.parse_args()
    try:
        ensure_signals_table()
        if args.backtest or args.stats: run_backtest(args.symbol)
        elif args.loop>0: run_loop(args.symbol,args.loop,paper=args.paper)
        else: run_strategy(args.symbol,paper=args.paper)
        sys.exit(0)
    except KeyboardInterrupt: log.info("Stopped."); sys.exit(0)
    except Exception as e: log.exception("Fatal: %s",e); sys.exit(2)


    