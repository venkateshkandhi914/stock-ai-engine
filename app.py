import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request
import joblib
import numpy as np
import pandas as pd
import requests
import pytz

# TradingView Real-Time Datafeed
from tvdatafeed import TvDatafeed, Interval
import yfinance as yf

app = Flask(__name__)
application = app

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "8834841152:AAF0bok9I01ylydeVc2RiAYz-F1BWvyDDLk"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "6057603813"

# 1. AI మోడల్స్ లోడింగ్
ai_model_fvg = None
if os.path.exists("fvg_ai_model.pkl"):
    try:
        ai_model_fvg = joblib.load("fvg_ai_model.pkl")
        print("Model 1 (FVG AI) Loaded Successfully")
    except Exception as e:
        print(f"FVG Model Load Error: {e}")

ai_model_pa = None
if os.path.exists("price_action_ai_model.pkl"):
    try:
        ai_model_pa = joblib.load("price_action_ai_model.pkl")
        print("Model 5 (Price Action 15 EMA AI) Loaded Successfully")
    except Exception as e:
        print(f"PA Model Load Error: {e}")

# TradingView క్లయింట్ (No-login మోడ్)
tv = TvDatafeed()

NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
    "TATAMOTORS", "BAJFINANCE", "ITC", "LT", "AXISBANK", "KOTAKBANK"
]
CRYPTO_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD"]

# 5 వేర్వేరు స్ట్రాటజీల ట్రాకింగ్ స్టేట్
STRATEGIES = ["AI", "QUANT", "HYBRID", "PRICE_ACTION", "PA_15EMA_AI"]
SENT_ALERTS = {s: set() for s in STRATEGIES}
ACTIVE_TRADES = {s: {} for s in STRATEGIES}
AUDIT_LOGS = {s: [] for s in STRATEGIES}
AUDIT_SENT_TODAY = False
LATEST_AUTO_SCAN_RESULTS = []

def send_telegram_msg(msg_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=8.0)
        return resp.status_code == 200
    except:
        return False

def check_market_session():
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    if ist_now.weekday() in [5, 6]:
        return False, "వీకెండ్ సెలవు (NSE క్లోజ్)"
    cur_mins = ist_now.hour * 60 + ist_now.minute
    if cur_mins < 555:
        return False, "మార్కెట్ ఇంకా ప్రారంభం కాలేదు (09:15 AM వరకు వేచి ఉండండి)"
    if cur_mins > 930:
        return False, "నేటి మార్కెట్ సమయం ముగిసింది (03:30 PM)"
    return True, "MARKET_LIVE"

# TradingView జీరో-డిలే ఫెచర్
def fetch_tv_candles(symbol, exchange="NSE", interval=Interval.in_5_minute, n_bars=60):
    try:
        df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=n_bars)
        if df is not None and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df
    except Exception as e:
        pass
    
    # Fallback yfinance (TradingView ఫెయిల్ అయితే బ్యాకప్)
    try:
        ticker_str = f"{symbol}.NS" if exchange == "NSE" else f"{symbol.replace('USD', '-USD')}"
        yf_df = yf.download(ticker_str, period="3d", interval="5m", progress=False)
        if not yf_df.empty:
            yf_df.columns = [c.lower() for c in yf_df.columns]
            return yf_df
    except:
        pass
    return None

class MasterPriceActionScanner:
    def __init__(self, df: pd.DataFrame, range_lookback: int = 20):
        self.df = df.copy()
        self.range_lookback = range_lookback
        self._prepare_indicators()

    def _prepare_indicators(self):
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3.0
        self.df['VWAP'] = (typical_price * self.df['volume']).cumsum() / (self.df['volume'].cumsum() + 1e-9)
        self.df['SMA20'] = self.df['close'].rolling(window=20).mean()

        self.df['Recent_High'] = self.df['high'].shift(2).rolling(window=5).max()
        self.df['Recent_Low'] = self.df['low'].shift(2).rolling(window=5).min()

        self.df['Range_High'] = self.df['high'].shift(2).rolling(window=self.range_lookback).max()
        self.df['Range_Low'] = self.df['low'].shift(2).rolling(window=self.range_lookback).min()

        self.df['Bullish_FVG'] = (self.df['low'] > self.df['high'].shift(2)) & (self.df['close'].shift(1) > self.df['open'].shift(1))
        self.df['Bearish_FVG'] = (self.df['high'] < self.df['low'].shift(2)) & (self.df['close'].shift(1) < self.df['open'].shift(1))

        self.df['Bullish_OB_Low'] = np.nan
        self.df['Bullish_OB_High'] = np.nan
        for i in range(2, len(self.df)):
            if (self.df['close'].iloc[i - 1] < self.df['open'].iloc[i - 1] and
                self.df['close'].iloc[i] > self.df['open'].iloc[i] and
                self.df['close'].iloc[i] > self.df['high'].iloc[i - 1]):
                self.df.loc[self.df.index[i], 'Bullish_OB_Low'] = self.df['low'].iloc[i - 1]
                self.df.loc[self.df.index[i], 'Bullish_OB_High'] = self.df['high'].iloc[i - 1]
        self.df['Bullish_OB_Low'] = self.df['Bullish_OB_Low'].ffill()
        self.df['Bullish_OB_High'] = self.df['Bullish_OB_High'].ffill()

        self.df['Extreme_High'] = self.df['high'].shift(3).rolling(window=20).max()

    def get_latest_signal(self):
        if len(self.df) < max(25, self.range_lookback + 2):
            return None
        i = len(self.df) - 1
        curr = self.df.iloc[i]
        prev = self.df.iloc[i - 1]
        candle_2 = self.df.iloc[i - 2]

        level_high = curr['Recent_High']
        if (prev['close'] >= level_high and curr['low'] <= level_high * 1.002 and curr['close'] > level_high and curr['close'] > curr['open']):
            if curr['close'] > curr['VWAP'] and curr['close'] > curr['SMA20']:
                entry = round(curr['close'], 2)
                sl = round(min(curr['low'], level_high) - 0.20, 2)
                risk = round(entry - sl, 2)
                if risk > 0:
                    return {'model': 'Breakout + Retest', 'setup': 'Support Retest above VWAP', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': round(entry + (risk * 2), 2), 'risk': risk, 'rr': '1:2.0'}

        r_high = curr['Range_High']
        r_low = curr['Range_Low']
        fakeout_down = prev['low'] < r_low and prev['close'] >= r_low
        bullish_markup = curr['close'] > r_high and curr['close'] > curr['open']
        if (fakeout_down or prev['close'] > r_high) and bullish_markup:
            entry = round(curr['close'], 2)
            sl = round(r_low, 2)
            risk = round(entry - sl, 2)
            if risk > 0:
                return {'model': 'Wyckoff Theory', 'setup': 'Accumulation Spring -> Markup Phase', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': round(entry + (risk * 2), 2), 'risk': risk, 'rr': '1:2.0'}

        swept_low = prev['low'] < candle_2['low'] and prev['close'] >= candle_2['low']
        if swept_low and curr['close'] > curr['open']:
            entry = round(curr['close'], 2)
            sl = round(min(curr['low'], prev['low']) - 0.20, 2)
            risk = round(entry - sl, 2)
            if risk > 0:
                return {'model': 'SMC Liquidity & FVG', 'setup': 'Low Sweep + Bullish FVG Retest', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': round(entry + (risk * 2), 2), 'risk': risk, 'rr': '1:2.0'}

        ob_high = curr['Bullish_OB_High']
        ob_low = curr['Bullish_OB_Low']
        tested_ob = (curr['low'] <= ob_high and curr['close'] >= ob_low) if pd.notna(ob_high) else False
        if tested_ob and curr['close'] > curr['open']:
            entry = round(curr['close'], 2)
            sl = round(min(curr['low'], prev['low']) - 0.20, 2)
            risk = round(entry - sl, 2)
            target = round(max(curr['Extreme_High'], entry + (risk * 2.0)), 2)
            if risk > 0:
                return {'model': 'Institutional SMC', 'setup': 'Bullish Order Block Mitigated', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': target, 'risk': risk, 'rr': '1:2.0'}
        return None

def evaluate_stock_universe(symbol, exchange="NSE"):
    try:
        df = fetch_tv_candles(symbol, exchange=exchange, interval=Interval.in_5_minute, n_bars=60)
        if df is None or len(df) < 30:
            return None

        curr_p = round(float(df['close'].iloc[-1]), 2)

        # Pivot Points
        high_val = float(df['high'].max())
        low_val = float(df['low'].min())
        pivot = (high_val + low_val + curr_p) / 3.0
        r1 = round((2 * pivot) - low_val, 2)
        s1 = round((2 * pivot) - high_val, 2)
        r2 = round(pivot + (high_val - low_val), 2)
        s2 = round(pivot - (high_val - low_val), 2)

        # 1. Type 1 (FVG AI Features)
        c1_high = df['high'].iloc[-3]
        c3_low = df['low'].iloc[-1]
        fvg_gap_pct = ((c3_low - c1_high) / curr_p) * 100.0

        tp = (df['high'] + df['low'] + df['close']) / 3.0
        cum_vol = df['volume'].rolling(75).sum()
        vwap_s = (tp * df['volume']).rolling(75).sum() / (cum_vol + 1e-9)
        curr_vwap = round(float(vwap_s.iloc[-1]), 2)
        dist_from_vwap = ((curr_p - curr_vwap) / curr_vwap) * 100.0

        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        curr_vol = df['volume'].iloc[-1]
        vol_ratio = float(curr_vol / (avg_vol + 1e-9))

        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-9)
        rsi_14 = float(100 - (100 / (1 + rs.iloc[-1])))

        tr = pd.concat([df['high'] - df['low'], (df['high'] - df['close'].shift()).abs(), (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
        atr_14 = float((tr.rolling(14).mean().iloc[-1] / curr_p) * 100.0)

        # Fast Indicators (EMA 5 & 13)
        ema_5 = float(df['close'].ewm(span=5, adjust=False).mean().iloc[-1])
        ema_13 = float(df['close'].ewm(span=13, adjust=False).mean().iloc[-1])

        # Type 1: FVG AI Model Evaluation
        quant_score_fvg = 50
        ai_buy = False
        if ai_model_fvg is not None:
            features = np.array([[fvg_gap_pct, dist_from_vwap, vol_ratio, rsi_14, atr_14]])
            win_prob = ai_model_fvg.predict_proba(features)[0][1]
            quant_score_fvg = int(win_prob * 100)
            if quant_score_fvg >= 60 and curr_p >= curr_vwap:
                ai_buy = True

        indicators_pass = (ema_5 > ema_13) and (curr_p > curr_vwap) and (45 <= rsi_14 <= 65) and (curr_vol > avg_vol)

        # Type 4: Master Price Action
        pa_scanner = MasterPriceActionScanner(df)
        pa_signal = pa_scanner.get_latest_signal()

        # Type 5: 15 EMA + Pure Price Action ML Features
        df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
        candle_range = df['high'] - df['low'] + 1e-9
        body_ratio = float(((df['close'] - df['open']).abs() / candle_range).iloc[-1])
        upper_wick = float(((df['high'] - df[['open', 'close']].max(axis=1)) / candle_range).iloc[-1])
        lower_wick = float(((df[['open', 'close']].min(axis=1) - df['low']) / candle_range).iloc[-1])
        dist_ema_15 = float((((df['close'] - df['ema_15']) / df['ema_15']) * 100.0).iloc[-1])
        ret_1 = float((df['close'].pct_change(1) * 100.0).iloc[-1])
        ret_3 = float((df['close'].pct_change(3) * 100.0).iloc[-1])
        roll_h = df['high'].rolling(20).max().iloc[-1]
        roll_l = df['low'].rolling(20).min().iloc[-1]
        dist_support = float((((curr_p - roll_l) / (roll_h - roll_l + 1e-9)) * 100.0))
        ema_15_val = float(df['ema_15'].iloc[-1])

        pa_ai_score = 50
        pa_ai_buy = False
        if ai_model_pa is not None:
            pa_features = np.array([[body_ratio, upper_wick, lower_wick, dist_ema_15, ret_1, ret_3, dist_support]])
            prob_pa = ai_model_pa.predict_proba(pa_features)[0][1]
            pa_ai_score = int(prob_pa * 100)
            if pa_ai_score >= 60 and curr_p >= ema_15_val:
                pa_ai_buy = True

        return {
            "stock": symbol,
            "exchange": exchange,
            "price": curr_p,
            "vwap": curr_vwap,
            "rsi": round(rsi_14, 2),
            "quant_score_fvg": quant_score_fvg,
            "ai_buy": ai_buy,
            "indicators_pass": indicators_pass,
            "pa_signal": pa_signal,
            "ema_15": round(ema_15_val, 2),
            "pa_ai_score": pa_ai_score,
            "pa_ai_buy": pa_ai_buy,
            "sl": round(curr_p * 0.993, 2),
            "target": round(curr_p * 1.01, 2),
            "s1": s1, "s2": s2, "r1": r1, "r2": r2
        }
    except Exception as e:
        return None

def background_scanner_and_audit():
    global SENT_ALERTS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS, LATEST_AUTO_SCAN_RESULTS
    ist = pytz.timezone("Asia/Kolkata")

    while True:
        try:
            ist_now = datetime.now(ist)

            # Daily Reset at 09:15 AM
            if ist_now.hour == 9 and ist_now.minute < 15:
                SENT_ALERTS = {s: set() for s in STRATEGIES}
                ACTIVE_TRADES = {s: {} for s in STRATEGIES}
                AUDIT_LOGS = {s: [] for s in STRATEGIES}
                AUDIT_SENT_TODAY = False

            # సాయంత్రం 03:30 PM తర్వాత 5 ప్రత్యేక ఆడిట్ రిపోర్టులు
            if ist_now.hour >= 15 and ist_now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in STRATEGIES:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append(f"🟢 {sym}: CLOSED AT 03:30 PM | P&L: ₹0.00 (Exit: ₹{pos['entry']})")
                    ACTIVE_TRADES[strat].clear()

                titles = {
                    "AI": "📊 *[AUDIT 1: PURE 365-DAY AI MODEL REPORT]*",
                    "QUANT": "📊 *[AUDIT 2: FAST QUANT INDICATORS REPORT]*",
                    "HYBRID": "📊 *[AUDIT 3: HYBRID CONFLUENCE (AI + QUANT) REPORT]*",
                    "PRICE_ACTION": "📊 *[AUDIT 4: INSTITUTIONAL PRICE ACTION REPORT]*",
                    "PA_15EMA_AI": "📊 *[AUDIT 5: 15 EMA + PRICE ACTION AI (24/7) REPORT]*"
                }

                for strat in STRATEGIES:
                    pnl = sum([float(l.split('P&L: ₹')[1].split(' ')[0]) for l in AUDIT_LOGS[strat] if 'P&L: ₹' in l])
                    msg = (
                        f"{titles[strat]}\n"
                        f"💰 *NET PROFIT: ₹{round(pnl, 2)}*\n"
                        f"────────────────────\n"
                        f"⏱️ *Audit Logs:*\n" + ("\n".join(AUDIT_LOGS[strat][-10:]) if AUDIT_LOGS[strat] else "No Trades Today")
                    )
                    send_telegram_msg(msg)
                    time.sleep(2)

                AUDIT_SENT_TODAY = True

            # మార్కెట్ స్కానింగ్
            is_nse, _ = check_market_session()
            scan_list = []
            
            # 24/7 క్రిప్టో ఎల్లప్పుడూ స్కాన్ అవుతుంది
            for c in CRYPTO_SYMBOLS:
                scan_list.append((c, "BINANCE"))

            # ఎన్‌ఎస్‌ఈ సమయాల్లో స్టాక్స్ స్కాన్ అవుతాయి
            if is_nse:
                for s in NSE_STOCKS:
                    scan_list.append((s, "NSE"))

            live_candidates = []
            for sym, exch in scan_list:
                res = evaluate_stock_universe(sym, exchange=exch)
                if not res:
                    continue

                price = res['price']

                # 1. TYPE 1: Pure AI
                if res['ai_buy'] and sym not in SENT_ALERTS["AI"] and exch == "NSE":
                    SENT_ALERTS["AI"].add(sym)
                    ACTIVE_TRADES["AI"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                    send_telegram_msg(
                        f"🟢 *[TYPE 1: PURE AI BUY ALERT]*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 Stock: `{sym}`\n"
                        f"🎯 AI Confidence: `{res['quant_score_fvg']}%`\n"
                        f"📥 Zero-Lag CMP: ₹{price}\n"
                        f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                        f"🛡️ S1: ₹{res['s1']} | 🚧 R1: ₹{res['r1']}\n"
                        f"━━━━━━━━━━━━━━━━━━━━"
                    )

                # 2. TYPE 2: Quant Indicators
                if res['indicators_pass'] and sym not in SENT_ALERTS["QUANT"] and exch == "NSE":
                    SENT_ALERTS["QUANT"].add(sym)
                    ACTIVE_TRADES["QUANT"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                    send_telegram_msg(
                        f"⚡ *[TYPE 2: FAST QUANT INDICATORS ALERT]*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 Stock: `{sym}`\n"
                        f"📊 RSI: {res['rsi']} | VWAP: ₹{res['vwap']}\n"
                        f"📥 Zero-Lag CMP: ₹{price}\n"
                        f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                        f"🛡️ S1: ₹{res['s1']} | 🚧 R1: ₹{res['r1']}\n"
                        f"━━━━━━━━━━━━━━━━━━━━"
                    )

                # 3. TYPE 3: Hybrid Confluence
                if res['ai_buy'] and res['indicators_pass'] and sym not in SENT_ALERTS["HYBRID"] and exch == "NSE":
                    SENT_ALERTS["HYBRID"].add(sym)
                    ACTIVE_TRADES["HYBRID"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                    live_candidates.append(res)
                    send_telegram_msg(
                        f"🚀 *[TYPE 3: HIGH CONFIDENCE CONFLUENCE]*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 Stock: `{sym}` (Dual Verified)\n"
                        f"🎯 AI Score: {res['quant_score_fvg']}% | RSI: {res['rsi']}\n"
                        f"📥 Zero-Lag CMP: ₹{price}\n"
                        f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                        f"🛡️ S1: ₹{res['s1']} | 🚧 R1: ₹{res['r1']}\n"
                        f"━━━━━━━━━━━━━━━━━━━━"
                    )

                # 4. TYPE 4: Institutional Price Action
                if res['pa_signal'] and sym not in SENT_ALERTS["PRICE_ACTION"] and exch == "NSE":
                    pa = res['pa_signal']
                    SENT_ALERTS["PRICE_ACTION"].add(sym)
                    ACTIVE_TRADES["PRICE_ACTION"][sym] = {"entry": pa['entry'], "sl": pa['sl'], "target": pa['target']}
                    send_telegram_msg(
                        f"🏛️ *[TYPE 4: INSTITUTIONAL PRICE ACTION]*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 Stock: `{sym}`\n"
                        f"⚡ Strategy: *{pa['model']}*\n"
                        f"🎯 Setup: _{pa['setup']}_\n"
                        f"📥 Signal: *{pa['signal']} @ ₹{pa['entry']}*\n"
                        f"🛑 SL: ₹{pa['sl']} | 🎯 Target: ₹{pa['target']}\n"
                        f"⚖️ Risk: ₹{pa['risk']} | R:R: {pa['rr']}\n"
                        f"🛡️ S1: ₹{res['s1']} | 🚧 R1: ₹{res['r1']}\n"
                        f"━━━━━━━━━━━━━━━━━━━━"
                    )

                # 5. TYPE 5: 15 EMA + Price Action AI (24/7 క్రిప్టో & స్టాక్స్)
                if res['pa_ai_buy'] and sym not in SENT_ALERTS["PA_15EMA_AI"]:
                    SENT_ALERTS["PA_15EMA_AI"].add(sym)
                    sl_pa = round(price * 0.992, 2)
                    tgt_pa = round(price * 1.015, 2)
                    ACTIVE_TRADES["PA_15EMA_AI"][sym] = {"entry": price, "sl": sl_pa, "target": tgt_pa}
                    asset_label = "CRYPTO (24/7)" if exch != "NSE" else "NSE STOCK"
                    curr_sym = "$" if exch != "NSE" else "₹"
                    send_telegram_msg(
                        f"🤖 *[TYPE 5: 15 EMA + PRICE ACTION AI]*\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"📌 Asset: `{sym}` ({asset_label})\n"
                        f"🎯 AI Score: `{res['pa_ai_score']}%`\n"
                        f"📈 15 EMA: {curr_sym}{res['ema_15']}\n"
                        f"📥 Live Entry: {curr_sym}{price}\n"
                        f"🛑 SL: {curr_sym}{sl_pa} | 🎯 Target: {curr_sym}{tgt_pa}\n"
                        f"━━━━━━━━━━━━━━━━━━━━\n"
                        f"⚡ _Zero-Lag TradingView Feed Active_"
                    )

                # Target / SL ఎగ్జిట్ ట్రాకింగ్
                for strat in STRATEGIES:
                    if sym in ACTIVE_TRADES[strat]:
                        trade = ACTIVE_TRADES[strat][sym]
                        if price >= trade["target"]:
                            pnl = round(price - trade["entry"], 2)
                            AUDIT_LOGS[strat].append(f"🟢 {sym}: TARGET HIT 🎯 @ {ist_now.strftime('%I:%M %p')} | P&L: ₹{pnl} (Exit: ₹{price})")
                            del ACTIVE_TRADES[strat][sym]
                        elif price <= trade["sl"]:
                            pnl = round(trade["entry"] - price, 2)
                            AUDIT_LOGS[strat].append(f"🔴 {sym}: STOP LOSS HIT 🛑 @ {ist_now.strftime('%I:%M %p')} | P&L: -₹{pnl} (Exit: ₹{price})")
                            del ACTIVE_TRADES[strat][sym]

            if live_candidates:
                LATEST_AUTO_SCAN_RESULTS = live_candidates

            time.sleep(60)
        except Exception as e:
            time.sleep(30)

threading.Thread(target=background_scanner_and_audit, daemon=True).start()

@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today_str = str(ist_now.date())
    if not LATEST_AUTO_SCAN_RESULTS:
        return jsonify([{"date": today_str, "pred": "Scanning Zero-Lag Feed", "actual": "Monitoring 5-Tier Confluence...", "accuracy": "Active"}])
    best = LATEST_AUTO_SCAN_RESULTS[0]
    return jsonify([{
        "date": today_str,
        "pred": f"{best['stock']} (Dual Confirmed BUY)",
        "actual": f"CMP: ₹{best['price']} | S1: ₹{best['s1']} | R1: ₹{best['r1']}",
        "accuracy": f"AI Score: {best['quant_score_fvg']}%"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: 5-Tier Engine & tvdatafeed Zero-Lag Feed Online!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "engine": "5-Tier Institutional & Crypto Hybrid Engine",
        "data_feed": "TradingView (tvdatafeed) Zero-Lag WebSocket",
        "strategies": [
            "Type 1: Pure 365-Days AI",
            "Type 2: Fast Quant Indicators",
            "Type 3: Hybrid Confluence",
            "Type 4: Master Price Action (Wyckoff/SMC)",
            "Type 5: 15 EMA + Price Action AI (24/7 Crypto + Stocks)"
        ],
        "audit_reports": "5 Independent Statements Scheduled at 03:30 PM"
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
