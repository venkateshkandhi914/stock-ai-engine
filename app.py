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
import yfinance as yf

app = Flask(__name__)
application = app

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "8834841152:AAF0bok9I01ylydeVc2RiAYz-F1BWvyDDLk"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "6057603813"

MODEL_FILE = "fvg_ai_model.pkl"
ai_model = None
if os.path.exists(MODEL_FILE):
    try:
        ai_model = joblib.load(MODEL_FILE)
        print("AI Model Loaded Successfully")
    except Exception as e:
        print(f"Model Error: {e}")

STOCK_UNIVERSE = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
    "TATAMOTORS", "BAJFINANCE", "ITC", "LT", "AXISBANK", "KOTAKBANK"
]

# 4 విభిన్న కేటగిరీల కోసం ట్రాకింగ్ స్టేట్
SENT_ALERTS = {"AI": set(), "QUANT": set(), "HYBRID": set(), "PRICE_ACTION": set()}
ACTIVE_TRADES = {"AI": {}, "QUANT": {}, "HYBRID": {}, "PRICE_ACTION": {}}
AUDIT_LOGS = {"AI": [], "QUANT": [], "HYBRID": [], "PRICE_ACTION": []}

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
        return False, "వీకెండ్ సెలవు (మార్కెట్ క్లోజ్)"
    cur_mins = ist_now.hour * 60 + ist_now.minute
    if cur_mins < 555:
        return False, "మార్కెట్ ఇంకా ప్రారంభం కాలేదు (09:15 AM వరకు వేచి ఉండండి)"
    if cur_mins > 930:
        return False, "మార్కెట్ సమయం ముగిసింది (03:30 PM)"
    return True, "MARKET_LIVE"

class MasterPriceActionScanner:
    def __init__(self, df: pd.DataFrame, range_lookback: int = 20):
        self.df = df.copy()
        self.range_lookback = range_lookback
        self._prepare_indicators()

    def _prepare_indicators(self):
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
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
        self.df['Bearish_OB_Low'] = np.nan
        self.df['Bearish_OB_High'] = np.nan

        for i in range(2, len(self.df)):
            if (self.df['close'].iloc[i - 1] < self.df['open'].iloc[i - 1] and
                self.df['close'].iloc[i] > self.df['open'].iloc[i] and
                self.df['close'].iloc[i] > self.df['high'].iloc[i - 1]):
                self.df.loc[self.df.index[i], 'Bullish_OB_Low'] = self.df['low'].iloc[i - 1]
                self.df.loc[self.df.index[i], 'Bullish_OB_High'] = self.df['high'].iloc[i - 1]

            if (self.df['close'].iloc[i - 1] > self.df['open'].iloc[i - 1] and
                self.df['close'].iloc[i] < self.df['open'].iloc[i] and
                self.df['close'].iloc[i] < self.df['low'].iloc[i - 1]):
                self.df.loc[self.df.index[i], 'Bearish_OB_Low'] = self.df['low'].iloc[i - 1]
                self.df.loc[self.df.index[i], 'Bearish_OB_High'] = self.df['high'].iloc[i - 1]

        self.df['Bullish_OB_Low'] = self.df['Bullish_OB_Low'].ffill()
        self.df['Bullish_OB_High'] = self.df['Bullish_OB_High'].ffill()
        self.df['Bearish_OB_Low'] = self.df['Bearish_OB_Low'].ffill()
        self.df['Bearish_OB_High'] = self.df['Bearish_OB_High'].ffill()

        self.df['Extreme_High'] = self.df['high'].shift(3).rolling(window=20).max()
        self.df['Extreme_Low'] = self.df['low'].shift(3).rolling(window=20).min()

    def get_latest_signal(self):
        if len(self.df) < max(25, self.range_lookback + 2):
            return None
        i = len(self.df) - 1
        curr = self.df.iloc[i]
        prev = self.df.iloc[i - 1]
        candle_2 = self.df.iloc[i - 2]

        # 1. Breakout & Retest
        level_high = curr['Recent_High']
        level_low = curr['Recent_Low']
        if (prev['close'] >= level_high and curr['low'] <= level_high * 1.002 and curr['close'] > level_high and curr['close'] > curr['open']):
            if curr['close'] > curr['VWAP'] and curr['close'] > curr['SMA20']:
                entry = round(curr['close'], 2)
                sl = round(min(curr['low'], level_high) - 0.20, 2)
                risk = round(entry - sl, 2)
                if risk > 0:
                    return {'model': 'Breakout + Retest', 'setup': 'Support Retest above VWAP', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': round(entry + (risk * 2), 2), 'risk': risk, 'rr': '1:2.0'}

        # 2. Wyckoff Accumulation Spring
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

        # 3. SMC Liquidity Sweep & FVG
        swept_low = prev['low'] < candle_2['low'] and prev['close'] >= candle_2['low']
        if swept_low and curr['close'] > curr['open']:
            entry = round(curr['close'], 2)
            sl = round(min(curr['low'], prev['low']) - 0.20, 2)
            risk = round(entry - sl, 2)
            if risk > 0:
                return {'model': 'SMC Liquidity & FVG', 'setup': 'Low Sweep + Bullish FVG Retest', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': round(entry + (risk * 2), 2), 'risk': risk, 'rr': '1:2.0'}

        # 4. Institutional Order Block
        ob_high = curr['Bullish_OB_High']
        ob_low = curr['Bullish_OB_Low']
        tested_bullish_ob = (curr['low'] <= ob_high and curr['close'] >= ob_low) if pd.notna(ob_high) else False
        if tested_bullish_ob and curr['close'] > curr['open']:
            entry = round(curr['close'], 2)
            sl = round(min(curr['low'], prev['low']) - 0.20, 2)
            risk = round(entry - sl, 2)
            target = round(max(curr['Extreme_High'], entry + (risk * 2.0)), 2)
            if risk > 0:
                return {'model': 'Institutional SMC', 'setup': 'Bullish Order Block Mitigated', 'signal': 'BUY', 'entry': entry, 'sl': sl, 'target': target, 'risk': risk, 'rr': '1:2.0'}

        return None

def evaluate_stock_full(symbol):
    clean_sym = symbol.replace('.NS', '').upper()
    try:
        ticker = yf.Ticker(f"{clean_sym}.NS")
        df = ticker.history(period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None

        df.columns = [c.lower() for c in df.columns]
        curr_p = round(float(df['close'].iloc[-1]), 2)

        # Pivots (S1, S2, R1, R2)
        high_val = float(df['high'].max())
        low_val = float(df['low'].min())
        pivot = (high_val + low_val + curr_p) / 3.0
        r1 = round((2 * pivot) - low_val, 2)
        s1 = round((2 * pivot) - high_val, 2)
        r2 = round(pivot + (high_val - low_val), 2)
        s2 = round(pivot - (high_val - low_val), 2)

        # AI Features
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

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_14 = float((tr.rolling(14).mean().iloc[-1] / curr_p) * 100.0)

        # Fast Indicators
        ema_5 = float(df['close'].ewm(span=5, adjust=False).mean().iloc[-1])
        ema_13 = float(df['close'].ewm(span=13, adjust=False).mean().iloc[-1])

        # 1. AI Score (Threshold 60%)
        quant_score = 50
        ai_buy = False
        if ai_model is not None:
            features = np.array([[fvg_gap_pct, dist_from_vwap, vol_ratio, rsi_14, atr_14]])
            win_prob = ai_model.predict_proba(features)[0][1]
            quant_score = int(win_prob * 100)
            if quant_score >= 60 and curr_p >= curr_vwap:
                ai_buy = True

        # 2. Indicators Pass
        indicators_pass = (ema_5 > ema_13) and (curr_p > curr_vwap) and (45 <= rsi_14 <= 65) and (curr_vol > avg_vol)

        # 3. Master Price Action Scan (Type 4)
        pa_scanner = MasterPriceActionScanner(df)
        pa_signal = pa_scanner.get_latest_signal()

        return {
            "stock": clean_sym,
            "price": curr_p,
            "vwap": curr_vwap,
            "rsi": round(rsi_14, 2),
            "quant_score": quant_score,
            "ai_buy": ai_buy,
            "indicators_pass": indicators_pass,
            "pa_signal": pa_signal,
            "sl": round(curr_p * 0.993, 2),
            "target": round(curr_p * 1.01, 2),
            "s1": s1, "s2": s2,
            "r1": r1, "r2": r2
        }
    except:
        return None

def background_auto_scanner():
    global SENT_ALERTS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS, LATEST_AUTO_SCAN_RESULTS
    ist = timezone(timedelta(hours=5, minutes=30))
    
    while True:
        try:
            ist_now = datetime.now(ist)

            # Daily Reset 09:15 AM
            if ist_now.hour == 9 and ist_now.minute < 15:
                SENT_ALERTS = {"AI": set(), "QUANT": set(), "HYBRID": set(), "PRICE_ACTION": set()}
                ACTIVE_TRADES = {"AI": {}, "QUANT": {}, "HYBRID": {}, "PRICE_ACTION": {}}
                AUDIT_LOGS = {"AI": [], "QUANT": [], "HYBRID": [], "PRICE_ACTION": []}
                AUDIT_SENT_TODAY = False

            # After 03:30 PM: 4 Separate Audit Reports
            if ist_now.hour >= 15 and ist_now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in ["AI", "QUANT", "HYBRID", "PRICE_ACTION"]:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append(f"🟢 {sym}: CLOSED AT 03:30 PM | P&L: ₹0.00 (Exit: ₹{pos['entry']})")
                    ACTIVE_TRADES[strat].clear()

                titles = {
                    "AI": "📊 *[AUDIT 1: PURE AI MODEL REPORT]*",
                    "QUANT": "📊 *[AUDIT 2: QUANT INDICATORS REPORT]*",
                    "HYBRID": "📊 *[AUDIT 3: HYBRID (AI + QUANT) REPORT]*",
                    "PRICE_ACTION": "📊 *[AUDIT 4: INSTITUTIONAL PRICE ACTION REPORT]*"
                }

                for strat in ["AI", "QUANT", "HYBRID", "PRICE_ACTION"]:
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

            # Live Scan
            is_live, _ = check_market_session()
            if is_live:
                live_candidates = []
                for sym in STOCK_UNIVERSE:
                    res = evaluate_stock_full(sym)
                    if not res:
                        continue
                    
                    price = res['price']

                    # 1. TYPE 1: AI ONLY
                    if res['ai_buy'] and sym not in SENT_ALERTS["AI"]:
                        SENT_ALERTS["AI"].add(sym)
                        ACTIVE_TRADES["AI"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                        send_telegram_msg(
                            f"🟢 *[TYPE 1: PURE AI BUY ALERT]*\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 Stock: `{sym}`\n"
                            f"🎯 AI Confidence: `{res['quant_score']}%`\n"
                            f"📥 CMP: ₹{price}\n"
                            f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                            f"🛡️ S1: ₹{res['s1']} | S2: ₹{res['s2']}\n"
                            f"🚧 R1: ₹{res['r1']} | R2: ₹{res['r2']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 2. TYPE 2: QUANT INDICATORS
                    if res['indicators_pass'] and sym not in SENT_ALERTS["QUANT"]:
                        SENT_ALERTS["QUANT"].add(sym)
                        ACTIVE_TRADES["QUANT"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                        send_telegram_msg(
                            f"⚡ *[TYPE 2: QUANT INDICATORS BUY ALERT]*\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 Stock: `{sym}`\n"
                            f"📊 RSI: {res['rsi']} | VWAP: ₹{res['vwap']}\n"
                            f"📥 CMP: ₹{price}\n"
                            f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                            f"🛡️ S1: ₹{res['s1']} | S2: ₹{res['s2']}\n"
                            f"🚧 R1: ₹{res['r1']} | R2: ₹{res['r2']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 3. TYPE 3: HYBRID CONFLUENCE
                    if res['ai_buy'] and res['indicators_pass'] and sym not in SENT_ALERTS["HYBRID"]:
                        SENT_ALERTS["HYBRID"].add(sym)
                        ACTIVE_TRADES["HYBRID"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                        live_candidates.append(res)
                        send_telegram_msg(
                            f"🚀 *[TYPE 3: HIGH CONFIDENCE CONFLUENCE]*\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 Stock: `{sym}` (Dual Verified)\n"
                            f"🎯 AI Score: {res['quant_score']}% | RSI: {res['rsi']}\n"
                            f"📥 CMP: ₹{price}\n"
                            f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                            f"🛡️ S1: ₹{res['s1']} | S2: ₹{res['s2']}\n"
                            f"🚧 R1: ₹{res['r1']} | R2: ₹{res['r2']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 4. TYPE 4: INSTITUTIONAL PRICE ACTION
                    if res['pa_signal'] and sym not in SENT_ALERTS["PRICE_ACTION"]:
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

                    # Target / SL Checker
                    for strat in ["AI", "QUANT", "HYBRID", "PRICE_ACTION"]:
                        if sym in ACTIVE_TRADES[strat]:
                            trade = ACTIVE_TRADES[strat][sym]
                            if price >= trade["target"]:
                                pnl = round(price - trade["entry"], 2)
                                AUDIT_LOGS[strat].append(f"🟢 {sym}: TARGET HIT 🎯 @ {ist_now.strftime('%I:%M %p')} | P&L: +₹{pnl} (Exit: ₹{price})")
                                del ACTIVE_TRADES[strat][sym]
                            elif price <= trade["sl"]:
                                pnl = round(trade["entry"] - price, 2)
                                AUDIT_LOGS[strat].append(f"🔴 {sym}: STOP LOSS HIT 🛑 @ {ist_now.strftime('%I:%M %p')} | P&L: -₹{pnl} (Exit: ₹{price})")
                                del ACTIVE_TRADES[strat][sym]

                if live_candidates:
                    LATEST_AUTO_SCAN_RESULTS = live_candidates

            time.sleep(60)
        except Exception:
            time.sleep(30)

threading.Thread(target=background_auto_scanner, daemon=True).start()

@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today_str = str(ist_now.date())
    if not LATEST_AUTO_SCAN_RESULTS:
        return jsonify([{"date": today_str, "pred": "Scanning NSE Universe", "actual": "Waiting for Setup...", "accuracy": "Active"}])
    best = LATEST_AUTO_SCAN_RESULTS[0]
    return jsonify([{
        "date": today_str,
        "pred": f"{best['stock']} (Dual Confirmed BUY)",
        "actual": f"CMP: ₹{best['price']} | S1: ₹{best['s1']} | R1: ₹{best['r1']}",
        "accuracy": f"AI Score: {best['quant_score']}%"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: 4-in-1 Engine Online!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "system": "4-Tier Institutional Stock AI Engine",
        "strategies": [
            "Type 1: Pure AI Model (60% Threshold)",
            "Type 2: Quant Indicators (RSI/VWAP/EMA 5-13/Vol)",
            "Type 3: Dual Confluence (AI + Indicators)",
            "Type 4: Master Price Action (Wyckoff/SMC/OB/FVG)"
        ],
        "audit_reports": "4 Individual Logs after 03:30 PM"
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
