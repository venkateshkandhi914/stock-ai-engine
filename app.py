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

# ప్రతి టైప్ కి విడివిడిగా ట్రాకింగ్
SENT_ALERTS = {"AI": set(), "QUANT": set(), "HYBRID": set()}
ACTIVE_TRADES = {"AI": {}, "QUANT": {}, "HYBRID": {}}
AUDIT_LOGS = {"AI": [], "QUANT": [], "HYBRID": []}

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
        return False, "నేటి మార్కెట్ సమయం ముగిసింది (03:30 PM)"
    return True, "MARKET_LIVE"

def evaluate_stock_full(symbol):
    clean_sym = symbol.replace('.NS', '').upper()
    try:
        ticker = yf.Ticker(f"{clean_sym}.NS")
        df = ticker.history(period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None

        df.columns = [c.lower() for c in df.columns]
        curr_p = round(float(df['close'].iloc[-1]), 2)

        # Support & Resistance Levels (Pivot Points)
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

        # Fast Indicators (EMA 5 & 13)
        ema_5 = float(df['close'].ewm(span=5, adjust=False).mean().iloc[-1])
        ema_13 = float(df['close'].ewm(span=13, adjust=False).mean().iloc[-1])

        quant_score = 50
        ai_buy = False
        if ai_model is not None:
            features = np.array([[fvg_gap_pct, dist_from_vwap, vol_ratio, rsi_14, atr_14]])
            win_prob = ai_model.predict_proba(features)[0][1]
            quant_score = int(win_prob * 100)
            if quant_score >= 65 and curr_p >= curr_vwap:
                ai_buy = True

        indicators_pass = (ema_5 > ema_13) and (curr_p > curr_vwap) and (45 <= rsi_14 <= 65) and (curr_vol > avg_vol)

        return {
            "stock": clean_sym,
            "price": curr_p,
            "vwap": curr_vwap,
            "rsi": round(rsi_14, 2),
            "quant_score": quant_score,
            "ai_buy": ai_buy,
            "indicators_pass": indicators_pass,
            "sl": round(curr_p * 0.993, 2),
            "target": round(curr_p * 1.01, 2),
            "s1": s1, "s2": s2, "r1": r1, "r2": r2
        }
    except:
        return None

def background_auto_scanner():
    global SENT_ALERTS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS, LATEST_AUTO_SCAN_RESULTS
    ist = timezone(timedelta(hours=5, minutes=30))
    
    while True:
        try:
            ist_now = datetime.now(ist)

            # ప్రతిరోజూ ఉదయం 9:15 కి డేటా క్లియర్
            if ist_now.hour == 9 and ist_now.minute < 15:
                SENT_ALERTS = {"AI": set(), "QUANT": set(), "HYBRID": set()}
                ACTIVE_TRADES = {"AI": {}, "QUANT": {}, "HYBRID": {}}
                AUDIT_LOGS = {"AI": [], "QUANT": [], "HYBRID": []}
                AUDIT_SENT_TODAY = False

            # సాయంత్రం 03:30 PM తర్వాత 3 ప్రత్యేక ఆడిట్ నివేదికలు
            if ist_now.hour >= 15 and ist_now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in ["AI", "QUANT", "HYBRID"]:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append(f"🟢 {sym}: CLOSED AT 03:30 PM | P&L: ₹0.00 (Exit: ₹{pos['entry']})")
                    ACTIVE_TRADES[strat].clear()

                # రిపోర్ట్ 1: Pure AI
                pnl_ai = sum([float(l.split('P&L: ₹')[1].split(' ')[0]) for l in AUDIT_LOGS["AI"] if 'P&L: ₹' in l])
                msg_ai = (
                    f"📊 *[AUDIT 1: PURE AI MODEL REPORT]*\n"
                    f"💰 *NET PROFIT: ₹{round(pnl_ai, 2)}*\n"
                    f"────────────────────\n"
                    f"⏱️ *Audit Logs:*\n" + ("\n".join(AUDIT_LOGS["AI"][-10:]) if AUDIT_LOGS["AI"] else "No Trades Today")
                )
                send_telegram_msg(msg_ai)
                time.sleep(3)

                # రిపోర్ట్ 2: Quant Indicators
                pnl_quant = sum([float(l.split('P&L: ₹')[1].split(' ')[0]) for l in AUDIT_LOGS["QUANT"] if 'P&L: ₹' in l])
                msg_quant = (
                    f"📊 *[AUDIT 2: QUANT INDICATORS REPORT]*\n"
                    f"💰 *NET PROFIT: ₹{round(pnl_quant, 2)}*\n"
                    f"────────────────────\n"
                    f"⏱️ *Audit Logs:*\n" + ("\n".join(AUDIT_LOGS["QUANT"][-10:]) if AUDIT_LOGS["QUANT"] else "No Trades Today")
                )
                send_telegram_msg(msg_quant)
                time.sleep(3)

                # రిపోర్ట్ 3: Hybrid (AI + Indicators)
                pnl_hyb = sum([float(l.split('P&L: ₹')[1].split(' ')[0]) for l in AUDIT_LOGS["HYBRID"] if 'P&L: ₹' in l])
                msg_hyb = (
                    f"📊 *[AUDIT 3: HYBRID (AI + QUANT) REPORT]*\n"
                    f"💰 *NET PROFIT: ₹{round(pnl_hyb, 2)}*\n"
                    f"────────────────────\n"
                    f"⏱️ *Audit Logs:*\n" + ("\n".join(AUDIT_LOGS["HYBRID"][-10:]) if AUDIT_LOGS["HYBRID"] else "No Trades Today")
                )
                send_telegram_msg(msg_hyb)

                AUDIT_SENT_TODAY = True

            # లైవ్ మార్కెట్ స్కాన్
            is_live, _ = check_market_session()
            if is_live:
                live_candidates = []
                for sym in STOCK_UNIVERSE:
                    res = evaluate_stock_full(sym)
                    if not res:
                        continue
                    
                    price = res['price']

                    # 1. TYPE 1: Pure AI Alert
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

                    # 2. TYPE 2: Quant Indicators Alert
                    if res['indicators_pass'] and sym not in SENT_ALERTS["QUANT"]:
                        SENT_ALERTS["QUANT"].add(sym)
                        ACTIVE_TRADES["QUANT"][sym] = {"entry": price, "sl": res['sl'], "target": res['target']}
                        send_telegram_msg(
                            f"⚡ *[TYPE 2: QUANT INDICATORS ALERT]*\n"
                            f"━━━━━━━━━━━━━━━━━━━━\n"
                            f"📌 Stock: `{sym}`\n"
                            f"📊 RSI: {res['rsi']} | VWAP: ₹{res['vwap']}\n"
                            f"📥 CMP: ₹{price}\n"
                            f"🛑 SL: ₹{res['sl']} | 🎯 Target: ₹{res['target']}\n"
                            f"🛡️ S1: ₹{res['s1']} | S2: ₹{res['s2']}\n"
                            f"🚧 R1: ₹{res['r1']} | R2: ₹{res['r2']}\n"
                            f"━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 3. TYPE 3: Hybrid Confluence Alert
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

                    # ఎగ్జిట్ ట్రాకింగ్ (Target & SL చెకింగ్)
                    for strat in ["AI", "QUANT", "HYBRID"]:
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
        return jsonify([{"date": today_str, "pred": "Scanning", "actual": "Waiting for Confluence...", "accuracy": "Active"}])
    best = LATEST_AUTO_SCAN_RESULTS[0]
    return jsonify([{
        "date": today_str,
        "pred": f"{best['stock']} (Dual Confirmed BUY)",
        "actual": f"CMP: ₹{best['price']} | S1: ₹{best['s1']} | R1: ₹{best['r1']}",
        "accuracy": f"AI Score: {best['quant_score']}%"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: Bot connection verified!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "system": "3-Category Stock AI Engine",
        "features": ["3 Separate Audits", "S1/S2 Supports", "R1/R2 Resistances", "Fast EMA 5/13"]
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
