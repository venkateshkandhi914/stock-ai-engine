import os
import sys
import threading
import time
import requests
import json
import numpy as np
import pandas as pd
import joblib
import yfinance as yf
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify, request

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

SENT_ALERTS_TODAY = set()
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

def evaluate_stock_ai(symbol):
    clean_sym = symbol.replace('.NS', '').upper()
    try:
        ticker = yf.Ticker(f"{clean_sym}.NS")
        df = ticker.history(period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None
            
        df.columns = [c.lower() for c in df.columns]
        curr_p = round(float(df['close'].iloc[-1]), 2)
        
        c1_high = df['high'].iloc[-3]
        c3_low = df['low'].iloc[-1]
        fvg_gap_pct = ((c3_low - c1_high) / curr_p) * 100.0

        tp = (df['high'] + df['low'] + df['close']) / 3.0
        cum_vol = df['volume'].rolling(75).sum()
        vwap = (tp * df['volume']).rolling(75).sum() / (cum_vol + 1e-9)
        curr_vwap = round(float(vwap.iloc[-1]), 2)
        dist_from_vwap = ((curr_p - curr_vwap) / curr_vwap) * 100.0

        avg_vol = df['volume'].rolling(20).mean().iloc[-1]
        vol_ratio = float(df['volume'].iloc[-1] / (avg_vol + 1e-9))

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

        quant_score = 50
        decision = "AVOID"
        if ai_model is not None:
            features = np.array([[fvg_gap_pct, dist_from_vwap, vol_ratio, rsi_14, atr_14]])
            win_prob = ai_model.predict_proba(features)[0][1]
            quant_score = int(win_prob * 100)

        sl = round(curr_p * 0.995, 2)
        t1 = round(curr_p * 1.008, 2)
        t2 = round(curr_p * 1.012, 2)

        if quant_score >= 65 and curr_p >= curr_vwap:
            decision = "BUY (AI Confirmed)"
        elif quant_score <= 35 and curr_p <= curr_vwap:
            decision = "SELL (AI Bearish)"
            sl = round(curr_p * 1.005, 2)
            t1 = round(curr_p * 0.992, 2)
            t2 = round(curr_p * 0.988, 2)

        return {
            "stock": clean_sym,
            "price": curr_p,
            "quant_score": quant_score,
            "decision": decision,
            "entry_zone": f"₹{curr_p}",
            "stop_loss": sl,
            "target_1": t1,
            "target_2": t2,
            "vwap": curr_vwap
        }
    except:
        return None

def background_auto_scanner():
    global SENT_ALERTS_TODAY, LATEST_AUTO_SCAN_RESULTS
    while True:
        try:
            is_live, _ = check_market_session()
            ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
            if ist_now.hour == 9 and ist_now.minute < 20:
                SENT_ALERTS_TODAY.clear()

            if is_live:
                live_candidates = []
                for sym in STOCK_UNIVERSE:
                    res = evaluate_stock_ai(sym)
                    if res and "BUY" in res['decision']:
                        live_candidates.append(res)
                        if sym not in SENT_ALERTS_TODAY:
                            msg = (
                                f"🟢 *AI AUTO-ALERT: HIGH CONFLUENCE*\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"📌 *స్టాక్:* `{res['stock']}`\n"
                                f"🎯 *AI Win Score:* `{res['quant_score']}%`\n"
                                f"⚡ *సిగ్నల్:* *{res['decision']}*\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"📥 *ధర (CMP):* ₹{res['price']}\n"
                                f"🛑 *స్టాప్‌లాస్:* ₹{res['stop_loss']}\n"
                                f"🎯 *టార్గెట్ 1:* ₹{res['target_1']}\n"
                                f"🚀 *టార్గెట్ 2:* ₹{res['target_2']}\n"
                                f"━━━━━━━━━━━━━━━━━━━━\n"
                                f"🤖 _Render Cloud AI Engine Active_"
                            )
                            if send_telegram_msg(msg):
                                SENT_ALERTS_TODAY.add(sym)

                LATEST_AUTO_SCAN_RESULTS = live_candidates
            time.sleep(120)
        except:
            time.sleep(30)

threading.Thread(target=background_auto_scanner, daemon=True).start()

@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    today_str = str(ist_now.date())
    
    if not LATEST_AUTO_SCAN_RESULTS:
        return jsonify([{
            "date": today_str,
            "pred": "మార్కెట్ స్కాన్ అవుతోంది",
            "actual": "సెటప్ కోసం వెయిటింగ్...",
            "accuracy": "AI Active"
        }])

    best = LATEST_AUTO_SCAN_RESULTS[0]
    return jsonify([{
        "date": today_str,
        "pred": f"{best['stock']} ({best['decision']})",
        "actual": f"CMP: ₹{best['price']} | SL: ₹{best['stop_loss']} | TGT: ₹{best['target_1']}",
        "accuracy": f"AI Score: {best['quant_score']}%"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: టెలిగ్రామ్ బాట్ విజయవంతంగా కనెక్ట్ అయింది!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return "Stock AI Engine 24/7 Live on Cloud!"

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)

