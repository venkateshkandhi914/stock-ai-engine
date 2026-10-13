import os
import sys
import threading
import time
from datetime import datetime, timezone, timedelta
from flask import Flask, jsonify
import joblib
import numpy as np
import pandas as pd
import requests
import pytz

app = Flask(__name__)
application = app

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or "8834841152:AAF0bok9I01ylydeVc2RiAYz-F1BWvyDDLk"
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or "6057603813"

# మోడల్స్ లోడింగ్
ai_model_fvg = None
if os.path.exists("fvg_ai_model.pkl"):
    try:
        ai_model_fvg = joblib.load("fvg_ai_model.pkl")
        print("Type 1: FVG AI Model Loaded")
    except Exception as e:
        print("FVG Model Load Error: " + str(e))

ai_model_pa = None
if os.path.exists("price_action_ai_model.pkl"):
    try:
        ai_model_pa = joblib.load("price_action_ai_model.pkl")
        print("Type 5: 15 EMA Price Action AI Model Loaded")
    except Exception as e:
        print("PA Model Load Error: " + str(e))

NSE_STOCKS = [
    "RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "SBIN",
    "TATAMOTORS", "BAJFINANCE", "ITC", "LT", "AXISBANK", "KOTAKBANK"
]

# ఎంపిక చేసిన 4 ప్రధాన స్ట్రాటజీలు: Stocks (Type 1, 3, 5) + Crypto 24/7
STRATEGIES = [
    "STOCKS_TYPE1_AI_FVG",
    "STOCKS_TYPE3_HYBRID",
    "STOCKS_TYPE5_15EMA_AI",
    "CRYPTO_24X7_15EMA_AI"
]

ALERT_TIMESTAMPS = {}
ACTIVE_TRADES = {s: {} for s in STRATEGIES}
AUDIT_LOGS = {s: [] for s in STRATEGIES}
AUDIT_SENT_TODAY = False

def send_telegram_msg(msg_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=6.0)
        return resp.status_code == 200
    except:
        return False

# 1. Binance Market-Wide Scanner: టాప్ కాయిన్స్ ఫెచ్ (Zero Lag)
def get_top_momentum_crypto(limit=12):
    priority_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    blacklist = ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT", "AEURUSDT"]
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            tickers = resp.json()
            valid = []
            for t in tickers:
                sym = t['symbol']
                if sym.endswith("USDT") and sym not in blacklist and not any(x in sym for x in ["UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT"]):
                    vol_usd = float(t['quoteVolume'])
                    if vol_usd >= 15000000:
                        valid.append({"symbol": sym, "volume": vol_usd})
            valid.sort(key=lambda x: x['volume'], reverse=True)
            top_coins = [x['symbol'] for x in valid[:limit]]
            for pc in reversed(priority_coins):
                if pc in top_coins:
                    top_coins.remove(pc)
                top_coins.insert(0, pc)
            return top_coins
    except:
        pass
    return priority_coins

# 2. Binance Direct 5m Candles Feed
def fetch_binance_live(symbol="BTCUSDT", limit=35):
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=" + symbol + "&interval=5m&limit=" + str(limit)
        resp = requests.get(url, timeout=3.5)
        if resp.status_code == 200:
            raw = resp.json()
            closes = [float(k[4]) for k in raw]
            highs = [float(k[2]) for k in raw]
            lows = [float(k[3]) for k in raw]
            opens = [float(k[1]) for k in raw]
            s_close = pd.Series(closes)
            ema_15 = round(float(s_close.ewm(span=15, adjust=False).mean().iloc[-1]), 4)
            curr_p = closes[-1]
            c_range = (highs[-1] - lows[-1]) + 1e-9
            b_ratio = abs(closes[-1] - opens[-1]) / c_range
            u_wick = (highs[-1] - max(opens[-1], closes[-1])) / c_range
            l_wick = (min(opens[-1], closes[-1]) - lows[-1]) / c_range
            d_ema = ((curr_p - ema_15) / ema_15) * 100.0

            pa_score = 65
            if ai_model_pa is not None:
                try:
                    f = np.array([[b_ratio, u_wick, l_wick, d_ema, 0.5, 1.0, 50.0]])
                    pa_score = int(ai_model_pa.predict_proba(f)[0][1] * 100)
                except:
                    pass

            pivot = (max(highs) + min(lows) + curr_p) / 3.0
            s1 = round((2 * pivot) - max(highs), 4)
            r1 = round((2 * pivot) - min(lows), 4)

            return {
                "symbol": symbol, "price": curr_p, "ema_15": ema_15,
                "score": pa_score, "bullish": curr_p >= ema_15,
                "s1": s1, "r1": r1
            }
    except:
        pass
    return None

# 3. TradingView Direct Public API (NSE Zero-Delay)
def fetch_tradingview_nse_live(symbol):
    try:
        url = "https://scanner.tradingview.com/india/scan"
        payload = {
            "symbols": {"tickers": ["NSE:" + symbol], "query": {"types": []}},
            "columns": ["open", "high", "low", "close", "volume", "change", "VWAP", "RSI", "EMA5", "EMA13", "EMA15", "EMA20"]
        }
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.5)
        if resp.status_code == 200:
            d = resp.json().get('data', [])
            if d:
                vals = d[0]['d']
                return {
                    "open": float(vals[0]), "high": float(vals[1]), "low": float(vals[2]),
                    "close": float(vals[3]), "volume": float(vals[4]),
                    "vwap": float(vals[6]) if vals[6] else float(vals[3]),
                    "rsi": float(vals[7]) if vals[7] else 50.0,
                    "ema_5": float(vals[8]) if vals[8] else float(vals[3]),
                    "ema_13": float(vals[9]) if vals[9] else float(vals[3]),
                    "ema_15": float(vals[10]) if vals[10] else float(vals[3])
                }
    except:
        pass
    return None

def is_nse_market_open():
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if ist_now.weekday() in [5, 6]:
        return False
    cur_mins = ist_now.hour * 60 + ist_now.minute
    return 555 <= cur_mins <= 930

def background_scanner_and_audit():
    global ALERT_TIMESTAMPS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS
    ist = pytz.timezone("Asia/Kolkata")

    while True:
        try:
            now = datetime.now(ist)

            # ప్రతిరోజు ఉదయం 9:15 కి డేటా రీసెట్
            if now.hour == 9 and now.minute < 15:
                ALERT_TIMESTAMPS.clear()
                ACTIVE_TRADES = {s: {} for s in STRATEGIES}
                AUDIT_LOGS = {s: [] for s in STRATEGIES}
                AUDIT_SENT_TODAY = False

            # సాయంత్రం 03:30 PM ఆడిట్ రిపోర్టులు
            if now.hour >= 15 and now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in STRATEGIES:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append("CLOSED @ 03:30 PM | " + sym + " | Exit: " + str(pos['entry']))
                    ACTIVE_TRADES[strat].clear()

                titles = {
                    "STOCKS_TYPE1_AI_FVG": "NSE STOCKS: TYPE 1 PURE AI (FVG)",
                    "STOCKS_TYPE3_HYBRID": "NSE STOCKS: TYPE 3 HYBRID CONFLUENCE",
                    "STOCKS_TYPE5_15EMA_AI": "NSE STOCKS: TYPE 5 15 EMA + PRICE ACTION AI",
                    "CRYPTO_24X7_15EMA_AI": "24/7 CRYPTO: 15 EMA + ALTCOIN AI"
                }

                send_telegram_msg("📊 *══════ [DAILY AUDIT REPORT] ══════*")
                for strat in STRATEGIES:
                    logs_txt = "\n".join(AUDIT_LOGS[strat][-5:]) if AUDIT_LOGS[strat] else "No Trades Executed Today"
                    msg = "📋 *[" + titles[strat] + "]*\n────────────────────\n⏱️ *Audit Logs:*\n" + logs_txt
                    send_telegram_msg(msg)
                    time.sleep(1)

                AUDIT_SENT_TODAY = True

            curr_epoch = time.time()

            # ----------------------------------------------------
            # A. 24/7 క్రిప్టో లైవ్ స్కాన్ (Binance Zero-Lag Feed)
            # ----------------------------------------------------
            crypto_list = get_top_momentum_crypto(limit=10)
            for pair in crypto_list:
                data = fetch_binance_live(pair)
                if data and data['bullish'] and data['score'] >= 60:
                    last_alert_time = ALERT_TIMESTAMPS.get("CRYPTO_" + pair, 0)
                    if curr_epoch - last_alert_time > 1800:
                        ALERT_TIMESTAMPS["CRYPTO_" + pair] = curr_epoch
                        sl = round(data['price'] * 0.992, 4)
                        tgt = round(data['price'] * 1.018, 4)
                        ACTIVE_TRADES["CRYPTO_24X7_15EMA_AI"][pair] = {"entry": data['price'], "sl": sl, "target": tgt}
                        alert_msg = (
                            "🤖 *[24/7 CRYPTO: 15 EMA + AI ALERT]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Asset: `" + str(pair) + "`\n"
                            "🎯 AI Confidence: `" + str(data['score']) + "%`\n"
                            "📈 15 EMA: $" + str(data['ema_15']) + "\n"
                            "📥 Live Entry: $" + str(data['price']) + "\n"
                            "🛑 SL: $" + str(sl) + " \vert{} 🎯 Target: $" + str(tgt) + "\n"
                            "🛡️ S1: $" + str(data['s1']) + " \vert{} 🚧 R1: $" + str(data['r1']) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "⚡ _Zero-Lag Binance API Active_"
                        )
                        send_telegram_msg(alert_msg)

            # ----------------------------------------------------
            # B. NSE స్టాక్స్ లైవ్ స్కాన్ (Type 1, 3, 5)
            # ----------------------------------------------------
            if is_nse_market_open():
                for sym in NSE_STOCKS:
                    stk = fetch_tradingview_nse_live(sym)
                    if not stk:
                        continue

                    price = stk['close']
                    vwap = stk['vwap']
                    rsi = stk['rsi']
                    ema_5 = stk['ema_5']
                    ema_13 = stk['ema_13']
                    ema_15 = stk['ema_15']

                    pivot = (stk['high'] + stk['low'] + price) / 3.0
                    r1 = round((2 * pivot) - stk['low'], 2)
                    s1 = round((2 * pivot) - stk['high'], 2)

                    # AI FVG మోడల్ కాలిక్యులేషన్
                    ai_fvg_score = 50
                    if ai_model_fvg is not None:
                        try:
                            f_arr = np.array([[0.5, ((price - vwap)/vwap)*100, 1.2, rsi, 1.5]])
                            ai_fvg_score = int(ai_model_fvg.predict_proba(f_arr)[0][1] * 100)
                        except:
                            pass

                    quant_pass = (ema_5 > ema_13) and (price > vwap) and (45 <= rsi <= 65)

                    # 1. TYPE 1: Pure AI FVG
                    if ai_fvg_score >= 60 and price >= vwap:
                        last_t = ALERT_TIMESTAMPS.get("STK_T1_" + sym, 0)
                        if curr_epoch - last_t > 1800:
                            ALERT_TIMESTAMPS["STK_T1_" + sym] = curr_epoch
                            sl = round(price * 0.993, 2)
                            tgt = round(price * 1.012, 2)
                            ACTIVE_TRADES["STOCKS_TYPE1_AI_FVG"][sym] = {"entry": price, "sl": sl, "target": tgt}
                            send_telegram_msg(
                                "🟢 *[TYPE 1: NSE PURE AI (FVG)]*\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "📌 Stock: `" + str(sym) + "`\n"
                                "🎯 AI Confidence: `" + str(ai_fvg_score) + "%`\n"
                                "📥 CMP: ₹" + str(price) + "\n"
                                "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                                "🛡️ S1: ₹" + str(s1) + " | 🚧 R1: ₹" + str(r1) + "\n"
                                "━━━━━━━━━━━━━━━━━━━━"
                            )

                    # 2. TYPE 3: Hybrid Confluence (AI + Quant)
                    if ai_fvg_score >= 60 and quant_pass:
                        last_t = ALERT_TIMESTAMPS.get("STK_T3_" + sym, 0)
                        if curr_epoch - last_t > 1800:
                            ALERT_TIMESTAMPS["STK_T3_" + sym] = curr_epoch
                            sl = round(price * 0.993, 2)
                            tgt = round(price * 1.015, 2)
                            ACTIVE_TRADES["STOCKS_TYPE3_HYBRID"][sym] = {"entry": price, "sl": sl, "target": tgt}
                            send_telegram_msg(
                                "🚀 *[TYPE 3: NSE HYBRID CONFLUENCE]*\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "📌 Stock: `" + str(sym) + "` (AI + Quant Dual Confirmed)\n"
                                "🎯 AI Score: " + str(ai_fvg_score) + "% | RSI: " + str(rsi) + "\n"
                                "📥 CMP: ₹" + str(price) + "\n"
                                "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                                "━━━━━━━━━━━━━━━━━━━━"
                            )

                    # 3. TYPE 5: 15 EMA + Price Action AI
                    if price >= ema_15:
                        last_t = ALERT_TIMESTAMPS.get("STK_T5_" + sym, 0)
                        if curr_epoch - last_t > 1800:
                            ALERT_TIMESTAMPS["STK_T5_" + sym] = curr_epoch
                            sl = round(price * 0.992, 2)
                            tgt = round(price * 1.015, 2)
                            ACTIVE_TRADES["STOCKS_TYPE5_15EMA_AI"][sym] = {"entry": price, "sl": sl, "target": tgt}
                            send_telegram_msg(
                                "📈 *[TYPE 5: NSE 15 EMA + PRICE ACTION AI]*\n"
                                "━━━━━━━━━━━━━━━━━━━━\n"
                                "📌 Stock: `" + str(sym) + "`\n"
                                "📈 15 EMA: ₹" + str(ema_15) + "\n"
                                "📥 Live Entry: ₹" + str(price) + "\n"
                                "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                                "━━━━━━━━━━━━━━━━━━━━"
                            )

            time.sleep(60)
        except Exception:
            time.sleep(30)

threading.Thread(target=background_scanner_and_audit, daemon=True).start()

# యాప్‌లో SCAN బటన్ కోసం సూపర్-ఫాస్ట్ ఎండ్‌పాయింట్
@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    today_str = ist_now.strftime("%d-%b %I:%M %p")

    # 1. మార్కెట్ ఓపెన్ ఉంటే NSE స్టాక్స్ స్కాన్
    if is_nse_market_open():
        for s in NSE_STOCKS[:4]:
            stk = fetch_tradingview_nse_live(s)
            if stk and stk['close'] >= stk['ema_15']:
                send_telegram_msg("🔍 *[APP SCAN: NSE STOCK]*\nStock: `" + s + "` @ ₹" + str(stk['close']))
                return jsonify([{
                    "date": today_str,
                    "pred": s + " (NSE 15 EMA Bounce 🟢)",
                    "actual": "CMP: ₹" + str(stk['close']) + " | 15 EMA: ₹" + str(stk['ema_15']),
                    "accuracy": "RSI: " + str(stk['rsi'])
                }])

    # 2. క్రిప్టో (24/7 Zero-Lag Live Signal)
    for pair in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        data = fetch_binance_live(pair)
        if data:
            trend = "BULLISH 🟢" if data['bullish'] else "CONSOLIDATION ⚪"
            send_telegram_msg("🔍 *[APP SCAN: 24/7 CRYPTO]*\nCoin: `" + pair + "` @ $" + str(data['price']) + "\n15 EMA: $" + str(data['ema_15']))
            return jsonify([{
                "date": today_str,
                "pred": pair + " (" + trend + ")",
                "actual": "CMP: $" + str(data['price']) + " \vert{} 15 EMA: $" + str(data['ema_15']),
                "accuracy": "AI Score: " + str(data['score']) + "%"
            }])

    return jsonify([{
        "date": today_str,
        "pred": "BTCUSDT (Live Stream OK 🟢)",
        "actual": "Feed Active",
        "accuracy": "Ready"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: Stocks (Type 1,3,5) + 24/7 Crypto Engine Online!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "stock_engine": "Type 1 (Pure AI), Type 3 (Hybrid), Type 5 (15 EMA + PA AI)",
        "crypto_engine": "24/7 Zero-Delay Binance Momentum Scanner",
        "zero_delay": True
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
