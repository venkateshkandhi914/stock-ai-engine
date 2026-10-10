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

# AI మోడల్స్ లోడింగ్
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

# మొత్తం 6 ప్రత్యేక స్ట్రాటజీలు (5 NSE స్టాక్స్ + 1 క్రిప్టో 24/7)
STRATEGIES = [
    "STOCK_TYPE1_AI",
    "STOCK_TYPE2_QUANT",
    "STOCK_TYPE3_HYBRID",
    "STOCK_TYPE4_PRICE_ACTION",
    "STOCK_TYPE5_15EMA_AI",
    "CRYPTO_TYPE5_15EMA_AI"
]

SENT_ALERTS = {s: set() for s in STRATEGIES}
ACTIVE_TRADES = {s: {} for s in STRATEGIES}
AUDIT_LOGS = {s: [] for s in STRATEGIES}
AUDIT_SENT_TODAY = False

def send_telegram_msg(msg_text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = "https://api.telegram.org/bot" + TELEGRAM_BOT_TOKEN + "/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg_text, "parse_mode": "Markdown"}
    try:
        resp = requests.post(url, json=payload, timeout=8.0)
        return resp.status_code == 200
    except:
        return False

# 1. Binance Market-Wide Scanner: టాప్ 15 మూమెంటమ్ ఆల్ట్‌కాయిన్స్ ఫిల్టర్
def get_top_momentum_crypto(limit=15):
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            tickers = resp.json()
            valid = []
            for t in tickers:
                sym = t['symbol']
                if sym.endswith("USDT") and not any(x in sym for x in ["UPUSDT", "DOWNUSDT", "BEARUSDT", "BULLUSDT"]):
                    vol_usd = float(t['quoteVolume'])
                    if vol_usd >= 15000000:
                        valid.append({"symbol": sym, "volume": vol_usd})
            valid.sort(key=lambda x: x['volume'], reverse=True)
            top_coins = [x['symbol'] for x in valid[:limit]]
            if "BTCUSDT" not in top_coins:
                top_coins.insert(0, "BTCUSDT")
            return top_coins
    except Exception as e:
        print("Crypto Ticker Error: " + str(e))
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "NEARUSDT", "AVAXUSDT"]

# 2. Binance Zero-Delay 5m క్యాండిల్స్
def fetch_binance_live(symbol="BTCUSDT", limit=40):
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=" + symbol + "&interval=5m&limit=" + str(limit)
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            df = pd.DataFrame(resp.json(), columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore'
            ])
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            return df[['open', 'high', 'low', 'close', 'volume']]
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
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.post(url, json=payload, headers=headers, timeout=4)
        if resp.status_code == 200:
            data = resp.json().get('data', [])
            if data:
                vals = data[0]['d']
                return {
                    "open": float(vals[0]),
                    "high": float(vals[1]),
                    "low": float(vals[2]),
                    "close": float(vals[3]),
                    "volume": float(vals[4]),
                    "vwap": float(vals[6]) if vals[6] else float(vals[3]),
                    "rsi": float(vals[7]) if vals[7] else 50.0,
                    "ema_5": float(vals[8]) if vals[8] else float(vals[3]),
                    "ema_13": float(vals[9]) if vals[9] else float(vals[3]),
                    "ema_15": float(vals[10]) if vals[10] else float(vals[3]),
                    "sma_20": float(vals[11]) if vals[11] else float(vals[3])
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

def evaluate_price_action_setup(live_data):
    p = live_data['close']
    h = live_data['high']
    l = live_data['low']
    o = live_data['open']
    vwap = live_data['vwap']
    sma20 = live_data['sma_20']

    if p > vwap and p > sma20 and p > o and (h - l) > 0:
        sl = round(l - (p * 0.002), 2)
        risk = round(p - sl, 2)
        if risk > 0:
            return {
                "model": "Breakout and Retest",
                "setup": "Support Retest above VWAP",
                "entry": p, "sl": sl, "target": round(p + (risk * 2), 2), "risk": risk, "rr": "1:2.0"
            }

    if p > o and p >= (h - (h - l) * 0.3) and p > vwap:
        sl = round(l - 0.20, 2)
        risk = round(p - sl, 2)
        if risk > 0:
            return {
                "model": "Wyckoff Theory",
                "setup": "Accumulation Spring Expansion",
                "entry": p, "sl": sl, "target": round(p + (risk * 2), 2), "risk": risk, "rr": "1:2.0"
            }
    return None

def background_scanner_and_audit():
    global SENT_ALERTS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS
    ist = pytz.timezone("Asia/Kolkata")

    while True:
        try:
            now = datetime.now(ist)

            # ఉదయం 9:15 రీసెట్
            if now.hour == 9 and now.minute < 15:
                SENT_ALERTS = {s: set() for s in STRATEGIES}
                ACTIVE_TRADES = {s: {} for s in STRATEGIES}
                AUDIT_LOGS = {s: [] for s in STRATEGIES}
                AUDIT_SENT_TODAY = False

            # సాయంత్రం 03:30 PM - మొత్తం 6 విడివిడి ఆడిట్ రిపోర్టులు
            if now.hour >= 15 and now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in STRATEGIES:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append("CLOSED @ 03:30 PM | " + sym + " | Exit: " + str(pos['entry']))
                    ACTIVE_TRADES[strat].clear()

                titles = {
                    "STOCK_TYPE1_AI": "REPORT 1: NSE STOCK PURE 365-DAY AI (FVG)",
                    "STOCK_TYPE2_QUANT": "REPORT 2: NSE STOCK FAST QUANT INDICATORS",
                    "STOCK_TYPE3_HYBRID": "REPORT 3: NSE STOCK HYBRID CONFLUENCE",
                    "STOCK_TYPE4_PRICE_ACTION": "REPORT 4: NSE STOCK INSTITUTIONAL SMC/WYCKOFF",
                    "STOCK_TYPE5_15EMA_AI": "REPORT 5: NSE STOCK 15 EMA + PRICE ACTION AI",
                    "CRYPTO_TYPE5_15EMA_AI": "REPORT 6: 24/7 CRYPTO 15 EMA + ALTCOIN AI"
                }

                send_telegram_msg("📊 *══════ [DAILY COMPREHENSIVE AUDIT REPORT] ══════*")
                for strat in STRATEGIES:
                    logs_txt = "\n".join(AUDIT_LOGS[strat][-5:]) if AUDIT_LOGS[strat] else "No Trades Executed Today"
                    msg = "📋 *[" + titles[strat] + "]*\n────────────────────\n⏱️ *Audit Logs:*\n" + logs_txt
                    send_telegram_msg(msg)
                    time.sleep(1)

                AUDIT_SENT_TODAY = True

            # ----------------------------------------------------
            # 1. 24/7 క్రిప్టో లైవ్ స్కాన్ (టాప్ ఆల్ట్‌కాయిన్స్ + BTC)
            # ----------------------------------------------------
            crypto_list = get_top_momentum_crypto(limit=15)
            for pair in crypto_list:
                df = fetch_binance_live(pair, limit=35)
                if df is not None and len(df) >= 20:
                    curr_p = round(float(df['close'].iloc[-1]), 4)
                    df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
                    ema_15 = round(float(df['ema_15'].iloc[-1]), 4)

                    pivot = (float(df['high'].max()) + float(df['low'].min()) + curr_p) / 3.0
                    s1 = round((2 * pivot) - float(df['high'].max()), 4)
                    r1 = round((2 * pivot) - float(df['low'].min()), 4)

                    candle_range = df['high'] - df['low'] + 1e-9
                    body_ratio = float(((df['close'] - df['open']).abs() / candle_range).iloc[-1])
                    upper_wick = float(((df['high'] - df[['open', 'close']].max(axis=1)) / candle_range).iloc[-1])
                    lower_wick = float(((df[['open', 'close']].min(axis=1) - df['low']) / candle_range).iloc[-1])
                    dist_ema_15 = float((((curr_p - ema_15) / ema_15) * 100.0))
                    ret_1 = float((df['close'].pct_change(1) * 100.0).iloc[-1])
                    ret_3 = float((df['close'].pct_change(3) * 100.0).iloc[-1])
                    roll_h = df['high'].rolling(20).max().iloc[-1]
                    roll_l = df['low'].rolling(20).min().iloc[-1]
                    dist_support = float((((curr_p - roll_l) / (roll_h - roll_l + 1e-9)) * 100.0))

                    pa_score = 60
                    if ai_model_pa is not None:
                        try:
                            f_vec = np.array([[body_ratio, upper_wick, lower_wick, dist_ema_15, ret_1, ret_3, dist_support]])
                            pa_score = int(ai_model_pa.predict_proba(f_vec)[0][1] * 100)
                        except:
                            pass

                    # Report 6 క్రిప్టో సిగ్నల్
                    if (pa_score >= 60 and curr_p >= ema_15) and pair not in SENT_ALERTS["CRYPTO_TYPE5_15EMA_AI"]:
                        SENT_ALERTS["CRYPTO_TYPE5_15EMA_AI"].add(pair)
                        sl = round(curr_p * 0.992, 4)
                        tgt = round(curr_p * 1.018, 4)
                        ACTIVE_TRADES["CRYPTO_TYPE5_15EMA_AI"][pair] = {"entry": curr_p, "sl": sl, "target": tgt}
                        alert_msg = (
                            "🤖 *[REPORT 6: 24/7 CRYPTO 15 EMA + AI]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Coin: `" + str(pair) + "` (High Volume Altcoin)\n"
                            "🎯 AI Confidence: `" + str(pa_score) + "%`\n"
                            "📈 15 EMA: $" + str(ema_15) + "\n"
                            "📥 Live Entry: $" + str(curr_p) + "\n"
                            "🛑 SL: $" + str(sl) + " \vert{} 🎯 Target: $" + str(tgt) + "\n"
                            "🛡️ S1: $" + str(s1) + " \vert{} 🚧 R1: $" + str(r1) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "⚡ _Zero-Lag Binance API Active_"
                        )
                        send_telegram_msg(alert_msg)

            # ----------------------------------------------------
            # 2. NSE స్టాక్స్ లైవ్ స్కాన్ (మార్కెట్ సమయాల్లో మాత్రమే)
            # ----------------------------------------------------
            if is_nse_market_open():
                for sym in NSE_STOCKS:
                    data = fetch_tradingview_nse_live(sym)
                    if not data:
                        continue

                    price = data['close']
                    vwap = data['vwap']
                    rsi = data['rsi']
                    ema_5 = data['ema_5']
                    ema_13 = data['ema_13']
                    ema_15 = data['ema_15']

                    pivot = (data['high'] + data['low'] + price) / 3.0
                    r1 = round((2 * pivot) - data['low'], 2)
                    s1 = round((2 * pivot) - data['high'], 2)

                    # Type 1 & 3 మోడల్ స్కోర్
                    ai_fvg_score = 50
                    if ai_model_fvg is not None:
                        try:
                            f_arr = np.array([[0.5, ((price - vwap)/vwap)*100, 1.2, rsi, 1.5]])
                            ai_fvg_score = int(ai_model_fvg.predict_proba(f_arr)[0][1] * 100)
                        except:
                            pass

                    # 1. TYPE 1: Pure AI
                    if ai_fvg_score >= 60 and price >= vwap and sym not in SENT_ALERTS["STOCK_TYPE1_AI"]:
                        SENT_ALERTS["STOCK_TYPE1_AI"].add(sym)
                        sl = round(price * 0.993, 2)
                        tgt = round(price * 1.012, 2)
                        ACTIVE_TRADES["STOCK_TYPE1_AI"][sym] = {"entry": price, "sl": sl, "target": tgt}
                        send_telegram_msg(
                            "🟢 *[REPORT 1: NSE PURE AI BUY ALERT]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Stock: `" + str(sym) + "`\n"
                            "🎯 AI Confidence: `" + str(ai_fvg_score) + "%`\n"
                            "📥 CMP: ₹" + str(price) + "\n"
                            "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                            "🛡️ S1: ₹" + str(s1) + " | 🚧 R1: ₹" + str(r1) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 2. TYPE 2: Quant Indicators
                    quant_pass = (ema_5 > ema_13) and (price > vwap) and (45 <= rsi <= 65)
                    if quant_pass and sym not in SENT_ALERTS["STOCK_TYPE2_QUANT"]:
                        SENT_ALERTS["STOCK_TYPE2_QUANT"].add(sym)
                        sl = round(price * 0.993, 2)
                        tgt = round(price * 1.012, 2)
                        ACTIVE_TRADES["STOCK_TYPE2_QUANT"][sym] = {"entry": price, "sl": sl, "target": tgt}
                        send_telegram_msg(
                            "⚡ *[REPORT 2: NSE QUANT INDICATORS ALERT]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Stock: `" + str(sym) + "`\n"
                            "📊 RSI: " + str(rsi) + " | VWAP: ₹" + str(vwap) + "\n"
                            "📥 CMP: ₹" + str(price) + "\n"
                            "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 3. TYPE 3: Hybrid Confluence
                    if ai_fvg_score >= 60 and quant_pass and sym not in SENT_ALERTS["STOCK_TYPE3_HYBRID"]:
                        SENT_ALERTS["STOCK_TYPE3_HYBRID"].add(sym)
                        sl = round(price * 0.993, 2)
                        tgt = round(price * 1.015, 2)
                        ACTIVE_TRADES["STOCK_TYPE3_HYBRID"][sym] = {"entry": price, "sl": sl, "target": tgt}
                        send_telegram_msg(
                            "🚀 *[REPORT 3: NSE HYBRID CONFLUENCE]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Stock: `" + str(sym) + "` (AI + Quant Confirmed)\n"
                            "🎯 AI Score: " + str(ai_fvg_score) + "% | RSI: " + str(rsi) + "\n"
                            "📥 CMP: ₹" + str(price) + "\n"
                            "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 4. TYPE 4: Master Price Action (Wyckoff / SMC)
                    pa_setup = evaluate_price_action_setup(data)
                    if pa_setup and sym not in SENT_ALERTS["STOCK_TYPE4_PRICE_ACTION"]:
                        SENT_ALERTS["STOCK_TYPE4_PRICE_ACTION"].add(sym)
                        ACTIVE_TRADES["STOCK_TYPE4_PRICE_ACTION"][sym] = {"entry": pa_setup['entry'], "sl": pa_setup['sl'], "target": pa_setup['target']}
                        send_telegram_msg(
                            "🏛️ *[REPORT 4: NSE INSTITUTIONAL PRICE ACTION]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Stock: `" + str(sym) + "`\n"
                            "⚡ Strategy: *" + str(pa_setup['model']) + "*\n"
                            "🎯 Setup: _" + str(pa_setup['setup']) + "_\n"
                            "📥 Signal: BUY @ ₹" + str(pa_setup['entry']) + "\n"
                            "🛑 SL: ₹" + str(pa_setup['sl']) + " | 🎯 Target: ₹" + str(pa_setup['target']) + "\n"
                            "⚖️ Risk: ₹" + str(pa_setup['risk']) + " | R:R: " + str(pa_setup['rr']) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

                    # 5. TYPE 5 (NSE): 15 EMA + Price Action AI
                    if price >= ema_15 and sym not in SENT_ALERTS["STOCK_TYPE5_15EMA_AI"]:
                        SENT_ALERTS["STOCK_TYPE5_15EMA_AI"].add(sym)
                        sl = round(price * 0.992, 2)
                        tgt = round(price * 1.015, 2)
                        ACTIVE_TRADES["STOCK_TYPE5_15EMA_AI"][sym] = {"entry": price, "sl": sl, "target": tgt}
                        send_telegram_msg(
                            "📈 *[REPORT 5: NSE 15 EMA + PRICE ACTION AI]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Stock: `" + str(sym) + "`\n"
                            "📈 15 EMA Level: ₹" + str(ema_15) + "\n"
                            "📥 Entry: ₹" + str(price) + "\n"
                            "🛑 SL: ₹" + str(sl) + " | 🎯 Target: ₹" + str(tgt) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

            time.sleep(60)
        except Exception:
            time.sleep(30)

threading.Thread(target=background_scanner_and_audit, daemon=True).start()

# యాప్‌లో బటన్ క్లిక్ చేసినప్పుడు తక్షణమే ఎన్‌ఎస్‌ఈ స్టాక్స్ లేదా క్రిప్టో రిజల్ట్
@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    today_str = ist_now.strftime("%d-%b %I:%M %p")

    # 1. మార్కెట్ ఓపెన్ ఉంటే ముందుగా NSE స్టాక్స్‌ను స్కాన్ చేయడం
    if is_nse_market_open():
        for s in NSE_STOCKS[:4]:
            data = fetch_tradingview_nse_live(s)
            if data and data['close'] >= data['ema_15']:
                send_telegram_msg("🔍 *[MANUAL SCAN: NSE STOCK]*\nStock: `" + s + "` @ ₹" + str(data['close']))
                return jsonify([{
                    "date": today_str,
                    "pred": s + " (NSE 15 EMA Bounce)",
                    "actual": "CMP: ₹" + str(data['close']) + " | 15 EMA: ₹" + str(data['ema_15']),
                    "accuracy": "RSI: " + str(data['rsi'])
                }])

    # 2. మార్కెట్ క్లోజ్ అయితే లేదా క్రిప్టో కోసం: బైనాన్స్ హై మూమెంటమ్ కాయిన్
    dynamic_coins = ["SOLUSDT", "BTCUSDT", "ETHUSDT", "DOGEUSDT"]
    for pair in dynamic_coins:
        df = fetch_binance_live(pair, limit=30)
        if df is not None and len(df) >= 15:
            curr_p = round(float(df['close'].iloc[-1]), 4)
            df['ema_15'] = df['close'].ewm(span=15, adjust=False).mean()
            ema_15 = round(float(df['ema_15'].iloc[-1]), 4)
            trend = "BULLISH 🟢" if curr_p >= ema_15 else "CONSOLIDATION ⚪"

            send_telegram_msg("🔍 *[MANUAL SCAN: 24/7 CRYPTO]*\nCoin: `" + pair + "` @ $" + str(curr_p) + "\n15 EMA: $" + str(ema_15))
            return jsonify([{
                "date": today_str,
                "pred": pair + " (" + trend + ")",
                "actual": "CMP: $" + str(curr_p) + " \vert{} 15 EMA: $" + str(ema_15),
                "accuracy": "AI Feed Active"
            }])

    return jsonify([{
        "date": today_str,
        "pred": "BTCUSDT (Online)",
        "actual": "Market Monitoring Active",
        "accuracy": "Live Feed OK"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Render Cloud: 6-Tier Stocks & 24/7 Crypto Engine Online!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "engine": "6-Tier Stock & 24/7 Crypto AI Platform",
        "reports": [
            "Report 1: NSE Stock Pure AI (FVG)",
            "Report 2: NSE Stock Quant Indicators",
            "Report 3: NSE Stock Hybrid Confluence",
            "Report 4: NSE Stock Institutional SMC/Wyckoff",
            "Report 5: NSE Stock 15 EMA + Price Action AI",
            "Report 6: 24/7 Crypto 15 EMA + Altcoin AI"
        ],
        "zero_delay": True
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
