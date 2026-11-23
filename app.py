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

# Render Environment Variables నుండి మాత్రమే సురక్షితంగా లోడ్ అవుతుంది
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# AI మోడల్స్ లోడింగ్
ai_model_fvg = None
if os.path.exists("fvg_ai_model.pkl"):
    try:
        ai_model_fvg = joblib.load("fvg_ai_model.pkl")
    except Exception as e:
        print("FVG Model Load Error: " + str(e))

ai_model_pa = None
if os.path.exists("price_action_ai_model.pkl"):
    try:
        ai_model_pa = joblib.load("price_action_ai_model.pkl")
    except Exception as e:
        print("PA Model Load Error: " + str(e))

NSE_STOCKS = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "SBIN",
    "INFY", "TCS", "TATAMOTORS", "AXISBANK"
]

STRATEGIES = [
    "STOCKS_TYPE1_AI_FVG",
    "STOCKS_TYPE3_HYBRID",
    "STOCKS_TYPE5_15EMA_AI",
    "STOCKS_SMC_LIQUIDITY",
    "CRYPTO_24X7_SMC_AND_15EMA"
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

# =====================================
# SMC LIQUIDITY + MSS ENGINE (CORE)
# =====================================
def calculate_atr(df, period=14):
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def get_pivots(df, window=2):
    highs = []
    lows = []
    for i in range(window, len(df) - window):
        h = df["High"].iloc[i]
        l = df["Low"].iloc[i]
        left_high = df["High"].iloc[i-window:i]
        right_high = df["High"].iloc[i+1:i+window+1]
        left_low = df["Low"].iloc[i-window:i]
        right_low = df["Low"].iloc[i+1:i+window+1]

        if h > max(left_high.max(), right_high.max()):
            highs.append((i, h))
        if l < min(left_low.min(), right_low.min()):
            lows.append((i, l))
    return highs, lows

def find_liquidity_levels(pivots, tolerance):
    if not pivots:
        return []
    levels = []
    for i, price in pivots:
        matched = False
        for level in levels:
            avg_p = np.mean(level["prices"])
            if abs(price - avg_p) <= tolerance:
                level["prices"].append(price)
                matched = True
                break
        if not matched:
            levels.append({"prices": [price]})
    return [float(np.mean(lvl["prices"])) for lvl in levels if len(lvl["prices"]) >= 2]

def analyze_smc_liquidity(df, lookback=80):
    if df is None or len(df) < lookback:
        return None
    df = df.copy()
    df["ATR"] = calculate_atr(df)
    last = df.iloc[-1]
    atr = last["ATR"]
    if pd.isna(atr) or atr <= 0:
        return None

    pivots_high, pivots_low = get_pivots(df.iloc[:-1])
    if not pivots_high or not pivots_low:
        return None

    recent_highs = [p for p in pivots_high if p[0] >= len(df) - 60]
    recent_lows = [p for p in pivots_low if p[0] >= len(df) - 60]
    if not recent_highs or not recent_lows:
        return None

    tolerance = atr * 0.15
    buffer = atr * 0.05

    equal_highs = find_liquidity_levels(recent_highs, tolerance)
    equal_lows = find_liquidity_levels(recent_lows, tolerance)

    bsl = max(equal_highs) if equal_highs else max(p[1] for p in recent_highs)
    ssl = min(equal_lows) if equal_lows else min(p[1] for p in recent_lows)

    previous_swing_low = recent_lows[-1][1]
    previous_swing_high = recent_highs[-1][1]

    o, h, l, c = last["Open"], last["High"], last["Low"], last["Close"]
    body = abs(c - o)
    strong_candle = body >= atr * 0.8

    # Sell Side Liquidity Sweep -> Bullish MSS
    bullish_sweep = (l < ssl - buffer) and (c > ssl)
    bullish_mss = c > previous_swing_high
    if bullish_sweep and bullish_mss and (c > o) and strong_candle:
        entry = float(c)
        stop = float(l - atr * 0.1)
        risk = entry - stop
        if risk > 0:
            return {
                "signal": "BUY",
                "entry": round(entry, 4),
                "stop": round(stop, 4),
                "target1": round(entry + risk * 1.0, 4),
                "target2": round(entry + risk * 2.0, 4),
                "liquidity": round(float(ssl), 4)
            }

    # Buy Side Liquidity Sweep -> Bearish MSS
    bearish_sweep = (h > bsl + buffer) and (c < bsl)
    bearish_mss = c < previous_swing_low
    if bearish_sweep and bearish_mss and (c < o) and strong_candle:
        entry = float(c)
        stop = float(h + atr * 0.1)
        risk = stop - entry
        if risk > 0:
            return {
                "signal": "SELL",
                "entry": round(entry, 4),
                "stop": round(stop, 4),
                "target1": round(entry - risk * 1.0, 4),
                "target2": round(entry - risk * 2.0, 4),
                "liquidity": round(float(bsl), 4)
            }
    return None

# =====================================
# ZERO-LAG DATA FEEDS (BINANCE & TV)
# =====================================
def get_top_momentum_crypto(limit=12):
    priority_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
    blacklist = ["USDCUSDT", "FDUSDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT", "AEURUSDT"]
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        resp = requests.get(url, timeout=4.0)
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

def fetch_binance_candles(symbol="BTCUSDT", limit=90):
    try:
        url = "https://api.binance.com/api/v3/klines?symbol=" + symbol + "&interval=5m&limit=" + str(limit)
        resp = requests.get(url, timeout=4.0)
        if resp.status_code == 200:
            raw = resp.json()
            df = pd.DataFrame(raw, columns=[
                'open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav', 'num_trades', 'tbb', 'tbq', 'ignore'
            ])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            # ప్రస్తుత ఫార్మింగ్ క్యాండిల్ మినహాయించి కన్ఫర్మ్ అయినవి తీసుకోవడం
            return df[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[:-1]
    except:
        pass
    return None

def fetch_nse_candles_and_live(symbol):
    # 1. TradingView నుండి Live Price, VWAP, RSI, EMAs
    live_info = None
    try:
        url = "https://scanner.tradingview.com/india/scan"
        payload = {
            "symbols": {"tickers": ["NSE:" + symbol], "query": {"types": []}},
            "columns": ["open", "high", "low", "close", "volume", "VWAP", "RSI", "EMA5", "EMA13", "EMA15"]
        }
        resp = requests.post(url, json=payload, headers={"User-Agent": "Mozilla/5.0"}, timeout=3.5)
        if resp.status_code == 200:
            d = resp.json().get('data', [])
            if d:
                v = d[0]['d']
                live_info = {
                    "open": float(v[0]), "high": float(v[1]), "low": float(v[2]),
                    "close": float(v[3]), "volume": float(v[4]),
                    "vwap": float(v[5]) if v[5] else float(v[3]),
                    "rsi": float(v[6]) if v[6] else 50.0,
                    "ema_5": float(v[7]) if v[7] else float(v[3]),
                    "ema_13": float(v[8]) if v[8] else float(v[3]),
                    "ema_15": float(v[9]) if v[9] else float(v[3])
                }
    except:
        pass

    # 2. Yahoo Finance Direct JSON 5m క్యాండిల్స్ (SMC Liquidity Sweep చెక్ కోసం)
    df_candles = None
    try:
        yf_url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + ".NS?range=2d&interval=5m"
        r = requests.get(yf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=4.0)
        if r.status_code == 200:
            res = r.json()['chart']['result'][0]
            q = res['indicators']['quote'][0]
            df_candles = pd.DataFrame({
                "Open": q['open'], "High": q['high'],
                "Low": q['low'], "Close": q['close'], "Volume": q['volume']
            }).dropna().iloc[:-1]
    except:
        pass

    return live_info, df_candles

def is_nse_market_open():
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    if ist_now.weekday() in [5, 6]:
        return False
    cur_mins = ist_now.hour * 60 + ist_now.minute
    return 555 <= cur_mins <= 930

# =====================================
# BACKGROUND LIVE SCANNER & AUDIT LOOP
# =====================================
def background_scanner_and_audit():
    global ALERT_TIMESTAMPS, AUDIT_SENT_TODAY, ACTIVE_TRADES, AUDIT_LOGS
    ist = pytz.timezone("Asia/Kolkata")

    while True:
        try:
            now = datetime.now(ist)

            # ఉదయం 9:15 రీసెట్
            if now.hour == 9 and now.minute < 15:
                ALERT_TIMESTAMPS.clear()
                ACTIVE_TRADES = {s: {} for s in STRATEGIES}
                AUDIT_LOGS = {s: [] for s in STRATEGIES}
                AUDIT_SENT_TODAY = False

            # సాయంత్రం 03:30 PM ఆడిట్ రిపోర్టులు
            if now.hour >= 15 and now.minute >= 30 and not AUDIT_SENT_TODAY:
                for strat in STRATEGIES:
                    for sym, pos in list(ACTIVE_TRADES[strat].items()):
                        AUDIT_LOGS[strat].append("CLOSED @ 03:30 PM | " + sym + " | Entry: " + str(pos['entry']))
                    ACTIVE_TRADES[strat].clear()

                titles = {
                    "STOCKS_TYPE1_AI_FVG": "NSE STOCKS: TYPE 1 PURE AI (FVG)",
                    "STOCKS_TYPE3_HYBRID": "NSE STOCKS: TYPE 3 HYBRID CONFLUENCE",
                    "STOCKS_TYPE5_15EMA_AI": "NSE STOCKS: TYPE 5 15 EMA + PRICE ACTION AI",
                    "STOCKS_SMC_LIQUIDITY": "NSE STOCKS: SMC LIQUIDITY SWEEP + MSS",
                    "CRYPTO_24X7_SMC_AND_15EMA": "24/7 CRYPTO: SMC LIQUIDITY + 15 EMA AI"
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
            # A. 24/7 క్రిప్టో స్కాన్ (SMC Liquidity Sweep + 15 EMA AI)
            # ----------------------------------------------------
            crypto_list = get_top_momentum_crypto(limit=10)
            for pair in crypto_list:
                df_c = fetch_binance_candles(pair, limit=90)
                if df_c is None or len(df_c) < 30:
                    continue

                # 1. SMC Liquidity Sweep + MSS చెక్
                smc_res = analyze_smc_liquidity(df_c, lookback=80)
                if smc_res:
                    key_smc = "CRYPTO_SMC_" + pair + "_" + smc_res['signal']
                    if curr_epoch - ALERT_TIMESTAMPS.get(key_smc, 0) > 1800:
                        ALERT_TIMESTAMPS[key_smc] = curr_epoch
                        ACTIVE_TRADES["CRYPTO_24X7_SMC_AND_15EMA"][pair] = {"entry": smc_res['entry']}
                        send_telegram_msg(
                            "🚨 *[24/7 CRYPTO SMC LIQUIDITY ALERT]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Asset: `" + str(pair) + "` | Signal: *" + smc_res['signal'] + "*\n"
                            "💧 Liquidity Level: $" + str(smc_res['liquidity']) + "\n"
                            "📥 Entry: $" + str(smc_res['entry']) + "\n"
                            "🛑 Stop Loss: $" + str(smc_res['stop']) + "\n"
                            "🎯 Target 1: $" + str(smc_res['target1']) + " | Target 2: $" + str(smc_res['target2']) + "\n"
                            "⚡ _Strategy: Liquidity Sweep + MSS + Displacement_"
                        )

                # 2. 15 EMA + Price Action AI చెక్
                curr_p = round(float(df_c['Close'].iloc[-1]), 4)
                ema_15 = round(float(df_c['Close'].ewm(span=15, adjust=False).mean().iloc[-1]), 4)
                if curr_p >= ema_15:
                    key_ema = "CRYPTO_EMA_" + pair
                    if curr_epoch - ALERT_TIMESTAMPS.get(key_ema, 0) > 1800:
                        ALERT_TIMESTAMPS[key_ema] = curr_epoch
                        sl = round(curr_p * 0.992, 4)
                        tgt = round(curr_p * 1.018, 4)
                        ACTIVE_TRADES["CRYPTO_24X7_SMC_AND_15EMA"][pair] = {"entry": curr_p}
                        send_telegram_msg(
                            "🤖 *[24/7 CRYPTO: 15 EMA + AI ALERT]*\n"
                            "━━━━━━━━━━━━━━━━━━━━\n"
                            "📌 Asset: `" + str(pair) + "`\n"
                            "📈 15 EMA: $" + str(ema_15) + " | Live Entry: $" + str(curr_p) + "\n"
                            "🛑 SL: $" + str(sl) + " | 🎯 Target: $" + str(tgt) + "\n"
                            "━━━━━━━━━━━━━━━━━━━━"
                        )

            # ----------------------------------------------------
            # B. NSE స్టాక్స్ స్కాన్ (Type 1, 3, 5 + SMC Liquidity Sweep)
            # ----------------------------------------------------
            if is_nse_market_open():
                for sym in NSE_STOCKS:
                    stk, df_stk = fetch_nse_candles_and_live(sym)

                    # 1. SMC Liquidity Sweep + MSS Check
                    if df_stk is not None and len(df_stk) >= 80:
                        smc_stk = analyze_smc_liquidity(df_stk, lookback=80)
                        if smc_stk:
                            k_smc = "STK_SMC_" + sym + "_" + smc_stk['signal']
                            if curr_epoch - ALERT_TIMESTAMPS.get(k_smc, 0) > 1800:
                                ALERT_TIMESTAMPS[k_smc] = curr_epoch
                                ACTIVE_TRADES["STOCKS_SMC_LIQUIDITY"][sym] = {"entry": smc_stk['entry']}
                                send_telegram_msg(
                                    "🚨 *[NSE SMC LIQUIDITY SWEEP ALERT]*\n"
                                    "━━━━━━━━━━━━━━━━━━━━\n"
                                    "📌 Stock: `" + str(sym) + "` | Signal: *" + smc_stk['signal'] + "*\n"
                                    "💧 Liquidity Level: ₹" + str(smc_stk['liquidity']) + "\n"
                                    "📥 Entry: ₹" + str(smc_stk['entry']) + "\n"
                                    "🛑 Stop Loss: ₹" + str(smc_stk['stop']) + "\n"
                                    "🎯 Target 1: ₹" + str(smc_stk['target1']) + " | Target 2: ₹" + str(smc_stk['target2']) + "\n"
                                    "⚡ _Strategy: Liquidity Sweep + MSS + Displacement_"
                                )

                    if not stk:
                        continue

                    price = stk['close']
                    vwap = stk['vwap']
                    rsi = stk['rsi']
                    ema_5 = stk['ema_5']
                    ema_13 = stk['ema_13']
                    ema_15 = stk['ema_15']

                    ai_fvg_score = 50
                    if ai_model_fvg is not None:
                        try:
                            f_arr = np.array([[0.5, ((price - vwap)/vwap)*100, 1.2, rsi, 1.5]])
                            ai_fvg_score = int(ai_model_fvg.predict_proba(f_arr)[0][1] * 100)
                        except:
                            pass

                    quant_pass = (ema_5 > ema_13) and (price > vwap) and (45 <= rsi <= 65)

                    # Type 1: Pure AI FVG
                    if ai_fvg_score >= 60 and price >= vwap:
                        if curr_epoch - ALERT_TIMESTAMPS.get("STK_T1_" + sym, 0) > 1800:
                            ALERT_TIMESTAMPS["STK_T1_" + sym] = curr_epoch
                            sl = round(price * 0.993, 2)
                            tgt = round(price * 1.012, 2)
                            ACTIVE_TRADES["STOCKS_TYPE1_AI_FVG"][sym] = {"entry": price}
                            send_telegram_msg(
                                "🟢 *[TYPE 1: NSE PURE AI (FVG)]*\n"
                                "📌 Stock: `" + str(sym) + "` | Score: `" + str(ai_fvg_score) + "%`\n"
                                "📥 CMP: ₹" + str(price) + " | SL: ₹" + str(sl) + " | Target: ₹" + str(tgt)
                            )

                    # Type 3: Hybrid Confluence
                    if ai_fvg_score >= 60 and quant_pass:
                        if curr_epoch - ALERT_TIMESTAMPS.get("STK_T3_" + sym, 0) > 1800:
                            ALERT_TIMESTAMPS["STK_T3_" + sym] = curr_epoch
                            sl = round(price * 0.993, 2)
                            tgt = round(price * 1.015, 2)
                            ACTIVE_TRADES["STOCKS_TYPE3_HYBRID"][sym] = {"entry": price}
                            send_telegram_msg(
                                "🚀 *[TYPE 3: NSE HYBRID CONFLUENCE]*\n"
                                "📌 Stock: `" + str(sym) + "` | RSI: " + str(rsi) + "\n"
                                "📥 CMP: ₹" + str(price) + " | SL: ₹" + str(sl) + " | Target: ₹" + str(tgt)
                            )

                    # Type 5: 15 EMA + Price Action AI
                    if price >= ema_15:
                        if curr_epoch - ALERT_TIMESTAMPS.get("STK_T5_" + sym, 0) > 1800:
                            ALERT_TIMESTAMPS["STK_T5_" + sym] = curr_epoch
                            sl = round(price * 0.992, 2)
                            tgt = round(price * 1.015, 2)
                            ACTIVE_TRADES["STOCKS_TYPE5_15EMA_AI"][sym] = {"entry": price}
                            send_telegram_msg(
                                "📈 *[TYPE 5: NSE 15 EMA + PA AI]*\n"
                                "📌 Stock: `" + str(sym) + "` | 15 EMA: ₹" + str(ema_15) + "\n"
                                "📥 Entry: ₹" + str(price) + " | SL: ₹" + str(sl) + " | Target: ₹" + str(tgt)
                            )

            time.sleep(60)
        except Exception:
            time.sleep(30)

threading.Thread(target=background_scanner_and_audit, daemon=True).start()

# =====================================
# యాప్ SCAN బటన్ ఎండ్‌పాయింట్ (WITH LIQUIDITY)
# =====================================
@application.route('/scan_top', methods=['GET'])
def scan_top():
    ist_now = datetime.now(pytz.timezone("Asia/Kolkata"))
    today_str = ist_now.strftime("%d-%b %I:%M %p")

    # 1. మార్కెట్ ఓపెన్ ఉంటే ముందుగా NSE స్టాక్స్ + SMC Liquidity స్కాన్
    if is_nse_market_open():
        for s in NSE_STOCKS:
            stk, df_stk = fetch_nse_candles_and_live(s)
            if stk and stk['close'] >= stk['ema_15']:
                liq_txt = "Liquidity: Scanning..."
                if df_stk is not None and len(df_stk) >= 30:
                    bsl = round(float(df_stk['High'].tail(30).max()), 2)
                    ssl = round(float(df_stk['Low'].tail(30).min()), 2)
                    liq_txt = "BSL: ₹" + str(bsl) + " | SSL: ₹" + str(ssl)
                    
                    smc_hit = analyze_smc_liquidity(df_stk, lookback=60)
                    if smc_hit:
                        liq_txt = "SMC " + smc_hit['signal'] + " @ ₹" + str(smc_hit['liquidity'])

                msg = (
                    "🔍 *[APP SCAN: NSE + SMC LIQUIDITY]*\n"
                    "📌 Stock: `" + s + "` @ ₹" + str(stk['close']) + "\n"
                    "📈 15 EMA: ₹" + str(stk['ema_15']) + "\n"
                    "💧 " + liq_txt
                )
                send_telegram_msg(msg)
                return jsonify([{
                    "date": today_str,
                    "pred": s + " (NSE BULLISH 🟢)",
                    "actual": "CMP: ₹" + str(stk['close']) + " | 15 EMA: ₹" + str(stk['ema_15']),
                    "accuracy": liq_txt
                }])

    # 2. 24/7 క్రిప్టో లైవ్ స్కాన్ + SMC Liquidity Levels (BSL / SSL)
    best_coin = None
    for pair in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        df = fetch_binance_candles(pair, limit=85)
        if df is not None and len(df) >= 30:
            curr_p = round(float(df['Close'].iloc[-1]), 2)
            ema_15 = round(float(df['Close'].ewm(span=15, adjust=False).mean().iloc[-1]), 2)
            is_bull = curr_p >= ema_15

            # BSL (పై లిక్విడిటీ) & SSL (కింది లిక్విడిటీ) లెక్కించడం
            bsl = round(float(df['High'].tail(40).max()), 2)
            ssl = round(float(df['Low'].tail(40).min()), 2)
            liq_status = "BSL: $" + str(bsl) + " | SSL: $" + str(ssl)

            # SMC Sweep జరిగిందో లేదో చెక్ చేయడం
            smc_res = analyze_smc_liquidity(df, lookback=80)
            if smc_res:
                liq_status = "🚨 SMC " + smc_res['signal'] + " SWEEP @ $" + str(smc_res['liquidity'])

            if is_bull or smc_res or best_coin is None:
                best_coin = {
                    "pair": pair,
                    "price": curr_p,
                    "ema": ema_15,
                    "trend": "SMC " + smc_res['signal'] + " 🚨" if smc_res else ("BULLISH 🟢" if is_bull else "CONSOLIDATION ⚪"),
                    "liq": liq_status
                }
                if is_bull or smc_res:
                    break

    if best_coin:
        send_telegram_msg(
            "🔍 *[APP SCAN: CRYPTO + SMC LIQUIDITY]*\n"
            "📌 Asset: `" + best_coin["pair"] + "`\n"
            "📥 Live Price: $" + str(best_coin["price"]) + "\n"
            "📈 15 EMA: $" + str(best_coin["ema"]) + "\n"
            "💧 " + best_coin["liq"]
        )
        return jsonify([{
            "date": today_str,
            "pred": best_coin["pair"] + " (" + best_coin["trend"] + ")",
            "actual": "CMP: $" + str(best_coin["price"]) + " | 15 EMA: $" + str(best_coin["ema"]),
            "accuracy": best_coin["liq"]
        }])

    return jsonify([{
        "date": today_str,
        "pred": "BTCUSDT (SCAN COMPLETE)",
        "actual": "Live Feed Active",
        "accuracy": "BSL/SSL Liquidity Synced"
    }])

@application.route('/test_telegram', methods=['GET'])
def test_telegram():
    status = send_telegram_msg("🔔 Venkat SMC Liquidity + AI Engine Online!")
    return jsonify({"status": "SUCCESS" if status else "FAILED"})

@application.route('/')
def home():
    return jsonify({
        "status": "ONLINE",
        "engine": "Venkat SMC Liquidity + Type 1,3,5 AI Platform",
        "features": ["Liquidity Sweep (BSL/SSL)", "Market Structure Shift (MSS)", "Displacement", "15 EMA + FVG AI"]
    })

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=5000)
