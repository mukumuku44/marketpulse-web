from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from datetime import datetime
import threading
import time
import os

app = Flask(__name__)
TICKERS = ["AAPL", "BTC-USD", "ETH-USD", "SPY", "TSLA", "NVDA"]
REFRESH_INTERVAL = 300
CACHE = {}

def get_data(ticker):
    try:
        df = yf.download(ticker, period="10y", interval="1d", progress=False, auto_adjust=True)
        if df.empty or len(df) < 200: return None
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except: return None

def add_indicators(df):
    df = df.copy()
    # Manual RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df['momentum_rsi'] = 100 - (100 / (1 + rs))
    
    # Manual MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['trend_macd'] = ema12 - ema26
    
    # Manual SMA
    df['trend_sma_fast'] = df['Close'].rolling(20).mean()
    df['trend_sma_slow'] = df['Close'].rolling(50).mean()
    
    # Manual Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    df['bb_std'] = df['Close'].rolling(20).std()
    df['volatility_bbh'] = df['bb_mid'] + 2 * df['bb_std']
    df['volatility_bbl'] = df['bb_mid'] - 2 * df['bb_std']
    
    # Manual ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['volatility_atr'] = tr.rolling(14).mean()
    
    return df

def create_signals(df):
    df = df.copy()
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    signals = []
    rsi = df['momentum_rsi']
    signals.append((30 - rsi.clip(0,30)) / 30)
    macd = df['trend_macd']
    signals.append(np.tanh(macd / (macd.std() + 1e-8)))
    sma_fast = df['trend_sma_fast']
    sma_slow = df['trend_sma_slow']
    signals.append((sma_fast > sma_slow).astype(float))
    price = df['Close']
    upper = df['volatility_bbh']
    lower = df['volatility_bbl']
    pos = (price - lower) / (upper - lower + 1e-8)
    signals.append(pos - 0.5)
    df['Composite_Score'] = np.nanmean(signals, axis=0)
    return df

def train_model(X, y):
    model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=1)
    tscv = TimeSeriesSplit(n_splits=2)
    for train_idx, _ in tscv.split(X):
        model.fit(X[train_idx], y[train_idx])
    return model

def analyze_ticker(ticker):
    try:
        df = get_data(ticker)
        if df is None: return {"ticker": ticker, "error": "No data"}
        df = add_indicators(df)
        df = create_signals(df)
        df.dropna(inplace=True)
        if len(df) < 50: return {"ticker": ticker, "error": "Not enough data"}
        features = ['Composite_Score', 'momentum_rsi', 'trend_macd', 'volatility_atr']
        X = df[features].values
        y = df['Target'].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = train_model(X_scaled, y)
        latest = X_scaled[-1].reshape(1, -1)
        proba = model.predict_proba(latest)[0][1]
        score = df['Composite_Score'].iloc[-1]
        df['pred'] = model.predict(X_scaled)
        df['return'] = df['Close'].pct_change().shift(-1)
        df['strategy'] = np.where(df['pred']==1, df['return'], 0)
        total_return = (1 + df['strategy']).cumprod().iloc[-2] - 1 if len(df) > 1 else 0
        win_rate = (df['strategy'] > 0).mean()
        accuracy = (df['pred'] == df['Target']).mean()
        signal = "BULLISH" if proba > 0.55 else "BEARISH" if proba < 0.45 else "NEUTRAL"
        color = "success" if signal == "BULLISH" else "danger" if signal == "BEARISH" else "warning"
        return {
            "ticker": ticker, "price": round(df['Close'].iloc[-1], 2),
            "composite": round(score, 3), "prob_up": round(proba, 3),
            "signal": signal, "color": color, "return": round(total_return, 3),
            "win_rate": round(win_rate, 3), "accuracy": round(accuracy, 3),
            "updated": datetime.now().strftime("%H:%M:%S")
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}

def update_cache():
    global CACHE
    while True:
        print(f"[{datetime.now()}] Updating...")
        results = [analyze_ticker(t) for t in TICKERS]
        CACHE = {"data": results, "time": datetime.now().isoformat()}
        time.sleep(REFRESH_INTERVAL)

threading.Thread(target=update_cache, daemon=True).start()

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/data")
def api_data(): return jsonify(CACHE)

@app.route("/health")
def health(): return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)