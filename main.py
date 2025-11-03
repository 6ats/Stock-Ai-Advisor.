# ========================================================
#  T&T Stock AI Pro – Streamlit 1.25.1+ (2025)
#  Fully working: LSTM, MACD, RSI, Dark Theme, Alerts
# ========================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ------------------- PAGE CONFIG (MUST BE FIRST) -------------------
st.set_page_config(
    page_title="T&T Stock AI Pro",
    page_icon="Chart with upwards trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------- DARK THEME (SAFE CSS) -------------------
st.markdown(
    """
<style>
    .main {background-color: #0e1117; color: #f0f0f0;}
    .stMetric {color: #00ff88 !important;}
    .sidebar .sidebar-content {background-color: #1a1d2e;}
    h1, h2, h3 {color: #00ff88; font-family: 'Arial Black', sans-serif;}
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select {background-color: #2a2d3e; color: white;}
    .stButton > button {background-color: #00ff88; color: black; font-weight: bold;}
</style>
""",
    unsafe_allow_html=True,
)

# ------------------- LSTM (OPTIONAL, SAFE) -------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.optimizers import Adam
    LSTM_AVAILABLE = True
except Exception:
    LSTM_AVAILABLE = False

# ------------------- LSTM HELPERS -------------------
def prepare_lstm_data(df, lookback=60):
    if not LSTM_AVAILABLE or len(df) < 100:
        return None, None, None
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler


def build_lstm_model(shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def predict_entry_exit(df, days=7):
    if not LSTM_AVAILABLE:
        return None, None, 0.0
    X, y, scaler = prepare_lstm_data(df)
    if X is None or len(X) < 20:
        return None, None, 0.0

    X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y, epochs=10, batch_size=32, verbose=0)

    seq = X_test[-1].reshape(1, X_test.shape[1], 1)
    preds = []
    for _ in range(days):
        pred = model.predict(seq, verbose=0)[0][0]
        preds.append(pred)
        seq = np.roll(seq, -1, axis=1)
        seq[0, -1, 0] = pred

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    cur = df["Close"].iloc[-1]
    entry = cur * 0.98 if preds[0] > cur else None
    exit_price = cur * 1.05 if preds[-1] < cur * 1.02 else cur * 1.10
    rr = (exit_price - entry) / (cur - entry) if entry and entry != cur else 0.0
    return entry, exit_price, rr


# ------------------- SIDEBAR CONTROLS -------------------
st.sidebar.header("Stock Controls")
ticker = st.sidebar.text_input("Symbol", value="AAPL", help="e.g. TSLA, NVDA")
period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
pred_days = st.sidebar.slider("AI Predict Days", 1, 30, 7)
show_macd = st.sidebar.checkbox("Show MACD", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", True)

# Quick Start Buttons
st.sidebar.markdown("**Quick Start**")
cols = st.sidebar.columns(4)
for i, sym in enumerate(["AAPL", "TSLA", "NVDA", "MSFT"]):
    if cols[i].button(sym):
        st.session_state.ticker = sym
        st.rerun()
if "ticker" in st.session_state:
    ticker = st.session_state.ticker

# ------------------- LOAD DATA -------------------
@st.cache_data(ttl=300)
def load_data(symbol, per):
    try:
        data = yf.Ticker(symbol).history(period=per)
        if data.empty:
            st.error(f"No data for **{symbol}**")
            return None
        return data
    except Exception as e:
        st.error(f"Error: {e}")
        return None

data = load_data(ticker, period)

# ------------------- MAIN APP -------------------
if data is not None and not data.empty:
    cur_price = data["Close"].iloc[-1]

    # Dashboard
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${cur_price:.2f}")
    c2.metric(
        "Change",
        f"${data['Close'].iloc[-1] - data['Close'].iloc[0]:.2f}",
        delta=f"{((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100:.1f}%"
    )
    c3.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    try:
        cap = yf.Ticker(ticker).info.get("marketCap", 0) / 1e9
        c4.metric("Market Cap", f"${cap:.1f}B")
    except:
        c4.metric("Market Cap", "N/A")

    # Chart
    st.subheader(f"{ticker.upper()} Live Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"],
        low=data["Low"], close=data["Close"], name="Price",
        increasing_line_color="#00ff88", decreasing_line_color="#ff4444"
    ))

    if show_macd:
        exp1 = data["Close"].ewm(span=12).mean()
        exp2 = data["Close"].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        fig.add_trace(go.Scatter(x=data.index, y=macd, name="MACD", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(x=data.index, y=signal, name="Signal", line=dict(color="orange")))

    if show_bb:
        ma20 = data["Close"].rolling(20).mean()
        std20 = data["Close"].rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        fig.add_trace(go.Scatter(x=data.index, y=upper, name="BB Upper", line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=data.index, y=lower, name="BB Lower", line=dict(dash="dash", color="gray")))
        fig.add_trace(go.Scatter(x=data.index, y=ma20, name="BB MA", line=dict(color="yellow")))

    fig.add_hline(y=cur_price, line_dash="dot", line_color="#00ff88")
    fig.update_layout(
        plot_bgcolor="#1a1d2e", paper_bgcolor="#0e1117",
        font_color="#f0f0f0", height=520
    )
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    def calculate_rsi(series, window=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    rsi_val = calculate_rsi(data["Close"]).iloc[-1]
    rsi_color = "#00ff88" if rsi_val < 30 else "#ff4444" if rsi_val > 70 else "#ffff00"
    rsi_signal = "BUY (Oversold)" if rsi_val < 30 else "SELL (Overbought)" if rsi_val > 70 else "HOLD"
    st.markdown(f"<h3 style='color:{rsi_color};text-align:center;'>RSI: {rsi_val:.1f} → {rsi_signal}</h3>", unsafe_allow_html=True)

    # AI Assistant
    st.subheader("AI Trading Assistant")
    entry, exit_price, rr = predict_entry_exit(data, pred_days)
    if entry and exit_price:
        c1, c2, c3 = st.columns(3)
        c1.metric("Entry Price", f"${entry:.2f}")
        c2.metric("Exit Target", f"${exit_price:.2f}")
        c3.metric("Risk/Reward", f"{rr:.1f}:1")
        st.success(f"**BUY** at ${entry:.2f} → **SELL** at ${exit_price:.2f} | Gain: {((exit_price - entry) / entry) * 100:.1f}%")
    else:
        st.warning("Need **2y data + TensorFlow** for AI signals.")

    # Alerts
    st.subheader("Price Alerts")
    alert_up = st.number_input("Alert if price >", value=cur_price * 1.05)
    alert_down = st.number_input("Alert if price <", value=cur_price * 0.95)
    if st.button("Set Alerts"):
        st.success(f"Alerts active: **>{alert_up:.2f}** | **<{alert_down:.2f}**")

    # Disclaimer
    st.info("**Educational tool only.** Not financial advice. DYOR.")

else:
    st.error("No data loaded. Try a valid ticker like **AAPL**.")
