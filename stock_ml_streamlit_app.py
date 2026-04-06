import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Stock Analyzer", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
.block-container { padding: 1.5rem 2rem; }
h1, h2, h3, h4 { color: #e6edf3 !important; }
p, label, div { color: #8b949e; }
.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e !important; border-radius: 6px; padding: 8px 20px; font-size: 0.85rem; }
.stTabs [aria-selected="true"] { background: #21262d !important; color: #e6edf3 !important; }
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 0.75rem; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 1.4rem; font-weight: 700; }
.stButton button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    width: 100%;
}
.stSelectbox > div > div { background: #21262d !important; border: 1px solid #30363d !important; color: #e6edf3 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df):
    df = df.copy()
    df['Price_Change']     = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    df['Daily_Return']     = df['Close'].pct_change()
    df['MA_7']   = df['Close'].rolling(7).mean()
    df['MA_14']  = df['Close'].rolling(14).mean()
    df['MA_30']  = df['Close'].rolling(30).mean()
    df['MA_50']  = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD']        = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Momentum']    = df['Close'] - df['Close'].shift(10)
    df['Volatility']  = df['Daily_Return'].rolling(14).std() * 100
    delta = df['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(14).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain / loss))
    df['BB_Middle'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_Upper']  = df['BB_Middle'] + bb_std * 2
    df['BB_Lower']  = df['BB_Middle'] - bb_std * 2
    df['BB_Width']  = df['BB_Upper'] - df['BB_Lower']
    df['HL_Range']  = df['High'] - df['Low']
    df['HL_Ratio']  = df['HL_Range'] / df['Close']
    if 'Volume' in df.columns:
        df['Volume_MA']    = df['Volume'].rolling(7).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    df['Price_to_MA7']  = df['Close'] / df['MA_7']
    df['Price_to_MA30'] = df['Close'] / df['MA_30']
    return df.dropna()


def prepare_ml_data(df, lookback=30):
    features = ['Open','Close','High','Low','Price_Change','MA_7','MA_14',
                'MA_30','EMA_12','EMA_26','MACD','Momentum','Volatility',
                'RSI','BB_Upper','BB_Lower','HL_Range','Price_to_MA7','Price_to_MA30']
    available = [f for f in features if f in df.columns]
    X = df[available].values
    y = df['Close'].values
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    Xs, ys = [], []
    for i in range(len(X_scaled) - lookback):
        Xs.append(X_scaled[i:i+lookback].flatten())
        ys.append(y[i+lookback])
    return np.array(Xs), np.array(ys), scaler, available


def train_models(X_train, y_train, X_test, y_test):
    models, results = {}, {}
    for name, mdl in [
        ('Linear Regression', LinearRegression()),
        ('Random Forest',     RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ]:
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        models[name]  = mdl
        results[name] = {
            'MAE':  mean_absolute_error(y_test, pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, pred)),
            'R2':   r2_score(y_test, pred),
            'predictions': pred
        }
    return models, results


# ── Pattern detection ─────────────────────────────────────────────────────────

def detect_patterns(df):
    patterns = []
    if len(df) < 50:
        return patterns

    close = df['Close'].values
    recent = close[-60:]

    # Golden / Death Cross
    if 'MA_7' in df.columns and 'MA_30' in df.columns:
        ma7  = df['MA_7'].values
        ma30 = df['MA_30'].values
        if ma7[-1] > ma30[-1] and ma7[-5] <= ma30[-5]:
            patterns.append({'name':'🟢 Golden Cross','type':'bullish',
                'desc':'MA7 just crossed above MA30 — short-term momentum turning bullish.','signal':'BUY'})
        elif ma7[-1] < ma30[-1] and ma7[-5] >= ma30[-5]:
            patterns.append({'name':'🔴 Death Cross','type':'bearish',
                'desc':'MA7 just crossed below MA30 — short-term momentum turning bearish.','signal':'SELL'})

    # RSI levels
    if 'RSI' in df.columns:
        rsi = df['RSI'].values
        if rsi[-1] < 30:
            patterns.append({'name':'🟢 RSI Oversold','type':'bullish',
                'desc':f'RSI at {rsi[-1]:.1f} — price may be oversold, potential reversal upward.','signal':'BUY'})
        elif rsi[-1] > 70:
            patterns.append({'name':'🔴 RSI Overbought','type':'bearish',
                'desc':f'RSI at {rsi[-1]:.1f} — price may be overbought, potential pullback.','signal':'SELL'})

    # Bollinger Band conditions
    if 'BB_Width' in df.columns:
        bw    = df['BB_Width'].values
        avg_bw = np.mean(bw[-30:])
        if bw[-1] < avg_bw * 0.6:
            patterns.append({'name':'🟡 Bollinger Squeeze','type':'neutral',
                'desc':'Bands are unusually narrow — a sharp breakout may be imminent.','signal':'WATCH'})
        elif df['Close'].values[-1] > df['BB_Upper'].values[-1]:
            patterns.append({'name':'🔴 BB Upper Breakout','type':'bearish',
                'desc':'Price closed above upper Bollinger Band — possible mean reversion.','signal':'CAUTION'})
        elif df['Close'].values[-1] < df['BB_Lower'].values[-1]:
            patterns.append({'name':'🟢 BB Lower Bounce','type':'bullish',
                'desc':'Price closed below lower Bollinger Band — potential bounce opportunity.','signal':'WATCH'})

    # Double Top
    window = recent[-40:]
    peaks = [i for i in range(2, len(window)-2)
             if window[i] > window[i-1] and window[i] > window[i-2]
             and window[i] > window[i+1] and window[i] > window[i+2]]
    if len(peaks) >= 2:
        p1, p2 = peaks[-2], peaks[-1]
        if abs(window[p1] - window[p2]) / window[p1] < 0.03 and (p2 - p1) >= 5:
            patterns.append({'name':'🔴 Double Top','type':'bearish',
                'desc':'Two similar peaks detected — classic bearish reversal pattern.','signal':'SELL'})

    # Double Bottom
    troughs = [i for i in range(2, len(window)-2)
               if window[i] < window[i-1] and window[i] < window[i-2]
               and window[i] < window[i+1] and window[i] < window[i+2]]
    if len(troughs) >= 2:
        t1, t2 = troughs[-2], troughs[-1]
        if abs(window[t1] - window[t2]) / window[t1] < 0.03 and (t2 - t1) >= 5:
            patterns.append({'name':'🟢 Double Bottom','type':'bullish',
                'desc':'Two similar troughs detected — classic bullish reversal pattern.','signal':'BUY'})

    # MACD crossover
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd = df['MACD'].values
        sig  = df['Signal_Line'].values
        if macd[-1] > sig[-1] and macd[-3] <= sig[-3]:
            patterns.append({'name':'🟢 MACD Bullish Crossover','type':'bullish',
                'desc':'MACD line crossed above the signal line — bullish momentum building.','signal':'BUY'})
        elif macd[-1] < sig[-1] and macd[-3] >= sig[-3]:
            patterns.append({'name':'🔴 MACD Bearish Crossover','type':'bearish',
                'desc':'MACD line crossed below the signal line — bearish momentum building.','signal':'SELL'})

    # Trend (linear slope)
    x     = np.arange(len(recent))
    slope = np.polyfit(x, recent, 1)[0]
    pct   = slope / recent[0] * 100
    if pct > 0.15:
        patterns.append({'name':'🟢 Uptrend','type':'bullish',
            'desc':f'Steady uptrend detected (+{pct:.2f}% per day avg over last 60 sessions).','signal':'BUY'})
    elif pct < -0.15:
        patterns.append({'name':'🔴 Downtrend','type':'bearish',
            'desc':f'Steady downtrend detected ({pct:.2f}% per day avg over last 60 sessions).','signal':'SELL'})

    return patterns


# ── Plotly helpers ────────────────────────────────────────────────────────────

BASE = dict(
    template='plotly_dark',
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(22,27,34,0.9)',
    font=dict(color='#8b949e'),
)

def range_buttons():
    return dict(
        buttons=[
            dict(count=1,  label='1M', step='month', stepmode='backward'),
            dict(count=3,  label='3M', step='month', stepmode='backward'),
            dict(count=6,  label='6M', step='month', stepmode='backward'),
            dict(count=1,  label='1Y', step='year',  stepmode='backward'),
            dict(step='all', label='All'),
        ],
        font=dict(color='#e6edf3'),
        bgcolor='#21262d',
        activecolor='#388bfd',
    )


def price_chart(df, stock):
    has_vol = 'Volume' in df.columns
    fig = make_subplots(
        rows=2 if has_vol else 1, cols=1,
        shared_xaxes=True,
        row_heights=[0.73, 0.27] if has_vol else [1.0],
        vertical_spacing=0.02
    )
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC',
        increasing=dict(line=dict(color='#26a69a'), fillcolor='#26a69a'),
        decreasing=dict(line=dict(color='#ef5350'), fillcolor='#ef5350'),
    ), row=1, col=1)
    if has_vol:
        bar_colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],
                             name='Volume', marker_color=bar_colors, opacity=0.7), row=2, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1, title_font_size=11)
    fig.update_layout(
        title=dict(text=f'<b>{stock}</b> — Candlestick Chart', font=dict(size=17, color='#e6edf3')),
        height=520,
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', y=1.04, font=dict(color='#e6edf3')),
        xaxis=dict(rangeselector=range_buttons(), gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d'),
        **BASE
    )
    return fig


def technical_chart(df):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.40, 0.20, 0.20, 0.20],
        vertical_spacing=0.025,
        subplot_titles=['Price + Moving Averages', 'Bollinger Bands', 'RSI (14)', 'MACD']
    )
    # Price + MAs
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close',
                             line=dict(color='#58a6ff', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_7'],  name='MA 7',
                             line=dict(color='#f0b429', width=1.5, dash='dot')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_30'], name='MA 30',
                             line=dict(color='#3fb950', width=1.5, dash='dash')), row=1, col=1)
    # Bollinger
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
                             line=dict(color='#8b949e', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
                             fill='tonexty', fillcolor='rgba(139,148,158,0.12)',
                             line=dict(color='#8b949e', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close',
                             line=dict(color='#58a6ff', width=1.5), showlegend=False), row=2, col=1)
    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
                             line=dict(color='#d2a8ff', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#ef5350', line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#26a69a', line_width=1, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='rgba(239,83,80,0.06)',  line_width=0, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='rgba(38,166,154,0.06)', line_width=0, row=3, col=1)
    # MACD
    hist = df['MACD'] - df['Signal_Line']
    fig.add_trace(go.Bar(x=df['Date'], y=hist, name='Histogram',
                         marker_color=['#26a69a' if v >= 0 else '#ef5350' for v in hist],
                         opacity=0.6), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'],        name='MACD',
                             line=dict(color='#58a6ff', width=1.8)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Signal_Line'], name='Signal',
                             line=dict(color='#f0b429', width=1.5)), row=4, col=1)

    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#8b949e', size=11)

    fig.update_layout(
        height=700,
        legend=dict(orientation='h', y=1.03, font=dict(color='#e6edf3', size=11)),
        xaxis=dict(rangeselector=range_buttons(), gridcolor='#21262d'),
        yaxis=dict(gridcolor='#21262d'),
        **BASE
    )
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    return fig


def predictions_chart(y_test, results):
    names  = list(results.keys())
    colors = ['#58a6ff', '#3fb950', '#f0b429']
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, horizontal_spacing=0.06)
    for i, (name, res) in enumerate(results.items()):
        x = list(range(len(y_test)))
        fig.add_trace(go.Scatter(x=x, y=y_test, name='Actual',
                                 line=dict(color='#8b949e', width=2),
                                 showlegend=(i == 0)), row=1, col=i+1)
        fig.add_trace(go.Scatter(x=x, y=res['predictions'], name=name,
                                 line=dict(color=colors[i], width=2)), row=1, col=i+1)
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color='#8b949e', size=12)
    fig.update_layout(height=380, legend=dict(orientation='h', y=1.1, font=dict(color='#e6edf3')), **BASE)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 📈 Stock Analyzer")
        st.markdown("<hr style='border-color:#30363d;margin:8px 0 16px'>", unsafe_allow_html=True)
        st.markdown("#### 📁 Upload Data")
        uploaded_files = st.file_uploader(
            "CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Columns needed: Date, Open, Close, High, Low, country, stock"
        )
        st.markdown("<hr style='border-color:#30363d;margin:12px 0'>", unsafe_allow_html=True)
        st.markdown("""
<div style='font-size:0.78rem;color:#8b949e;line-height:1.9'>
<b style='color:#e6edf3'>Required columns</b><br>
• <code>Date</code> — YYYY-MM-DD<br>
• <code>Open / Close / High / Low</code><br>
• <code>country</code> — e.g. USA, IND<br>
• <code>stock</code> — e.g. AAPL, TCS<br>
• <code>Volume</code> — optional
</div>""", unsafe_allow_html=True)

    # Header
    st.markdown("""
<div style='margin-bottom:1.5rem'>
  <h1 style='color:#e6edf3;margin:0;font-size:2rem;font-weight:800'>📊 Stock Price Analyzer</h1>
  <p style='color:#8b949e;margin:4px 0 0;font-size:0.9rem'>
    Technical analysis &nbsp;·&nbsp; Chart patterns &nbsp;·&nbsp; ML predictions
  </p>
</div>""", unsafe_allow_html=True)

    # Empty state
    if not uploaded_files:
        st.markdown("""
<div style='background:#161b22;border:1px dashed #30363d;border-radius:14px;
            padding:60px 40px;text-align:center;margin-top:60px'>
  <div style='font-size:3.5rem;margin-bottom:12px'>📂</div>
  <h3 style='color:#e6edf3;margin:0 0 8px'>Upload your stock CSV to get started</h3>
  <p style='color:#8b949e;margin:0'>Use the sidebar uploader · Required: Date, Open, Close, High, Low, country, stock</p>
</div>""", unsafe_allow_html=True)
        st.stop()

    # Load
    dfs = []
    for f in uploaded_files:
        tmp = pd.read_csv(f)
        if 'Unnamed: 0' in tmp.columns:
            tmp = tmp.drop('Unnamed: 0', axis=1)
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)

    if df is None or len(df) == 0:
        st.error("❌ No data loaded.")
        st.stop()

    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        st.error("❌ Could not parse the 'Date' column.")
        st.stop()

    for col in ['country', 'stock']:
        if col not in df.columns:
            st.error(f"❌ Missing required column: '{col}'")
            st.stop()

    # Stock selector
    c1, c2, _ = st.columns([1, 1, 3])
    with c1:
        selected_country = st.selectbox("🌍 Country", sorted(df['country'].unique()))
    country_data = df[df['country'] == selected_country]
    with c2:
        selected_stock = st.selectbox("📌 Stock", sorted(country_data['stock'].unique()))

    stock_data = country_data[country_data['stock'] == selected_stock].copy()
    stock_data = stock_data.sort_values('Date').reset_index(drop=True)

    LOOKBACK = 30
    if len(stock_data) < LOOKBACK + 10:
        st.error(f"❌ Not enough data for {selected_stock}. Need at least {LOOKBACK + 10} rows.")
        st.stop()

    # KPI row
    latest  = stock_data['Close'].iloc[-1]
    first   = stock_data['Close'].iloc[0]
    change  = latest - first
    pct_chg = change / first * 100
    vol     = stock_data['Close'].pct_change().std() * 100
    high52  = stock_data['Close'].tail(252).max()
    low52   = stock_data['Close'].tail(252).min()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Current Price", f"${latest:.2f}")
    k2.metric("Total Change",  f"${change:.2f}", delta=f"{pct_chg:.2f}%")
    k3.metric("Volatility",    f"{vol:.2f}%")
    k4.metric("52W High",      f"${high52:.2f}")
    k5.metric("52W Low",       f"${low52:.2f}")

    st.markdown("<div style='margin:14px 0'></div>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Chart",
        "📈 Technical Indicators",
        "🔍 Chart Patterns",
        "🤖 ML Models",
        "📋 Data Overview",
    ])

    # ── Tab 1: Price Chart ──────────────────────────────────────────────
    with tab1:
        st.markdown(f"#### {selected_stock} — Interactive Candlestick")
        st.markdown("<p style='color:#8b949e;font-size:0.82rem'>🖱 Drag to zoom · Scroll to pan · Use range buttons above chart</p>",
                    unsafe_allow_html=True)
        st.plotly_chart(price_chart(stock_data, selected_stock), use_container_width=True)

    # ── Tab 2: Technical Indicators ─────────────────────────────────────
    with tab2:
        eng = engineer_features(stock_data)
        if len(eng) == 0:
            st.warning("Not enough data to compute indicators.")
        else:
            st.markdown(f"#### {selected_stock} — Technical Indicators")
            st.markdown("<p style='color:#8b949e;font-size:0.82rem'>🖱 All panels share the same x-axis — drag any panel to zoom all</p>",
                        unsafe_allow_html=True)
            st.plotly_chart(technical_chart(eng), use_container_width=True)

    # ── Tab 3: Chart Patterns ───────────────────────────────────────────
    with tab3:
        eng3     = engineer_features(stock_data)
        patterns = detect_patterns(eng3)

        st.markdown(f"#### {selected_stock} — Detected Chart Patterns")

        if not patterns:
            st.info("No strong patterns detected in the current data window.")
        else:
            bull = [p for p in patterns if p['type'] == 'bullish']
            bear = [p for p in patterns if p['type'] == 'bearish']
            neut = [p for p in patterns if p['type'] == 'neutral']

            bc1, bc2, bc3 = st.columns(3)
            bc1.metric("🟢 Bullish Signals", len(bull))
            bc2.metric("🔴 Bearish Signals", len(bear))
            bc3.metric("🟡 Watch Signals",   len(neut))

            st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)

            SIG_BG  = {'BUY':'#1a3a2a','SELL':'#3a1a1a','WATCH':'#2a2a10','CAUTION':'#3a2a10'}
            SIG_COL = {'BUY':'#3fb950','SELL':'#ef5350','WATCH':'#f0b429','CAUTION':'#ffa657'}
            TYPE_BORDER = {'bullish':'#26a69a','bearish':'#ef5350','neutral':'#f0b429'}

            for p in patterns:
                border = TYPE_BORDER[p['type']]
                sbg    = SIG_BG.get(p['signal'], '#21262d')
                scol   = SIG_COL.get(p['signal'], '#e6edf3')
                st.markdown(f"""
<div style='background:#161b22;border:1px solid #30363d;border-left:4px solid {border};
            border-radius:10px;padding:16px 20px;margin-bottom:10px;
            display:flex;align-items:center;gap:16px'>
  <div style='flex:1'>
    <div style='font-size:0.97rem;font-weight:700;color:#e6edf3;margin-bottom:3px'>{p['name']}</div>
    <div style='font-size:0.83rem;color:#8b949e'>{p['desc']}</div>
  </div>
  <div style='background:{sbg};color:{scol};border-radius:6px;
              padding:4px 12px;font-size:0.78rem;font-weight:700;white-space:nowrap'>
    {p['signal']}
  </div>
</div>""", unsafe_allow_html=True)

        # Pattern context chart
        st.markdown("#### Price with Pattern Context")
        pfig = go.Figure()
        pfig.add_trace(go.Scatter(x=eng3['Date'], y=eng3['Close'],   name='Close',
                                  line=dict(color='#58a6ff', width=2)))
        pfig.add_trace(go.Scatter(x=eng3['Date'], y=eng3['MA_7'],    name='MA 7',
                                  line=dict(color='#f0b429', width=1.5, dash='dot')))
        pfig.add_trace(go.Scatter(x=eng3['Date'], y=eng3['MA_30'],   name='MA 30',
                                  line=dict(color='#3fb950', width=1.5, dash='dash')))
        pfig.add_trace(go.Scatter(x=eng3['Date'], y=eng3['BB_Upper'],name='BB Upper',
                                  line=dict(color='#8b949e', width=1, dash='dot')))
        pfig.add_trace(go.Scatter(x=eng3['Date'], y=eng3['BB_Lower'],name='BB Lower',
                                  fill='tonexty', fillcolor='rgba(139,148,158,0.08)',
                                  line=dict(color='#8b949e', width=1, dash='dot')))
        pfig.update_layout(
            height=380,
            legend=dict(orientation='h', y=1.06, font=dict(color='#e6edf3', size=11)),
            xaxis=dict(rangeselector=range_buttons(), gridcolor='#21262d'),
            yaxis=dict(gridcolor='#21262d'),
            **BASE
        )
        st.plotly_chart(pfig, use_container_width=True)

    # ── Tab 4: ML Models ────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Machine Learning Price Prediction")
        st.markdown("<p style='color:#8b949e;font-size:0.85rem'>Trains Linear Regression, Random Forest and Gradient Boosting on engineered technical features.</p>",
                    unsafe_allow_html=True)

        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run = st.button("🚀 Train Models", use_container_width=True)

        if run:
            with st.spinner("Engineering features & training models…"):
                eng4 = engineer_features(stock_data)
                if len(eng4) < LOOKBACK + 10:
                    st.warning("Not enough data after feature engineering.")
                else:
                    X, y, scaler, feats = prepare_ml_data(eng4, lookback=LOOKBACK)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, shuffle=False
                    )
                    mdls, res = train_models(X_train, y_train, X_test, y_test)
                    st.success("✅ Models trained successfully!")

                    res_df = pd.DataFrame({
                        'Model':    list(res.keys()),
                        'MAE':      [f"{res[m]['MAE']:.4f}"  for m in res],
                        'RMSE':     [f"{res[m]['RMSE']:.4f}" for m in res],
                        'R² Score': [f"{res[m]['R2']:.4f}"   for m in res],
                    })
                    st.markdown("##### Model Performance")
                    st.dataframe(res_df, use_container_width=True, hide_index=True)

                    best = max(res, key=lambda m: res[m]['R2'])
                    m1, m2, m3 = st.columns(3)
                    m1.metric("🏆 Best Model", best)
                    m2.metric("Best R²",       f"{res[best]['R2']:.4f}")
                    m3.metric("Best RMSE",     f"${res[best]['RMSE']:.4f}")

                    st.markdown("##### Predictions vs Actual")
                    st.plotly_chart(predictions_chart(y_test, res), use_container_width=True)

    # ── Tab 5: Data Overview ────────────────────────────────────────────
    with tab5:
        st.markdown("#### Raw Data Overview")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**First 10 rows**")
            st.dataframe(stock_data.head(10), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Price Statistics**")
            cols_stat = [c for c in ['Open','Close','High','Low'] if c in stock_data.columns]
            st.dataframe(stock_data[cols_stat].describe().round(4), use_container_width=True)
        st.markdown(f"**Shape:** `{stock_data.shape[0]} rows × {stock_data.shape[1]} columns`")
        st.dataframe(pd.DataFrame({
            'Column': stock_data.columns,
            'Type':   stock_data.dtypes.astype(str)
        }), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()

