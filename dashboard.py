import os
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, time
import re
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# --- Page Config & Auto-refresh ---
st.set_page_config(layout="wide", page_title="Order Flow Dashboard")
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="refresh", limit=None)

# --- CSS for Full-Width Responsive Charts & Layout ---
def inject_full_width_chart_css():
    st.markdown("""
    <style>
    /* Remove default padding from main area */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
    }
    /* Full width container for charts */
    .element-container {
        width: 100% !important;
        max-width: 100% !important;
    }
    /* Chart container styling for responsiveness */
    .lightweight-chart-container {
        width: 100% !important;
        height: auto;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        margin-top: 20px;
        position: relative;
    }
    /* Header style */
    .trading-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    /* Responsive for mobile */
    @media(max-width:768px){
        .trading-header { padding:15px; }
        .stock-name { font-size:20px; }
        .delta-boxes { flex-direction: column; align-items:center; }
        .delta-box { min-width:80px; padding:15px; }
        .delta-value { font-size:24px; }
    }
    /* Delta boxes styles */
    .delta-boxes {
        display:flex;
        justify-content:center;
        gap:20px;
        margin:20px 0;
        flex-wrap:wrap;
    }
    .delta-box {
        background:#fff;
        border:2px solid #e5e7eb;
        border-radius:12px;
        padding:20px;
        min-width:120px;
        text-align:center;
        transition: all 0.3s;
    }
    .delta-positive {
        border-color:#22c55e;
        background:linear-gradient(135deg,#f0fdf4,#dcfce7);
    }
    .delta-negative {
        border-color:#ef4444;
        background:linear-gradient(135deg,#fef2f2,#fee2e2);
    }
    .delta-neutral {
        border-color:#6b7280;
        background:linear-gradient(135deg,#f9fafb,#f3f4f6);
    }
    .delta-value { font-size:28px; font-weight:700; margin:0; color:#111; }
    .delta-label { font-size:14px; color:#6b7280; margin-top:8px; }
    </style>
    """, unsafe_allow_html=True)

inject_full_width_chart_css()

# --- Load stock list and mappings ---
STOCK_LIST_PATH = "stock_list.csv"
@st.cache_data
def load_stock_list():
    try:
        df = pd.read_csv(STOCK_LIST_PATH)
        return {str(row['security_id']): row['symbol'] for idx, row in df.iterrows()}
    except:
        return {}
stock_mapping = load_stock_list()

# --- Fetch list of security options ---
@st.cache_data(ttl=6000)
def get_security_options():
    try:
        df = pd.read_csv(STOCK_LIST_PATH)
        options = [f"{row['symbol']} ({row['security_id']})" for idx, row in df.iterrows()]
        return options
    except:
        return ["0 (0)"]

security_options = get_security_options()

# --- User inputs ---
selected_option = st.sidebar.selectbox("🎯 Security", security_options)
match_security_id = re.search(r'\((\d+)\)', selected_option)
security_id = int(match_security_id.group(1)) if match_security_id else 0
if security_id == 0:
    st.error("⚠️ Invalid security selected.")
    st.stop()

interval = st.sidebar.selectbox("⏱️ Interval (min)", [1,3,5,15,30,60], index=2)
mobile_mode = st.sidebar.toggle("📱 Mobile Mode", value=True)

# --- API URL ---
API_BASE = "https://comoflo.onrender.com/api"

# --- Data fetch functions ---
@st.cache_data
def fetch_live_data(security_id):
    url = f"{API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            df = pd.DataFrame(r.json())
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                return df
    except:
        pass
    return pd.DataFrame()

@st.cache_data
def aggregate_df(df, interval):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    res = df.resample(f"{interval}min").agg({
        'buy_initiated':'sum',
        'sell_initiated':'sum',
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'buy_volume':'sum',
        'sell_volume':'sum'
    }).dropna().reset_index()
    res['tick_delta'] = res['buy_initiated'] - res['sell_initiated']
    res['cumulative_tick_delta'] = res['tick_delta'].cumsum()
    res['delta'] = res['buy_volume'] - res['sell_volume']
    res['cumulative_delta'] = res['delta'].cumsum()
    return res

# --- Fetch historical data / cache ---
@st.cache_data
def load_cache(security_id):
    path = f"local_cache/cache_{security_id}.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def save_cache(df, security_id):
    os.makedirs("local_cache", exist_ok=True)
    df.to_csv(f"local_cache/cache_{security_id}.csv", index=False)

# --- Make full dataset ---
historical_df = load_cache(security_id)
live_df = fetch_live_data(security_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates('timestamp').sort_values('timestamp')

# Filter current day 9am - 11:59pm
today = datetime.now().date()
start_dt = datetime.combine(today, time(9,0))
end_dt = datetime.combine(today, time(23,59))
full_df = full_df[(full_df['timestamp']>=start_dt)&(full_df['timestamp']<=end_dt)]
agg_df = aggregate_df(full_df, interval)

# --- Function to create the lightweight chart HTML ---
def create_lightweight_chart(stock_name, df, interval):
    if df.empty:
        return '<div style="text-align:center; padding:50px; color:#888;">No data available</div>'
    candles = []
    for _, row in df.tail(200).iterrows():
        ts_unix = int(pd.to_datetime(row['timestamp']).timestamp())
        candles.append({
            'time': ts_unix,
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close'])
        })
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    html = f"""
    <div class="lightweight-chart-container">
        <div id="{chart_id}" style="width:100%; height:500px;"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const container = document.getElementById('{chart_id}');
        if (!container || typeof LightweightCharts === 'undefined') {{
            container.innerHTML='<div style="text-align:center; padding:50px; color:#888;">Chart library not loaded</div>';
            return;
        }}
        container.innerHTML = '';
        const chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: 500,
            layout: {{
                background: {{ color: '#fff' }},
                textColor: '#333'
            }},
            grid: {{
                vertLines: {{ color: '#f0f0f0' }},
                horzLines: {{ color: '#f0f0f0' }}
            }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{
                borderColor: '#ccc',
                scaleMargins: {{ top: 0.1, bottom: 0.1 }}
            }},
            timeScale: {{
                borderColor: '#ccc',
                timeVisible: true,
                secondsVisible: false
            }}
        }});
        const candlestickSeries = chart.addCandlestickSeries({{
            upColor: '#22c55e',
            downColor: '#ef4444'
        }});
        candlestickSeries.setData({json.dumps(candles)});
        chart.timeScale().fitContent();
        const resizeObserver = new ResizeObserver(entries => {{
            if (entries.length > 0) {{
                const rect = entries[0].contentRect;
                chart.applyOptions({{ width: rect.width }});
            }}
        }});
        resizeObserver.observe(container);
        window.addEventListener('beforeunload', () => resizeObserver.disconnect());
    }})();
    </script>
    """
    return html

# --- Create delta boxes HTML ---
def create_delta_boxes(df):
    if df.empty:
        return ""
    latest = df.iloc[-1]
    tick_delta = int(latest.get('tick_delta', 0))
    cum_delta = int(latest.get('cumulative_tick_delta', 0))
    def delta_class(val):
        if val > 0:
            return "delta-positive"
        elif val < 0:
            return "delta-negative"
        return "delta-neutral"
    window = f"""
    <div class="delta-boxes">
        <div class="delta-box {delta_class(tick_delta)}">
            <div class="delta-value">{tick_delta:+,}</div>
            <div class="delta-label">Tick Δ</div>
        </div>
        <div class="delta-box {delta_class(cum_delta)}">
            <div class="delta-value">{cum_delta:+,}</div>
            <div class="delta-label">Cumulative Δ</div>
        </div>
    </div>
    """
    return window

# --- Render header ---
if not agg_df.empty:
    latest = agg_df.iloc[-1]
    close_price = f"{latest['close']:.1f}"
    prev_close = agg_df.iloc[-2]['close'] if len(agg_df)>1 else latest['open']
    change = latest['close'] - prev_close
    change_pct = (change/prev_close)*100 if prev_close!=0 else 0
    change_str = f"↑ +{change:.1f}" if change>0 else f"↓ {change:.1f}"
    change_pct_str = f"({change_pct:+.2f}%)"
    price_color = "price-positive" if change>0 else "price-negative"
else:
    close_price = "---"
    change_str = "---"
    change_pct_str = "---"
    price_color = "price-negative"

st.markdown(f"""
<div class="trading-header">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div style="display:flex; align-items:center;">
      <div style="background:#3b82f6; padding:6px 12px; border-radius:6px; font-size:12px; font-weight:bold; margin-right:10px;">NSE</div>
      <div class="stock-name" style="font-size:20px; font-weight:bold;">{selected_option.split('(')[0].strip()}</div>
    </div>
    <div class="{price_color}" style="font-size:24px;">{close_price}</div>
    <div style="margin-left:20px; font-size:16px;">{change_str} {change_pct_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# --- Delta boxes ---
if not agg_df.empty:
    st.markdown(create_delta_boxes(agg_df), unsafe_allow_html=True)

# --- Chart ---
if not agg_df.empty:
    chart_html = create_lightweight_chart(agg_df['close'].name, agg_df, interval)
    components.html(chart_html, height=600, width=0)  # width=0 for full container
else:
    st.info("No data to display.")

# --- Metrics Below Chart ---
if not agg_df.empty:
    latest = agg_df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Buy Vol.", f"{int(latest['buy_volume']):,}")
    col2.metric("Sell Vol.", f"{int(latest['sell_volume']):,}")
    total_vol = int(latest['buy_volume']) + int(latest['sell_volume'])
    col3.metric("Total Vol.", f"{total_vol:,}")
    col4.metric("Net Δ", f"{int(latest['tick_delta']):,}")
else:
    st.write("No data available.")

# --- Final footer ---
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; font-size:12px; color:#6c757d; padding:10px;">
Order Flow Dashboard &nbsp; | &nbsp; Refresh every {refresh_interval}s &nbsp; | &nbsp; Interval: {interval} min
</div>
""", unsafe_allow_html=True)
