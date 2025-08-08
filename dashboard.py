import os
import json
import re
import requests
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta, time
from streamlit_autorefresh import st_autorefresh
import streamlit.components.v1 as components

# ----------------------- CONFIG AND SETUP -----------------------
st.set_page_config(layout="wide", page_title="Order Flow Dashboard")
REFRESH_ENABLED = st.sidebar.toggle('🔄 Auto-refresh', value=True)
REFRESH_INTERVAL = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if REFRESH_ENABLED:
    st_autorefresh(interval=REFRESH_INTERVAL * 1000, key="auto_refresh")

# Paths and constants
LOCAL_CACHE_DIR = "local_cache"
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"

# Ensure cache directory
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

# ------------------ CSS: Full width, responsive ------------------
def inject_full_width_css():
    st.markdown("""
    <style>
    /* Remove default container padding for full width layout */
    .main > div {
        padding-top: 0rem;
        padding-bottom: 0rem;
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100% !important;
    }

    /* Trading header style */
    .trading-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: white;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        align-items: center;
    }
    .stock-info {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .stock-name {
        font-size: 24px;
        font-weight: bold;
    }
    .price-change-positive {
        color: #22c55e;
        font-weight: 600;
        font-size: 18px;
    }
    .price-change-negative {
        color: #ef4444;
        font-weight: 600;
        font-size: 18px;
    }

    /* Delta boxes container */
    .delta-boxes {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin: 20px 0;
        flex-wrap: wrap;
    }
    .delta-box {
        background: #f0f0f0;
        border-radius: 12px;
        border: 2px solid #d0d0d0;
        padding: 20px 30px;
        min-width: 150px;
        text-align: center;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        font-size: 18px;
    }
    .delta-positive {
        border-color: #22c55e;
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        color: #16a34a;
    }
    .delta-negative {
        border-color: #ef4444;
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        color: #dc2626;
    }
    .delta-neutral {
        border-color: #6b7280;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        color: #6b7280;
    }

    /* Lightweight Chart container */
    .lightweight-chart-container {
        width: 100% !important;
        height: 600px;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin: 20px 0;
        position: relative;
    }

    /* Responsive adjustments for smaller screens */
    @media(max-width:768px){
        .trading-header {
            flex-direction: column;
            align-items: flex-start;
            padding: 15px;
        }
        .stock-info {
            flex-direction: column;
            gap: 10px;
        }
        .delta-boxes {
            gap: 15px;
        }
        .delta-box {
            min-width: 120px;
            padding:15px 20px;
            font-size: 16px;
        }
        .delta-positive, .delta-negative, .delta-neutral {
            font-size: 16px;
        }
        .lightweight-chart-container {
            height: 500px !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)

inject_full_width_css()

# ------------------ Data and Auth ------------------
@st.cache_data
def load_stock_mapping():
    try:
        df = pd.read_csv(STOCK_LIST_FILE)
        return {str(k): v for k, v in zip(df['security_id'], df['symbol'])}
    except Exception:
        return {}

stock_mapping = load_stock_mapping()

# ------------------ Sidebar - Security & Options ------------------
def fetch_security_ids():
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
        )
        if resp.status_code == 200:
            files = resp.json()
            ids = set()
            for f in files:
                if f['name'].endswith('.csv'):
                    url = f['download_url']
                    df = pd.read_csv(url)
                    ids.update(df['security_id'].astype(str).unique())
            return sorted(list(ids))
    except:
        pass
    # fallback to stock list
    try:
        df = pd.read_csv(STOCK_LIST_FILE)
        return sorted(df['security_id'].astype(str).unique())
    except:
        return []

security_ids = fetch_security_ids()
selected_option = st.sidebar.selectbox("🎯 Security", ["Loading..."] if not security_ids else 
                                         [f"{stock_mapping.get(str(i), 'Unknown')} ({i})" for i in security_ids])
match = re.search(r'\((\d+)\)', selected_option)
if match:
    selected_id = int(match.group(1))
else:
    st.error("Invalid selection")
    st.stop()

interval_mins = st.sidebar.selectbox("⏱️ Interval (Minutes)", [1,3,5,15,30], index=2)

# ------------------ Data Fetching Utilities ------------------
def fetch_live_data(security_id):
    try:
        url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if len(df):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            return df
    except:
        pass
    return pd.DataFrame()

def load_local_cache(security_id):
    fp = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
    if os.path.exists(fp):
        try:
            df = pd.read_csv(fp)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df.sort_values('timestamp')
        except:
            pass
    return pd.DataFrame()

def save_local_cache(df, security_id):
    fp = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
    df.to_csv(fp, index=False)

def get_historical_data(security_id):
    # For simplicity, use cache only
    return load_local_cache(security_id)

# ------------------ Data Aggregation ------------------
def aggregate_data(df, interval):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    agg = df.resample(f'{interval}min').agg({
        'buy_initiated':'sum',
        'sell_initiated':'sum',
        'open':'first',
        'high':'max',
        'low':'min',
        'close':'last',
        'buy_volume':'sum',
        'sell_volume':'sum'
    }).dropna().reset_index()
    agg['tick_delta'] = agg['buy_initiated'] - agg['sell_initiated']
    agg['cumulative_tick_delta'] = agg['tick_delta'].cumsum()
    agg['delta'] = agg['buy_volume'] - agg['sell_volume']
    agg['cumulative_delta'] = agg['delta'].cumsum()
    return agg

# ------------------ Chart Functions ------------------
def create_lightweight_chart(title, df):
    if df.empty:
        return '<div style="text-align:center; padding: 50px; font-size:16px; color:#999;">No data available</div>'
    candles = []
    for _, row in df.tail(100).iterrows():
        try:
            t = int(pd.to_datetime(row['timestamp']).timestamp())
            candles.append({
                'time': t,
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close'])
            })
        except:
            continue
    chart_id = f"chart_{title.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    html = f"""
    <div class="lightweight-chart-container">
        <div id="{chart_id}"></div>
    </div>
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const container = document.getElementById('{chart_id}');
        if(!container){{
            container.innerHTML='Chart container not found';
            return;
        }}
        container.innerHTML='';
        const chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: 600,
            layout: {{ background: {{ color: '#ffffff' }}, textColor: '#000' }},
            grid: {{ vertLines: {{ color:'#f0f0f0' }}, horzLines: {{ color:'#f0f0f0' }} }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
            rightPriceScale: {{ borderColor: '#ccc', scaleMargins: {{ top: 0.1, bottom: 0.1 }} }},
            timeScale: {{ borderColor:'#ccc', secondsVisible: false, }}
        }});
        const candleSeries=chart.addCandlestickSeries({{
            upColor:'#22c55e',
            downColor:'#ef4444',
            borderVisible:false,
            wickUpColor:'#22c55e',
            wickDownColor:'#ef4444'
        }});
        candleSeries.setData({json.dumps(candles)});
        chart.timeScale().fitContent();

        const resizeObserver = new ResizeObserver(entries => {{
            if(!entries.length)return;
            chart.applyOptions({{ width: container.clientWidth }});
        }});
        resizeObserver.observe(container);
        window.addEventListener('beforeunload', ()=>{{
            resizeObserver.disconnect();
            chart.remove();
        }});
    }})();
    </script>
    """
    return html

def create_delta_boxes(df):
    if df.empty:
        return ""
    latest = df.iloc[-1]
    tick_delta = int(latest.get('tick_delta',0))
    cum_delta = int(latest.get('cumulative_tick_delta',0))
    def delta_class(val):
        if val > 0: return "delta-positive"
        elif val < 0: return "delta-negative"
        else: return "delta-neutral"
    def format_val(val):
        sign = "+" if val>0 else ""
        return f"{sign}{val:,}"
    return f"""
    <div class="delta-boxes">
        <div class="delta-box {delta_class(tick_delta)}">
            <div class="delta-value">{format_val(tick_delta)}</div>
            <div class="delta-label">Tick Δ</div>
        </div>
        <div class="delta-box {delta_class(cum_delta)}">
            <div class="delta-value">{format_val(cum_delta)}</div>
            <div class="delta-label">Cumulative Δ</div>
        </div>
    </div>
    """

# ------------------ Main App ------------------
# Header
if 'selected_option' not in st.session_state:
    st.session_state['selected_option'] = None

# Sidebar security selection
security_list = ["Loading..."] if not security_ids else [f"{stock_mapping.get(str(i), 'Unknown')} ({i})" for i in security_ids]
selected_option = st.sidebar.selectbox("🎯 Security", security_list)
match = re.search(r'\((\d+)\)', selected_option)
if match:
    security_id=int(match.group(1))
else:
    st.error("Invalid selection")
    st.stop()

# Fetch data
with st.spinner("Loading data..."):
    live_data = fetch_live_data(security_id)
    today = datetime.now().date()
    start_time = datetime.combine(today, time(9))
    end_time = datetime.combine(today, time(23,59))
    live_data = live_data[(live_data['timestamp']>=pd.Timestamp(start_time)) &
                          (live_data['timestamp']<=pd.Timestamp(end_time))]
    agg_df = aggregate_data(live_data, interval_mins)

# Prepare header info
if not agg_df.empty:
    latest = agg_df.iloc[-1]
    current_price = f"{latest.get('close',0):.1f}"
    prev_close = agg_df.iloc[-2]['close'] if len(agg_df)>1 else latest.get('open',0)
    change = latest.get('close',0) - prev_close
    change_pct = (change/prev_close*100) if prev_close else 0
    change_str = f"↑ +{change:.1f}" if change>0 else f"↓ {change:.1f}"
    change_pct_str = f"({change_pct:+.2f}%)"
    price_class = "price-change-positive" if change>0 else "price-change-negative"
else:
    current_price = "---"
    change_str = "---"
    change_pct_str = "---"
    price_class = ""

# Render header
st.markdown(f"""
<div class="trading-header">
    <div class="stock-info">
        <div style="font-size:20px; font-weight:600;">{selected_option.split('(')[0].strip()}</div>
        <div class="{price_class}" style="font-size:16px;">{current_price} {change_str} {change_pct_str}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Delta boxes
if not agg_df.empty:
    st.markdown(create_delta_boxes(agg_df), unsafe_allow_html=True)

# Chart
if not agg_df.empty:
    chart_html = create_lightweight_chart(selected_option.split('(')[0].strip(), agg_df)
    components.html(chart_html, height=650, width=0)  # width=0 for full width
else:
    st.info("No data available for this security.")

# Additional Metrics/Table (if any)
if not agg_df.empty:
    # Example: Show latest session stats
    latest = agg_df.iloc[-1]
    st.metric("Buy Volume", f"{int(latest['buy_initiated']):,}")
    st.metric("Sell Volume", f"{int(latest['sell_initiated']):,}")
    total_vol = int(latest['buy_initiated']) + int(latest['sell_initiated'])
    st.metric("Total Volume", f"{total_vol:,}")
    net_delta = int(agg_df['tick_delta'].sum())
    st.metric("Net Δ", f"{net_delta:+,}")
else:
    st.write("No session data available.")

# End -----------------------
