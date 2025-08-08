import os
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, time
import re
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# Configure page
st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Auto-refresh logic
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh", limit=None)

# Inject CSS for full width and mobile-optimized layout
def inject_full_width_chart_css():
    st.markdown("""
    <style>
        /* Reset default padding and make all containers full width */
        .main > div {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        .element-container {
            width: 100% !important;
            max-width: 100% !important;
        }
        /* Header styling */
        .trading-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stock-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 20px;
        }
        .stock-name {
            font-size: 24px;
            font-weight: bold;
        }
        .price-positive {
            color: #22c55e;
            font-weight: bold;
        }
        .price-negative {
            color: #ef4444;
            font-weight: bold;
        }
        /* Delta boxes styling */
        .delta-boxes {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 20px 0;
            flex-wrap: wrap;
        }
        .delta-box {
            background: white;
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 20px 30px;
            min-width: 120px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: transform 0.2s;
        }
        .delta-positive {
            border-color: #22c55e;
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        }
        .delta-negative {
            border-color: #ef4444;
            background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        }
        .delta-neutral {
            border-color: #6b7280;
            background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        }
        .delta-value {
            font-size: 28px;
            font-weight: bold;
            margin: 0;
            line-height: 1;
        }
        .delta-label {
            font-size: 14px;
            color: #6b7280;
            margin: 8px 0 0 0;
            font-weight: 500;
        }
        .delta-positive .delta-value {
            color: #16a34a;
        }
        .delta-negative .delta-value {
            color: #dc2626;
        }
        .delta-neutral .delta-value {
            color: #6b7280;
        }
        /* Chart container styling */
        .lightweight-chart-container {
            width: 100% !important;
            height: 500px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            margin: 20px 0;
            position: relative;
        }
        /* Responsive for mobile */
        @media (max-width: 768px) {
            .trading-header {
                padding: 15px;
            }
            .stock-name {
                font-size: 20px;
            }
            .stock-info {
                flex-direction: column;
                align-items: flex-start;
                gap: 15px;
            }
            .delta-boxes {
                gap: 15px;
            }
            .delta-box {
                min-width: 100px;
                padding: 15px 20px;
            }
            .delta-value {
                font-size: 24px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

inject_full_width_chart_css()

# Create cache directory
os.makedirs("local_cache", exist_ok=True)
GITHUB_API_URL = "https://api.github.com/repos/Vishtheendodoc/ComOflo/contents/data_snapshots"
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")

# Load stock mapping
@st.cache_data
def load_stock_mapping():
    try:
        df = pd.read_csv("stock_list.csv")
        return {str(k): v for k, v in zip(df['security_id'], df['symbol'])}
    except:
        return {}
stock_mapping = load_stock_mapping()

# Fetch security options
@st.cache_data(ttl=6000)
def get_security_options():
    try:
        resp = requests.get(GITHUB_API_URL, headers={"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {})
        files = resp.json()
        ids = set()
        for file in files:
            if file['name'].endswith('.csv'):
                df = pd.read_csv(file['download_url'])
                ids.update(df['security_id'].astype(str))
        ids = list(ids)
        ids.sort()
        return [f"{stock_mapping.get(i, f'Stock {i}')} ({i})" for i in ids]
    except:
        # Fallback: load from local stock list
        try:
            df = pd.read_csv("stock_list.csv")
            ids = df['security_id'].astype(str).unique()
            return [f"{stock_mapping.get(i, f'Stock {i}')} ({i})" for i in ids]
        except:
            return ["No Data Available (0)"]
security_options = get_security_options()

# Sidebar inputs
st.sidebar.title("📊 Order Flow")
selected_option = st.sidebar.selectbox("🎯 Security", security_options)
match = re.search(r'\((\d+)\)', selected_option)
if match:
    security_id = int(match.group(1))
else:
    st.error("Invalid selection")
    st.stop()
interval = st.sidebar.selectbox("⏱️ Interval (min)", [1,3,5,15,30,60], index=2)

# Load and process data
def fetch_live_data(sec_id):
    url = f"https://comoflo.onrender.com/api/delta_data/{sec_id}?interval=1"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
        return df
    except:
        return pd.DataFrame()

def aggregate_df(df, interval_mins):
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    df.set_index('timestamp', inplace=True)
    agg = df.resample(f"{interval_mins}min").agg({
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

with st.spinner("Loading data..."):
    live_df = fetch_live_data(security_id)
    today = datetime.now().date()
    start_day = datetime.combine(today, time(9,0))
    end_day = datetime.combine(today, time(23,59))
    live_df = live_df[(live_df['timestamp'] >= start_day) & (live_df['timestamp'] <= end_day)]
    agg_df = aggregate_df(live_df, interval)

# Display header
stock_name = re.sub(r' \(.+\)$', '', selected_option.split('(')[0]).strip()
if not agg_df.empty:
    last_row = agg_df.iloc[-1]
    current_price = f"{last_row.get('close', '---'):.1f}"
    if len(agg_df) > 1:
        prev_price = agg_df.iloc[-2].get('close', last_row.get('open', 0))
        change = last_row.get('close') - prev_price
        change_pct = (change/prev_price*100) if prev_price !=0 else 0
        change_str = f"↑ +{change:.1f}" if change>0 else f"↓ {change:.1f}"
        change_pct_str = f"({change_pct:+.2f}%)"
        price_class = "price-positive" if change>0 else "price-negative"
    else:
        change_str = "---"
        change_pct_str = "---"
        price_class = "price-negative"
else:
    current_price = "---"
    change_str = "---"
    change_pct_str = "---"
    price_class = "price-negative"

st.markdown(f"""
<div class="trading-header">
<div class="stock-info">
    <div style="display:flex; align-items:center;">
        <span style="background:#3b82f6; padding:6px 12px; border-radius:6px; font-size:12px; font-weight:bold; margin-right:15px;">NSE</span>
        <span class="stock-name">{stock_name}</span>
    </div>
    <div style="display:flex; align-items:center; margin-top:10px;">
        <span style="font-size:32px; font-weight:bold;">{current_price}</span>
        <span class="{price_class}" style="font-size:18px; margin-left:15px;">{change_str} {change_pct_str}</span>
    </div>
</div>
</div>
""", unsafe_allow_html=True)

# Delta Boxes
def create_delta_boxes(df):
    if df.empty:
        return ""
    latest = df.iloc[-1]
    tick_delta = int(latest['tick_delta'])
    cum_delta = int(latest['cumulative_tick_delta'])
    def box_class(v):
        if v>0:
            return "delta-positive"
        elif v<0:
            return "delta-negative"
        return "delta-neutral"
    tick_class = box_class(tick_delta)
    cum_class = box_class(cum_delta)
    def format_value(v):
        sign = "+" if v>0 else ""
        return f"{sign}{v:,}"
    return f"""
    <div class="delta-boxes">
        <div class="delta-box {tick_class}">
            <div class="delta-value">{format_value(tick_delta)}</div>
            <div class="delta-label">Tick Δ</div>
        </div>
        <div class="delta-box {cum_class}">
            <div class="delta-value">{format_value(cum_delta)}</div>
            <div class="delta-label">Cumulative Δ</div>
        </div>
    </div>
    """

if not agg_df.empty:
    st.markdown(create_delta_boxes(agg_df), unsafe_allow_html=True)

# Create lightweight chart HTML
def create_tradingview_chart(stock_name, chart_data, interval):
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    candle_data = []
    for _, row in chart_data.tail(100).iterrows():
        try:
            timestamp = int(pd.to_datetime(row['timestamp']).timestamp())
            candle_data.append({
                'time': timestamp,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0))
            })
        except:
            continue
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    chart_html = f"""
<div class="lightweight-chart-container">
    <div id="{chart_id}" style="width: 100%; height: 100%;"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    const container = document.getElementById('{chart_id}');
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">Chart library not loaded</div>';
        return;
    }}
    container.innerHTML = '';
    const chart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 500,
        layout: {{
            background: {{ type: 'solid', color: '#ffffff' }},
            textColor: '#333'
        }},
        grid: {{
            vertLines: {{ color: '#f0f0f0' }},
            horzLines: {{ color: '#f0f0f0' }}
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal
        }},
        rightPriceScale: {{
            borderColor: '#ccc',
            scaleMargins: {{ top: 0.1, bottom: 0.1 }}
        }},
        timeScale: {{
            borderColor: '#ccc',
            timeVisible: true,
            secondsVisible: false
        }},
        autoSize: true
    }});
    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444'
    }});
    candleSeries.setData({json.dumps(candle_data)});
    chart.timeScale().fitContent();
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries.contentRect;
        chart.applyOptions({{ width: rect.width, height: rect.height }});
    }});
    resizeObserver.observe(container);
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        chart.remove();
    }});
}})();
</script>
    """
    return chart_html

# Display the chart drawer
if not agg_df.empty:
    chart_html = create_tradingview_chart(stock_name, agg_df, interval)
    components.html(chart_html, height=600, width=0)

# Optionally, add some metrics or footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6b7280; font-size: 12px; padding: 20px;">
    📊 Order Flow Dashboard | Responsive TradingView-style charts
</div>
""", unsafe_allow_html=True)

