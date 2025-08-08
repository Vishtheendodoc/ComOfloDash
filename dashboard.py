import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
import json
from datetime import datetime, timedelta, time as dtime
import re

# --- Constants and Configurations ---
LOCAL_CACHE_DIR = "local_cache"
ALERT_CACHE_DIR = "alert_cache"
STOCK_LIST_FILE = "stock_list.csv"
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"

# --- Startup and Directory Checks ---
os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
os.makedirs(ALERT_CACHE_DIR, exist_ok=True)

# --- Streamlit Config ---
st.set_page_config(layout="wide", page_title="Order Flow Dashboard")
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh", limit=None)

# --- Load Stock Mapping ---
@st.cache_data
def load_stock_mapping():
    try:
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        return {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}
stock_mapping = load_stock_mapping()

# --- Mobile CSS ---
def inject_mobile_css():
    mobile_css = """
    <style>
    @media (max-width: 768px) {
        .main .block-container {padding-top: 1rem !important;padding-left: 1rem !important;padding-right: 1rem !important;max-width: 100% !important;}
        h1 {font-size: 1.5rem !important;margin-bottom: 0.5rem !important;}
        .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);border-radius: 10px;padding: 12px;color: white;text-align: center;margin: 4px 0;}
        .metric-value {font-size: 18px;font-weight: bold;margin: 0;}
        .metric-label {font-size: 11px;opacity: 0.9;margin: 0;}
    }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)

# --- Security ID Load for Sidebar ---
@st.cache_data(ttl=6000)
def fetch_security_ids():
    try:
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        ids = sorted(list(stock_df['security_id'].unique()))
        return [f"{stock_mapping.get(str(i), f'Stock {i}')} ({i})" for i in ids]
    except Exception:
        return ["No Data Available (0)"]

security_options = fetch_security_ids()
selected_option = st.sidebar.selectbox("🎯 Security", security_options)
match = re.search(r'\((\d+)\)', selected_option)
if match:
    selected_id = int(match.group(1))
    if selected_id == 0:
        st.error("⚠️ No security data available. Please check your data source.")
        st.stop()
else:
    st.error(f"⚠️ Selected option '{selected_option}' does not contain a valid ID")
    st.stop()

interval = st.sidebar.selectbox("⏱️ Interval", [1, 3, 5, 15, 30, 60, 90, 120, 180, 240, 360, 480], index=2)
mobile_view = st.sidebar.toggle("📱 Mobile Mode", value=True)

if mobile_view:
    inject_mobile_css()

# --- Data Fetching Utilities ---
def save_to_local_cache(df, security_id):
    if not df.empty:
        cache_file = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
        df.to_csv(cache_file, index=False)

def load_from_local_cache(security_id):
    cache_file = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            return df
        except Exception as e:
            st.warning(f"Failed to load local cache: {e}")
    return pd.DataFrame()

def fetch_historical_data(security_id):
    # Only load from local cache & stock_list for simplicity (remove github version for brevity)
    return load_from_local_cache(security_id)

def fetch_live_data(security_id):
    api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(api_url, timeout=20)
        r.raise_for_status()
        live_data = pd.DataFrame(r.json())
        if not live_data.empty:
            live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
            live_data.sort_values('timestamp', inplace=True)
            save_to_local_cache(live_data, security_id)
            return live_data
    except Exception:
        pass
    return pd.DataFrame()

def aggregate_data(df, interval_minutes):
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    df_agg = df_copy.resample(f"{interval_minutes}min").agg({
        'buy_initiated': 'sum',
        'sell_initiated': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'buy_volume': 'sum',
        'sell_volume': 'sum'
    }).dropna().reset_index()
    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    df_agg['cumulative_tick_delta'] = df_agg['tick_delta'].cumsum()
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    return df_agg

# --- Mobile TradingView-style Chart + Delta Boxes ---
def tradingview_style_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['close'],
        mode='lines',
        line=dict(color='#1a57b4', width=2),
        name='Close Price'
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=30, b=30),
        template="plotly_white",
        showlegend=False,
        uirevision="orderflow-chart"
    )
    fig.update_xaxes(showgrid=False, showticklabels=True, tickformat='%H:%M')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#f0f0f0')
    return fig

def styled_delta_boxes(df):
    if df.empty:
        return
    latest = df.iloc[-1]
    tick_delta = int(latest['tick_delta']) if pd.notna(latest['tick_delta']) else 0
    cum_delta = int(latest['cumulative_tick_delta']) if pd.notna(latest['cumulative_tick_delta']) else 0

    def box(value, label):
        color = "#26a69a" if value > 0 else "#ef5350" if value < 0 else "#757575"
        sign = "+" if value > 0 else ""
        return f"""
            <div style="
                display:inline-block;
                min-width:90px;
                padding:10px 18px;
                background:{color};
                border-radius:8px;
                margin:8px 8px 0 0;
                color:white;
                font-weight:bold;
                font-size:20px;
                text-align:center;
            ">{label}: {sign}{value}</div>
        """
    st.markdown(
        f"<div style='text-align:center;'>{box(tick_delta,'TΔ')}{box(cum_delta,'CumΔ')}</div>",
        unsafe_allow_html=True
    )

# --- Mobile Metrics ---
def create_mobile_metrics(df):
    if df.empty: return
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card"><p class="metric-value">{float(latest['close']):.1f}</p>
        <p class="metric-label">Price</p></div>""", unsafe_allow_html=True)
    with col2:
        tick_delta = int(latest['tick_delta'])
        col = "#26a69a" if tick_delta >= 0 else "#ef5350"
        sign = "+" if tick_delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="background: {col};">
            <p class="metric-value">{sign}{tick_delta}</p>
            <p class="metric-label">Tick Δ</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        cum_delta = int(latest['cumulative_tick_delta'])
        col = "#26a69a" if cum_delta >= 0 else "#ef5350"
        sign = "+" if cum_delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="background: {col};">
            <p class="metric-value">{sign}{cum_delta}</p>
            <p class="metric-label">Cum Δ</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        buy_vol = float(latest['buy_initiated'])
        sell_vol = float(latest['sell_initiated'])
        vol_total = buy_vol + sell_vol
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{vol_total:,.0f}</p>
            <p class="metric-label">Volume</p>
        </div>
        """, unsafe_allow_html=True)

# --- Fetch and Process Data ---
historical_df = fetch_historical_data(selected_id)
live_df = fetch_live_data(selected_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

today = datetime.now().date()
start_time = datetime.combine(today, dtime(9, 0))
end_time = datetime.combine(today, dtime(23, 59, 59))
full_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]
agg_df = aggregate_data(full_df, interval)

# --- Main Mobile-Optimized Display ---
if mobile_view:
    stock_name = selected_option.split(' (')[0]
    st.markdown(f"# 📊 {stock_name}")
    st.caption(f"🔄 Updates every {refresh_interval}s • {interval}min intervals")
    if not agg_df.empty:
        create_mobile_metrics(agg_df)
        st.markdown("---")
        st.markdown("### 📈 Chart (Mobile Optimized)")
        fig = tradingview_style_chart(agg_df)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
        styled_delta_boxes(agg_df)
        st.markdown("---")
        # Optional: add recent table etc.
        csv = agg_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Data",
            csv,
            f"orderflow_{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.error("📵 No data available for this security")

else:
    # Desktop - simple chart and table for demonstration
    st.title(f"Order Flow Dashboard: {selected_option}")
    if not agg_df.empty:
        st.caption("Full history + live updates")
        fig = tradingview_style_chart(agg_df)
        st.plotly_chart(fig, use_container_width=True)
        styled_delta_boxes(agg_df)
        # Optionally, show table, download, etc.
        csv = agg_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
    else:
        st.warning("No data available for this security.")
