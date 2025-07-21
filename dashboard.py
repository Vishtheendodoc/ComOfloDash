import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import time

# 🩹 Preload Plotly and Streamlit JS to fix dynamic import errors
st.markdown("""
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script type="module" src="https://cdn.jsdelivr.net/npm/streamlit"></script>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="data_refresh")

# --- Config ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"

# --- Load stock mapping ---
@st.cache_data
def load_stock_mapping():
    try:
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        mapping = {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
        return mapping
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}

stock_mapping = load_stock_mapping()

# --- Enhanced Mobile CSS ---
def inject_mobile_css():
    mobile_css = """
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .css-1d391kg { padding: 0.5rem !important; }
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        h1 { font-size: 1.5rem !important; margin-bottom: 0.5rem !important; }
        .mobile-table { font-size: 11px !important; overflow-x: auto; white-space: nowrap; }
        .mobile-table th {
            padding: 4px 2px !important;
            font-size: 10px !important;
            font-weight: bold !important;
            background-color: #f0f2f6 !important;
        }
        .mobile-table td { padding: 3px 2px !important; border-bottom: 1px solid #e6e6e6 !important; }
        .positive { color: #26a69a !important; font-weight: bold !important; }
        .negative { color: #ef5350 !important; font-weight: bold !important; }
        .neutral { color: #757575 !important; }
        .stTabs [data-baseweb="tab-list"] { gap: 4px; }
        .stTabs [data-baseweb="tab"] { padding: 8px 12px !important; font-size: 12px !important; }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 12px;
            color: white;
            text-align: center;
            margin: 4px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .metric-value { font-size: 18px; font-weight: bold; margin: 0; }
        .metric-label { font-size: 11px; opacity: 0.9; margin: 0; }
        .stDownloadButton button {
            width: 100% !important;
            border-radius: 20px !important;
            font-size: 12px !important;
            padding: 8px !important;
        }
    }
    </style>
    """
    st.markdown(mobile_css, unsafe_allow_html=True)

# --- Sidebar Controls ---
st.sidebar.title("📱 Order Flow")
@st.cache_data(ttl=600)
def fetch_security_ids():
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
    r = requests.get(base_url, headers=headers)
    r.raise_for_status()
    files = r.json()
    ids = set()
    for file in files:
        if file['name'].endswith('.csv'):
            df = pd.read_csv(file['download_url'])
            ids.update(df['security_id'].unique())
    ids = sorted(list(ids))
    return [f"{stock_mapping.get(str(i), 'Unknown')} ({i})" for i in ids]

try:
    security_options = fetch_security_ids()
    selected_option = st.sidebar.selectbox("🎯 Security", security_options)
    selected_id = int(selected_option.split('(')[-1].strip(')'))
except Exception as e:
    st.sidebar.error("❌ Error loading security list")
    selected_id = 438425  # Default fallback

interval = st.sidebar.selectbox("⏱️ Interval", [1, 3, 5, 15, 30], index=2)
mobile_view = st.sidebar.toggle("📱 Mobile Mode", value=True)
if st.sidebar.button("🔄 Refresh All Data"):
    st.cache_data.clear()
    st.rerun()

# --- Data Fetching Functions ---
@st.cache_data(ttl=300)
def fetch_historical_data_enhanced(security_id):
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {}
    if 'GITHUB_TOKEN' in st.secrets:
        headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    try:
        resp = requests.get(base_url, headers=headers, timeout=30)
        resp.raise_for_status()
        files = resp.json()
    except Exception as e:
        st.error(f"❌ GitHub API error: {e}")
        return pd.DataFrame()
    csv_files = [f for f in files if f['name'].endswith('.csv')]
    all_dataframes = []
    for file_info in csv_files:
        try:
            df = pd.read_csv(file_info['download_url'])
            if 'security_id' in df.columns:
                sec_data = df[df['security_id'] == security_id]
                if not sec_data.empty:
                    all_dataframes.append(sec_data)
        except Exception as e:
            continue
    if not all_dataframes:
        st.warning("📁 No historical data found.")
        return pd.DataFrame()
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    combined_df = combined_df.dropna(subset=['timestamp'])
    combined_df.sort_values('timestamp', inplace=True)
    combined_df.drop_duplicates('timestamp', keep='last', inplace=True)
    return combined_df

@st.cache_data(ttl=30)
def fetch_live_data(security_id):
    """Fetch live data with fixes for API mismatch"""
    api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(api_url, timeout=15)
        r.raise_for_status()
        live_data = pd.DataFrame(r.json())
        live_data['security_id'] = security_id
        live_data['timestamp'] = pd.to_datetime(
            date.today().strftime('%Y-%m-%d') + ' ' + live_data['timestamp'],
            errors='coerce'
        )
        required_columns = ['timestamp', 'buy_initiated', 'buy_volume', 'close', 'delta', 'high',
                            'inference', 'low', 'open', 'sell_initiated', 'sell_volume',
                            'tick_delta', 'security_id']
        for col in required_columns:
            if col not in live_data.columns:
                live_data[col] = 0
        live_data = live_data.dropna(subset=['timestamp'])
        live_data.sort_values('timestamp', inplace=True)
        st.info(f"✅ Live API returned {len(live_data)} records")
        return live_data
    except Exception as e:
        st.error(f"❌ Live API error: {e}")
        return pd.DataFrame()

def merge_historical_and_live_data(historical_df, live_df):
    """Merge historical and live data"""
    if not historical_df.empty and not live_df.empty:
        latest_hist = historical_df['timestamp'].max()
        live_df = live_df[live_df['timestamp'] > latest_hist]
    combined_df = pd.concat([historical_df, live_df], ignore_index=True)
    combined_df.sort_values('timestamp', inplace=True)
    combined_df.drop_duplicates('timestamp', keep='last', inplace=True)
    return combined_df

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

def create_market_profile_chart(df):
    if df.empty:
        st.warning("📊 No data to render market profile chart.")
        return go.Figure()
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ))
    fig.update_layout(height=500, template="plotly_white")
    return fig

def create_mobile_charts(df):
    if df.empty:
        st.warning("📊 No data to render mobile charts.")
        return
    # Your original mobile chart creation logic here
    ...

# --- Main Display ---
with st.spinner("📚 Loading historical data..."):
    historical_df = fetch_historical_data_enhanced(selected_id)
with st.spinner("📡 Fetching live updates..."):
    live_df = fetch_live_data(selected_id)
full_df = merge_historical_and_live_data(historical_df, live_df)

if not full_df.empty:
    agg_df = aggregate_data(full_df, interval)
else:
    agg_df = pd.DataFrame()

st.write("🔍 Debug Data:", agg_df)

if mobile_view:
    st.markdown(f"# 📊 {selected_option}")
    if not agg_df.empty:
        create_mobile_charts(agg_df)
    else:
        st.error("📵 No data available for charts.")
else:
    st.title(f"Order Flow Dashboard: {selected_option}")
    if not agg_df.empty:
        st.subheader("Candlestick Chart")
        fig = create_market_profile_chart(agg_df)
        st.plotly_chart(fig, use_container_width=True, config={'responsive': True})
    else:
        st.warning("📊 No data to render charts.")
