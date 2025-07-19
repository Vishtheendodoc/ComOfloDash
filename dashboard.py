import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="data_refresh")

# --- Config ---
GITHUB_USER = "Vishtheendodoc"       # 🔥 Replace with your GitHub username
GITHUB_REPO = "ComOflo"              # 🔥 Replace with your GitHub repo name
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"  # 🔥 Replace with your Flask API URL
STOCK_LIST_FILE = "stock_list.csv"   # 🔥 CSV mapping security_id → stock_name

# --- Load stock mapping ---
@st.cache_data
def load_stock_mapping():
    try:
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        # Force security_id to string for consistent matching
        mapping = {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
        return mapping
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}

stock_mapping = load_stock_mapping()

# --- Sidebar Controls ---
st.sidebar.title("Order Flow Controls")

@st.cache_data(ttl=600)
def fetch_security_ids():
    """Get unique security IDs from GitHub and map to stock names"""
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
    # Map to stock names or fallback to Unknown
    return [f"{stock_mapping.get(str(i), 'Unknown')} ({i})" for i in ids]

security_options = fetch_security_ids()
selected_option = st.sidebar.selectbox("Select Security", security_options)
selected_id = int(selected_option.split('(')[-1].strip(')'))
interval = st.sidebar.selectbox("Interval (minutes)", [1, 3, 5, 15, 30])
show_volume_overlay = st.sidebar.checkbox("Show Volume Overlay", value=False)

# --- Mobile/Desktop Toggle ---
mobile_view = st.sidebar.toggle("📱 Mobile View", value=False)

# --- Fetch Historical Data ---
@st.cache_data(ttl=600)
def fetch_historical_data(security_id):
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    try:
        resp = requests.get(base_url, headers=headers)
        if resp.status_code == 404:
            st.warning("📂 No historical data yet. Showing live data only.")
            return pd.DataFrame()
        resp.raise_for_status()
        files = resp.json()
    except Exception as e:
        st.error(f"GitHub API error: {e}")
        return pd.DataFrame()

    combined_df = pd.DataFrame()
    for file_info in files:
        if file_info['name'].endswith('.csv'):
            df = pd.read_csv(file_info['download_url'])
            df = df[df['security_id'] == str(security_id)]
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    if not combined_df.empty:
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df.sort_values('timestamp', inplace=True)
    return combined_df

# --- Fetch Live Data ---
def fetch_live_data(security_id):
    api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(api_url, timeout=20)
        r.raise_for_status()
        live_data = pd.DataFrame(r.json())
        if not live_data.empty:
            live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
            live_data.sort_values('timestamp', inplace=True)
            return live_data
    except Exception as e:
        st.warning(f"Live API fetch failed: {e}")
    return pd.DataFrame()

historical_df = fetch_historical_data(selected_id)
live_df = fetch_live_data(selected_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# --- Aggregate Data ---
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
    df_agg['inference'] = df_agg['tick_delta'].apply(
        lambda x: 'Buy Dominant' if x > 0 else ('Sell Dominant' if x < 0 else 'Neutral')
    )
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    
    return df_agg

agg_df = aggregate_data(full_df, interval)

# --- Format Data ---
agg_df_formatted = agg_df.copy()

# Round close price to 1 decimal
agg_df_formatted['close'] = agg_df_formatted['close'].round(1)
agg_df_formatted['close'] = agg_df_formatted['close'].map("{:.1f}".format)

# Round volumes and deltas to whole numbers
for col in ['buy_volume', 'sell_volume', 'buy_initiated', 'sell_initiated',
            'delta', 'cumulative_delta', 'tick_delta', 'cumulative_tick_delta']:
    agg_df_formatted[col] = agg_df_formatted[col].round(0).astype(int)

# Keep only selected columns for table (remove open, high, low)
columns_to_show = [
    'timestamp', 'close', 'buy_volume', 'sell_volume',
    'buy_initiated', 'sell_initiated', 'tick_delta',
    'cumulative_tick_delta', 'delta', 'cumulative_delta', 'inference'
]
agg_df_table = agg_df_formatted[columns_to_show]

# --- Display ---
# --- Display ---
st.title(f"Order Flow Dashboard: {selected_option}")

if not agg_df_formatted.empty:
    st.caption("Full history + live updates every 5s")

    # Compact table styling for better mobile visibility
    if mobile_view:
        compact_table_css = """
        <style>
        div[data-testid="stDataFrame"] div[data-testid="stHorizontalBlock"] {
            overflow-x: auto;
            font-size: 12px; /* Smaller font for compact view */
        }
        </style>
        """
        st.markdown(compact_table_css, unsafe_allow_html=True)

        # Apply smaller font and tighter padding
        agg_df_table_styled = agg_df_table.style \
            .background_gradient(cmap="RdYlGn", subset=['tick_delta', 'cumulative_tick_delta']) \
            .set_table_styles([{
                'selector': 'th, td',
                'props': [('font-size', '12px'), ('padding', '2px')]
            }])
    else:
        # Normal styling for desktop
        agg_df_table_styled = agg_df_table.style.background_gradient(
            cmap="RdYlGn", subset=['tick_delta', 'cumulative_tick_delta']
        )

    st.dataframe(
        agg_df_table_styled,
        use_container_width=True,
        height=300 if mobile_view else 600
    )

    if mobile_view:
        # Mobile tabs: Compact charts
        tab1, tab2, tab3 = st.tabs(["📊 Candlestick", "📈 Volume", "📉 Tick Delta"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=agg_df['timestamp'],
                open=agg_df['open'],
                high=agg_df['high'],
                low=agg_df['low'],
                close=agg_df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))
            fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=agg_df['buy_initiated'],
                                     name="Buy Initiated", marker_color='green', opacity=0.6))
            fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=-agg_df['sell_initiated'],
                                     name="Sell Initiated", marker_color='red', opacity=0.6))
            fig_vol.update_layout(barmode='overlay', height=300, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)

        with tab3:
            fig_tick = go.Figure()
            fig_tick.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['cumulative_tick_delta'],
                                          mode='lines', line=dict(color='blue', width=3)))
            fig_tick.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), template="plotly_white")
            st.plotly_chart(fig_tick, use_container_width=True)

    else:
        # Desktop full view
        st.subheader("Candlestick Chart with Order Flow")
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=agg_df['timestamp'],
            open=agg_df['open'],
            high=agg_df['high'],
            low=agg_df['low'],
            close=agg_df['close'],
            name='OHLC',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ))
        fig.update_layout(height=700, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        if show_volume_overlay:
            st.subheader("Volume Overlay")
            fig_vol = go.Figure()
            fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=agg_df['buy_initiated'],
                                     name="Buy Initiated", marker_color='green', opacity=0.6))
            fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=-agg_df['sell_initiated'],
                                     name="Sell Initiated", marker_color='red', opacity=0.6))
            fig_vol.update_layout(barmode='overlay', height=500, template="plotly_white")
            st.plotly_chart(fig_vol, use_container_width=True)

        st.subheader("Cumulative Tick Delta")
        fig_tick = go.Figure()
        fig_tick.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['cumulative_tick_delta'],
                                      mode='lines', line=dict(color='blue', width=3)))
        fig_tick.update_layout(height=500, template="plotly_white")
        st.plotly_chart(fig_tick, use_container_width=True)

    # Download Button
    csv = agg_df_table.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
else:
    st.warning("No data available for this security.")
