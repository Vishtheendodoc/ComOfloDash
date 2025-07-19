import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Auto-refresh every 5 seconds
st_autorefresh(interval=50000, key="refresh")  # 5s refresh

# --- Config ---
GITHUB_USER = "Vishtheendodoc"       # 🔥 Replace with your GitHub username
GITHUB_REPO = "ComOfloDash"     # 🔥 Replace with your GitHub repo name
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"  # 🔥 Replace with your Flask API URL

# --- Sidebar Controls ---
st.sidebar.title("Order Flow Controls")

@st.cache_data(ttl=600)
def fetch_security_ids():
    """Get unique security IDs from GitHub historical data"""
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
    return sorted(list(ids))

security_ids = fetch_security_ids()
selected_id = st.sidebar.selectbox("Select Security ID", security_ids)
interval = st.sidebar.selectbox("Interval (minutes)", [1, 3, 5, 15, 30])
show_volume_overlay = st.sidebar.checkbox("Show Volume Overlay", value=False)

# --- Fetch Historical Data ---
@st.cache_data(ttl=600)
def fetch_historical_data(security_id):
    """Combine all CSVs from GitHub for selected security"""
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_PAT']}"}
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
    """Fetch latest live data from Flask API"""
    api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(api_url, timeout=10)
        r.raise_for_status()
        live_data = pd.DataFrame(r.json())
        if not live_data.empty:
            live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
            live_data.sort_values('timestamp', inplace=True)
            return live_data
    except Exception as e:
        st.warning(f"Live API fetch failed: {e}")
    return pd.DataFrame()

live_df = fetch_live_data(selected_id)

# --- Combine Historical + Live ---
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# --- Aggregate Data ---
def aggregate_data(df, interval_minutes):
    df_copy = df.copy()
    df_copy.set_index('timestamp', inplace=True)
    df_agg = df_copy.resample(f"{interval_minutes}min").agg({
        'buy_volume': 'sum',
        'sell_volume': 'sum',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'buy_initiated': 'sum',
        'sell_initiated': 'sum'
    }).dropna().reset_index()

    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    df_agg['cumulative_tick_delta'] = df_agg['tick_delta'].cumsum()
    df_agg['inference'] = df_agg['tick_delta'].apply(
        lambda x: 'Buy Dominant' if x > 0 else ('Sell Dominant' if x < 0 else 'Neutral')
    )
    return df_agg

agg_df = aggregate_data(full_df, interval)

# --- Display ---
st.title(f"Order Flow Dashboard: Security {selected_id}")

if not agg_df.empty:
    st.caption("Full history + live updates every 10s")
    st.dataframe(agg_df)

    # --- Candlestick Chart with Labels ---
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

    # Add Buy/Sell Initiated labels
    for i, row in agg_df.iterrows():
        if row['buy_initiated'] > 0:
            fig.add_annotation(x=row['timestamp'], y=row['high'],
                               text=f"B: {int(row['buy_initiated'])}",
                               showarrow=False, font=dict(color='green', size=10))
        if row['sell_initiated'] > 0:
            fig.add_annotation(x=row['timestamp'], y=row['low'],
                               text=f"S: {int(row['sell_initiated'])}",
                               showarrow=False, font=dict(color='red', size=10))

    # Add Inference Arrows
    for i, row in agg_df.iterrows():
        if row['inference'] == 'Buy Dominant':
            fig.add_annotation(x=row['timestamp'], y=row['high'] * 1.01,
                               text="▲", showarrow=False,
                               font=dict(color='green', size=14))
        elif row['inference'] == 'Sell Dominant':
            fig.add_annotation(x=row['timestamp'], y=row['low'] * 0.99,
                               text="▼", showarrow=False,
                               font=dict(color='red', size=14))

    fig.update_layout(
        height=700,
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Optional Volume Overlay ---
    if show_volume_overlay:
        st.subheader("Volume Overlay")
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=agg_df['buy_initiated'],
                                 name="Buy Initiated", marker_color='green', opacity=0.6))
        fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=-agg_df['sell_initiated'],
                                 name="Sell Initiated", marker_color='red', opacity=0.6))
        fig_vol.update_layout(barmode='overlay', height=400, template="plotly_white")
        st.plotly_chart(fig_vol, use_container_width=True)

    # --- Cumulative Tick Delta ---
    st.subheader("Cumulative Tick Delta")
    fig_tick = go.Figure()
    fig_tick.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['cumulative_tick_delta'],
                                  mode='lines', line=dict(color='blue', width=3)))
    fig_tick.update_layout(height=400, template="plotly_white")
    st.plotly_chart(fig_tick, use_container_width=True)

    # --- Download Button ---
    csv = agg_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
else:
    st.warning("No data available for this security.")
