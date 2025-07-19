import os
import streamlit as st
import pandas as pd
import requests
from datetime import datetime
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# --- Config ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"

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

historical_df = fetch_historical_data(selected_id)

# --- Fetch Live Data ---
def fetch_live_data(security_id):
    """Fetch latest live data from Flask API"""
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
        st.warning(f"⚠️ Live API fetch failed: {e}")
    return pd.DataFrame()

# --- Color Highlighting ---
def make_custom_cmap(neg_color, pos_color):
    """Create gradient colormap from negative to positive"""
    return LinearSegmentedColormap.from_list("custom_cmap", [neg_color, "#ffffff", pos_color])

def highlight_columns(df):
    """Style dataframe with colors"""
    styled_df = df.style

    # Solid Colors for Buy/Sell Initiated
    styled_df = styled_df.applymap(
        lambda v: 'background-color: #e0f2f1; color: #00796b' if v > 0 else '',
        subset=['buy_initiated']
    )
    styled_df = styled_df.applymap(
        lambda v: 'background-color: #ffebee; color: #c62828' if v > 0 else '',
        subset=['sell_initiated']
    )

    # Solid Colors for Tick Delta and Cumulative Tick Delta
    styled_df = styled_df.background_gradient(
        cmap=make_custom_cmap('#ef5350', '#26a69a'),
        subset=['tick_delta', 'cumulative_tick_delta'],
        axis=None
    )

    
    return styled_df

# --- Aggregation ---
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
    
    df_agg['tick_delta'] = df_agg['buy_initiated'] - df_agg['sell_initiated']
    df_agg['cumulative_tick_delta'] = df_agg['tick_delta'].cumsum()
    df_agg['inference'] = df_agg['tick_delta'].apply(
        lambda x: 'Buy Dominant' if x > 0 else ('Sell Dominant' if x < 0 else 'Neutral')
    )
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    
    return df_agg

# --- Live Auto-Refresh Loop ---
placeholder = st.empty()
while True:
    live_df = fetch_live_data(selected_id)
    full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    agg_df = aggregate_data(full_df, interval)

    with placeholder.container():
        if not agg_df.empty:
            st.caption("Full history + live updates every 5s")
            st.dataframe(highlight_columns(agg_df))

            # Candlestick Chart
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

            # Add Annotations
            for _, row in agg_df.iterrows():
                if row['buy_initiated'] > 0:
                    fig.add_annotation(x=row['timestamp'], y=row['high'],
                                       text=f"B: {int(row['buy_initiated'])}",
                                       showarrow=False, font=dict(color='#26a69a', size=10))
                if row['sell_initiated'] > 0:
                    fig.add_annotation(x=row['timestamp'], y=row['low'],
                                       text=f"S: {int(row['sell_initiated'])}",
                                       showarrow=False, font=dict(color='#ef5350', size=10))

            fig.update_layout(height=700, xaxis_title="Time", yaxis_title="Price", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

            # Volume Overlay
            if show_volume_overlay:
                st.subheader("Volume Overlay")
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=agg_df['buy_initiated'],
                                         name="Buy Initiated", marker_color='#26a69a', opacity=0.6))
                fig_vol.add_trace(go.Bar(x=agg_df['timestamp'], y=-agg_df['sell_initiated'],
                                         name="Sell Initiated", marker_color='#ef5350', opacity=0.6))
                fig_vol.update_layout(barmode='overlay', height=400, template="plotly_white")
                st.plotly_chart(fig_vol, use_container_width=True)

            # Cumulative Tick Delta
            st.subheader("Cumulative Tick Delta")
            fig_tick = go.Figure()
            fig_tick.add_trace(go.Scatter(x=agg_df['timestamp'], y=agg_df['cumulative_tick_delta'],
                                          mode='lines', line=dict(color='blue', width=3)))
            fig_tick.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig_tick, use_container_width=True)

            # Download Button
            csv = agg_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
        else:
            st.warning("❌ No data available for this security.")
