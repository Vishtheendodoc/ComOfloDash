# Add pandas import at the top
import os
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import requests
import json
from datetime import datetime, timedelta, time

# Place auto-refresh controls and call at the very top, before any other Streamlit widgets
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh", limit=None)

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Add local cache configuration
LOCAL_CACHE_DIR = "local_cache"
if not os.path.exists(LOCAL_CACHE_DIR):
    os.makedirs(LOCAL_CACHE_DIR)

# --- Config ---
GITHUB_USER = "Vishtheendodoc"
GITHUB_REPO = "ComOflo"
DATA_FOLDER = "data_snapshots"
FLASK_API_BASE = "https://comoflo.onrender.com/api"
STOCK_LIST_FILE = "stock_list.csv"

# --- Telegram Config ---
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")
ALERT_CACHE_DIR = "alert_cache"

# Create alert cache directory
if not os.path.exists(ALERT_CACHE_DIR):
    os.makedirs(ALERT_CACHE_DIR)

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

# INTEGRATION GUIDE: Where to Add Telegram Alert Functions
# ================================================================

# STEP 1: Add these imports at the top of your main dashboard file (paste-2.txt)
# Add after the existing imports (around line 10):

import json
from datetime import datetime, timedelta

# STEP 2: Add configuration variables
# Add these after your existing config variables (around line 30, after STOCK_LIST_FILE):

# --- Telegram Alert Configuration ---
TELEGRAM_BOT_TOKEN = st.secrets.get("TELEGRAM_BOT_TOKEN", "")  # Add to your Streamlit secrets
TELEGRAM_CHAT_ID = st.secrets.get("TELEGRAM_CHAT_ID", "")     # Add to your Streamlit secrets

# Alert cache directory
ALERT_CACHE_DIR = "alert_cache"
if not os.path.exists(ALERT_CACHE_DIR):
    os.makedirs(ALERT_CACHE_DIR)

# STEP 3: Add all the Telegram alert functions
# Add these functions after your load_stock_mapping() function (around line 50):

# --- Telegram Alert Functions ---
def send_telegram_alert(message):
    """Send message to Telegram"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        st.warning("Telegram credentials not configured")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        st.error(f"Failed to send Telegram alert: {e}")
        return False

def get_last_alert_state(security_id):
    """Get the last alert state for a security"""
    alert_file = os.path.join(ALERT_CACHE_DIR, f"alert_state_{security_id}.json")
    if os.path.exists(alert_file):
        try:
            with open(alert_file, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return None

def save_alert_state(security_id, state, timestamp):
    """Save the current alert state for a security"""
    alert_file = os.path.join(ALERT_CACHE_DIR, f"alert_state_{security_id}.json")
    alert_data = {
        'state': state,
        'timestamp': timestamp.isoformat(),
        'last_alert_time': datetime.now().isoformat()
    }
    try:
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f)
    except Exception as e:
        st.error(f"Failed to save alert state: {e}")

def determine_gradient_state(cumulative_delta):
    """Determine if cumulative delta is in bullish or bearish state"""
    if cumulative_delta > 0:
        return "bullish"  # Green gradient
    elif cumulative_delta < 0:
        return "bearish"  # Red gradient
    else:
        return "neutral"

def check_gradient_change(security_id, df):
    """Check if there's a gradient change and send alert if needed"""
    if df.empty:
        return False
    
    # Get the latest cumulative tick delta
    latest_row = df.iloc[-1]
    current_cum_delta = latest_row['cumulative_tick_delta']
    current_state = determine_gradient_state(current_cum_delta)
    current_timestamp = latest_row['timestamp']
    
    # Get last known state
    last_alert = get_last_alert_state(security_id)
    
    # Check if state changed and enough time has passed (prevent spam)
    if last_alert:
        last_state = last_alert.get('state')
        last_alert_time = datetime.fromisoformat(last_alert.get('last_alert_time'))
        
        # Only alert if state changed and at least 5 minutes have passed
        if (current_state != last_state and 
            current_state != "neutral" and 
            last_state != "neutral" and  # Must be a real reversal, not from neutral
            datetime.now() - last_alert_time > timedelta(minutes=5)):
            
            # Send alert
            stock_name = stock_mapping.get(str(security_id), f"Stock {security_id}")
            
            if current_state == "bullish":
                emoji = "🟢"
                direction = "BULLISH"
                color = "Green"
            else:  # bearish
                emoji = "🔴"
                direction = "BEARISH"
                color = "Red"
            
            message = f"""
{emoji} <b>GRADIENT REVERSAL ALERT</b> {emoji}

📈 <b>Stock:</b> {stock_name}
🔄 <b>From:</b> {last_state.upper()} → <b>{direction}</b>
📊 <b>Cumulative Tick Delta:</b> {int(current_cum_delta)}
⏰ <b>Time:</b> {current_timestamp.strftime('%H:%M:%S')}
💰 <b>Price:</b> ₹{latest_row['close']:.1f}

Order flow has reversed to <b>{direction}</b>! 🚨
            """.strip()
            
            if send_telegram_alert(message):
                save_alert_state(security_id, current_state, current_timestamp)
                return True
    else:
        # First time - just save the state without alerting
        if current_state != "neutral":
            save_alert_state(security_id, current_state, current_timestamp)
    
    return False

def monitor_all_stocks():
    """Monitor all stocks for gradient changes - EFFICIENT VERSION"""
    try:
        # Get all unique security IDs from your stock list
        stock_df = pd.read_csv(STOCK_LIST_FILE)
        all_security_ids = stock_df['security_id'].unique()
        
        alerts_sent = 0
        processed_count = 0
        
        # Add progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, security_id in enumerate(all_security_ids):
            try:
                # Update progress
                progress = (i + 1) / len(all_security_ids)
                progress_bar.progress(progress)
                stock_name = stock_mapping.get(str(security_id), f"Stock {security_id}")
                status_text.text(f"Checking {stock_name}... ({i+1}/{len(all_security_ids)})")
                
                # Skip if we recently checked this stock (within last 3 minutes)
                last_check_file = os.path.join(ALERT_CACHE_DIR, f"last_check_{security_id}.txt")
                if os.path.exists(last_check_file):
                    try:
                        with open(last_check_file, 'r') as f:
                            last_check_time = datetime.fromisoformat(f.read().strip())
                            if datetime.now() - last_check_time < timedelta(minutes=3):
                                continue  # Skip this stock
                    except Exception:
                        pass
                
                # Fetch live data only (more efficient than historical + live)
                live_df = fetch_live_data(security_id)
                
                # If no live data, try to get recent cached data
                if live_df.empty:
                    cache_df = load_from_local_cache(security_id)
                    if not cache_df.empty:
                        # Use last 50 records from cache
                        live_df = cache_df.tail(50).copy()
                
                if not live_df.empty:
                    # Filter for current day
                    today = datetime.now().date()
                    start_time = datetime.combine(today, datetime.time(9, 0))
                    end_time = datetime.combine(today, datetime.time(23, 59, 59))
                    day_df = live_df[(live_df['timestamp'] >= pd.Timestamp(start_time)) & 
                                   (live_df['timestamp'] <= pd.Timestamp(end_time))]
                    
                    if not day_df.empty:
                        # Aggregate data (use 5-minute intervals for efficiency)
                        agg_df = aggregate_data(day_df, 5)
                        
                        # Check for gradient changes
                        if check_gradient_change(security_id, agg_df):
                            alerts_sent += 1
                        
                        processed_count += 1
                        
                        # Save last check time
                        with open(last_check_file, 'w') as f:
                            f.write(datetime.now().isoformat())
                        
            except Exception as e:
                # Don't show individual stock errors to avoid spam
                continue
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        return alerts_sent, processed_count
        
    except Exception as e:
        st.error(f"Error in monitor_all_stocks: {e}")
        return 0, 0


# --- Enhanced Mobile CSS ---
def inject_mobile_css():
    mobile_css = """
    <style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        /* Compact sidebar */
        .css-1d391kg { 
            padding: 0.5rem !important;
        }
        
        /* Smaller main content padding */
        .main .block-container {
            padding-top: 1rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* Compact title */
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Mobile table styling */
        .mobile-table {
            font-size: 11px !important;
            overflow-x: auto;
            white-space: nowrap;
        }
        
        .mobile-table th {
            padding: 4px 2px !important;
            font-size: 10px !important;
            font-weight: bold !important;
            background-color: #f0f2f6 !important;
        }
        
        .mobile-table td {
            padding: 3px 2px !important;
            border-bottom: 1px solid #e6e6e6 !important;
        }
        
        /* Color coding for mobile table */
        .positive { color: #26a69a !important; font-weight: bold !important; }
        .negative { color: #ef5350 !important; font-weight: bold !important; }
        .neutral { color: #757575 !important; }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 12px !important;
        }
        
        /* Metrics styling */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            padding: 12px;
            color: white;
            text-align: center;
            margin: 4px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .metric-value {
            font-size: 18px;
            font-weight: bold;
            margin: 0;
        }
        
        .metric-label {
            font-size: 11px;
            opacity: 0.9;
            margin: 0;
        }
        
        /* Button styling */
        .stDownloadButton button {
            width: 100% !important;
            border-radius: 20px !important;
            font-size: 12px !important;
            padding: 8px !important;
        }
    }
    
    /* Tooltip styles */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
        border-bottom: 1px dotted #666;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 11px;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
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

security_options = fetch_security_ids()
selected_option = st.sidebar.selectbox("🎯 Security", security_options)
selected_id = int(selected_option.split('(')[-1].strip(')'))
interval = st.sidebar.selectbox("⏱️ Interval", [1, 3, 5, 15, 30, 60, 90, 120, 180, 240, 360, 480], index=2)

mobile_view = st.sidebar.toggle("📱 Mobile Mode", value=True)

if mobile_view:
    inject_mobile_css()

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Alert System")

# Enable/disable monitoring
alert_enabled = st.sidebar.toggle("Enable Alerts", value=False)

# 🆕 NEW toggle for continuous monitoring
continuous_monitoring = st.sidebar.toggle("Continuous All Stocks Monitoring", value=False)
monitor_interval = st.sidebar.selectbox("Monitor Interval (seconds)", [30, 60, 120, 300], index=1)

if alert_enabled:
    if st.sidebar.button("🔍 Monitor All Stocks"):
        with st.spinner("Monitoring all stocks for gradient changes..."):
            alerts_sent, processed = monitor_all_stocks()
        
        if alerts_sent > 0:
            st.sidebar.success(f"✅ Sent {alerts_sent} alerts from {processed} stocks")
        else:
            st.sidebar.info(f"ℹ️ No alerts triggered from {processed} stocks")

# 🆕 Continuous All Stocks Monitoring
if alert_enabled and continuous_monitoring:
    st_autorefresh(interval=monitor_interval * 1000, key="all_stock_monitor")
    with st.spinner("🔄 Monitoring all stocks for gradient changes..."):
        alerts_sent, processed = monitor_all_stocks()

    if alerts_sent > 0:
        st.sidebar.success(f"✅ Sent {alerts_sent} alerts from {processed} stocks")
    else:
        st.sidebar.info(f"ℹ️ No alerts triggered from {processed} stocks")


    # Test alert button
    if st.sidebar.button("🧪 Test Alert"):
        test_message = f"""
🟢 <b>TEST ALERT</b> 🟢

📈 <b>Stock:</b> Test Stock
🔄 <b>Status:</b> Alert system working
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

This is a test message! 🚨
        """.strip()
        
        if send_telegram_alert(test_message):
            st.sidebar.success("✅ Test alert sent!")
        else:
            st.sidebar.error("❌ Failed to send test alert")

# --- Data Fetching Functions with Local Cache ---
def save_to_local_cache(df, security_id):
    """Save data to local cache file"""
    if not df.empty:
        cache_file = os.path.join(LOCAL_CACHE_DIR, f"cache_{security_id}.csv")
        df.to_csv(cache_file, index=False)

def load_from_local_cache(security_id):
    """Load data from local cache file"""
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
    """Fetch historical data from GitHub and merge with local cache"""
    # Load from GitHub
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    github_df = pd.DataFrame()
    
    try:
        resp = requests.get(base_url, headers=headers)
        if resp.status_code != 404:  # Only process if data exists
            resp.raise_for_status()
            files = resp.json()
            
            for file_info in files:
                if file_info['name'].endswith('.csv'):
                    df = pd.read_csv(file_info['download_url'], dtype=str)  # Force all columns to string
                    df.columns = df.columns.str.strip()  # Strip spaces from column names
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Strip spaces from all values
                    df = df[df['security_id'] == str(security_id)]
                    # Convert relevant columns to numeric
                    numeric_cols = [
                        'buy_initiated', 'buy_volume', 'close', 'delta', 'high', 'low', 'open',
                        'sell_initiated', 'sell_volume', 'tick_delta'
                    ]
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    github_df = pd.concat([github_df, df], ignore_index=True)

            if not github_df.empty:
                github_df['timestamp'] = pd.to_datetime(github_df['timestamp'])
                github_df.sort_values('timestamp', inplace=True)
    except Exception as e:
        st.error(f"GitHub API error: {e}")

    # Load from local cache
    cache_df = load_from_local_cache(security_id)
    
    # Merge GitHub data with local cache
    combined_df = pd.concat([github_df, cache_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    return combined_df

def fetch_live_data(security_id):
    """Fetch live data and update local cache"""
    api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
    try:
        r = requests.get(api_url, timeout=20)
        r.raise_for_status()
        live_data = pd.DataFrame(r.json())
        if not live_data.empty:
            live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
            live_data.sort_values('timestamp', inplace=True)
            
            # Load existing cache
            cache_df = load_from_local_cache(security_id)
            
            # Merge with new live data and save
            updated_df = pd.concat([cache_df, live_data]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            save_to_local_cache(updated_df, security_id)
            
            return live_data
    except Exception as e:
        st.warning(f"Live API fetch failed: {e}")
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
    df_agg['inference'] = df_agg['tick_delta'].apply(
        lambda x: 'Buy Dominant' if x > 0 else ('Sell Dominant' if x < 0 else 'Neutral')
    )
    df_agg['delta'] = df_agg['buy_volume'] - df_agg['sell_volume']
    df_agg['cumulative_delta'] = df_agg['delta'].cumsum()
    
    return df_agg

# --- Fetch and process data ---
historical_df = fetch_historical_data(selected_id)
live_df = fetch_live_data(selected_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# Filter for current day between 9:00 and 23:59
today = datetime.now().date()
start_time = datetime.combine(today, datetime.time(9, 0))
end_time = datetime.combine(today, datetime.time(23, 59, 59))
full_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]

agg_df = aggregate_data(full_df, interval)

# Automatic gradient change detection for current stock
if not agg_df.empty and alert_enabled:
    try:
        alert_sent = check_gradient_change(selected_id, agg_df)
        if alert_sent:
            st.success("🚨 Gradient reversal alert sent!")
    except Exception as e:
        st.warning(f"Alert check failed: {e}")

# --- Mobile Optimized Display Functions ---
def create_mobile_metrics(df):
    """Create compact metric cards for mobile"""
    if df.empty:
        return
    
    latest = df.iloc[-1]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        close_price = float(latest['close']) if pd.notna(latest['close']) else 0.0
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{close_price:.1f}</p>
            <p class="metric-label">Price</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        tick_delta = int(latest['tick_delta']) if pd.notna(latest['tick_delta']) else 0
        delta_color = "#26a69a" if tick_delta >= 0 else "#ef5350"
        sign = "+" if tick_delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="background: {delta_color};">
            <p class="metric-value">{sign}{tick_delta}</p>
            <p class="metric-label">Tick Δ</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        cum_delta = int(latest['cumulative_tick_delta']) if pd.notna(latest['cumulative_tick_delta']) else 0
        cum_delta_color = "#26a69a" if cum_delta >= 0 else "#ef5350"
        sign = "+" if cum_delta > 0 else ""
        st.markdown(f"""
        <div class="metric-card" style="background: {cum_delta_color};">
            <p class="metric-value">{sign}{cum_delta}</p>
            <p class="metric-label">Cum Δ</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        buy_vol = float(latest['buy_initiated']) if pd.notna(latest['buy_initiated']) else 0.0
        sell_vol = float(latest['sell_initiated']) if pd.notna(latest['sell_initiated']) else 0.0
        vol_total = buy_vol + sell_vol
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{vol_total:,.0f}</p>
            <p class="metric-label">Volume</p>
        </div>
        """, unsafe_allow_html=True)

def create_mobile_table(df):
    """Create a highly optimized mobile table"""
    if df.empty:
        return
    
    # Get today's date
    today = datetime.now().date()
    start_time = datetime.combine(today, datetime.time(9, 0))
    end_time = datetime.combine(today, datetime.time(23, 59, 59))
    
    # Filter for today and between 9:00 and 23:59
    mobile_df = df[(df['timestamp'] >= pd.Timestamp(start_time)) & (df['timestamp'] <= pd.Timestamp(end_time))].copy()
    
    # Format data for mobile display with safe type conversion
    mobile_df['Time'] = mobile_df['timestamp'].dt.strftime('%H:%M')
    mobile_df['Price'] = mobile_df['close'].fillna(0).round(1)
    mobile_df['BI'] = mobile_df['buy_initiated'].fillna(0).astype(int)
    mobile_df['SI'] = mobile_df['sell_initiated'].fillna(0).astype(int)
    mobile_df['TΔ'] = mobile_df['tick_delta'].fillna(0).astype(int)
    mobile_df['CumΔ'] = mobile_df['cumulative_tick_delta'].fillna(0).astype(int)
    
    # Select only essential columns
    display_df = mobile_df[['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']].copy()
    
    # Apply color coding
    def apply_color_coding(val, col_name):
        if col_name in ['TΔ', 'CumΔ']:
            val = int(val) if pd.notna(val) else 0
            if val > 0:
                return f'<span class="positive">+{val}</span>'
            elif val < 0:
                return f'<span class="negative">{val}</span>'
            else:
                return f'<span class="neutral">{val}</span>'
        return str(val)
    
    # Create HTML table
    html_table = '<table class="mobile-table" style="width:100%; border-collapse: collapse;">'
    html_table += '<thead><tr>'
    
    # Column headers with tooltips
    headers = {
        'Time': 'Timestamp',
        'Price': 'Close Price', 
        'BI': 'Buy Initiated Trades',
        'SI': 'Sell Initiated Trades',
        'TΔ': 'Tick Delta (BI - SI)',
        'CumΔ': 'Cumulative Tick Delta'
    }
    
    for col, tooltip in headers.items():
        html_table += f'<th><div class="tooltip">{col}<span class="tooltiptext">{tooltip}</span></div></th>'
    html_table += '</tr></thead><tbody>'
    
    # Table rows
    for _, row in display_df.iterrows():
        html_table += '<tr>'
        for col in display_df.columns:
            if col in ['TΔ', 'CumΔ']:
                html_table += f'<td>{apply_color_coding(row[col], col)}</td>'
            else:
                html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'
    
    html_table += '</tbody></table>'
    st.markdown(html_table, unsafe_allow_html=True)

# REPLACE THIS ENTIRE FUNCTION WITH THE NEW MARKET PROFILE VERSION
def create_market_profile_chart(df):
    """Create market profile style chart exactly like the reference image"""
    if df.empty:
        return go.Figure()
        
    fig = go.Figure()
    
    # Main candlestick chart
    fig.add_trace(go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='rgba(38, 166, 154, 0.1)',
        decreasing_fillcolor='rgba(239, 83, 80, 0.1)',
        line=dict(width=1)
    ))
    
    # Calculate price range for volume profile positioning
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_range = price_max - price_min
    
    # Add volume profile bars and annotations for each candle
    for i, row in df.iterrows():
        timestamp = row['timestamp']
        buy_vol = int(row['buy_initiated'])
        sell_vol = int(row['sell_initiated'])
        tick_delta = int(row['tick_delta'])
        close_price = row['close']
        high_price = row['high']
        low_price = row['low']
        
        # Calculate bar width proportional to volume
        max_volume = max(df['buy_initiated'].max(), df['sell_initiated'].max())
        if max_volume > 0:
            bar_width = (buy_vol + sell_vol) / max_volume * (price_range * 0.05)
        else:
            bar_width = 0
        
        # Add volume profile bar (teal horizontal bar)
        if bar_width > 0:
            fig.add_shape(
                type="rect",
                x0=timestamp - pd.Timedelta(minutes=1),
                x1=timestamp + pd.Timedelta(minutes=1),
                y0=close_price - bar_width/2,
                y1=close_price + bar_width/2,
                fillcolor="rgba(68, 183, 172, 0.8)",
                line=dict(width=0),
            )
        
        # Add buy dominant triangle (green up arrow)
        if tick_delta > 0:
            fig.add_trace(go.Scatter(
                x=[timestamp],
                y=[high_price + price_range * 0.01],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='#26a69a'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Delta annotation (Δ+xx)
            fig.add_annotation(
                x=timestamp,
                y=close_price + price_range * 0.005,
                text=f"Δ+{tick_delta}",
                showarrow=False,
                font=dict(size=8, color='#26a69a', family="Arial"),
                bgcolor="rgba(38, 166, 154, 0.3)",
                bordercolor="#26a69a",
                borderwidth=1
            )
            
        # Add sell dominant triangle (red down arrow)  
        elif tick_delta < 0:
            fig.add_trace(go.Scatter(
                x=[timestamp],
                y=[low_price - price_range * 0.01],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='#ef5350'
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Delta annotation (Δ-xx)
            fig.add_annotation(
                x=timestamp,
                y=close_price - price_range * 0.005,
                text=f"Δ{tick_delta}",  # tick_delta already negative
                showarrow=False,
                font=dict(size=8, color='#ef5350', family="Arial"),
                bgcolor="rgba(239, 83, 80, 0.3)",
                bordercolor="#ef5350",
                borderwidth=1
            )
        
        # Buy volume annotation (B: xx)
        fig.add_annotation(
            x=timestamp,
            y=high_price + price_range * 0.02,
            text=f"B: {buy_vol}",
            showarrow=False,
            font=dict(size=8, color='#26a69a'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#26a69a",
            borderwidth=1
        )
        
        # Sell volume annotation (S: xx)
        fig.add_annotation(
            x=timestamp,
            y=low_price - price_range * 0.02,
            text=f"S: {sell_vol}",
            showarrow=False,
            font=dict(size=8, color='#ef5350'),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ef5350",
            borderwidth=1
        )
    
    # Chart layout
    fig.update_layout(
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        template="plotly_white",
        showlegend=True,
        font=dict(size=10),
        title="Order Flow Analysis - Market Profile Style",
        xaxis_title="Time",
        yaxis_title="Price"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0', tickformat='%H:%M')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
    
    return fig

def create_mobile_charts(df):
    """Create optimized charts for mobile"""
    if df.empty:
        return
    
    # Chart configuration for mobile
    mobile_config = {
        'displayModeBar': False,
        'responsive': True,
        'staticPlot': False
    }
    
    mobile_layout = {
        'height': 250,
        'margin': dict(l=30, r=20, t=20, b=30),
        'template': "plotly_white",
        'showlegend': False,
        'font': dict(size=10)
    }
    
    tab1, tab2, tab3 = st.tabs(["🕯️ Candles", "📊 Volume", "📈 Delta"])
    
    with tab1:
        # Option to toggle between candlestick and line chart
        chart_type = st.radio("Chart Type:", ["Candlestick", "Line"], horizontal=True, key="mobile_chart_type")
        
        fig = go.Figure()
        
        if chart_type == "Candlestick":
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ))
        else:  # Line chart
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['close'],
                mode='lines+markers',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=3),
                name='Close Price'
            ))
        
        fig.update_layout(**mobile_layout)
        fig.update_xaxes(showgrid=False, showticklabels=True, tickformat='%H:%M')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True, config=mobile_config)

    with tab2:
        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=df['timestamp'], 
            y=df['buy_initiated'],
            name="Buy", 
            marker_color='#26a69a',
            opacity=0.7,
            width=60000 * interval  # Adjust bar width based on interval
        ))
        fig_vol.add_trace(go.Bar(
            x=df['timestamp'], 
            y=-df['sell_initiated'],
            name="Sell", 
            marker_color='#ef5350',
            opacity=0.7,
            width=60000 * interval
        ))
        fig_vol.update_layout(barmode='overlay', **mobile_layout)
        fig_vol.update_xaxes(showgrid=False, showticklabels=True, tickformat='%H:%M')
        fig_vol.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig_vol, use_container_width=True, config=mobile_config)

    with tab3:
        fig_delta = go.Figure()
        
        # Add cumulative delta line
        fig_delta.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['cumulative_tick_delta'],
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=3),
            name='Cumulative Delta'
        ))
        
        # Add zero line
        fig_delta.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig_delta.update_layout(**mobile_layout)
        fig_delta.update_xaxes(showgrid=False, showticklabels=True, tickformat='%H:%M')
        fig_delta.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f0f0f0')
        st.plotly_chart(fig_delta, use_container_width=True, config=mobile_config)

# --- Main Display ---
if mobile_view:
    # Compact title
    stock_name = selected_option.split(' (')[0]
    st.markdown(f"# 📊 {stock_name}")
    st.caption(f"🔄 Updates every 5s • {interval}min intervals")
    
    if not agg_df.empty:
        # Mobile metrics
        create_mobile_metrics(agg_df)
        
        st.markdown("---")
        
        # Mobile table
        st.markdown("### 📋 Recent Activity")
        create_mobile_table(agg_df)
        
        st.markdown("---")
        
        # Mobile charts - THIS IS THE SECTION TO REPLACE
        st.markdown("### 📈 Charts")
        
        # Add chart style selector
        chart_style = st.radio("Chart Style:", ["Market Profile", "Traditional"], horizontal=True)
        
        if chart_style == "Market Profile":
            fig = create_market_profile_chart(agg_df)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False, 'responsive': True})
        else:
            create_mobile_charts(agg_df)
        
        # Download button
        st.markdown("---")
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
    # Desktop view (your original implementation)
    st.title(f"Order Flow Dashboard: {selected_option}")
    
    if not agg_df.empty:
        st.caption("Full history + live updates every 5s")
        
        # Format data for desktop display
        agg_df_formatted = agg_df.copy()
        agg_df_formatted['close'] = agg_df_formatted['close'].round(1)
        
        for col in ['buy_volume', 'sell_volume', 'buy_initiated', 'sell_initiated',
                    'delta', 'cumulative_delta', 'tick_delta', 'cumulative_tick_delta']:
            agg_df_formatted[col] = agg_df_formatted[col].round(0).astype(int)
        
        columns_to_show = [
            'timestamp', 'close', 'buy_initiated', 'sell_initiated', 'tick_delta',
            'cumulative_tick_delta', 'inference'
        ]
        
        column_abbreviations = {
            'timestamp': 'Time',
            'close': 'Close',
            'buy_initiated': 'Buy Initiated',
            'sell_initiated': 'Sell Initiated',
            'tick_delta': 'Tick Delta',
            'cumulative_tick_delta': 'Cumulative Tick Delta',
            'inference': 'Inference'
        }
        
        agg_df_table = agg_df_formatted[columns_to_show]
        agg_df_table = agg_df_table.rename(columns=column_abbreviations)
        
        styled_table = agg_df_table.style.background_gradient(
            cmap="RdYlGn", subset=['Tick Delta', 'Cumulative Tick Delta']
        )
        
        st.dataframe(styled_table, use_container_width=True, height=600)
        
        # Desktop charts
        st.subheader("Candlestick Chart")
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
        fig.update_layout(height=600, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Cumulative Tick Delta")
        fig_delta = go.Figure()
        fig_delta.add_trace(go.Scatter(
            x=agg_df['timestamp'], 
            y=agg_df['cumulative_tick_delta'],
            mode='lines', 
            line=dict(color='blue', width=3)
        ))
        fig_delta.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig_delta, use_container_width=True)
        
        # Download
        csv = agg_df_table.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
    else:
        st.warning("No data available for this security.")

# Add y-axis range controls for main chart (mobile and desktop)
def get_yaxis_range(df):
    y_min = float(df['low'].min()) if not df.empty else 0
    y_max = float(df['high'].max()) if not df.empty else 1
    return y_min, y_max

def yaxis_range_controls(df, label_prefix=""):
    y_min, y_max = get_yaxis_range(df)
    col1, col2 = st.columns(2)
    with col1:
        user_ymin = st.number_input(f"{label_prefix}Y-axis min", value=y_min, step=1.0, format="%.1f")
    with col2:
        user_ymax = st.number_input(f"{label_prefix}Y-axis max", value=y_max, step=1.0, format="%.1f")
    return user_ymin, user_ymax

# Set Plotly uirevision to preserve zoom/pan
PLOTLY_UIREVISION = "orderflow-chart"

# For mobile charts:
# In the mobile chart section, before plotting, add:
# user_ymin, user_ymax = yaxis_range_controls(agg_df, label_prefix="Mobile ")
# Then, in fig.update_yaxes, add: range=[user_ymin, user_ymax]
# and in fig.update_layout, add: uirevision=PLOTLY_UIREVISION

# For desktop charts:
# In the desktop chart section, before plotting, add:
# user_ymin, user_ymax = yaxis_range_controls(agg_df, label_prefix="Desktop ")
# Then, in fig.update_yaxes, add: range=[user_ymin, user_ymax]
# and in fig.update_layout, add: uirevision=PLOTLY_UIREVISION
