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
import re
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import streamlit.components.v1 as components
import pytz

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("error.log"),  # Logs saved here
        logging.StreamHandler()           # Also shows in terminal
    ]
)

# Define the log_error function
def log_error(message):
    logging.error(message)

# Place auto-refresh controls and call at the very top, before any other Streamlit widgets
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh", limit=None)

st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Custom CSS for mobile-responsive lightweight chart styling
def inject_lightweight_chart_css():
    st.markdown("""
    <style>
        .main > div {
            padding-top: 0rem;
            padding-bottom: 0rem;
        }
        
        .trading-header {
            background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .stock-info {
            display: flex;
            align-items: center;
            color: white;
            font-size: 18px;
            font-weight: bold;
            flex-wrap: wrap;
        }
        
        .price-info {
            display: flex;
            gap: 20px;
            margin-left: 20px;
            color: white;
            flex-wrap: wrap;
        }
        
        .price-negative {
            color: #e74c3c;
        }
        
        .price-positive {
            color: #27ae60;
        }
        
        .control-panel {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #e9ecef;
        }
        
        .control-label {
            font-size: 12px;
            font-weight: bold;
            color: #495057;
            text-transform: uppercase;
            margin-bottom: 5px;
        }
        
        /* Chart container optimizations */
        .chart-container {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto;
            margin-bottom: 20px;
        }
        
        .chart-wrapper {
            position: relative;
            width: 100%;
            min-height: 400px;
        }
        
        /* Delta overlay optimizations */
        .delta-overlay {
            position: relative;
            height: 100px;
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-top: none;
            border-radius: 0 0 8px 8px;
            overflow-x: auto;
            overflow-y: hidden;
            padding: 5px;
        }
        
        .delta-box {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #ccc;
            border-radius: 3px;
            padding: 2px 4px;
            font-size: 8px;
            font-weight: bold;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
            min-width: 35px;
            text-align: center;
            position: absolute;
            display: block;
            line-height: 1.1;
            white-space: nowrap;
        }
        
        .delta-row-1 {
            top: 8px;
        }
        
        .delta-row-2 {
            top: 35px;
        }
        
        .delta-positive {
            color: #27ae60;
            border-color: #27ae60;
            background: rgba(39, 174, 96, 0.1);
        }
        
        .delta-negative {
            color: #e74c3c;
            border-color: #e74c3c;
            background: rgba(231, 76, 60, 0.1);
        }
        
        .delta-neutral {
            color: #6c757d;
            border-color: #6c757d;
            background: rgba(108, 117, 125, 0.1);
        }

        /* Optimized table styling */
        .data-table-section {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }
        
        .simplified-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            background: white;
        }
        
        .simplified-table th {
            background: #f8f9fa;
            padding: 8px 6px;
            text-align: center;
            font-weight: bold;
            color: #495057;
            border: 1px solid #dee2e6;
            font-size: 11px;
            white-space: nowrap;
        }
        
        .simplified-table td {
            padding: 6px 4px;
            text-align: center;
            border: 1px solid #dee2e6;
            font-size: 12px;
        }
        
        .simplified-table tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        .positive-delta {
            color: #27ae60;
            font-weight: bold;
            background-color: rgba(39, 174, 96, 0.1) !important;
        }
        
        .negative-delta {
            color: #e74c3c;
            font-weight: bold;
            background-color: rgba(231, 76, 60, 0.1) !important;
        }

        /* Mobile optimizations */
        @media (max-width: 768px) {
            .trading-header {
                padding: 10px 15px;
            }
            
            .stock-info {
                font-size: 16px;
                flex-direction: column;
                align-items: flex-start;
            }
            
            .price-info {
                gap: 10px;
                margin-left: 0;
                margin-top: 5px;
            }
            
            .control-panel {
                padding: 10px;
            }
            
            .chart-wrapper {
                min-height: 350px;
            }
            
            .delta-overlay {
                height: 80px;
            }
            
            .delta-box {
                font-size: 7px;
                padding: 1px 3px;
                min-width: 30px;
            }
            
            .delta-row-1 {
                top: 6px;
            }
            
            .delta-row-2 {
                top: 28px;
            }
            
            .simplified-table {
                font-size: 11px;
            }
            
            .simplified-table th {
                padding: 6px 4px;
                font-size: 10px;
            }
            
            .simplified-table td {
                padding: 4px 3px;
                font-size: 10px;
            }
        }
        
        @media (max-width: 480px) {
            .chart-wrapper {
                min-height: 300px;
            }
            
            .delta-overlay {
                height: 70px;
            }
            
            .simplified-table {
                font-size: 10px;
            }
            
            .simplified-table th,
            .simplified-table td {
                padding: 3px 2px;
                font-size: 9px;
            }
        }
    </style>
    """, unsafe_allow_html=True)

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

# --- Enhanced Alert Configuration ---
ALERT_BATCH_SIZE = 10  # Process stocks in batches
MAX_WORKERS = 5  # Concurrent API calls
ALERT_COOLDOWN_MINUTES = 5  # Minimum time between alerts for same stock
MONITOR_COOLDOWN_MINUTES = 2  # Minimum time between checks for same stock

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

# --- All Telegram Alert Functions (keeping existing ones) ---
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
    """Determine if cumulative delta is positive or negative relative to zero"""
    if cumulative_delta > 0:
        return "positive"
    elif cumulative_delta < 0:
        return "negative"
    else:
        return "zero"

def check_gradient_change(security_id, df):
    """Check if cumulative delta crosses zero and send alert if needed"""
    if df.empty:
        return False
    
    latest_row = df.iloc[-1]
    current_cum_delta = latest_row['cumulative_tick_delta']
    current_state = determine_gradient_state(current_cum_delta)
    current_timestamp = latest_row['timestamp']
    
    last_alert = get_last_alert_state(security_id)
    
    if last_alert:
        last_state = last_alert.get('state')
        last_alert_time = datetime.fromisoformat(last_alert.get('last_alert_time'))
        
        zero_cross_occurred = (
            (last_state == "positive" and current_state == "negative") or
            (last_state == "negative" and current_state == "positive")
        )
        
        if (zero_cross_occurred and 
            datetime.now() - last_alert_time > timedelta(minutes=5)):
            
            stock_name = stock_mapping.get(str(security_id), f"Stock {security_id}")
            
            if current_state == "positive":
                emoji = "🟢"
                direction = "POSITIVE"
                cross_direction = "CROSSED ABOVE ZERO"
            else:
                emoji = "🔴"
                direction = "NEGATIVE"
                cross_direction = "CROSSED BELOW ZERO"
            
            message = f"""
{emoji} <b>ZERO CROSS ALERT</b> {emoji}

📈 <b>Stock:</b> {stock_name}
🔄 <b>Transition:</b> {last_state.upper()} → <b>{direction}</b>
⚡ <b>Event:</b> {cross_direction}
📊 <b>Cumulative Tick Delta:</b> {int(current_cum_delta)}
⏰ <b>Time:</b> {current_timestamp.strftime('%H:%M:%S')}
💰 <b>Price:</b> ₹{latest_row['close']:.1f}

Cumulative delta has {cross_direction.lower()}! 🚨
            """.strip()
            
            if send_telegram_alert(message):
                save_alert_state(security_id, current_state, current_timestamp)
                return True
    else:
        save_alert_state(security_id, current_state, current_timestamp)
    
    return False

# --- Enhanced Sidebar Controls ---
def enhanced_alert_controls():
    """Enhanced alert controls in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Enhanced Alert System")
    
    alert_enabled = st.sidebar.toggle("Enable Smart Alerts", value=False, key="enhanced_alerts")
    
    alert_status_file = os.path.join(ALERT_CACHE_DIR, "alert_status.txt")
    with open(alert_status_file, 'w') as f:
        f.write(str(alert_enabled))
    
    if alert_enabled:
        monitor_mode = st.sidebar.radio(
            "Monitoring Mode:",
            ["Auto (Every 2 min)", "Manual Check"],
            key="monitor_mode"
        )
        
        sensitivity = st.sidebar.selectbox(
            "Alert Sensitivity:",
            ["High (Any change)", "Medium (Significant changes)", "Low (Major changes only)"],
            index=1,
            key="alert_sensitivity"
        )
        
        if st.sidebar.button("🧪 Test Alert"):
            test_message = f"""
🟢 <b>TEST ZERO CROSS ALERT</b> 🟢
📈 <b>Stock:</b> TEST STOCK
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
This is a test alert! 🚨
            """.strip()
            
            if send_telegram_alert(test_message):
                st.sidebar.success("✅ Test alert sent!")
            else:
                st.sidebar.error("❌ Failed to send test alert")

enhanced_alert_controls()
st.sidebar.markdown("---")

# --- Data Fetching Functions ---
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
    base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
    headers = {"Authorization": f"token {st.secrets['GITHUB_TOKEN']}"}
    github_df = pd.DataFrame()
    
    try:
        resp = requests.get(base_url, headers=headers)
        if resp.status_code != 404:
            resp.raise_for_status()
            files = resp.json()
            
            for file_info in files:
                if file_info['name'].endswith('.csv'):
                    df = pd.read_csv(file_info['download_url'], dtype=str)
                    df.columns = df.columns.str.strip()
                    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                    df = df[df['security_id'] == str(security_id)]
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

    cache_df = load_from_local_cache(security_id)
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
            
            cache_df = load_from_local_cache(security_id)
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

# --- Security ID Fetching ---
@st.cache_data(ttl=6000)
def fetch_security_ids():
    try:
        base_url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FOLDER}"
        headers = {"Authorization": f"token {os.environ['GITHUB_TOKEN']}"}
        r = requests.get(base_url, headers=headers)
        
        ids = set()
        if r.status_code == 200:
            files = r.json()
            for file in files:
                if file['name'].endswith('.csv'):
                    df = pd.read_csv(file['download_url'])
                    ids.update(df['security_id'].unique())
        
        if not ids:
            st.info("📋 No data snapshots found, loading from stock list...")
            try:
                stock_df = pd.read_csv(STOCK_LIST_FILE)
                ids.update(stock_df['security_id'].unique())
            except Exception as stock_error:
                st.error(f"Failed to load stock list: {stock_error}")
                return ["No Data Available (0)"]
        
        if ids:
            ids = sorted(list(ids))
            return [f"{stock_mapping.get(str(i), f'Stock {i}')} ({i})" for i in ids]
        else:
            return ["No Data Available (0)"]
            
    except Exception as e:
        st.error(f"Failed to fetch security IDs: {e}")
        try:
            stock_df = pd.read_csv(STOCK_LIST_FILE)
            ids = sorted(list(stock_df['security_id'].unique()))
            return [f"{stock_mapping.get(str(i), f'Stock {i}')} ({i})" for i in ids]
        except:
            return ["No Data Available (0)"]

# --- Optimized Lightweight Chart Functions ---
def parse_ist_time(timestamp_str):
    """Convert timestamp to Unix timestamp for lightweight charts"""
    try:
        if isinstance(timestamp_str, str):
            # Try parsing as time only first
            try:
                time_obj = datetime.strptime(timestamp_str, "%H:%M").time()
                ist_tz = pytz.timezone('Asia/Kolkata')
                today = datetime.now(ist_tz).date()
                dt = datetime.combine(today, time_obj)
                dt_ist = ist_tz.localize(dt)
                return int(dt_ist.timestamp())
            except:
                # Try parsing as full datetime
                dt = pd.to_datetime(timestamp_str)
                return int(dt.timestamp())
        else:
            # Already a datetime object
            dt = pd.to_datetime(timestamp_str)
            return int(dt.timestamp())
    except Exception as e:
        st.warning(f"Error parsing timestamp {timestamp_str}: {e}")
        return None

def create_responsive_lightweight_chart(stock_name, chart_data, interval):
    """Create a responsive lightweight chart with optimized delta display"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6c757d;">No data available</div>'
    
    # Convert DataFrame to chart data
    candle_data = []
    delta_boxes = []
    cumulative_delta = 0
    
    for i, row in chart_data.iterrows():
        try:
            ts_unix = parse_ist_time(row["timestamp"])
            if ts_unix is None:
                continue
            
            # OHLC data
            open_price = float(row.get("open", 0))
            high_price = float(row.get("high", 0))
            low_price = float(row.get("low", 0))
            close_price = float(row.get("close", 0))
            
            # Delta data
            tick_delta = float(row.get("tick_delta", 0))
            cumulative_delta = float(row.get("cumulative_tick_delta", cumulative_delta + tick_delta))
            
            candle_data.append({
                'time': ts_unix,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price
            })
            
            # Delta box styling
            delta_class = "delta-positive" if tick_delta > 0 else ("delta-negative" if tick_delta < 0 else "delta-neutral")
            cum_delta_class = "delta-positive" if cumulative_delta > 0 else ("delta-negative" if cumulative_delta < 0 else "delta-neutral")
            
            delta_boxes.append({
                'time': ts_unix,
                'tick_delta': tick_delta,
                'cumulative_delta': cumulative_delta,
                'tick_class': delta_class,
                'cum_class': cum_delta_class
            })
            
        except Exception as e:
            continue
    
    chart_id = f"chart_{stock_name.replace('-', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
    
    # Create delta overlay HTML for recent data only
    delta_overlay_html = ""
    recent_boxes = delta_boxes[-15:]  # Show only last 15 for better mobile performance
    
    for i, box in enumerate(recent_boxes):
        delta_overlay_html += f"""
        <div class="delta-info" data-time="{box['time']}" data-index="{i}" style="position: absolute; display: none;">
            <div class="delta-box delta-row-1 {box['tick_class']}">
                Δ{box['tick_delta']:+.0f}
            </div>
            <div class="delta-box delta-row-2 {box['cum_class']}">
                Σ{box['cumulative_delta']:+.0f}
            </div>
        </div>
        """
    
    return f"""
    <div class="chart-container">
        <div class="chart-wrapper">
            <div id="{chart_id}" style="width: 100%; height: 100%; min-height: inherit;"></div>
        </div>
        
        <div id="delta-container-{chart_id}" class="delta-overlay">
            <div style="position: relative; height: 100%; width: 100%;">
                {delta_overlay_html}
            </div>
        </div>
    </div>
    
    <script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
    <script>
    (function() {{
        const chartContainer = document.getElementById('{chart_id}');
        const deltaContainer = document.getElementById('delta-container-{chart_id}');
        
        if (!chartContainer || typeof LightweightCharts === 'undefined') {{
            chartContainer.innerHTML = '<div style="text-align: center; padding: 40px; color: #6c757d;">Chart library not loaded</div>';
            return;
        }}
        
        chartContainer.innerHTML = '';
        
        // Responsive chart options
        const isMobile = window.innerWidth < 768;
        const chartHeight = isMobile ? 350 : 450;
        
        const chart = LightweightCharts.createChart(chartContainer, {{
            width: chartContainer.clientWidth,
            height: chartHeight,
            layout: {{
                background: {{ type: 'solid', color: '#ffffff' }},
                textColor: '#333333',
                fontSize: isMobile ? 10 : 12
            }},
            grid: {{
                vertLines: {{ 
                    color: '#f0f0f0',
                    visible: !isMobile
                }},
                horzLines: {{ color: '#f0f0f0' }}
            }},
            crosshair: {{
                mode: LightweightCharts.CrosshairMode.Normal,
            }},
            rightPriceScale: {{
                borderColor: '#cccccc',
                scaleMargins: {{ top: 0.1, bottom: 0.1 }},
                visible: true
            }},
            leftPriceScale: {{
                visible: false
            }},
            timeScale: {{
                borderColor: '#cccccc',
                timeVisible: true,
                secondsVisible: false,
                rightOffset: isMobile ? 5 : 12,
                barSpacing: isMobile ? 6 : 8
            }},
            handleScroll: {{
                mouseWheel: true,
                pressedMouseMove: true,
                horzTouchDrag: true,
                vertTouchDrag: true
            }},
            handleScale: {{
                axisPressedMouseMove: true,
                mouseWheel: true,
                pinch: true
            }}
        }});

        const candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350',
        }});
        
        candleSeries.setData({json.dumps(candle_data)});
        
        const deltaData = {json.dumps(recent_boxes)};
        
        chart.timeScale().fitContent();
        
        function positionDeltaBoxes() {{
            const timeScale = chart.timeScale();
            const visibleRange = timeScale.getVisibleRange();
            const chartWidth = chartContainer.clientWidth;
            
            if (!visibleRange || !deltaData.length) return;
            
            deltaData.forEach((data, index) => {{
                const deltaInfo = deltaContainer.querySelector(`[data-time="${{data.time}}"]`);
                if (!deltaInfo) return;
                
                const x = timeScale.timeToCoordinate(data.time);
                if (x !== null && x >= 0 && x <= chartWidth) {{
                    const boxWidth = isMobile ? 35 : 45;
                    const leftPos = Math.max(5, Math.min(x - boxWidth/2, chartWidth - boxWidth - 5));
                    deltaInfo.style.left = leftPos + 'px';
                    deltaInfo.style.display = 'block';
                }} else {{
                    deltaInfo.style.display = 'none';
                }}
            }});
        }}
        
        setTimeout(positionDeltaBoxes, 200);
        chart.timeScale().subscribeVisibleTimeRangeChange(positionDeltaBoxes);
        
        // Responsive resize observer
        const resizeObserver = new ResizeObserver(entries => {{
            if (entries.length === 0 || entries[0].target !== chartContainer) return;
            const newRect = entries[0].contentRect;
            const newIsMobile = window.innerWidth < 768;
            const newHeight = newIsMobile ? 350 : 450;
            
            chart.applyOptions({{ 
                width: newRect.width,
                height: newHeight,
                layout: {{
                    fontSize: newIsMobile ? 10 : 12
                }},
                grid: {{
                    vertLines: {{ visible: !newIsMobile }}
                }},
                timeScale: {{
                    rightOffset: newIsMobile ? 5 : 12,
                    barSpacing: newIsMobile ? 6 : 8
                }}
            }});
            
            setTimeout(positionDeltaBoxes, 100);
        }});
        
        resizeObserver.observe(chartContainer);
        
        // Handle orientation change on mobile
        window.addEventListener('orientationchange', () => {{
            setTimeout(() => {{
                chart.applyOptions({{ width: chartContainer.clientWidth }});
                positionDeltaBoxes();
            }}, 300);
        }});
        
    }})();
    </script>
    """

def create_simplified_table(df):
    """Create a simplified HTML table with only required columns"""
    if df.empty:
        return '<div style="text-align: center; padding: 20px; color: #6c757d;">No data available</div>'
    
    # Get last 20 rows for better mobile performance
    display_df = df.tail(20).copy()
    
    html = '''
    <table class="simplified-table">
        <thead>
            <tr>
                <th>Time</th>
                <th>Close</th>
                <th>BI</th>
                <th>SI</th>
                <th>TD</th>
                <th>CTD</th>
            </tr>
        </thead>
        <tbody>
    '''
    
    for _, row in display_df.iterrows():
        time_str = row['timestamp'].strftime('%H:%M')
        close_price = f"{row['close']:.1f}"
        buy_initiated = int(row['buy_initiated'])
        sell_initiated = int(row['sell_initiated'])
        tick_delta = int(row['tick_delta'])
        cum_tick_delta = int(row['cumulative_tick_delta'])
        
        # Apply styling classes
        td_class = "positive-delta" if tick_delta > 0 else ("negative-delta" if tick_delta < 0 else "")
        ctd_class = "positive-delta" if cum_tick_delta > 0 else ("negative-delta" if cum_tick_delta < 0 else "")
        
        html += f'''
        <tr>
            <td>{time_str}</td>
            <td>{close_price}</td>
            <td>{buy_initiated}</td>
            <td>{sell_initiated}</td>
            <td class="{td_class}">{tick_delta:+d}</td>
            <td class="{ctd_class}">{cum_tick_delta:+d}</td>
        </tr>
        '''
    
    html += '''
        </tbody>
    </table>
    '''
    
    return html

# --- Main Application ---
inject_lightweight_chart_css()

# Sidebar Controls
st.sidebar.title("📱 Order Flow")
st.sidebar.markdown("---")

security_options = fetch_security_ids()

if not security_options:
    security_options = ["No Data Available (0)"]

selected_option = st.sidebar.selectbox("🎯 Security", security_options)

if selected_option is None:
    selected_option = "No Data Available (0)"

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

# --- Fetch and process data ---
historical_df = fetch_historical_data(selected_id)
live_df = fetch_live_data(selected_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# Filter for current day between 9:00 and 23:59
today = datetime.now().date()
start_time = datetime.combine(today, time(9, 0))
end_time = datetime.combine(today, time(23, 59, 59))
full_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]

agg_df = aggregate_data(full_df, interval)

# --- Main Display ---
stock_name = selected_option.split(' (')[0]

# Calculate current price and change for header
current_price = "---"
price_change = "---"
price_change_pct = "---"
price_class = "price-negative"

if not agg_df.empty:
    latest = agg_df.iloc[-1]
    current_price = f"{latest.get('close', 0):.1f}"
    
    if len(agg_df) > 1:
        prev_close = agg_df.iloc[-2].get('close', latest.get('open', 0))
        change = latest.get('close', 0) - prev_close
        change_pct = (change / prev_close * 100) if prev_close != 0 else 0
        price_change = f"↑ +{change:.1f}" if change > 0 else f"↓ {change:.1f}"
        price_change_pct = f"({change_pct:+.2f}%)"
        price_class = "price-positive" if change > 0 else "price-negative"

# Header with stock info
st.markdown(f"""
<div class="trading-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div class="stock-info">
            <span style="background: #007bff; padding: 4px 8px; border-radius: 4px; margin-right: 10px; font-size: 12px;">NSE</span>
            <span style="margin-right: 15px;">{stock_name}</span>
            <div class="price-info">
                <span class="{price_class}">{current_price} {price_change} {price_change_pct}</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Simplified Control Panel
st.markdown('<div class="control-panel">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.markdown('<div class="control-label">Interval (min)</div>', unsafe_allow_html=True)
    interval_display = st.selectbox(
        "Interval", 
        options=[1, 3, 5, 15, 30],
        index=[1, 3, 5, 15, 30].index(interval) if interval in [1, 3, 5, 15, 30] else 0,
        key="interval_selector",
        label_visibility="collapsed"
    )
    if interval_display != interval:
        interval = interval_display
        st.rerun()

with col2:
    st.markdown('<div class="control-label">Chart Type</div>', unsafe_allow_html=True)
    chart_type = st.selectbox(
        "Chart Type", 
        options=["Candlestick"],
        key="chart_type_selector",
        label_visibility="collapsed"
    )

with col3:
    st.markdown('<div class="control-label">Auto Refresh</div>', unsafe_allow_html=True)
    st.caption(f"🔄 Updates every {refresh_interval}s • {interval}min intervals")

st.markdown('</div>', unsafe_allow_html=True)

# Responsive chart display
if not agg_df.empty:
    with st.spinner("Loading chart data..."):
        chart_html = create_responsive_lightweight_chart(stock_name, agg_df, interval)
        components.html(chart_html, height=550)
else:
    st.warning("No data available for the selected stock and interval.")

# Simplified data table section
if not agg_df.empty:
    st.markdown('<div class="data-table-section">', unsafe_allow_html=True)
    
    # Create two columns for table header and download button
    col_table, col_download = st.columns([2, 1])
    
    with col_table:
        st.markdown("### 📋 Trading Activity")
    
    with col_download:
        # Download button
        csv = agg_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 CSV",
            csv,
            f"orderflow_{stock_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Display simplified HTML table
    table_html = create_simplified_table(agg_df)
    st.markdown(table_html, unsafe_allow_html=True)
    
    # Compact summary statistics
    st.markdown("#### 📊 Session Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_buy_vol = agg_df['buy_volume'].sum()
    total_sell_vol = agg_df['sell_volume'].sum()
    total_volume = total_buy_vol + total_sell_vol
    net_delta = agg_df['tick_delta'].sum()
    
    with col1:
        st.metric("Buy Vol", f"{total_buy_vol:,.0f}")
    
    with col2:
        st.metric("Sell Vol", f"{total_sell_vol:,.0f}")
    
    with col3:
        st.metric("Total Vol", f"{total_volume:,.0f}")
    
    with col4:
        st.metric("Net TD", f"{net_delta:+,.0f}")
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="data-table-section">', unsafe_allow_html=True)
    st.markdown("### 📋 Trading Activity")
    st.markdown('<div style="text-align: center; padding: 40px; color: #6c757d;">No data available for the selected stock and interval.</div>')
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6c757d; font-size: 12px; padding: 10px;">
    📊 Order Flow Dashboard | 🔄 Auto-refresh: {refresh_interval}s | ⏱️ Interval: {interval}min
</div>
""", unsafe_allow_html=True)
