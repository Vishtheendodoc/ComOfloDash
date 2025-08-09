import os
import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta
import re
import threading
import time
import logging
import streamlit.components.v1 as components


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
        'last_alert_time': datetime.datetime.now().isoformat()
    }
    try:
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f)
    except Exception as e:
        st.error(f"Failed to save alert state: {e}")

def determine_gradient_state(cumulative_delta):
    """Determine if cumulative delta is positive or negative relative to zero"""
    if cumulative_delta > 0:
        return "positive"  # Above zero
    elif cumulative_delta < 0:
        return "negative"  # Below zero
    else:
        return "zero"  # Exactly at zero

def check_gradient_change(security_id, df):
    """Check if cumulative delta crosses zero and send alert if needed"""
    if df.empty:
        return False
    
    # Get the latest cumulative tick delta
    latest_row = df.iloc[-1]
    current_cum_delta = latest_row['cumulative_tick_delta']
    current_state = determine_gradient_state(current_cum_delta)
    current_timestamp = latest_row['timestamp']
    
    # Get last known state
    last_alert = get_last_alert_state(security_id)
    
    # Check if state changed from positive to negative or vice versa
    if last_alert:
        last_state = last_alert.get('state')
        last_alert_time = datetime.datetime.fromisoformat(last_alert.get('last_alert_time'))
        
        # Only alert on zero-crossing transitions and if 5 min have passed
        zero_cross_occurred = (
            (last_state == "positive" and current_state == "negative") or
            (last_state == "negative" and current_state == "positive")
        )
        
        if (zero_cross_occurred and 
            datetime.datetime.now() - last_alert_time > datetime.timedelta(minutes=5)):
            
            stock_name = stock_mapping.get(str(security_id), f"Stock {security_id}")
            
            if current_state == "positive":
                emoji = "🟢"
                direction = "POSITIVE"
                cross_direction = "CROSSED ABOVE ZERO"
            else:  # negative
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
        # First time - just save the state without alerting
        save_alert_state(security_id, current_state, current_timestamp)
    
    return False

def check_gradient_change_enhanced(security_id, df, sensitivity_threshold=50):
    """Enhanced zero cross detection with sensitivity threshold"""
    if df.empty:
        return False
    
    # Get the latest cumulative tick delta
    latest_row = df.iloc[-1]
    current_cum_delta = latest_row['cumulative_tick_delta']
    current_state = determine_gradient_state(current_cum_delta)
    current_timestamp = latest_row['timestamp']
    
    # Get last known state
    last_alert = get_last_alert_state(security_id)
    
    # Check if state changed from positive to negative or vice versa
    if last_alert:
        last_state = last_alert.get('state')
        last_alert_time = datetime.datetime.fromisoformat(last_alert.get('last_alert_time'))
        
        # Only alert on zero-crossing transitions with sufficient magnitude
        zero_cross_occurred = (
            (last_state == "positive" and current_state == "negative" and abs(current_cum_delta) >= sensitivity_threshold) or
            (last_state == "negative" and current_state == "positive" and abs(current_cum_delta) >= sensitivity_threshold)
        )
        
        if (zero_cross_occurred and 
            datetime.datetime.now() - last_alert_time > datetime.timedelta(minutes=5)):
            
            stock_name = stock_mapping.get(str(security_id), f"Stock {security_id}")
            
            if current_state == "positive":
                emoji = "🟢"
                direction = "POSITIVE"
                cross_direction = "CROSSED ABOVE ZERO"
            else:  # negative
                emoji = "🔴"
                direction = "NEGATIVE"
                cross_direction = "CROSSED BELOW ZERO"
            
            # Calculate momentum (how far from zero)
            momentum = abs(current_cum_delta)
            momentum_text = f"Strong momentum ({momentum})" if momentum > sensitivity_threshold * 2 else f"Moderate momentum ({momentum})"
            
            message = f"""
{emoji} <b>ZERO CROSS ALERT</b> {emoji}

📈 <b>Stock:</b> {stock_name}
🔄 <b>Transition:</b> {last_state.upper()} → <b>{direction}</b>
⚡ <b>Event:</b> {cross_direction}
📊 <b>Cumulative Tick Delta:</b> {int(current_cum_delta)}
🚀 <b>Momentum:</b> {momentum_text}
⏰ <b>Time:</b> {current_timestamp.strftime('%H:%M:%S')}
💰 <b>Price:</b> ₹{latest_row['close']:.1f}

Cumulative delta has {cross_direction.lower()}! 🚨
            """.strip()
            
            if send_telegram_alert(message):
                save_alert_state(security_id, current_state, current_timestamp)
                return True
    else:
        # First time - just save the state without alerting
        save_alert_state(security_id, current_state, current_timestamp)
    
    return False

def fetch_stock_data_efficient(security_id, timeout=10):
    """Efficiently fetch data for a single stock with timeout"""
    try:
        # Try live API first (fastest)
        api_url = f"{FLASK_API_BASE}/delta_data/{security_id}?interval=1"
        response = requests.get(api_url, timeout=timeout)
        
        if response.status_code == 200:
            live_data = pd.DataFrame(response.json())
            if not live_data.empty:
                live_data['timestamp'] = pd.to_datetime(live_data['timestamp'])
                live_data.sort_values('timestamp', inplace=True)
                
                # Filter for current day
                today = datetime.now().date()
                start_time = datetime.combine(today, time(9, 0))
                end_time = datetime.combine(today, time(23, 59, 59))
                day_data = live_data[
                    (live_data['timestamp'] >= pd.Timestamp(start_time)) & 
                    (live_data['timestamp'] <= pd.Timestamp(end_time))
                ]
                
                if not day_data.empty:
                    return day_data
        
        # Fallback to local cache if API fails
        cache_df = load_from_local_cache(security_id)
        if not cache_df.empty:
            today = datetime.now().date()
            start_time = datetime.combine(today, time(9, 0))
            end_time = datetime.combine(today, time(23, 59, 59))
            day_data = cache_df[
                (cache_df['timestamp'] >= pd.Timestamp(start_time)) & 
                (cache_df['timestamp'] <= pd.Timestamp(end_time))
            ]
            return day_data.tail(50)  # Last 50 records
            
    except Exception as e:
        # Silent fail for individual stocks to avoid spam
        pass
    
    return pd.DataFrame()

def should_check_stock(security_id):
    """Check if enough time has passed since last check"""
    last_check_file = os.path.join(ALERT_CACHE_DIR, f"last_check_{security_id}.txt")
    
    if os.path.exists(last_check_file):
        try:
            with open(last_check_file, 'r') as f:
                last_check_time = datetime.fromisoformat(f.read().strip())
                time_diff = datetime.now() - last_check_time
                return time_diff > timedelta(minutes=MONITOR_COOLDOWN_MINUTES)
        except Exception:
            pass
    
    return True

def update_last_check_time(security_id):
    """Update the last check time for a stock"""
    last_check_file = os.path.join(ALERT_CACHE_DIR, f"last_check_{security_id}.txt")
    try:
        with open(last_check_file, 'w') as f:
            f.write(datetime.now().isoformat())
    except Exception:
        pass

def process_single_stock(security_id, use_enhanced=False, sensitivity=50):
    """Process a single stock for zero cross changes"""
    try:
        # Skip if recently checked
        if not should_check_stock(security_id):
            return False, f"Recently checked"
        
        # Fetch data
        df = fetch_stock_data_efficient(security_id, timeout=8)
        
        if df.empty:
            return False, "No data"
        
        # Aggregate data (use 3-minute intervals for efficiency)
        agg_df = aggregate_data(df, 3)
        
        if agg_df.empty:
            return False, "No aggregated data"
        
        # Check for zero cross changes
        if use_enhanced:
            alert_sent = check_gradient_change_enhanced(security_id, agg_df, sensitivity)
        else:
            alert_sent = check_gradient_change(security_id, agg_df)
        
        # Update last check time
        update_last_check_time(security_id)
        
        return alert_sent, "Processed successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"
    
def monitor_all_stocks_enhanced():
    """Enhanced monitoring of all stocks with concurrent processing"""
    try:
        # Load all security IDs
        stock_df = pd.read_csv(STOCK_LIST_FILE)

        # ✅ Always scan all stocks, regardless of trading hour
        all_security_ids = stock_df['security_id'].unique()
        st.sidebar.info(f"🧮 Monitoring {len(all_security_ids)} stocks in this cycle")

        alerts_sent = 0
        processed = 0

        def worker(security_id):
            nonlocal alerts_sent, processed
            try:
                if not should_check_stock(security_id):
                    return
                processed += 1
                result = process_single_stock(security_id)
                if result:
                    alerts_sent += 1
            except Exception as e:
                log_error(f"❌ Error processing stock {security_id}: {str(e)}")

        threads = []
        for sec_id in all_security_ids:
            t = threading.Thread(target=worker, args=(sec_id,))
            threads.append(t)
            t.start()

            # Throttle thread creation to prevent resource exhaustion
            while threading.active_count() > API_BATCH_SIZE:
                time.sleep(0.1)

        for t in threads:
            t.join()

        # ✅ Log the result for dashboard
        log_file = os.path.join(ALERT_CACHE_DIR, "monitoring_log.txt")
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()}: {alerts_sent} alerts, {processed} processed\n")

        return alerts_sent, processed

    except Exception as e:
        log_error(f"❌ Failed in enhanced monitoring: {str(e)}")
        return 0, 0


# --- Background Alert System (Advanced Option) ---
def start_background_monitoring():
    """Start background monitoring in a separate thread"""
    def background_monitor():
        while True:
            try:
                # Check if alerts are enabled (you'll need to store this in a file or session state)
                alert_status_file = os.path.join(ALERT_CACHE_DIR, "alert_status.txt")
                if os.path.exists(alert_status_file):
                    with open(alert_status_file, 'r') as f:
                        alerts_enabled = f.read().strip() == "True"
                else:
                    alerts_enabled = False
                
                if alerts_enabled:
                    alerts_sent, processed = monitor_all_stocks_enhanced()
                    
                    # Log monitoring activity
                    log_file = os.path.join(ALERT_CACHE_DIR, "monitoring_log.txt")
                    with open(log_file, 'a') as f:
                        f.write(f"{datetime.now().isoformat()}: {alerts_sent} alerts, {processed} processed\n")
                
                # Wait for next cycle (configurable)
                time.sleep(120)  # 2 minutes between cycles
                
            except Exception as e:
                # Log errors but continue monitoring
                time.sleep(60)  # Wait 1 minute on error
    
    # Start background thread
    thread = threading.Thread(target=background_monitor, daemon=True)
    thread.start()
    return thread


# --- CSS from charting per.py ---
def inject_full_width_chart_css():
    st.markdown("""
    <style>
        .main > div {padding-top: 0rem; padding-bottom: 0rem;}
        .element-container {width: 100% !important; max-width: 100% !important;}
        .trading-header {
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            padding: 20px; border-radius: 8px; margin-bottom: 20px;
            color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stock-info {display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;}
        .stock-name {font-size: 24px; font-weight: bold;}
        .price-positive {color: #22c55e; font-weight: bold;}
        .price-negative {color: #ef4444; font-weight: bold;}
        .delta-boxes {display: flex; justify-content: center; gap: 20px; margin: 20px 0; flex-wrap: wrap;}
        .delta-box {
            background: white; border: 2px solid #e5e7eb; border-radius: 12px;
            padding: 20px 30px; min-width: 120px; text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;
        }
        .delta-positive {border-color: #22c55e; background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);}
        .delta-negative {border-color: #ef4444; background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);}
        .delta-neutral {border-color: #6b7280; background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);}
        .delta-value {font-size: 28px; font-weight: bold; margin: 0; line-height: 1;}
        .delta-label {font-size: 14px; color: #6b7280; margin: 8px 0 0 0; font-weight: 500;}
        .delta-positive .delta-value {color: #16a34a;}
        .delta-negative .delta-value {color: #dc2626;}
        .delta-neutral .delta-value {color: #6b7280;}
        .lightweight-chart-container {width: 100% !important; height: 800px; border: 1px solid #e5e7eb; border-radius: 8px; margin: 20px 0;}
        @media (max-width: 768px) {
            .trading-header {padding: 15px;}
            .stock-name {font-size: 20px;}
            .stock-info {flex-direction: column; align-items: flex-start; gap: 15px;}
            .delta-boxes {gap: 15px;}
            .delta-box {min-width: 100px; padding: 15px 20px;}
            .delta-value {font-size: 24px;}
        }
    </style>
    """, unsafe_allow_html=True)
inject_full_width_chart_css()

# --- Keep your mobile CSS ---
def inject_mobile_css():
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .css-1d391kg {padding: 0.5rem !important;}
        .main .block-container {padding-top: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important;}
        .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 12px; color: white; text-align: center; margin: 4px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .metric-value {font-size: 18px; font-weight: bold; margin: 0;}
        .metric-label {font-size: 11px; opacity: 0.9; margin: 0;}
    }
    </style>
    """, unsafe_allow_html=True)

# --- TradingView chart function ---
def create_tradingview_chart_with_delta_boxes(stock_name, chart_data, interval):
    """Enhanced chart with GoCharting-style tick delta and cumulative delta boxes"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Prepare all data series
    candle_data = []
    tick_delta_values = []
    cumulative_delta_values = []
    
    for _, row in chart_data.tail(100).iterrows():
        try:
            timestamp = int(pd.to_datetime(row['timestamp']).timestamp())
            
            # Candlestick data
            candle_data.append({
                'time': timestamp,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0))
            })
            
            # Store delta values for box creation
            tick_delta = float(row.get('tick_delta', 0))
            cum_delta = float(row.get('cumulative_tick_delta', 0))
            
            tick_delta_values.append({
                'timestamp': timestamp,
                'value': tick_delta,
                'formatted': f"{tick_delta:+,.0f}" if tick_delta != 0 else "0"
            })
            
            cumulative_delta_values.append({
                'timestamp': timestamp,
                'value': cum_delta,
                'formatted': f"{cum_delta:+,.0f}" if cum_delta != 0 else "0"
            })
            
        except:
            continue
    
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    
    chart_html = f"""
<div class="chart-with-delta-container" style="width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 8px;">
    <!-- Main Chart -->
    <div id="{chart_id}" style="width: 100%; height: 500px;"></div>
    
    <!-- Delta Boxes Container -->
    <div style="padding: 10px; background: #f8fafc; border-top: 1px solid #e5e7eb;">
        <!-- Tick Delta Row -->
        <div style="margin-bottom: 8px;">
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 4px;">
                Tick Delta
            </div>
            <div class="delta-row" id="tick-delta-row" style="display: flex; gap: 2px; overflow-x: auto; padding: 2px 0;">
                <!-- Tick delta boxes will be inserted here -->
            </div>
        </div>
        
        <!-- Cumulative Delta Row -->
        <div>
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 4px;">
                Cumulative Delta
            </div>
            <div class="delta-row" id="cumulative-delta-row" style="display: flex; gap: 2px; overflow-x: auto; padding: 2px 0;">
                <!-- Cumulative delta boxes will be inserted here -->
            </div>
        </div>
    </div>
</div>

<style>
.delta-row {{
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 #f1f5f9;
}}
.delta-row::-webkit-scrollbar {{
    height: 6px;
}}
.delta-row::-webkit-scrollbar-track {{
    background: #f1f5f9;
    border-radius: 3px;
}}
.delta-row::-webkit-scrollbar-thumb {{
    background: #cbd5e1;
    border-radius: 3px;
}}
.delta-box {{
    min-width: 60px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 600;
    border-radius: 4px;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
    white-space: nowrap;
    cursor: default;
    transition: transform 0.1s ease;
}}
.delta-box:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
}}
.delta-positive {{
    background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
    border: 1px solid #15803d;
}}
.delta-negative {{
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
    border: 1px solid #b91c1c;
}}
.delta-zero {{
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
    border: 1px solid #374151;
}}
</style>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    const container = document.getElementById('{chart_id}');
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">Chart library not loaded</div>';
        return;
    }}
    
    container.innerHTML = '';
    
    // Create main candlestick chart
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
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {{
                width: 1,
                color: '#9B7DFF',
                style: LightweightCharts.LineStyle.Solid,
            }},
            horzLine: {{
                width: 1,
                color: '#9B7DFF', 
                style: LightweightCharts.LineStyle.Solid,
            }},
        }},
        rightPriceScale: {{
            borderColor: '#D6DCDE',
        }},
        timeScale: {{
            borderColor: '#D6DCDE',
            timeVisible: true,
            secondsVisible: false
        }},
        autoSize: true
    }});
    
    // Add candlestick series
    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350'
    }});
    
    candleSeries.setData({json.dumps(candle_data)});
    
    // Create delta boxes
    const tickDeltaData = {json.dumps(tick_delta_values)};
    const cumulativeDeltaData = {json.dumps(cumulative_delta_values)};
    
    function createDeltaBoxes(data, containerId) {{
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        
        data.forEach((item, index) => {{
            const box = document.createElement('div');
            box.className = 'delta-box';
            
            // Determine color class based on value
            if (item.value > 0) {{
                box.classList.add('delta-positive');
            }} else if (item.value < 0) {{
                box.classList.add('delta-negative');
            }} else {{
                box.classList.add('delta-zero');
            }}
            
            // Set text content
            box.textContent = item.formatted;
            
            // Add tooltip
            box.title = `Time: ${{new Date(item.timestamp * 1000).toLocaleTimeString()}}\\nValue: ${{item.formatted}}`;
            
            container.appendChild(box);
        }});
    }}
    
    // Create both delta box rows
    createDeltaBoxes(tickDeltaData, 'tick-delta-row');
    createDeltaBoxes(cumulativeDeltaData, 'cumulative-delta-row');
    
    // Fit content
    chart.timeScale().fitContent();
    
    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({{ 
            width: rect.width, 
            height: 500
        }});
    }});
    
    resizeObserver.observe(container);
    
    // Cleanup
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        chart.remove();
    }});
}})();
</script>
    """
    return chart_html
    """Enhanced chart with GoCharting-style tick delta and cumulative delta rows"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Prepare all data series
    candle_data = []
    tick_delta_data = []
    cumulative_delta_data = []
    volume_data = []
    
    for _, row in chart_data.tail(100).iterrows():
        try:
            timestamp = int(pd.to_datetime(row['timestamp']).timestamp())
            
            # Candlestick data
            candle_data.append({
                'time': timestamp,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0))
            })
            
            # Tick delta data (can be positive or negative)
            tick_delta = float(row.get('tick_delta', 0))
            tick_delta_data.append({
                'time': timestamp,
                'value': tick_delta
            })
            
            # Cumulative delta data (can be positive or negative)
            cum_delta = float(row.get('cumulative_tick_delta', 0))
            cumulative_delta_data.append({
                'time': timestamp,
                'value': cum_delta
            })
            
            # Volume data for reference
            buy_vol = float(row.get('buy_volume', 0))
            sell_vol = float(row.get('sell_volume', 0))
            total_vol = buy_vol + sell_vol
            volume_data.append({
                'time': timestamp,
                'value': total_vol,
                'color': '#22c55e' if buy_vol > sell_vol else '#ef4444'
            })
            
        except:
            continue
    
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    
    chart_html = f"""
<div class="lightweight-chart-container" style="position: relative;">
    <div id="{chart_id}" style="width: 100%; height: 100%;"></div>
    <div id="{chart_id}_labels" style="position: absolute; left: 5px; top: 50%; transform: translateY(-50%); background: rgba(255,255,255,0.9); padding: 8px; border-radius: 4px; font-size: 11px; z-index: 10;">
        <div style="margin-bottom: 20px;"><strong>Price</strong></div>
        <div style="margin-bottom: 20px;"><strong>Volume</strong></div>
        <div style="margin-bottom: 20px;"><strong>Tick Δ</strong></div>
        <div><strong>Cum Δ</strong></div>
    </div>
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
    
    // Create main chart with multiple panes
    const chart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 800,
        layout: {{
            background: {{ type: 'solid', color: '#ffffff' }},
            textColor: '#333',
            fontSize: 11
        }},
        grid: {{
            vertLines: {{ color: '#f0f0f0' }},
            horzLines: {{ color: '#f0f0f0' }}
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: {{
                width: 1,
                color: '#9B7DFF',
                style: LightweightCharts.LineStyle.Solid,
            }},
            horzLine: {{
                width: 1,
                color: '#9B7DFF', 
                style: LightweightCharts.LineStyle.Solid,
            }},
        }},
        rightPriceScale: {{
            borderColor: '#D6DCDE',
            scaleMargins: {{ top: 0.05, bottom: 0.7 }}
        }},
        timeScale: {{
            borderColor: '#D6DCDE',
            timeVisible: true,
            secondsVisible: false
        }},
        autoSize: true
    }});
    
    // 1. Main candlestick series (top pane - 40% of chart)
    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        priceScaleId: 'price'
    }});
    candleSeries.setData({json.dumps(candle_data)});
    
    // 2. Volume series (second pane - 15% of chart) 
    const volumeSeries = chart.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: 'volume',
        scaleMargins: {{ top: 0.5, bottom: 0.55 }},
    }});
    volumeSeries.setData({json.dumps(volume_data)});
    
    // 3. Tick Delta series (third pane - 20% of chart)
    const tickDeltaSeries = chart.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: 'tick-delta', 
        scaleMargins: {{ top: 0.7, bottom: 0.3 }},
    }});
    
    // Color-code tick delta bars based on positive/negative values
    const tickDeltaWithColors = {json.dumps(tick_delta_data)}.map(item => ({{
        time: item.time,
        value: Math.abs(item.value),
        color: item.value >= 0 ? '#26a69a' : '#ef5350'
    }}));
    tickDeltaSeries.setData(tickDeltaWithColors);
    
    // 4. Cumulative Delta series (bottom pane - 25% of chart)
    const cumulativeDeltaSeries = chart.addHistogramSeries({{
        priceFormat: {{ type: 'volume' }},
        priceScaleId: 'cumulative-delta',
        scaleMargins: {{ top: 0.85, bottom: 0.05 }},
    }});
    
    // Color-code cumulative delta bars
    const cumDeltaWithColors = {json.dumps(cumulative_delta_data)}.map(item => ({{
        time: item.time,
        value: Math.abs(item.value),
        color: item.value >= 0 ? '#26a69a' : '#ef5350'
    }}));
    cumulativeDeltaSeries.setData(cumDeltaWithColors);
    
    // Configure price scales for each pane
    chart.priceScale('price').applyOptions({{
        scaleMargins: {{ top: 0.05, bottom: 0.7 }},
        borderColor: '#D6DCDE',
    }});
    
    chart.priceScale('volume').applyOptions({{
        scaleMargins: {{ top: 0.5, bottom: 0.55 }},
        borderColor: '#D6DCDE',
    }});
    
    chart.priceScale('tick-delta').applyOptions({{
        scaleMargins: {{ top: 0.7, bottom: 0.3 }},
        borderColor: '#D6DCDE',
    }});
    
    chart.priceScale('cumulative-delta').applyOptions({{
        scaleMargins: {{ top: 0.85, bottom: 0.05 }},
        borderColor: '#D6DCDE',
    }});
    
    // Add zero line for delta indicators
    const tickDeltaZeroLine = chart.addLineSeries({{
        color: '#666666',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceScaleId: 'tick-delta',
    }});
    
    const cumDeltaZeroLine = chart.addLineSeries({{
        color: '#666666', 
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
        priceScaleId: 'cumulative-delta',
    }});
    
    // Create zero line data points
    const firstTime = {json.dumps(candle_data)}[0]?.time;
    const lastTime = {json.dumps(candle_data)}[{json.dumps(candle_data)}.length - 1]?.time;
    
    if (firstTime && lastTime) {{
        const zeroLineData = [
            {{ time: firstTime, value: 0 }},
            {{ time: lastTime, value: 0 }}
        ];
        tickDeltaZeroLine.setData(zeroLineData);
        cumDeltaZeroLine.setData(zeroLineData);
    }}
    
    // Fit content
    chart.timeScale().fitContent();
    
    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({{ 
            width: rect.width, 
            height: Math.max(800, rect.height)
        }});
    }});
    
    resizeObserver.observe(container);
    
    // Cleanup
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        chart.remove();
    }});
}})();
</script>
    """
    return chart_html



@st.cache_data(ttl=6000)
def fetch_security_ids():
    try:
        # First try to get IDs from data snapshots
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
        
        # If no data snapshots exist, fall back to stock_list.csv
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
        # Final fallback - try to load from stock list
        try:
            stock_df = pd.read_csv(STOCK_LIST_FILE)
            ids = sorted(list(stock_df['security_id'].unique()))
            return [f"{stock_mapping.get(str(i), f'Stock {i}')} ({i})" for i in ids]
        except:
            return ["No Data Available (0)"]

security_options = fetch_security_ids()

# Ensure security_options is not empty
if not security_options:
    security_options = ["No Data Available (0)"]

selected_option = st.sidebar.selectbox("🎯 Security", security_options)

# Handle None or invalid selected_option
if selected_option is None:
    selected_option = "No Data Available (0)"

# Extract security ID safely
match = re.search(r'\((\d+)\)', selected_option)
if match:
    selected_id = int(match.group(1))
    if selected_id == 0:  # Fallback case
        st.error("⚠️ No security data available. Please check your data source.")
        st.stop()
else:
    st.error(f"⚠️ Selected option '{selected_option}' does not contain a valid ID")
    st.stop()

match = re.search(r'\((\d+)\)', selected_option)
if match:
    selected_id = int(match.group(1))
else:
    selected_id = None
    st.error(f"⚠️ Selected option '{selected_option}' does not contain an ID")

interval = st.sidebar.selectbox("⏱️ Interval", [1, 3, 5, 15, 30, 60, 90, 120, 180, 240, 360, 480], index=2)

mobile_view = st.sidebar.toggle("📱 Mobile Mode", value=True)

if mobile_view:
    inject_mobile_css()

# --- Sidebar Controls ---
st.sidebar.title("📱 Order Flow")
st.sidebar.markdown("---")



# --- Enhanced Sidebar Controls ---
def enhanced_alert_controls():
    """Enhanced alert controls in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Enhanced Alert System")
    
    # Alert toggle
    alert_enabled = st.sidebar.toggle("Enable Smart Alerts", value=False, key="enhanced_alerts")
    
    # Save alert status for background monitoring
    alert_status_file = os.path.join(ALERT_CACHE_DIR, "alert_status.txt")
    with open(alert_status_file, 'w') as f:
        f.write(str(alert_enabled))
    
    if alert_enabled:
        # Monitoring options
        monitor_mode = st.sidebar.radio(
            "Monitoring Mode:",
            ["Auto (Every 2 min)", "Manual Check", "Background Mode"],
            key="monitor_mode"
        )
        
        # Stock filtering options
        stock_filter = st.sidebar.selectbox(
            "Monitor Which Stocks:",
            ["All Stocks", "NIFTY Indices Only", "Top 50 by Volume", "Custom List"],
            key="stock_filter"
        )
        
        # Alert sensitivity
        sensitivity = st.sidebar.selectbox(
            "Alert Sensitivity:",
            ["High (Any change)", "Medium (Significant changes)", "Low (Major changes only)"],
            index=1,
            key="alert_sensitivity"
        )
        
        # Manual check button
        if st.sidebar.button("🔍 Check All Stocks Now", key="manual_check"):
            with st.spinner("🔄 Checking all stocks for gradient changes..."):
                alerts_sent, processed = monitor_all_stocks_enhanced()
        
        # Auto monitoring
        if monitor_mode == "Auto (Every 2 min)":
            # Use streamlit auto-refresh for monitoring
            st_autorefresh(interval=120000, key="enhanced_all_stock_monitor")
            with st.spinner("🔄 Auto-monitoring all stocks..."):
                alerts_sent, processed = monitor_all_stocks_enhanced()
        
        elif monitor_mode == "Background Mode":
            if st.sidebar.button("🚀 Start Background Monitoring"):
                thread = start_background_monitoring()
                st.sidebar.success("✅ Background monitoring started!")
                st.sidebar.info("💡 Monitoring will continue even when viewing different stocks")
        
        # Show monitoring stats
        st.sidebar.markdown("#### 📊 Monitoring Stats")
        
        # Read recent monitoring log
        log_file = os.path.join(ALERT_CACHE_DIR, "monitoring_log.txt")
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()[-5:]  # Last 5 entries
                for line in lines:
                    if line.strip():
                        parts = line.strip().split(": ")
                        if len(parts) == 2:
                            timestamp = parts[0].split("T")[1][:5]  # Extract time
                            stats = parts[1]
                            st.sidebar.caption(f"🕒 {timestamp}: {stats}")
            except Exception:
                pass
        
        # Test alert button
        if st.sidebar.button("🧪 Test Zero Cross Alert"):
            test_message = f"""
🟢 <b>ZERO CROSS TEST ALERT</b> 🟢

📈 <b>Stock:</b> TEST STOCK
🔄 <b>Transition:</b> NEGATIVE → <b>POSITIVE</b>
⚡ <b>Event:</b> CROSSED ABOVE ZERO
📊 <b>Cumulative Tick Delta:</b> +75
🚀 <b>Momentum:</b> Moderate momentum (75)
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}
💰 <b>Price:</b> ₹1250.5

This is a test of the zero cross alert system! 🚨
            """.strip()
            
            if send_telegram_alert(test_message):
                st.sidebar.success("✅ Zero cross test alert sent!")
            else:
                st.sidebar.error("❌ Failed to send zero cross test alert")

def add_sensitivity_control_to_sidebar():
    """Add this code block inside the enhanced_alert_controls function after the alert_enabled toggle"""
    
    if alert_enabled:  # This should be inside the existing if alert_enabled block
        # Add sensitivity control
        st.sidebar.markdown("#### ⚙️ Alert Configuration")
        
        # Zero cross sensitivity
        sensitivity_threshold = st.sidebar.slider(
            "Zero Cross Sensitivity:",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Minimum cumulative delta magnitude required to trigger zero cross alert",
            key="zero_cross_sensitivity"
        )
        
        # Alert mode selection
        alert_mode = st.sidebar.radio(
            "Alert Mode:",
            ["Basic Zero Cross", "Enhanced (with sensitivity)"],
            index=1,
            key="alert_mode"
        )
        
        st.sidebar.caption(f"💡 Current threshold: ±{sensitivity_threshold}")
        
        # Save settings to file for background monitoring
        settings = {
            "sensitivity_threshold": sensitivity_threshold,
            "enhanced_mode": alert_mode == "Enhanced (with sensitivity)"
        }
        settings_file = os.path.join(ALERT_CACHE_DIR, "alert_settings.json")
        with open(settings_file, 'w') as f:
            json.dump(settings, f)

enhanced_alert_controls()
st.sidebar.markdown("---")

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
import datetime
today = datetime.datetime.now().date()
start_time = datetime.datetime.combine(today, datetime.time(9, 0))
end_time = datetime.datetime.combine(today, datetime.time(23, 59, 59))
full_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]

agg_df = aggregate_data(full_df, interval)

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
    """Create a highly optimized mobile table for mobile view, with single-row header and no tooltips."""
    if df.empty:
        return

    import datetime
    # Today's time window
    today = datetime.datetime.now().date()
    start_time = datetime.datetime.combine(today, datetime.time(9, 0))
    end_time = datetime.datetime.combine(today, datetime.time(23, 59, 59))

    # Filter for today
    mobile_df = df[(df['timestamp'] >= pd.Timestamp(start_time)) & (df['timestamp'] <= pd.Timestamp(end_time))].copy()

    # Format columns for display
    mobile_df['Time'] = mobile_df['timestamp'].dt.strftime('%H:%M')
    mobile_df['Price'] = mobile_df['close'].fillna(0).round(1)
    mobile_df['BI'] = mobile_df['buy_initiated'].fillna(0).astype(int)
    mobile_df['SI'] = mobile_df['sell_initiated'].fillna(0).astype(int)
    mobile_df['TΔ'] = mobile_df['tick_delta'].fillna(0).astype(int)
    mobile_df['CumΔ'] = mobile_df['cumulative_tick_delta'].fillna(0).astype(int)

    display_df = mobile_df[['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']]

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

    # --- Build Table ---
    html_table = '<table class="mobile-table" style="width:100%; border-collapse: collapse;">'

    # Clean header (no tooltips, single line, no wrap)
    headers = ['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']
    html_table += '<thead><tr>'
    for h in headers:
        html_table += f'<th>{h}</th>'
    html_table += '</tr></thead><tbody>'

    # Rows
    for _, row in display_df.iterrows():
        html_table += '<tr>'
        for col in display_df.columns:
            if col in ['TΔ', 'CumΔ']:
                html_table += f'<td>{apply_color_coding(row[col], col)}</td>'
            else:
                html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'

    html_table += '</tbody></table>'

    # Caption for abbreviations
    st.markdown(html_table, unsafe_allow_html=True)
    st.caption("BI=Buy Initiated, SI=Sell Initiated, TΔ=Tick Delta, CumΔ=Cumulative Tick Delta")



# --- MAIN DISPLAY ---
if mobile_view:
    inject_mobile_css()
    stock_name = selected_option.split(' (')[0]
    st.markdown(f"# 📊 {stock_name}")
    st.caption(f"🔄 Updates every {refresh_interval}s • {interval}min intervals")
    if not agg_df.empty:
        create_mobile_metrics(agg_df)
        st.markdown("---")
        st.markdown("### 📋 Recent Activity")
        create_mobile_table(agg_df)
        st.markdown("---")
        st.markdown("### 📈 Charts")
        chart_html = create_tradingview_chart_with_delta_boxes(stock_name, agg_df, interval)
        components.html(chart_html, height=650, width=0)        st.markdown("---")
        csv = agg_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Data", csv, f"orderflow_{stock_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv", use_container_width=True)

    else:
        st.error("📵 No data available for this security")
else:
    st.title(f"Order Flow Dashboard: {selected_option}")
    if not agg_df.empty:
        st.caption("Full history + live updates")
        agg_df_formatted = agg_df.copy()
        agg_df_formatted['close'] = agg_df_formatted['close'].round(1)
        for col in ['buy_volume', 'sell_volume', 'buy_initiated', 'sell_initiated', 'delta', 'cumulative_delta', 'tick_delta', 'cumulative_tick_delta']:
            agg_df_formatted[col] = agg_df_formatted[col].round(0).astype(int)
        columns_to_show = ['timestamp', 'close', 'buy_initiated', 'sell_initiated', 'tick_delta', 'cumulative_tick_delta', 'inference']
        column_abbreviations = {'timestamp': 'Time', 'close': 'Close', 'buy_initiated': 'Buy Initiated', 'sell_initiated': 'Sell Initiated', 'tick_delta': 'Tick Delta', 'cumulative_tick_delta': 'Cumulative Tick Delta', 'inference': 'Inference'}
        agg_df_table = agg_df_formatted[columns_to_show].rename(columns=column_abbreviations)
        styled_table = agg_df_table.style.background_gradient(cmap="RdYlGn", subset=['Tick Delta', 'Cumulative Tick Delta'])
        st.dataframe(styled_table, use_container_width=True, height=600)
        st.subheader("Candlestick Chart")
        chart_html = create_tradingview_chart(selected_option, agg_df, interval)
        components.html(chart_html, height=650, width=0)        csv = agg_df_table.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
    else:
        st.warning("No data available for this security.")

