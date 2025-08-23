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
                return live_data  # Return all data for alert system
        
        # Fallback to local cache if API fails
        cache_df = load_from_local_cache(security_id)
        if not cache_df.empty:
            return cache_df  # Return all cached data
            
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

# --- Enhanced Visual Indicators CSS ---
def inject_enhanced_css():
    st.markdown("""
    <style>
    /* Enhanced Visual Indicators */
    .smart-summary-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 16px;
        color: white;
        margin: 8px 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-left: 4px solid #4CAF50;
    }
    
    .smart-summary-card.bearish {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        border-left-color: #f44336;
    }
    
    .smart-summary-card.neutral {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        border-left-color: #9ca3af;
    }
    
    .summary-metric {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 8px 0;
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
    }
    
    .summary-metric:last-child {
        border-bottom: none;
    }
    
    .summary-label {
        font-size: 16px;
        font-weight: 600;
        opacity: 1.0;
        color: #ffffff !important; /* force white */
    }

    
    .summary-value {
        font-size: 22px;          /* larger numbers */
        font-weight: 800;         /* extra bold */
        color: #ffffff !important; /* force white */
        text-shadow: 1px 1px 3px rgba(0,0,0,0.6); /* slight glow for visibility */
    }

    
    .summary-value.positive { color: #4CAF50; }
    .summary-value.negative { color: #f44336; }
    .summary-value.neutral { color: #FFC107; }
    
    /* Enhanced Chart Legend */
    .enhanced-legend {
        background: rgba(255,255,255,0.95);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0;
        font-size: 12px;
    }
    
    .legend-line {
        width: 20px;
        height: 2px;
        border-radius: 1px;
    }
    
    .legend-line.strong-resistance { background: #d32f2f; }
    .legend-line.medium-resistance { background: #ff8a80; }
    .legend-line.strong-support { background: #00796b; }
    .legend-line.medium-support { background: #80cbc4; }
    .legend-line.pivot { background: #ffa726; border-top: 1px dotted #ffa726; }
    
    /* Enhanced Mobile Responsiveness */
    @media (max-width: 768px) {
        .css-1d391kg {padding: 0.5rem !important;}
        .main .block-container {padding-top: 1rem !important; padding-left: 1rem !important; padding-right: 1rem !important; max-width: 100% !important;}
        .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 12px; color: white; text-align: center; margin: 4px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
        .metric-value {font-size: 18px; font-weight: bold; margin: 0;}
        .metric-label {font-size: 11px; opacity: 0.9; margin: 0;}
        
        .smart-summary-card {
            padding: 12px;
            margin: 6px 0;
        }
        
        .summary-metric {
            flex-direction: column;
            align-items: flex-start;
            gap: 4px;
        }
        
        .summary-value {
            font-size: 14px;
        }
    }
    
    # Add to your existing CSS:
    .trading-signal-high { 
        border-left: 4px solid #ef4444 !important; 
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%); 
    }
    .trading-signal-medium { 
        border-left: 4px solid #f59e0b !important; 
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%); 
    }
    .premium-sr-line {
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    </style>
    """, unsafe_allow_html=True)

# --- Keep your mobile CSS ---
def inject_mobile_css():
    pass
# --- Smart Data Summary Panel ---
def create_smart_data_summary(df, sr_levels):
    """Create enhanced data summary with key insights - handles missing columns gracefully"""
    if df.empty:
        return {}
    
    summary = {}
    
    # Basic stats
    summary['total_records'] = len(df)
    summary['date_range'] = f"{df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')}"
    
    # Price analysis
    if 'close' in df.columns:
        latest_price = df['close'].iloc[-1]
        price_change = df['close'].iloc[-1] - df['close'].iloc[0]
        price_change_pct = (price_change / df['close'].iloc[0]) * 100 if df['close'].iloc[0] != 0 else 0
        
        summary['current_price'] = latest_price
        summary['price_change'] = price_change
        summary['price_change_pct'] = price_change_pct
        summary['price_trend'] = 'Bullish' if price_change > 0 else 'Bearish' if price_change < 0 else 'Neutral'
    else:
        summary['current_price'] = 0
        summary['price_change'] = 0
        summary['price_change_pct'] = 0
        summary['price_trend'] = 'Neutral'
    
    # Volume analysis
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        total_volume = df['buy_volume'].sum() + df['sell_volume'].sum()
        avg_volume = total_volume / len(df) if len(df) > 0 else 0
        summary['total_volume'] = total_volume
        summary['avg_volume'] = avg_volume
        summary['volume_trend'] = 'High' if avg_volume > 1000 else 'Medium' if avg_volume > 500 else 'Low'
    else:
        summary['total_volume'] = 0
        summary['avg_volume'] = 0
        summary['volume_trend'] = 'Low'
    
    # Delta analysis - handle both raw and aggregated data
    if 'tick_delta' in df.columns:
        latest_delta = df['tick_delta'].iloc[-1]
        summary['latest_delta'] = latest_delta
    else:
        summary['latest_delta'] = 0
    
    if 'cumulative_tick_delta' in df.columns:
        cumulative_delta = df['cumulative_tick_delta'].iloc[-1]
        summary['cumulative_delta'] = cumulative_delta
        summary['delta_sentiment'] = 'Bullish' if cumulative_delta > 0 else 'Bearish' if cumulative_delta < 0 else 'Neutral'
    elif 'tick_delta' in df.columns:
        # Calculate cumulative delta if not present
        cumulative_delta = df['tick_delta'].cumsum().iloc[-1]
        summary['cumulative_delta'] = cumulative_delta
        summary['delta_sentiment'] = 'Bullish' if cumulative_delta > 0 else 'Bearish' if cumulative_delta < 0 else 'Neutral'
    else:
        summary['cumulative_delta'] = 0
        summary['delta_sentiment'] = 'Neutral'
    
    # Support/Resistance analysis
    if sr_levels:
        strong_levels = [level for level in sr_levels if level.get('strength') == 'high']
        summary['strong_levels'] = len(strong_levels)
        summary['total_levels'] = len(sr_levels)
        summary['level_strength'] = f"{len(strong_levels)}/{len(sr_levels)} strong"
    else:
        summary['strong_levels'] = 0
        summary['total_levels'] = 0
        summary['level_strength'] = "No levels"
    
    # Market session analysis
    if 'buy_volume' in df.columns and 'sell_volume' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        morning_volume = df[df['hour'].between(9, 11)]['buy_volume'].sum() + df[df['hour'].between(9, 11)]['sell_volume'].sum()
        afternoon_volume = df[df['hour'].between(14, 16)]['buy_volume'].sum() + df[df['hour'].between(14, 16)]['sell_volume'].sum()
        summary['session_activity'] = 'Morning' if morning_volume > afternoon_volume else 'Afternoon'
    else:
        summary['session_activity'] = 'Unknown'
    
    return summary

# --- Support and Resistance Calculation Functions ---
def calculate_support_resistance_levels_enhanced(df, lookback_periods=20, min_touches=2, strength_threshold=0.002):
    """
    Enhanced support and resistance calculation with high conviction levels
    """
    if df.empty or len(df) < lookback_periods:
        return []
    
    levels = []
    df = df.copy()
    
    # Calculate pivot points with enhanced accuracy
    df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
    df['r1'] = 2 * df['pivot'] - df['low']
    df['s1'] = 2 * df['pivot'] - df['high']
    df['r2'] = df['pivot'] + (df['high'] - df['low'])
    df['s2'] = df['pivot'] - (df['high'] - df['low'])
    df['r3'] = df['high'] + 2 * (df['pivot'] - df['low'])
    df['s3'] = df['low'] - 2 * (df['high'] - df['pivot'])
    
    # Enhanced volume calculation
    def calculate_volume_at_level(price_level, tolerance_pct=0.2):
        tolerance = price_level * (tolerance_pct / 100)
        volume = 0
        touches = 0
        for _, row in df.iterrows():
            price_range = [row['low'], row['high'], row['close']]
            if any(abs(price - price_level) <= tolerance for price in price_range):
                volume += row.get('buy_volume', 0) + row.get('sell_volume', 0)
                touches += 1
        return volume, touches
    
    # Find swing highs and lows with enhanced criteria
    swing_levels = []
    
    for i in range(lookback_periods, len(df) - lookback_periods):
        current_high = df['high'].iloc[i]
        current_low = df['low'].iloc[i]
        
        # Enhanced swing high detection
        left_highs = df['high'].iloc[i-lookback_periods:i].max()
        right_highs = df['high'].iloc[i+1:i+lookback_periods+1].max()
        
        if current_high >= left_highs and current_high >= right_highs:
            volume, touches = calculate_volume_at_level(current_high)
            if touches >= min_touches:
                swing_levels.append({
                    'price': current_high,
                    'time': int(pd.to_datetime(df['timestamp'].iloc[i]).timestamp()),
                    'type': 'Resistance',
                    'volume': volume,
                    'touches': touches,
                    'strength_score': touches * (volume / 1000) if volume > 0 else touches
                })
        
        # Enhanced swing low detection
        left_lows = df['low'].iloc[i-lookback_periods:i].min()
        right_lows = df['low'].iloc[i+1:i+lookback_periods+1].min()
        
        if current_low <= left_lows and current_low <= right_lows:
            volume, touches = calculate_volume_at_level(current_low)
            if touches >= min_touches:
                swing_levels.append({
                    'price': current_low,
                    'time': int(pd.to_datetime(df['timestamp'].iloc[i]).timestamp()),
                    'type': 'Support',
                    'volume': volume,
                    'touches': touches,
                    'strength_score': touches * (volume / 1000) if volume > 0 else touches
                })
    
    # Add current pivot levels with enhanced analysis
    if not df.empty:
        latest_data = df.tail(1)
        pivot_levels = {
            'PP': latest_data['pivot'].iloc[0],
            'R1': latest_data['r1'].iloc[0],
            'R2': latest_data['r2'].iloc[0],
            'R3': latest_data['r3'].iloc[0],
            'S1': latest_data['s1'].iloc[0],
            'S2': latest_data['s2'].iloc[0],
            'S3': latest_data['s3'].iloc[0]
        }
        
        for level_name, price in pivot_levels.items():
            volume, touches = calculate_volume_at_level(price)
            levels.append({
                'price': price,
                'time': int(pd.to_datetime(latest_data['timestamp'].iloc[0]).timestamp()),
                'type': level_name,
                'volume': volume,
                'touches': touches,
                'strength_score': touches * (volume / 1000) if volume > 0 else 1,
                'is_pivot': True
            })
    
    # Add swing levels to main levels
    levels.extend(swing_levels)
    
    # Calculate strength categories based on multiple factors
    if levels:
        scores = [level['strength_score'] for level in levels]
        high_threshold = sorted(scores, reverse=True)[min(len(scores)//3, len(scores)-1)] if len(scores) > 3 else max(scores)
        medium_threshold = sorted(scores, reverse=True)[min(2*len(scores)//3, len(scores)-1)] if len(scores) > 2 else max(scores)/2
        
        for level in levels:
            score = level['strength_score']
            touches = level['touches']
            volume = level['volume']
            
            # High conviction criteria
            if (score >= high_threshold and touches >= 3 and volume > 5000) or touches >= 5:
                level['strength'] = 'very_high'
                level['conviction'] = 'High Conviction'
            elif score >= high_threshold or (touches >= 2 and volume > 2000):
                level['strength'] = 'high'
                level['conviction'] = 'Strong'
            elif score >= medium_threshold or touches >= 2:
                level['strength'] = 'medium'
                level['conviction'] = 'Medium'
            else:
                level['strength'] = 'low'
                level['conviction'] = 'Weak'
    
    return levels

def create_support_resistance_series_enhanced(levels, chart_data, show_sr=True, show_only_high_conviction=False):
    """
    Create enhanced TradingView series with conviction-based filtering
    """
    if not levels or not show_sr:
        return []
    
    series = []
    
    # Filter levels based on conviction if requested
    if show_only_high_conviction:
        levels = [level for level in levels if level.get('strength') in ['very_high', 'high']]
    
    if chart_data.empty:
        return []
    
    start_time = int(pd.to_datetime(chart_data['timestamp'].min()).timestamp())
    end_time = int(pd.to_datetime(chart_data['timestamp'].max()).timestamp())
    
    # Enhanced color scheme based on conviction
    color_scheme = {
        'very_high': {
            'resistance': '#b71c1c',  # Deep red
            'support': '#1b5e20',     # Deep green
            'pivot': '#e65100',       # Deep orange
            'line_width': 4
        },
        'high': {
            'resistance': '#d32f2f',  # Red
            'support': '#2e7d32',     # Green
            'pivot': '#f57c00',       # Orange
            'line_width': 3
        },
        'medium': {
            'resistance': '#f44336',  # Light red
            'support': '#4caf50',     # Light green
            'pivot': '#ff9800',       # Light orange
            'line_width': 2
        },
        'low': {
            'resistance': '#ef5350',  # Very light red
            'support': '#66bb6a',     # Very light green
            'pivot': '#ffb74d',       # Very light orange
            'line_width': 1
        }
    }
    
    for level in levels:
        price = level['price']
        level_type = level.get('type', 'Level')
        strength = level.get('strength', 'low')
        conviction = level.get('conviction', 'Weak')
        volume = level.get('volume', 0)
        touches = level.get('touches', 0)
        is_pivot = level.get('is_pivot', False)
        
        # Determine color and style based on type and strength
        if any(x in level_type.upper() for x in ['R1', 'R2', 'R3', 'RESISTANCE']):
            color = color_scheme[strength]['resistance']
            line_style = 'resistance'
        elif any(x in level_type.upper() for x in ['S1', 'S2', 'S3', 'SUPPORT']):
            color = color_scheme[strength]['support']
            line_style = 'support'
        else:  # Pivot
            color = color_scheme[strength]['pivot']
            line_style = 'pivot'
        
        line_width = color_scheme[strength]['line_width']
        
        # Enhanced line style based on strength
        if strength == 'very_high':
            linestyle = 0  # Solid
        elif strength == 'high':
            linestyle = 0  # Solid
        elif strength == 'medium':
            linestyle = 2  # Dashed
        else:
            linestyle = 1  # Dotted
        
        # Create series name with conviction info
        series_name = f"{level_type} {price:.2f} [{conviction}] (T:{touches}, V:{volume:,.0f})"
        
        line_series = {
            'name': series_name,
            'type': 'line',
            'data': [
                {'time': start_time, 'value': price},
                {'time': end_time, 'value': price}
            ],
            'color': color,
            'linewidth': line_width,
            'linestyle': linestyle,
            'priceLineVisible': False,
            'priceFormat': {'type': 'price', 'precision': 2},
            'volume': volume,
            'touches': touches,
            'strength': strength,
            'conviction': conviction,
            'level_type': level_type
        }
        series.append(line_series)
    
    return series

# --- TradingView chart function ---
def create_tradingview_chart_with_delta_boxes(stock_name, chart_data, interval):
    """Enhanced chart with perfectly aligned tick delta and cumulative delta boxes"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Use pre-calculated support and resistance levels
    sr_series = create_support_resistance_series(sr_levels, chart_data)
    
    # Prepare all data series
    candle_data = []
    tick_delta_values = []
    cumulative_delta_values = []
    
    # Format number function for K/M display
    def format_number(num):
        if abs(num) >= 1000000:
            return f"{num/1000000:.1f}M".replace('.0M', 'M')
        elif abs(num) >= 1000:
            return f"{num/1000:.1f}K".replace('.0K', 'K')
        else:
            return str(int(num))
    
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
                'formatted': f"+{format_number(tick_delta)}" if tick_delta > 0 else format_number(tick_delta)
            })
            
            cumulative_delta_values.append({
                'timestamp': timestamp,
                'value': cum_delta,
                'formatted': f"+{format_number(cum_delta)}" if cum_delta > 0 else format_number(cum_delta)
            })
            
        except:
            continue
    
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    
    chart_html = f"""
<div class="chart-with-delta-container" style="width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 8px;">
    <!-- Main Chart -->
    <div id="{chart_id}" style="width: 100%; height: 500px;"></div>
    
    <!-- Delta Boxes Container -->
    <div id="{chart_id}_delta_container" style="padding: 10px; background: #f8fafc; border-top: 1px solid #e5e7eb;">
        <!-- Tick Delta Row -->
        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Tick Delta
            </div>
            <div class="delta-row" id="tick-delta-row" style="position: relative; height: 32px; overflow: visible;">
                <!-- Tick delta boxes will be inserted here -->
            </div>
        </div>
        
        <!-- Cumulative Delta Row -->
        <div>
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Cumulative Delta
            </div>
            <div class="delta-row" id="cumulative-delta-row" style="position: relative; height: 32px; overflow: visible;">
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
    height: 26px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 600;
    border-radius: 6px;
    color: white;
    text-shadow: 0 1px 2px rgba(0,0,0,0.4);
    white-space: nowrap;
    cursor: default;
    transition: all 0.2s ease;
    position: relative;
    transform: translateZ(0);
}}

.delta-box:hover {{
    transform: translateY(-1px) translateZ(0);
    box-shadow: 0 3px 6px rgba(0,0,0,0.25);
    z-index: 10;
}}

.delta-positive {{
    background: linear-gradient(135deg, #26a69a 0%, #1e8c82 100%);
    border: 1px solid #1e8c82;
}}

.delta-negative {{
    background: linear-gradient(135deg, #ef5350 0%, #d84343 100%);
    border: 1px solid #c62828;
}}

.delta-zero {{
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
    border: 1px solid #374151;
}}

.delta-alignment-line {{
    position: absolute;
    top: -5px;
    bottom: -5px;
    width: 1px;
    background: rgba(155, 125, 255, 0.3);
    pointer-events: none;
    transition: opacity 0.2s ease;
}}
</style>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    const container = document.getElementById('{chart_id}');
    const deltaContainer = document.getElementById('{chart_id}_delta_container');
    
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">Chart library not loaded</div>';
        return;
    }}
    
    container.innerHTML = '';
    
    let chart;
    let candleSeries;
    let deltaBoxes = {{}};
    let alignmentLines = [];
    
    // Chart data
    const candleData = {json.dumps(candle_data)};
    const tickDeltaData = {json.dumps(tick_delta_values)};
    const cumulativeDeltaData = {json.dumps(cumulative_delta_values)};
    const srSeriesData = {json.dumps(sr_series)};
    
    // Initialize chart
    function initChart() {{
        chart = LightweightCharts.createChart(container, {{
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
                secondsVisible: false,
                rightOffset: 5,
                barSpacing: 8,
                minBarSpacing: 4
            }},
            autoSize: false
        }});
        
        // Add candlestick series
        candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        }});
        
        candleSeries.setData(candleData);
        
        // Add support and resistance lines
        srSeriesData.forEach(srSeries => {{
            const lineSeries = chart.addLineSeries({{
                color: srSeries.color,
                lineWidth: srSeries.linewidth,
                lineStyle: srSeries.linestyle,
                priceLineVisible: srSeries.priceLineVisible,
                title: srSeries.name
            }});
            lineSeries.setData(srSeries.data);
        }});
        
        chart.timeScale().fitContent();
        
        // Create delta boxes with alignment
        createAlignedDeltaBoxes();
        
        // Subscribe to chart events for alignment updates
        chart.timeScale().subscribeVisibleTimeRangeChange(updateDeltaBoxAlignment);
    }}
    
    function createAlignedDeltaBoxes() {{
        createDeltaBoxes(tickDeltaData, 'tick-delta-row', 'tick');
        createDeltaBoxes(cumulativeDeltaData, 'cumulative-delta-row', 'cumulative');
        updateDeltaBoxAlignment();
    }}
    
    function createDeltaBoxes(data, containerId, type) {{
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        deltaBoxes[type] = [];
        
        data.forEach((item, index) => {{
            const box = document.createElement('div');
            box.className = 'delta-box';
            box.dataset.timestamp = item.timestamp;
            box.dataset.type = type;
            box.dataset.index = index;
            
            // Determine color class based on value
            if (item.value > 0) {{
                box.classList.add('delta-positive');
            }} else if (item.value < 0) {{
                box.classList.add('delta-negative');
            }} else {{
                box.classList.add('delta-zero');
            }}
            
            // Set text content with K/M formatting
            box.textContent = item.formatted;
            
            // Add tooltip with full value and time
            const date = new Date(item.timestamp * 1000);
            const fullValue = item.value.toLocaleString();
            box.title = `Time: ${{date.toLocaleTimeString()}}\\nValue: ${{fullValue >= 0 ? '+' : ''}}${{fullValue}}`;
            
            // Add hover effect for alignment line
            box.addEventListener('mouseenter', () => showAlignmentLine(item.timestamp));
            box.addEventListener('mouseleave', () => hideAlignmentLines());
            
            container.appendChild(box);
            deltaBoxes[type].push(box);
        }});
    }}
    
    function updateDeltaBoxAlignment() {{
        if (!chart || !candleSeries) return;
        
        const timeScale = chart.timeScale();
        const visibleRange = timeScale.getVisibleRange();
        
        if (!visibleRange) return;
        
        // Get chart dimensions
        const chartRect = container.getBoundingClientRect();
        const chartWidth = chartRect.width;
        
        // Update both delta box types
        ['tick', 'cumulative'].forEach(type => {{
            if (!deltaBoxes[type]) return;
            
            deltaBoxes[type].forEach((box, index) => {{
                const timestamp = parseInt(box.dataset.timestamp);
                
                // Calculate position based on timestamp
                const logicalPosition = timeScale.timeToCoordinate(timestamp);
                
                if (logicalPosition !== null) {{
                    // Calculate box width based on visible time range and available space
                    const visibleTimeSpan = visibleRange.to - visibleRange.from;
                    const pixelsPerSecond = chartWidth / visibleTimeSpan;
                    const barSpacing = Math.max(4, Math.min(12, pixelsPerSecond * 60)); // Assuming 1-minute bars
                    
                    const boxWidth = Math.max(40, Math.min(80, barSpacing - 2));
                    
                    box.style.width = boxWidth + 'px';
                    box.style.minWidth = boxWidth + 'px';
                    box.style.position = 'absolute';
                    box.style.left = (logicalPosition - boxWidth/2) + 'px';
                    box.style.opacity = '1';
                    box.style.display = 'flex';
                    box.style.alignItems = 'center';
                    box.style.justifyContent = 'center';
                    
                    // Adjust font size and ensure visibility
                    const fontSize = boxWidth < 50 ? '10px' : '11px';
                    box.style.fontSize = fontSize;
                    box.style.color = 'white';
                    box.style.textShadow = '1px 1px 2px rgba(0,0,0,0.9)';
                }} else {{
                    box.style.opacity = '0.3';
                }}
            }});
        }});
    }}
    
    function showAlignmentLine(timestamp) {{
        hideAlignmentLines();
        
        const logicalPosition = chart.timeScale().timeToCoordinate(timestamp);
        if (logicalPosition === null) return;
        
        // Create alignment line for both delta rows
        ['tick-delta-row', 'cumulative-delta-row'].forEach(rowId => {{
            const row = document.getElementById(rowId);
            if (!row) return;
            
            const line = document.createElement('div');
            line.className = 'delta-alignment-line';
            line.style.left = logicalPosition + 'px';
            row.appendChild(line);
            alignmentLines.push(line);
        }});
    }}
    
    function hideAlignmentLines() {{
        alignmentLines.forEach(line => line.remove());
        alignmentLines = [];
    }}
    
    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({{ 
            width: rect.width, 
            height: 500
        }});
        // Delay alignment update to ensure chart has resized
        setTimeout(updateDeltaBoxAlignment, 100);
    }});
    
    // Initialize everything
    initChart();
    resizeObserver.observe(container);
    
    // Cleanup
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        if (chart) chart.remove();
    }});
    
    // Update alignment periodically to handle any drift
    setInterval(updateDeltaBoxAlignment, 1000);
}})();
</script>
    """
    return chart_html

# STEP 1: Add these functions RIGHT AFTER your existing create_tradingview_chart_with_delta_boxes function
# (Around line 700 in your code, after the "return chart_html" line)

def save_chart_state(chart_id, visible_range, zoom_level):
    """Save chart view state to local storage equivalent"""
    state_file = os.path.join(LOCAL_CACHE_DIR, f"chart_state_{chart_id}.json")
    chart_state = {
        'visible_range': visible_range,
        'zoom_level': zoom_level,
        'timestamp': datetime.now().isoformat()
    }
    try:
        with open(state_file, 'w') as f:
            json.dump(chart_state, f)
    except Exception as e:
        logging.warning(f"Failed to save chart state: {e}")

def load_chart_state(chart_id):
    """Load chart view state"""
    state_file = os.path.join(LOCAL_CACHE_DIR, f"chart_state_{chart_id}.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load chart state: {e}")
    return None

def create_tradingview_chart_with_delta_boxes_persistent_enhanced(stock_name, chart_data, interval, chart_options):
    """Enhanced chart with view state persistence across refreshes"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Use pre-calculated support and resistance levels
    # Calculate enhanced support and resistance levels
    sr_levels_enhanced = calculate_support_resistance_levels_enhanced(chart_data)

    # Create series based on user preferences  
    sr_series = create_support_resistance_series_enhanced(
        sr_levels_enhanced, 
        chart_data, 
        show_sr=chart_options['show_sr_lines'],
        show_only_high_conviction=chart_options['high_conviction_only']
    )
    
    # Prepare all data series (same as your existing function)
    candle_data = []
    tick_delta_values = []
    cumulative_delta_values = []
    
    def format_number(num):
        if abs(num) >= 1000000:
            return f"{num/1000000:.1f}M".replace('.0M', 'M')
        elif abs(num) >= 1000:
            return f"{num/1000:.1f}K".replace('.0K', 'K')
        else:
            return str(int(num))
    
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
            
            tick_delta = float(row.get('tick_delta', 0))
            cum_delta = float(row.get('cumulative_tick_delta', 0))
            
            tick_delta_values.append({
                'timestamp': timestamp,
                'value': tick_delta,
                'formatted': f"+{format_number(tick_delta)}" if tick_delta > 0 else format_number(tick_delta)
            })
            
            cumulative_delta_values.append({
                'timestamp': timestamp,
                'value': cum_delta,
                'formatted': f"+{format_number(cum_delta)}" if cum_delta > 0 else format_number(cum_delta)
            })
            
        except:
            continue
    
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    
    # Load previous chart state
    saved_state = load_chart_state(chart_id)
    # Get chart height from options
    chart_height = chart_options.get('chart_height', 600)
    
    chart_html = f"""
<div class="chart-with-delta-container" style="width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 8px;">
    <!-- Main Chart -->
    <div id="{chart_id}" style="width: 100%; height: {chart_height}px;"></div>
    
    <!-- Delta Boxes Container -->
    <div id="{chart_id}_delta_container" style="padding: 10px; background: #f8fafc; border-top: 1px solid #e5e7eb;">
        <!-- Tick Delta Row -->
        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Tick Delta
            </div>
            <div class="delta-row" id="tick-delta-row" style="position: relative; height: 32px; overflow: visible;">
                <!-- Tick delta boxes will be inserted here -->
            </div>
        </div>
        
        <!-- Cumulative Delta Row -->
        <div>
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Cumulative Delta
            </div>
            <div class="delta-row" id="cumulative-delta-row" style="position: relative; height: 32px; overflow: visible;">
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
    transition: all 0.2s ease;
    position: relative;
}}
.delta-box:hover {{
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    z-index: 10;
}}
.delta-positive {{
    background: linear-gradient(135deg, #26a69a 0%, #1e8c82 100%);
    border: 1px solid #1e8c82;
}}
.delta-negative {{
    background: linear-gradient(135deg, #ef5350 0%, #d84343 100%);
    border: 1px solid #d84343;
}}

.delta-zero {{
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
    border: 1px solid #374151;
}}
.delta-alignment-line {{
    position: absolute;
    top: -5px;
    bottom: -5px;
    width: 1px;
    background: rgba(155, 125, 255, 0.3);
    pointer-events: none;
    transition: opacity 0.2s ease;
}}
</style>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    const container = document.getElementById('{chart_id}');
    const deltaContainer = document.getElementById('{chart_id}_delta_container');
    
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">Chart library not loaded</div>';
        return;
    }}
    
    container.innerHTML = '';
    
    let chart;
    let candleSeries;
    let deltaBoxes = {{}};
    let alignmentLines = [];
    let isUpdating = false;
    
    // Chart data
    const candleData = {json.dumps(candle_data)};
    const tickDeltaData = {json.dumps(tick_delta_values)};
    const cumulativeDeltaData = {json.dumps(cumulative_delta_values)};
    const srSeriesData = {json.dumps(sr_series)};
    
    // Saved state from server
    const savedState = {json.dumps(saved_state) if saved_state else 'null'};
    
    // Chart state management
    let chartState = {{
        visibleRange: null,
        isFirstLoad: savedState ? false : true
    }};
    
    // Initialize chart
    function initChart() {{
        chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: {chart_height},  // <-- UPDATED LINE
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
                secondsVisible: false,
                rightOffset: 5,
                barSpacing: 8,
                minBarSpacing: 4
            }},
            autoSize: false
        }});
        
        // Add candlestick series
        candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        }});
        
        candleSeries.setData(candleData);
        
        // Add support and resistance lines
        srSeriesData.forEach(srSeries => {{
            const lineSeries = chart.addLineSeries({{
                color: srSeries.color,
                lineWidth: srSeries.linewidth,
                lineStyle: srSeries.linestyle,
                priceLineVisible: srSeries.priceLineVisible,
                title: srSeries.name
            }});
            lineSeries.setData(srSeries.data);
        }});
        
        // Restore previous view state or fit content for first load
        if (savedState && savedState.visible_range) {{
            try {{
                setTimeout(() => {{
                    chart.timeScale().setVisibleRange(savedState.visible_range);
                }}, 100);
            }} catch (e) {{
                console.warn('Failed to restore visible range:', e);
                chart.timeScale().fitContent();
            }}
        }} else {{
            chart.timeScale().fitContent();
        }}
        
        // Create delta boxes with alignment
        createAlignedDeltaBoxes();
        
        // Subscribe to chart events for state persistence
        chart.timeScale().subscribeVisibleTimeRangeChange((newVisibleRange) => {{
            if (!isUpdating && newVisibleRange) {{
                chartState.visibleRange = newVisibleRange;
                // Save to sessionStorage immediately
                try {{
                    sessionStorage.setItem('chart_state_{chart_id}', JSON.stringify({{
                        visible_range: newVisibleRange,
                        timestamp: new Date().toISOString()
                    }}));
                }} catch (e) {{}}
            }}
            updateDeltaBoxAlignment();
        }});
    }}
    
    function createAlignedDeltaBoxes() {{
        createDeltaBoxes(tickDeltaData, 'tick-delta-row', 'tick');
        createDeltaBoxes(cumulativeDeltaData, 'cumulative-delta-row', 'cumulative');
        updateDeltaBoxAlignment();
    }}
    
    function createDeltaBoxes(data, containerId, type) {{
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        deltaBoxes[type] = [];
        
        data.forEach((item, index) => {{
            const box = document.createElement('div');
            box.className = 'delta-box';
            box.dataset.timestamp = item.timestamp;
            box.dataset.type = type;
            box.dataset.index = index;
            
            // Determine color class based on value
            if (item.value > 0) {{
                box.classList.add('delta-positive');
            }} else if (item.value < 0) {{
                box.classList.add('delta-negative');
            }} else {{
                box.classList.add('delta-zero');
            }}
            
            // Set text content with K/M formatting
            box.textContent = item.formatted;
            
            // Add tooltip with full value and time
            const date = new Date(item.timestamp * 1000);
            const fullValue = item.value.toLocaleString();
            box.title = `Time: ${{date.toLocaleTimeString()}}\\nValue: ${{fullValue >= 0 ? '+' : ''}}${{fullValue}}`;
            
            // Add hover effect for alignment line
            box.addEventListener('mouseenter', () => showAlignmentLine(item.timestamp));
            box.addEventListener('mouseleave', () => hideAlignmentLines());
            
            container.appendChild(box);
            deltaBoxes[type].push(box);
        }});
    }}
    
    function updateDeltaBoxAlignment() {{
        if (!chart || !candleSeries) return;
        
        const timeScale = chart.timeScale();
        const visibleRange = timeScale.getVisibleRange();
        
        if (!visibleRange) return;
        
        // Get chart dimensions
        const chartRect = container.getBoundingClientRect();
        const chartWidth = chartRect.width;
        
        // Update both delta box types
        ['tick', 'cumulative'].forEach(type => {{
            if (!deltaBoxes[type]) return;
            
            deltaBoxes[type].forEach((box, index) => {{
                const timestamp = parseInt(box.dataset.timestamp);
                
                // Calculate position based on timestamp
                const logicalPosition = timeScale.timeToCoordinate(timestamp);
                
                if (logicalPosition !== null) {{
                    // Calculate box width based on visible time range and available space
                    const visibleTimeSpan = visibleRange.to - visibleRange.from;
                    const pixelsPerSecond = chartWidth / visibleTimeSpan;
                    const barSpacing = Math.max(4, Math.min(12, pixelsPerSecond * 60)); // Assuming 1-minute bars
                    
                    const boxWidth = Math.max(40, Math.min(80, barSpacing - 2));
                    
                    box.style.width = boxWidth + 'px';
                    box.style.minWidth = boxWidth + 'px';
                    box.style.position = 'absolute';
                    box.style.left = (logicalPosition - boxWidth/2) + 'px';
                    box.style.opacity = '1';
                    box.style.display = 'flex';
                    box.style.alignItems = 'center';
                    box.style.justifyContent = 'center';
                    
                    // Adjust font size and ensure visibility
                    const fontSize = boxWidth < 50 ? '10px' : '11px';
                    box.style.fontSize = fontSize;
                    box.style.color = 'white';
                    box.style.textShadow = '1px 1px 2px rgba(0,0,0,0.9)';
                }} else {{
                    box.style.opacity = '0.3';
                }}
            }});
        }});
    }}
    
    function showAlignmentLine(timestamp) {{
        hideAlignmentLines();
        
        const logicalPosition = chart.timeScale().timeToCoordinate(timestamp);
        if (logicalPosition === null) return;
        
        // Create alignment line for both delta rows
        ['tick-delta-row', 'cumulative-delta-row'].forEach(rowId => {{
            const row = document.getElementById(rowId);
            if (!row) return;
            
            const line = document.createElement('div');
            line.className = 'delta-alignment-line';
            line.style.left = logicalPosition + 'px';
            row.appendChild(line);
            alignmentLines.push(line);
        }});
    }}
    
    function hideAlignmentLines() {{
        alignmentLines.forEach(line => line.remove());
        alignmentLines = [];
    }}
    
    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({{ 
            width: rect.width, 
            height: {chart_height}  // <-- UPDATED LINE
        }});
        // Delay alignment update to ensure chart has resized
        setTimeout(updateDeltaBoxAlignment, 100);
    }});
    
    // Initialize everything
    initChart();
    resizeObserver.observe(container);
    
    // Cleanup
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        if (chart) chart.remove();
    }});
    
    // Update alignment periodically to handle any drift
    setInterval(updateDeltaBoxAlignment, 1000);
    
    // Load session storage state as backup if no saved state
    if (!savedState) {{
        try {{
            const sessionState = sessionStorage.getItem('chart_state_{chart_id}');
            if (sessionState) {{
                const parsed = JSON.parse(sessionState);
                if (parsed.visible_range) {{
                    setTimeout(() => {{
                        try {{
                            chart.timeScale().setVisibleRange(parsed.visible_range);
                        }} catch (e) {{}}
                    }}, 200);
                }}
            }}
        }} catch (e) {{}}
    }}
}})();
</script>
    """
    return chart_html

def add_chart_persistence_controls():
    """Add chart persistence controls to sidebar"""
    st.sidebar.markdown("#### 📊 Chart Settings")
    
    # Reset chart view button
    if st.sidebar.button("🔄 Reset Chart View", help="Reset chart to fit all data"):
        # Clear saved chart states
        chart_state_files = [f for f in os.listdir(LOCAL_CACHE_DIR) if f.startswith('chart_state_')]
        for file in chart_state_files:
            try:
                os.remove(os.path.join(LOCAL_CACHE_DIR, file))
            except:
                pass
        st.sidebar.success("✅ Chart view reset!")
        st.rerun()  # new API

    
    return True

def add_enhanced_chart_controls():
    """Add enhanced chart controls to sidebar"""
    st.sidebar.markdown("#### 🎯 Chart Controls")
    
    # S/R Line controls
    show_sr_lines = st.sidebar.toggle("📊 Show S/R Lines", value=True, key="show_sr_lines")
    
    if show_sr_lines:
        show_conviction_filter = st.sidebar.toggle(
            "🔥 High Conviction Only", 
            value=False, 
            key="high_conviction_only",
            help="Show only high conviction support/resistance levels"
        )
        
        st.sidebar.markdown("##### Line Strength Legend:")
        st.sidebar.markdown("""
        <div style="font-size: 11px; line-height: 1.4;">
        🔴 <b>Very High</b>: Deep colors, 5+ touches<br>
        🟠 <b>High</b>: Strong colors, 3+ touches<br>
        🟡 <b>Medium</b>: Normal colors, 2+ touches<br>
        ⚪ <b>Low</b>: Light colors, few touches
        </div>
        """, unsafe_allow_html=True)
    
    # Chart display options
    st.sidebar.markdown("#### 📈 Display Options")
    
    chart_height = st.sidebar.selectbox(
        "Chart Height",
        [500, 600, 700, 800],
        index=1,
        key="chart_height"
    )
    
    show_volume_profile = st.sidebar.toggle(
        "📊 Volume Profile Info", 
        value=True, 
        key="show_volume_profile",
        help="Show volume and touches in S/R line names"
    )
    
    return {
        'show_sr_lines': show_sr_lines,
        'high_conviction_only': show_conviction_filter if show_sr_lines else False,
        'chart_height': chart_height,
        'show_volume_profile': show_volume_profile
    }

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
persist_view = add_chart_persistence_controls()
chart_options = add_enhanced_chart_controls()

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
            
            # Check if timestamp column exists and needs date reconstruction
            if 'timestamp' in df.columns:
                # Check if timestamps are time-only (contain only HH:MM format)
                sample_timestamp = str(df['timestamp'].iloc[0]) if not df.empty else ""
                if ':' in sample_timestamp and len(sample_timestamp.split(' ')[0]) <= 5:
                    # Timestamps are time-only, need to add current date
                    current_date = datetime.now().strftime('%Y-%m-%d')
                    df['timestamp'] = df['timestamp'].apply(lambda x: f"{current_date} {x}:00" if ':' in str(x) else x)
            
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
                    # Extract date from filename (orderflow_20250821_10.csv -> 2025-08-21)
                    filename = file_info['name']
                    date_match = re.search(r'orderflow_(\d{4})(\d{2})(\d{2})_(\d{2})\.csv', filename)
                    
                    if date_match:
                        year, month, day, hour = date_match.groups()
                        file_date = f"{year}-{month}-{day}"
                        
                        df = pd.read_csv(file_info['download_url'], dtype=str)  # Force all columns to string
                        df.columns = df.columns.str.strip()  # Strip spaces from column names
                        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)  # Strip spaces from all values
                        df = df[df['security_id'] == str(security_id)]
                        
                        # Convert time-only timestamps to full datetime by combining file date with time
                        if not df.empty and 'timestamp' in df.columns:
                            # Combine file date with time from data
                            df['timestamp'] = df['timestamp'].apply(lambda x: f"{file_date} {x}:00" if ':' in str(x) else x)
                        
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

# Process data before creating summary
# Create two dataframes: one for all days (graph) and one for latest day only (table)
import datetime

# All days data for graph (no date filtering)
all_days_df = full_df.copy()
agg_df_all_days = aggregate_data(all_days_df, interval) if not all_days_df.empty else pd.DataFrame()

# Latest day data for table (use the most recent date in the data instead of current calendar date)
if not full_df.empty:
    # Get the latest date from the data
    latest_date = full_df['timestamp'].dt.date.max()
    
    # Create time range for the latest day (9:00 AM to 11:59 PM)
    start_time = datetime.datetime.combine(latest_date, datetime.time(9, 0))
    end_time = datetime.datetime.combine(latest_date, datetime.time(23, 59, 59))
    
    # Filter for the latest day
    current_day_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]
    agg_df_current_day = aggregate_data(current_day_df, interval) if not current_day_df.empty else pd.DataFrame()
    
    # Store the latest date for display
    latest_date_str = latest_date.strftime('%Y-%m-%d')
else:
    # If no data, create empty dataframes
    current_day_df = pd.DataFrame()
    agg_df_current_day

# Calculate support and resistance levels for smart summary
sr_levels = []
if not agg_df_all_days.empty:
    sr_levels = calculate_support_resistance_levels_enhanced(agg_df_all_days)

# Create smart data summary using the appropriate dataframe
if not agg_df_current_day.empty:
    # Use current day data for summary if available
    smart_summary = create_smart_data_summary(agg_df_current_day, sr_levels)
elif not agg_df_all_days.empty:
    # Fallback to all days data if current day is empty
    smart_summary = create_smart_data_summary(agg_df_all_days, sr_levels)
else:
    # Create empty summary if no data
    smart_summary = {
        'current_price': 0,
        'price_change': 0,
        'price_change_pct': 0,
        'price_trend': 'Neutral',
        'total_volume': 0,
        'session_activity': 'Unknown',
        'total_records': 0,
        'delta_sentiment': 'Neutral',
        'level_strength': 'No levels'
    }

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
    """Create a highly optimized mobile table for mobile view, with single-row header, smaller font, and color coding."""
    if df.empty:
        return

    # ===== CSS for mobile table =====
    st.markdown("""
    <style>
    /* Table styling */
    .mobile-table { 
        width:100%; 
        border-collapse: collapse; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
    }
    .mobile-table th, .mobile-table td {
        font-size: 11px;        /* smaller font size */
        padding: 3px 6px;       /* tighter padding */
        text-align: left;
        vertical-align: middle;
    }
    .mobile-table thead th {
        font-weight: 600;
        font-size: 11px;
        color: #374151;
        padding-bottom: 6px;
    }
    .mobile-table td { 
        border-bottom: 1px solid #f1f5f9; 
        color: #111827; 
    }

    /* Color coding for delta spans */
    .mobile-table .positive { color: #16a34a; font-weight:700; }
    .mobile-table .negative { color: #dc2626; font-weight:700; }
    .mobile-table .neutral  { color: #6b7280; font-weight:600; }

    /* Right-align numeric columns */
    .mobile-table td.numeric { text-align: right; }
    </style>
    """, unsafe_allow_html=True)
    # ================================

    # Use the dataframe as-is since it's already filtered for current day
    mobile_df = df.copy()

    # Format columns for display
    mobile_df['Time'] = mobile_df['timestamp'].dt.strftime('%H:%M')
    mobile_df['Price'] = mobile_df['close'].fillna(0).round(1)
    mobile_df['BI'] = mobile_df['buy_initiated'].fillna(0).astype(int)
    mobile_df['SI'] = mobile_df['sell_initiated'].fillna(0).astype(int)
    mobile_df['TΔ'] = mobile_df['tick_delta'].fillna(0).astype(int)
    mobile_df['CumΔ'] = mobile_df['cumulative_tick_delta'].fillna(0).astype(int)

    display_df = mobile_df[['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']]

    def apply_color_coding(val, col_name):
        val = int(val) if pd.notna(val) else 0
        if col_name in ['TΔ', 'CumΔ']:
            if val > 0:
                return f'<span class="positive">+{val}</span>'
            elif val < 0:
                return f'<span class="negative">{val}</span>'
            else:
                return f'<span class="neutral">{val}</span>'
        return str(val)

    # --- Build Table ---
    html_table = '<table class="mobile-table">'

    # Header row
    headers = ['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']
    html_table += '<thead><tr>'
    for h in headers:
        html_table += f'<th>{h}</th>'
    html_table += '</tr></thead><tbody>'

    # Data rows
    for _, row in display_df.iterrows():
        html_table += '<tr>'
        for col in display_df.columns:
            if col in ['TΔ', 'CumΔ']:
                html_table += f'<td class="numeric">{apply_color_coding(row[col], col)}</td>'
            elif col in ['Price', 'BI', 'SI']:
                html_table += f'<td class="numeric">{row[col]}</td>'
            else:
                html_table += f'<td>{row[col]}</td>'
        html_table += '</tr>'

    html_table += '</tbody></table>'

    # Render in Streamlit
    st.markdown(html_table, unsafe_allow_html=True)
    st.caption("BI=Buy Initiated, SI=Sell Initiated, TΔ=Tick Delta, CumΔ=Cumulative Tick Delta")

# Enhanced Order Flow Dashboard with Advanced Trading Signals
# Add these functions to your existing code

import numpy as np
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN

# --- Enhanced Trading Signal Functions ---
def calculate_volume_profile(df, price_levels=50):
    """Calculate volume profile to identify high-volume areas"""
    if df.empty or 'close' not in df.columns:
        return {}
    
    # Get price range
    price_min = df['low'].min() if 'low' in df.columns else df['close'].min()
    price_max = df['high'].max() if 'high' in df.columns else df['close'].max()
    
    # Create price bins
    price_bins = np.linspace(price_min, price_max, price_levels)
    volume_profile = {}
    
    for i in range(len(price_bins) - 1):
        lower = price_bins[i]
        upper = price_bins[i + 1]
        mid_price = (lower + upper) / 2
        
        # Filter data within this price range
        mask = ((df['low'] <= upper) & (df['high'] >= lower)) if 'low' in df.columns and 'high' in df.columns else (df['close'].between(lower, upper))
        volume_in_range = df[mask]['buy_volume'].sum() + df[mask]['sell_volume'].sum() if mask.any() else 0
        
        if volume_in_range > 0:
            volume_profile[mid_price] = volume_in_range
    
    return volume_profile

def identify_order_flow_divergence(df, lookback=10):
    """Identify divergences between price and order flow"""
    if df.empty or len(df) < lookback * 2:
        return []
    
    divergences = []
    
    for i in range(lookback, len(df) - lookback):
        # Price trend (last 'lookback' periods)
        recent_prices = df['close'].iloc[i-lookback:i+1]
        price_trend = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
        
        # Delta trend (last 'lookback' periods) 
        recent_deltas = df['cumulative_tick_delta'].iloc[i-lookback:i+1]
        delta_trend = 1 if recent_deltas.iloc[-1] > recent_deltas.iloc[0] else -1
        
        # Divergence detection
        if price_trend != delta_trend:
            divergence_type = "Bearish" if price_trend > 0 and delta_trend < 0 else "Bullish"
            divergences.append({
                'timestamp': df['timestamp'].iloc[i],
                'price': df['close'].iloc[i],
                'type': divergence_type,
                'strength': abs(recent_prices.pct_change().sum()) + abs(recent_deltas.pct_change().sum())
            })
    
    return divergences

def calculate_market_microstructure_signals(df):
    """Calculate advanced microstructure signals"""
    if df.empty:
        return df.copy()
    
    df = df.copy()
    
    # 1. Order Flow Imbalance Ratio
    df['flow_ratio'] = (df['buy_initiated'] - df['sell_initiated']) / (df['buy_initiated'] + df['sell_initiated'] + 1)
    
    # 2. Volume-Weighted Delta
    total_volume = df['buy_volume'] + df['sell_volume']
    df['vw_delta'] = df['tick_delta'] * (total_volume / total_volume.rolling(window=20).mean())
    
    # 3. Delta Momentum (rate of change)
    df['delta_momentum'] = df['cumulative_tick_delta'].diff(5)
    
    # 4. Absorption (large volume, small price movement)
    price_change = abs(df['close'].pct_change())
    volume_spike = total_volume > total_volume.rolling(window=20).mean() * 1.5
    df['absorption'] = volume_spike & (price_change < price_change.rolling(window=20).mean())
    
    # 5. Breakout Confirmation
    price_ma = df['close'].rolling(window=10).mean()
    df['above_ma'] = df['close'] > price_ma
    df['breakout_signal'] = (df['above_ma'] != df['above_ma'].shift(1)) & (abs(df['tick_delta']) > df['tick_delta'].rolling(window=20).std())
    
    return df

def calculate_premium_sr_levels(df, min_strength=3, cluster_distance=0.5):
    """Calculate only the most significant S/R levels using clustering and statistical analysis"""
    if df.empty or len(df) < 50:
        return []
    
    levels = []
    
    # 1. Identify significant swing points using statistical methods
    highs = df['high'].values if 'high' in df.columns else df['close'].values
    lows = df['low'].values if 'low' in df.columns else df['close'].values
    closes = df['close'].values
    volumes = (df['buy_volume'] + df['sell_volume']).values
    timestamps = df['timestamp'].values
    
    # Find peaks and valleys with minimum prominence
    price_range = highs.max() - lows.min()
    min_prominence = price_range * 0.002  # 0.2% of price range
    
    # Resistance levels (peaks)
    resistance_peaks, peak_properties = find_peaks(highs, prominence=min_prominence, distance=5)
    
    # Support levels (valleys) 
    support_valleys, valley_properties = find_peaks(-lows, prominence=min_prominence, distance=5)
    
    # 2. Calculate volume at each level and test frequency
    def calculate_level_strength(price_level, tolerance_pct=0.3):
        tolerance = price_level * (tolerance_pct / 100)
        touches = 0
        total_volume = 0
        rejection_count = 0
        
        for i in range(len(df)):
            # Check if price touched this level
            high_val = highs[i]
            low_val = lows[i]
            close_val = closes[i]
            volume_val = volumes[i]
            
            if (low_val <= price_level + tolerance and high_val >= price_level - tolerance):
                touches += 1
                total_volume += volume_val
                
                # Check for rejection (price touched level but closed away from it)
                if abs(close_val - price_level) > tolerance:
                    rejection_count += 1
        
        # Strength score based on touches, volume, and rejections
        strength_score = touches * 2 + (total_volume / 10000) + rejection_count * 1.5
        return strength_score, touches, total_volume, rejection_count
    
    # Process resistance levels
    for idx in resistance_peaks:
        price = highs[idx]
        timestamp = timestamps[idx]
        strength, touches, volume, rejections = calculate_level_strength(price)
        
        if touches >= min_strength:
            levels.append({
                'price': price,
                'timestamp': int(pd.to_datetime(timestamp).timestamp()),
                'type': 'Resistance',
                'strength_score': strength,
                'touches': touches,
                'volume': volume,
                'rejections': rejections,
                'level_type': 'Swing High'
            })
    
    # Process support levels
    for idx in support_valleys:
        price = lows[idx]
        timestamp = timestamps[idx] 
        strength, touches, volume, rejections = calculate_level_strength(price)
        
        if touches >= min_strength:
            levels.append({
                'price': price,
                'timestamp': int(pd.to_datetime(timestamp).timestamp()),
                'type': 'Support', 
                'strength_score': strength,
                'touches': touches,
                'volume': volume,
                'rejections': rejections,
                'level_type': 'Swing Low'
            })
    
    # 3. Add key volume profile levels (Point of Control)
    volume_profile = calculate_volume_profile(df, price_levels=20)
    if volume_profile:
        # Get top 3 volume areas
        sorted_vp = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for price, volume in sorted_vp:
            strength, touches, _, rejections = calculate_level_strength(price)
            if touches >= 2:  # Lower threshold for volume-based levels
                levels.append({
                    'price': price,
                    'timestamp': int(pd.to_datetime(df['timestamp'].iloc[-1]).timestamp()),
                    'type': 'Volume Node',
                    'strength_score': strength + (volume / 5000),  # Boost score for volume
                    'touches': touches,
                    'volume': volume,
                    'rejections': rejections,
                    'level_type': 'POC'
                })
    
    # 4. Cluster nearby levels to avoid redundancy
    if len(levels) > 0:
        prices = np.array([level['price'] for level in levels]).reshape(-1, 1)
        price_range = prices.max() - prices.min()
        eps = price_range * (cluster_distance / 100)  # Cluster levels within 0.5% of each other
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(prices)
        
        # Keep only the strongest level from each cluster
        clustered_levels = []
        for cluster_id in set(clustering.labels_):
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_levels = [levels[i] for i in cluster_indices]
            
            # Keep the level with highest strength score
            strongest_level = max(cluster_levels, key=lambda x: x['strength_score'])
            clustered_levels.append(strongest_level)
        
        levels = clustered_levels
    
    # 5. Rank and keep only top levels
    levels.sort(key=lambda x: x['strength_score'], reverse=True)
    
    # Assign strength categories based on ranking
    total_levels = len(levels)
    for i, level in enumerate(levels):
        if i < total_levels * 0.2:  # Top 20%
            level['strength'] = 'very_high'
            level['conviction'] = 'Extreme'
        elif i < total_levels * 0.4:  # Top 40%
            level['strength'] = 'high' 
            level['conviction'] = 'Strong'
        elif i < total_levels * 0.7:  # Top 70%
            level['strength'] = 'medium'
            level['conviction'] = 'Medium'
        else:
            level['strength'] = 'low'
            level['conviction'] = 'Weak'
    
    # Return only top 8-12 levels to avoid clutter
    return levels[:min(12, len(levels))]

def create_trading_opportunity_signals(df, sr_levels):
    """Generate specific trading opportunity signals"""
    if df.empty:
        return []
    
    signals = []
    current_price = df['close'].iloc[-1]
    current_delta = df['cumulative_tick_delta'].iloc[-1]
    
    # 1. Support/Resistance proximity signals
    for level in sr_levels[:6]:  # Check only top 6 levels
        price_diff = abs(current_price - level['price'])
        price_diff_pct = (price_diff / current_price) * 100
        
        if price_diff_pct <= 0.5:  # Within 0.5% of level
            if level['type'] == 'Resistance' and current_delta < -50:
                signals.append({
                    'type': 'SHORT_SETUP',
                    'message': f'🔴 SHORT: Price at resistance {level["price"]:.1f} with negative delta ({current_delta})',
                    'confidence': level['conviction'],
                    'price_level': level['price']
                })
            elif level['type'] == 'Support' and current_delta > 50:
                signals.append({
                    'type': 'LONG_SETUP', 
                    'message': f'🟢 LONG: Price at support {level["price"]:.1f} with positive delta ({current_delta})',
                    'confidence': level['conviction'],
                    'price_level': level['price']
                })
    
    # 2. Delta momentum signals
    recent_data = df.tail(10)
    delta_acceleration = recent_data['cumulative_tick_delta'].diff().mean()
    
    if abs(delta_acceleration) > 20:
        direction = "bullish momentum" if delta_acceleration > 0 else "bearish momentum"
        signals.append({
            'type': 'MOMENTUM',
            'message': f'⚡ Strong {direction} detected (acceleration: {delta_acceleration:.1f})',
            'confidence': 'Medium',
            'price_level': current_price
        })
    
    # 3. Order flow divergence
    divergences = identify_order_flow_divergence(df, lookback=5)
    if divergences:
        latest_div = divergences[-1]
        if (pd.to_datetime(latest_div['timestamp']) - df['timestamp'].iloc[-1]).total_seconds() < 300:  # Within 5 minutes
            signals.append({
                'type': 'DIVERGENCE',
                'message': f'🔄 {latest_div["type"]} divergence detected - Watch for reversal',
                'confidence': 'High' if latest_div['strength'] > 0.1 else 'Medium',
                'price_level': current_price
            })
    
    return signals

def create_enhanced_chart_with_signals(stock_name, chart_data, interval, chart_options, trading_signals):
    """Enhanced chart with trading signals and premium S/R levels"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Calculate premium S/R levels
    sr_levels_premium = calculate_premium_sr_levels(chart_data)
    
    # Create series for premium S/R levels only
    sr_series = create_support_resistance_series_premium(sr_levels_premium, chart_data)
    
    # Add microstructure signals to data
    enhanced_data = calculate_market_microstructure_signals(chart_data)
    
    # Prepare chart data (same as before but with enhanced data)
    candle_data = []
    tick_delta_values = []
    cumulative_delta_values = []
    signal_markers = []
    
    def format_number(num):
        if abs(num) >= 1000000:
            return f"{num/1000000:.1f}M".replace('.0M', 'M')
        elif abs(num) >= 1000:
            return f"{num/1000:.1f}K".replace('.0K', 'K')
        else:
            return str(int(num))
    
    for _, row in enhanced_data.tail(100).iterrows():
        try:
            timestamp = int(pd.to_datetime(row['timestamp']).timestamp())
            
            candle_data.append({
                'time': timestamp,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)), 
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0))
            })
            
            tick_delta = float(row.get('tick_delta', 0))
            cum_delta = float(row.get('cumulative_tick_delta', 0))
            
            # Enhanced delta values with additional info
            tick_delta_values.append({
                'timestamp': timestamp,
                'value': tick_delta,
                'formatted': f"+{format_number(tick_delta)}" if tick_delta > 0 else format_number(tick_delta),
                'flow_ratio': row.get('flow_ratio', 0),
                'absorption': row.get('absorption', False)
            })
            
            cumulative_delta_values.append({
                'timestamp': timestamp,
                'value': cum_delta,
                'formatted': f"+{format_number(cum_delta)}" if cum_delta > 0 else format_number(cum_delta),
                'momentum': row.get('delta_momentum', 0)
            })
            
            # Add signal markers
            if row.get('breakout_signal', False):
                signal_markers.append({
                    'time': timestamp,
                    'position': 'aboveBar' if cum_delta > 0 else 'belowBar',
                    'color': '#ff6b35',
                    'shape': 'arrowUp' if cum_delta > 0 else 'arrowDown',
                    'text': 'Breakout'
                })
                
        except Exception as e:
            continue
    
    chart_id = f"enhanced_chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    chart_height = chart_options.get('chart_height', 600)
    
    # Create trading signals panel HTML
    signals_html = create_trading_signals_panel(trading_signals)
    
    chart_html = f"""
<div class="enhanced-chart-container" style="width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 8px;">
    <!-- Trading Signals Panel -->
    {signals_html}
    
    <!-- Main Chart -->
    <div id="{chart_id}" style="width: 100%; height: {chart_height}px;"></div>
    
    <!-- Enhanced Delta Boxes with Microstructure Info -->
    <div id="{chart_id}_delta_container" style="padding: 10px; background: #f8fafc; border-top: 1px solid #e5e7eb;">
        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                📊 Tick Delta (with Flow Ratio)
            </div>
            <div class="delta-row" id="tick-delta-row" style="position: relative; height: 34px; overflow: visible;"></div>
        </div>
        
        <div>
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                📈 Cumulative Delta (with Momentum)
            </div>
            <div class="delta-row" id="cumulative-delta-row" style="position: relative; height: 34px; overflow: visible;"></div>
        </div>
    </div>
</div>

<style>
{create_enhanced_chart_styles()}
</style>

<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
{create_enhanced_chart_script(chart_id, candle_data, sr_series, tick_delta_values, cumulative_delta_values, signal_markers, chart_height)}
</script>
    """
    return chart_html

def create_support_resistance_series_premium(levels, chart_data):
    """Create series for premium S/R levels with enhanced styling"""
    if not levels or chart_data.empty:
        return []
    
    series = []
    start_time = int(pd.to_datetime(chart_data['timestamp'].min()).timestamp())
    end_time = int(pd.to_datetime(chart_data['timestamp'].max()).timestamp())
    
    # Premium color scheme
    color_scheme = {
        'very_high': {'resistance': '#8B0000', 'support': '#006400', 'width': 3, 'style': 0},  # Dark red/green, solid
        'high': {'resistance': '#DC143C', 'support': '#228B22', 'width': 2, 'style': 0},       # Red/green, solid  
        'medium': {'resistance': '#FF6347', 'support': '#32CD32', 'width': 2, 'style': 2},     # Light red/green, dashed
        'low': {'resistance': '#FFA07A', 'support': '#90EE90', 'width': 1, 'style': 1}        # Very light, dotted
    }
    
    for level in levels:
        price = level['price']
        level_type = level.get('type', 'Level')
        strength = level.get('strength', 'low')
        
        # Determine color based on type and strength
        if 'resistance' in level_type.lower():
            color = color_scheme[strength]['resistance']
        else:
            color = color_scheme[strength]['support']
        
        width = color_scheme[strength]['width']
        style = color_scheme[strength]['style']
        
        # Enhanced series name with key info
        touches = level.get('touches', 0)
        volume = level.get('volume', 0)
        conviction = level.get('conviction', 'Unknown')
        
        series_name = f"{level_type} {price:.1f} [{conviction}] T:{touches}"
        
        series.append({
            'name': series_name,
            'type': 'line',
            'data': [
                {'time': start_time, 'value': price},
                {'time': end_time, 'value': price}
            ],
            'color': color,
            'linewidth': width,
            'linestyle': style,
            'priceLineVisible': False
        })
    
    return series

def create_trading_signals_panel(signals):
    """Create HTML for trading signals panel"""
    if not signals:
        return '<div style="padding: 8px; background: #f0fdf4; border-bottom: 1px solid #e5e7eb; text-align: center; color: #16a34a;">📊 No active signals</div>'
    
    signals_html = '<div style="padding: 10px; background: #fffbeb; border-bottom: 2px solid #f59e0b;">'
    signals_html += '<div style="font-weight: 600; margin-bottom: 8px; color: #92400e;">🎯 Live Trading Signals</div>'
    signals_html += '<div style="display: flex; gap: 15px; flex-wrap: wrap;">'
    
    for signal in signals[:3]:  # Show max 3 signals
        confidence_color = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#6b7280'}
        bg_color = {'HIGH': '#fef2f2', 'LONG_SETUP': '#f0fdf4', 'SHORT_SETUP': '#fef2f2', 
                   'MOMENTUM': '#eff6ff', 'DIVERGENCE': '#fdf4ff'}
        
        signals_html += f'''
        <div style="background: {bg_color.get(signal["type"], "#f9fafb")}; 
                    padding: 8px 12px; border-radius: 6px; border-left: 3px solid {confidence_color.get(signal["confidence"], "#6b7280")};">
            <div style="font-size: 12px; font-weight: 600;">{signal["message"]}</div>
            <div style="font-size: 10px; color: #6b7280;">Confidence: {signal["confidence"]}</div>
        </div>
        '''
    
    signals_html += '</div></div>'
    return signals_html

def create_enhanced_chart_styles():
    """CSS styles for enhanced chart"""
    return """
    .enhanced-chart-container {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .delta-row {
        scrollbar-width: thin;
        scrollbar-color: #cbd5e1 #f1f5f9;
    }
    
    .delta-box {
        min-width: 65px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 10px;
        font-weight: 600;
        border-radius: 5px;
        color: white;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4);
        white-space: nowrap;
        cursor: default;
        transition: all 0.2s ease;
        position: relative;
    }
    
    .delta-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
        z-index: 10;
    }
    
    .delta-positive {
        background: linear-gradient(135deg, #26a69a 0%, #1e8c82 100%);
        border: 1px solid #1e8c82;
    }
    
    .delta-negative {
        background: linear-gradient(135deg, #ef5350 0%, #d84343 100%);
        border: 1px solid #d84343;
    }
    
    .delta-zero {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
        border: 1px solid #374151;
    }
    
    .delta-absorption {
        box-shadow: 0 0 0 2px #fbbf24;
    }
    
    .delta-momentum-high {
        box-shadow: inset 0 0 0 2px rgba(255,255,255,0.3);
    }
    """

def create_enhanced_chart_script(chart_id, candle_data, sr_series, tick_delta_values, cumulative_delta_values, signal_markers, chart_height):
    """JavaScript for enhanced chart"""
    return f"""
(function() {{
    const container = document.getElementById('{chart_id}');
    
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align: center; padding: 40px; color: #6b7280;">Chart library not loaded</div>';
        return;
    }}
    
    container.innerHTML = '';
    
    let chart;
    let candleSeries;
    let deltaBoxes = {{}};
    
    const candleData = {candle_data};
    const tickDeltaData = {tick_delta_values};
    const cumulativeDeltaData = {cumulative_delta_values};
    const srSeriesData = {sr_series};
    const signalMarkers = {signal_markers};
    
    function initChart() {{
        chart = LightweightCharts.createChart(container, {{
            width: container.clientWidth,
            height: {chart_height},
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
                vertLine: {{ width: 1, color: '#9B7DFF', style: LightweightCharts.LineStyle.Solid }},
                horzLine: {{ width: 1, color: '#9B7DFF', style: LightweightCharts.LineStyle.Solid }},
            }},
            rightPriceScale: {{ borderColor: '#D6DCDE' }},
            timeScale: {{
                borderColor: '#D6DCDE',
                timeVisible: true,
                secondsVisible: false,
                rightOffset: 5,
                barSpacing: 10,
                minBarSpacing: 6
            }}
        }});
        
        candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        }});
        
        candleSeries.setData(candleData);
        
        // Add signal markers
        if (signalMarkers.length > 0) {{
            candleSeries.setMarkers(signalMarkers);
        }}
        
        // Add premium S/R lines
        srSeriesData.forEach(srSeries => {{
            const lineSeries = chart.addLineSeries({{
                color: srSeries.color,
                lineWidth: srSeries.linewidth,
                lineStyle: srSeries.linestyle,
                priceLineVisible: false,
                title: srSeries.name
            }});
            lineSeries.setData(srSeries.data);
        }});
        
        chart.timeScale().fitContent();
        createEnhancedDeltaBoxes();
        
        chart.timeScale().subscribeVisibleTimeRangeChange(updateDeltaBoxAlignment);
    }}
    
    function createEnhancedDeltaBoxes() {{
        createDeltaBoxes(tickDeltaData, 'tick-delta-row', 'tick');
        createDeltaBoxes(cumulativeDeltaData, 'cumulative-delta-row', 'cumulative');
        updateDeltaBoxAlignment();
    }}
    
    function createDeltaBoxes(data, containerId, type) {{
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '';
        deltaBoxes[type] = [];
        
        data.forEach((item, index) => {{
            const box = document.createElement('div');
            box.className = 'delta-box';
            box.dataset.timestamp = item.timestamp;
            
            // Enhanced styling based on additional data
            if (item.value > 0) {{
                box.classList.add('delta-positive');
            }} else if (item.value < 0) {{
                box.classList.add('delta-negative');
            }} else {{
                box.classList.add('delta-zero');
            }}
            
            // Add special styling for absorption or high momentum
            if (item.absorption) {{
                box.classList.add('delta-absorption');
            }}
            if (Math.abs(item.momentum || 0) > 100) {{
                box.classList.add('delta-momentum-high');
            }}
            
            box.textContent = item.formatted;
            
            // Enhanced tooltip
            const date = new Date(item.timestamp * 1000);
            let tooltip = `Time: ${{date.toLocaleTimeString()}}\\nValue: ${{item.value >= 0 ? '+' : ''}}${{item.value}}`;
            
            if (type === 'tick' && item.flow_ratio !== undefined) {{
                tooltip += `\\nFlow Ratio: ${{(item.flow_ratio * 100).toFixed(1)}}%`;
            }}
            if (type === 'cumulative' && item.momentum !== undefined) {{
                tooltip += `\\nMomentum: ${{item.momentum}}`;
            }}
            
            box.

# Display active trading signals
if trading_signals and trading_options.get('enable_signals', True):
    st.markdown("### 🚨 Live Trading Opportunities")
    
    signal_cols = st.columns(min(len(trading_signals), 3))
    for i, signal in enumerate(trading_signals[:3]):
        with signal_cols[i]:
            confidence_colors = {'High': '🔴', 'Medium': '🟡', 'Low': '⚪'}
            st.info(f"{confidence_colors.get(signal['confidence'], '📊')} **{signal['type'].replace('_', ' ').title()}**\n\n{signal['message']}")

# --- MAIN DISPLAY ---
if mobile_view:
    inject_mobile_css()
    inject_enhanced_css()
    stock_name = selected_option.split(' (')[0]
    st.markdown(f"# 📊 {stock_name}")
    st.caption(f"🔄 Updates every {refresh_interval}s • {interval}min intervals")
    
    # Smart Data Summary Panel
    if smart_summary:
        trend_class = smart_summary.get('price_trend', 'neutral').lower()
        st.markdown(f"""
            <div class="smart-summary-card {trend_class}">
                <div class="summary-metric">
                    <span class="summary-label">💰 Current Price</span>
                    <span class="summary-value">{smart_summary.get('current_price', 0):.2f}</span>
                </div>
                <div class="summary-metric">
                    <span class="summary-label">📈 Price Change</span>
                    <span class="summary-value {'positive' if smart_summary.get('price_change', 0) > 0 else 'negative' if smart_summary.get('price_change', 0) < 0 else 'neutral'}">
                        {smart_summary.get('price_change', 0):+.2f} ({smart_summary.get('price_change_pct', 0):+.1f}%)
                    </span>
                </div>
                <div class="summary-metric">
                    <span class="summary-label">📊 Delta Sentiment</span>
                    <span class="summary-value {'positive' if smart_summary.get('delta_sentiment') == 'Bullish' else 'negative' if smart_summary.get('delta_sentiment') == 'Bearish' else 'neutral'}">
                        {smart_summary.get('delta_sentiment', 'Neutral')}
                    </span>
                </div>
                <div class="summary-metric">
                    <span class="summary-label">🛡️ S/R Levels</span>
                    <span class="summary-value">{smart_summary.get('level_strength', 'No levels')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Data summary
    if not agg_df_all_days.empty:
        earliest_date = agg_df_all_days['timestamp'].min().strftime('%Y-%m-%d')
        latest_date = agg_df_all_days['timestamp'].max().strftime('%Y-%m-%d')
        total_records = len(agg_df_all_days)
        today_records = len(agg_df_current_day)
        
        st.info(f"📊 **Data Summary:** Chart shows {total_records} records from {earliest_date} to {latest_date} • Table shows {today_records} records from {latest_date_str}")
        
        st.markdown("---")
        st.markdown("### 📈 Charts (All Days Data)")
        # NEW CODE:
        # Add premium trading controls
        trading_options = add_premium_trading_controls()

        # Generate trading signals
        if not agg_df_all_days.empty:
            premium_sr_levels = calculate_premium_sr_levels(agg_df_all_days, min_strength=3)
            trading_signals = create_trading_opportunity_signals(agg_df_all_days, premium_sr_levels)
        else:
            trading_signals = []

        # Create enhanced chart
        chart_html = create_enhanced_chart_with_signals(
            stock_name, 
            agg_df_all_days, 
            interval, 
            {**chart_options, **trading_options}, 
            trading_signals
        )
        components.html(chart_html, height=chart_options.get('chart_height', 600) + 150, width=0)
        st.markdown("---")
        st.markdown(f"### 📋 {latest_date_str} Activity")
        st.markdown("""
        <style>
        .mobile-table th, .mobile-table td {
            font-size: 11px;   /* Smaller font size */
            padding: 3px 4px;  /* Tighter cell padding */
        }
        </style>
        """, unsafe_allow_html=True)
        create_mobile_table(agg_df_current_day)        
        st.markdown("---")
        csv = agg_df_current_day.to_csv(index=False).encode('utf-8')
        st.download_button(f"📥 Download {latest_date_str} Data", csv, f"orderflow_{stock_name}_{latest_date_str}.csv", "text/csv", use_container_width=True)

    else:
        st.error("📵 No data available for this security")
else:
    inject_enhanced_css()
    st.title(f"Order Flow Dashboard: {selected_option}")
    if not agg_df_all_days.empty:
        # Smart Data Summary Panel
        if smart_summary:
            col1, col2 = st.columns([2, 1])
            with col1:
                trend_class = smart_summary.get('price_trend', 'neutral').lower()
                st.markdown(f"""
                <div class="smart-summary-card {trend_class}">
                    <div class="summary-metric">
                        <span class="summary-label">💰 Current Price</span>
                        <span class="summary-value">{smart_summary.get('current_price', 0):.2f}</span>
                    </div>
                    <div class="summary-metric">
                        <span class="summary-label">📈 Price Change</span>
                        <span class="summary-value {'positive' if smart_summary.get('price_change', 0) > 0 else 'negative' if smart_summary.get('price_change', 0) < 0 else 'neutral'}">
                            {smart_summary.get('price_change', 0):+.2f} ({smart_summary.get('price_change_pct', 0):+.1f}%)
                        </span>
                    </div>
                    <div class="summary-metric">
                        <span class="summary-label">📊 Delta Sentiment</span>
                        <span class="summary-value {'positive' if smart_summary.get('delta_sentiment') == 'Bullish' else 'negative' if smart_summary.get('delta_sentiment') == 'Bearish' else 'neutral'}">
                            {smart_summary.get('delta_sentiment', 'Neutral')}
                        </span>
                    </div>
                    <div class="summary-metric">
                        <span class="summary-label">🛡️ S/R Levels</span>
                        <span class="summary-value">{smart_summary.get('level_strength', 'No levels')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div style="background: #f8fafc; padding: 12px; border-radius: 8px; border: 1px solid #e5e7eb;">
                    <div style="font-weight: 600; margin-bottom: 8px;">📈 Quick Stats</div>
                    <div style="font-size: 12px; line-height: 1.4;">
                        <div>📊 Total Volume: {:,}</div>
                        <div>📅 Session: {}</div>
                        <div>🔄 Records: {}</div>
                    </div>
                </div>
                """.format(
                    int(smart_summary.get('total_volume', 0)),
                    smart_summary.get('session_activity', 'Unknown'),
                    smart_summary.get('total_records', 0)
                ), unsafe_allow_html=True)
        
        # Data summary
        earliest_date = agg_df_all_days['timestamp'].min().strftime('%Y-%m-%d')
        latest_date = agg_df_all_days['timestamp'].max().strftime('%Y-%m-%d')
        total_records = len(agg_df_all_days)
        today_records = len(agg_df_current_day)
        
        st.info(f"📊 **Data Summary:** Chart shows {total_records} records from {earliest_date} to {latest_date} • Table shows {today_records} records from {latest_date_str}")
        
        st.subheader("Candlestick Chart (All Days Data)")
        # NEW CODE:
        # Add premium trading controls
        trading_options = add_premium_trading_controls()

        # Generate trading signals
        if not agg_df_all_days.empty:
            premium_sr_levels = calculate_premium_sr_levels(agg_df_all_days, min_strength=3)
            trading_signals = create_trading_opportunity_signals(agg_df_all_days, premium_sr_levels)
        else:
            trading_signals = []

        # Create enhanced chart
        chart_html = create_enhanced_chart_with_signals(
            stock_name, 
            agg_df_all_days, 
            interval, 
            {**chart_options, **trading_options}, 
            trading_signals
        )
        components.html(chart_html, height=chart_options.get('chart_height', 600) + 150, width=0) 
        st.caption("Full history + live updates")
        
        st.subheader(f"{latest_date_str} Data Table")
        agg_df_formatted = agg_df_current_day.copy()
        agg_df_formatted['close'] = agg_df_formatted['close'].round(1)
        for col in ['buy_volume', 'sell_volume', 'buy_initiated', 'sell_initiated', 'delta', 'cumulative_delta', 'tick_delta', 'cumulative_tick_delta']:
            agg_df_formatted[col] = agg_df_formatted[col].round(0).astype(int)
        columns_to_show = ['timestamp', 'close', 'buy_initiated', 'sell_initiated', 'tick_delta', 'cumulative_tick_delta', 'inference']
        column_abbreviations = {'timestamp': 'Time', 'close': 'Close', 'buy_initiated': 'Buy Initiated', 'sell_initiated': 'Sell Initiated', 'tick_delta': 'Tick Delta', 'cumulative_tick_delta': 'Cumulative Tick Delta', 'inference': 'Inference'}
        agg_df_table = agg_df_formatted[columns_to_show].rename(columns=column_abbreviations)
        styled_table = agg_df_table.style.background_gradient(cmap="RdYlGn", subset=['Tick Delta', 'Cumulative Tick Delta'])
        st.dataframe(styled_table, use_container_width=True, height=600)       
        csv = agg_df_table.to_csv(index=False).encode('utf-8')
        st.download_button(f"Download {latest_date_str} Data", csv, f"orderflow_{latest_date_str}.csv", "text/csv")
    else:
        st.warning("No data available for this security.")

# Session Analysis
if trading_options.get('session_analysis', True) and not agg_df_current_day.empty:
    st.markdown("---")
    session_stats = create_market_session_analysis(agg_df_current_day)
    create_session_analysis_display(session_stats)
