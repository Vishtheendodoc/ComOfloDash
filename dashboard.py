import os
import streamlit as st
import pandas as pd
import requests
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
from datetime import datetime, timedelta, time
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
        logging.FileHandler("error.log"),
        logging.StreamHandler()
    ]
)

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
ALERT_BATCH_SIZE = 10
MAX_WORKERS = 5
ALERT_COOLDOWN_MINUTES = 5
MONITOR_COOLDOWN_MINUTES = 2

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

# --- CSS Styling ---
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

def calculate_support_resistance_levels(chart_data, sensitivity=0.7):
    """
    Calculate support and resistance levels based on delta activity and price action
    Fixed to handle JSON serialization properly
    """
    if chart_data.empty:
        return {'support_levels': [], 'resistance_levels': []}
    
    df = chart_data.copy()
    
    # Calculate significant delta levels
    df['abs_tick_delta'] = abs(df['tick_delta'].fillna(0))
    df['abs_cum_delta'] = abs(df['cumulative_tick_delta'].fillna(0))
    
    # Define thresholds for significant activity
    tick_delta_threshold = df['abs_tick_delta'].quantile(1 - sensitivity)
    cum_delta_threshold = df['abs_cum_delta'].quantile(1 - sensitivity)
    
    # Find significant delta events
    significant_events = df[
        (df['abs_tick_delta'] >= tick_delta_threshold) | 
        (df['abs_cum_delta'] >= cum_delta_threshold)
    ].copy()
    
    if significant_events.empty:
        return {'support_levels': [], 'resistance_levels': []}
    
    support_levels = []
    resistance_levels = []
    
    # Group price levels and analyze delta behavior
    try:
        price_groups = significant_events.groupby(
            pd.cut(significant_events['close'], bins=20, duplicates='drop')
        )
        
        for price_range, group in price_groups:
            if len(group) < 2:
                continue
                
            avg_price = float(group['close'].mean())
            total_buy_pressure = float(group[group['tick_delta'] > 0]['tick_delta'].sum())
            total_sell_pressure = float(abs(group[group['tick_delta'] < 0]['tick_delta'].sum()))
            
            # Determine level significance
            touches = len(group)
            volume_significance = float((total_buy_pressure + total_sell_pressure) / len(group))
            
            # Convert timestamps to JSON-serializable format
            timestamps = []
            for ts in group['timestamp'].tolist():
                if hasattr(ts, 'isoformat'):
                    timestamps.append(ts.isoformat())
                elif hasattr(ts, 'timestamp'):
                    timestamps.append(int(ts.timestamp()))
                else:
                    timestamps.append(str(ts))
            
            level_data = {
                'price': round(avg_price, 2),
                'touches': int(touches),
                'strength': round(volume_significance, 2),
                'buy_pressure': round(total_buy_pressure, 2),
                'sell_pressure': round(total_sell_pressure, 2),
                'timestamps': timestamps
            }
            
            # Classify as support or resistance based on delta behavior
            if total_buy_pressure > total_sell_pressure * 1.2:
                support_levels.append(level_data)
            elif total_sell_pressure > total_buy_pressure * 1.2:
                resistance_levels.append(level_data)
            else:
                level_data_copy = level_data.copy()
                level_data_copy['strength'] = round(level_data_copy['strength'] * 0.6, 2)
                support_levels.append(level_data)
                resistance_levels.append(level_data_copy)
    
    except Exception as e:
        logging.warning(f"Error calculating S/R levels: {e}")
        return {'support_levels': [], 'resistance_levels': []}
    
    # Sort by strength and limit to top levels
    support_levels = sorted(support_levels, key=lambda x: x['strength'], reverse=True)[:5]
    resistance_levels = sorted(resistance_levels, key=lambda x: x['strength'], reverse=True)[:5]
    
    return {
        'support_levels': support_levels,
        'resistance_levels': resistance_levels
    }

def create_tradingview_chart_with_sr_levels(stock_name, chart_data, interval):
    """Enhanced chart with support/resistance lines based on delta analysis - Fixed JSON serialization"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6b7280;">No data available</div>'
    
    # Calculate support/resistance levels
    sr_levels = calculate_support_resistance_levels(chart_data)
    
    # Prepare all data series
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
            # Convert pandas timestamp to unix timestamp
            if hasattr(row['timestamp'], 'timestamp'):
                timestamp = int(row['timestamp'].timestamp())
            else:
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
            
        except Exception as e:
            logging.warning(f"Error processing row: {e}")
            continue
    
    chart_id = f"chart_{stock_name.replace(' ','_').replace('(','').replace(')','').replace('-','_')}"
    
    chart_html = f"""
<div class="chart-with-delta-container" style="width: 100%; background: white; border: 1px solid #e5e7eb; border-radius: 8px;">
    <!-- Chart Controls -->
    <div style="padding: 10px; background: #f8fafc; border-bottom: 1px solid #e5e7eb; display: flex; gap: 10px; align-items: center;">
        <label style="font-size: 12px; font-weight: 600; color: #374151;">
            <input type="checkbox" id="toggle-sr" checked style="margin-right: 5px;">
            Support/Resistance Lines
        </label>
        <label style="font-size: 12px; color: #6b7280;">
            Sensitivity:
            <select id="sr-sensitivity" style="margin-left: 5px; padding: 2px;">
                <option value="0.5">Low</option>
                <option value="0.7" selected>Medium</option>
                <option value="0.9">High</option>
            </select>
        </label>
        <div style="font-size: 11px; color: #6b7280; margin-left: auto;">
            🟢 Support: {len(sr_levels['support_levels'])} | 🔴 Resistance: {len(sr_levels['resistance_levels'])}
        </div>
    </div>
    
    <!-- Main Chart -->
    <div id="{chart_id}" style="width: 100%; height: 500px;"></div>
    
    <!-- Delta Boxes Container -->
    <div id="{chart_id}_delta_container" style="padding: 10px; background: #f8fafc; border-top: 1px solid #e5e7eb;">
        <div style="margin-bottom: 12px;">
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Tick Delta
            </div>
            <div class="delta-row" id="tick-delta-row" style="position: relative; height: 32px; overflow: visible;">
            </div>
        </div>
        
        <div>
            <div style="font-size: 12px; font-weight: 600; color: #374151; margin-bottom: 6px;">
                Cumulative Delta
            </div>
            <div class="delta-row" id="cumulative-delta-row" style="position: relative; height: 32px; overflow: visible;">
            </div>
        </div>
    </div>
    
    <!-- S/R Levels Legend -->
    <div style="padding: 10px; background: #f1f5f9; border-top: 1px solid #e5e7eb; font-size: 11px;">
        <div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 2px; background: #16a34a; border-radius: 1px;"></div>
                <span style="color: #374151;">Support (Buy Pressure)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 2px; background: #dc2626; border-radius: 1px;"></div>
                <span style="color: #374151;">Resistance (Sell Pressure)</span>
            </div>
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 12px; height: 2px; background: #6b7280; border-radius: 1px;"></div>
                <span style="color: #374151;">Mixed Activity</span>
            </div>
        </div>
    </div>
</div>

<style>
.delta-row {{
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 #f1f5f9;
}}
.delta-row::-webkit-scrollbar {{ height: 6px; }}
.delta-row::-webkit-scrollbar-track {{ background: #f1f5f9; border-radius: 3px; }}
.delta-row::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 3px; }}

.delta-box {{
    min-width: 60px; height: 26px;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; border-radius: 6px;
    color: white; text-shadow: 0 1px 2px rgba(0,0,0,0.4);
    white-space: nowrap; cursor: default;
    transition: all 0.2s ease; position: relative;
}}
.delta-box:hover {{ transform: translateY(-1px); box-shadow: 0 3px 6px rgba(0,0,0,0.25); z-index: 10; }}
.delta-positive {{ background: linear-gradient(135deg, #26a69a 0%, #1e8c82 100%); border: 1px solid #1e8c82; }}
.delta-negative {{ background: linear-gradient(135deg, #ef5350 0%, #d84343 100%); border: 1px solid #c62828; }}
.delta-zero {{ background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%); border: 1px solid #374151; }}
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
    
    let chart;
    let candleSeries;
    let deltaBoxes = {{}};
    let srLines = [];
    
    // Data - Now properly JSON serializable
    const candleData = {json.dumps(candle_data)};
    const tickDeltaData = {json.dumps(tick_delta_values)};
    const cumulativeDeltaData = {json.dumps(cumulative_delta_values)};
    const supportLevels = {json.dumps(sr_levels['support_levels'])};
    const resistanceLevels = {json.dumps(sr_levels['resistance_levels'])};
    
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
                vertLine: {{ width: 1, color: '#9B7DFF', style: LightweightCharts.LineStyle.Solid }},
                horzLine: {{ width: 1, color: '#9B7DFF', style: LightweightCharts.LineStyle.Solid }}
            }},
            rightPriceScale: {{ borderColor: '#D6DCDE' }},
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
        
        candleSeries = chart.addCandlestickSeries({{
            upColor: '#26a69a',
            downColor: '#ef5350',
            borderVisible: false,
            wickUpColor: '#26a69a',
            wickDownColor: '#ef5350'
        }});
        
        candleSeries.setData(candleData);
        
        // Add support/resistance lines
        addSupportResistanceLines();
        
        chart.timeScale().fitContent();
        createAlignedDeltaBoxes();
        chart.timeScale().subscribeVisibleTimeRangeChange(updateDeltaBoxAlignment);
        
        // Event listeners
        document.getElementById('toggle-sr').addEventListener('change', toggleSRLines);
        document.getElementById('sr-sensitivity').addEventListener('change', updateSensitivity);
    }}
    
    function addSupportResistanceLines() {{
        // Clear existing lines
        srLines.forEach(line => chart.removeSeries(line));
        srLines = [];
        
        const toggleSR = document.getElementById('toggle-sr');
        if (!toggleSR.checked) return;
        
        // Add support lines (green)
        supportLevels.forEach(level => {{
            const line = chart.addLineSeries({{
                color: '#16a34a',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: true,
                priceLineVisible: true,
                lastValueVisible: true,
                title: `Support ₹${{level.price}} (${{level.touches}} touches)`
            }});
            
            const lineData = candleData.map(candle => ({{
                time: candle.time,
                value: level.price
            }}));
            
            line.setData(lineData);
            srLines.push(line);
        }});
        
        // Add resistance lines (red)
        resistanceLevels.forEach(level => {{
            const line = chart.addLineSeries({{
                color: '#dc2626',
                lineWidth: 2,
                lineStyle: LightweightCharts.LineStyle.Solid,
                crosshairMarkerVisible: true,
                priceLineVisible: true,
                lastValueVisible: true,
                title: `Resistance ₹${{level.price}} (${{level.touches}} touches)`
            }});
            
            const lineData = candleData.map(candle => ({{
                time: candle.time,
                value: level.price
            }}));
            
            line.setData(lineData);
            srLines.push(line);
        }});
    }}
    
    function toggleSRLines(event) {{
        if (event.target.checked) {{
            addSupportResistanceLines();
        }} else {{
            srLines.forEach(line => chart.removeSeries(line));
            srLines = [];
        }}
    }}
    
    function updateSensitivity() {{
        addSupportResistanceLines();
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
            
            if (item.value > 0) {{
                box.classList.add('delta-positive');
            }} else if (item.value < 0) {{
                box.classList.add('delta-negative');
            }} else {{
                box.classList.add('delta-zero');
            }}
            
            box.textContent = item.formatted;
            
            const date = new Date(item.timestamp * 1000);
            const fullValue = item.value.toLocaleString();
            box.title = `Time: ${{date.toLocaleTimeString()}}\\nValue: ${{fullValue >= 0 ? '+' : ''}}${{fullValue}}`;
            
            container.appendChild(box);
            deltaBoxes[type].push(box);
        }});
    }}
    
    function updateDeltaBoxAlignment() {{
        if (!chart || !candleSeries) return;
        
        const timeScale = chart.timeScale();
        const visibleRange = timeScale.getVisibleRange();
        if (!visibleRange) return;
        
        const chartRect = container.getBoundingClientRect();
        const chartWidth = chartRect.width;
        
        ['tick', 'cumulative'].forEach(type => {{
            if (!deltaBoxes[type]) return;
            
            deltaBoxes[type].forEach((box) => {{
                const timestamp = parseInt(box.dataset.timestamp);
                const logicalPosition = timeScale.timeToCoordinate(timestamp);
                
                if (logicalPosition !== null) {{
                    const visibleTimeSpan = visibleRange.to - visibleRange.from;
                    const pixelsPerSecond = chartWidth / visibleTimeSpan;
                    const barSpacing = Math.max(4, Math.min(12, pixelsPerSecond * 60));
                    const boxWidth = Math.max(40, Math.min(80, barSpacing - 2));
                    
                    box.style.width = boxWidth + 'px';
                    box.style.minWidth = boxWidth + 'px';
                    box.style.position = 'absolute';
                    box.style.left = (logicalPosition - boxWidth/2) + 'px';
                    box.style.opacity = '1';
                    box.style.display = 'flex';
                    box.style.alignItems = 'center';
                    box.style.justifyContent = 'center';
                    
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
    
    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {{
        if (entries.length === 0 || entries[0].target !== container) return;
        const rect = entries[0].contentRect;
        chart.applyOptions({{ width: rect.width, height: 500 }});
        setTimeout(updateDeltaBoxAlignment, 100);
    }});
    
    // Initialize
    initChart();
    resizeObserver.observe(container);
    
    // Cleanup
    window.addEventListener('beforeunload', () => {{
        resizeObserver.disconnect();
        if (chart) chart.remove();
    }});
    
    setInterval(updateDeltaBoxAlignment, 1000);
}})();
</script>
    """
    
    return chart_html

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

# --- Mobile Display Functions ---
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

    st.markdown("""
    <style>
    .mobile-table { 
        width:100%; 
        border-collapse: collapse; 
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; 
    }
    .mobile-table th, .mobile-table td {
        font-size: 11px;
        padding: 3px 6px;
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
    .mobile-table .positive { color: #16a34a; font-weight:700; }
    .mobile-table .negative { color: #dc2626; font-weight:700; }
    .mobile-table .neutral  { color: #6b7280; font-weight:600; }
    .mobile-table td.numeric { text-align: right; }
    </style>
    """, unsafe_allow_html=True)

    today = datetime.now().date()
    start_time = datetime.combine(today, time(9, 0))
    end_time = datetime.combine(today, time(23, 59, 59))

    mobile_df = df[(df['timestamp'] >= pd.Timestamp(start_time)) & 
                   (df['timestamp'] <= pd.Timestamp(end_time))].copy()

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

    html_table = '<table class="mobile-table">'
    
    headers = ['Time', 'Price', 'BI', 'SI', 'TΔ', 'CumΔ']
    html_table += '<thead><tr>'
    for h in headers:
        html_table += f'<th>{h}</th>'
    html_table += '</tr></thead><tbody>'

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
    st.markdown(html_table, unsafe_allow_html=True)
    st.caption("BI=Buy Initiated, SI=Sell Initiated, TΔ=Tick Delta, CumΔ=Cumulative Tick Delta")

# --- Sidebar Controls ---
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

# --- Main UI Setup ---
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
mobile_view = st.sidebar.toggle("📱 Mobile Mode", value=True)

if mobile_view:
    inject_mobile_css()

# --- Enhanced Alert Controls ---
def enhanced_alert_controls():
    st.sidebar.markdown("---")
    st.sidebar.subheader("🚨 Alert System")
    
    alert_enabled = st.sidebar.toggle("Enable Zero Cross Alerts", value=False, key="enhanced_alerts")
    
    if alert_enabled:
        if st.sidebar.button("🔍 Check Current Stock", key="manual_check"):
            if not st.session_state.get('agg_df', pd.DataFrame()).empty:
                alert_sent = check_gradient_change(selected_id, st.session_state['agg_df'])
                if alert_sent:
                    st.sidebar.success("✅ Alert sent!")
                else:
                    st.sidebar.info("ℹ️ No alert needed")
        
        if st.sidebar.button("🧪 Test Alert"):
            test_message = f"""
🟢 <b>TEST ZERO CROSS ALERT</b> 🟢

📈 <b>Stock:</b> TEST STOCK
🔄 <b>Transition:</b> NEGATIVE → <b>POSITIVE</b>
⚡ <b>Event:</b> CROSSED ABOVE ZERO
📊 <b>Cumulative Tick Delta:</b> +75
⏰ <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

This is a test alert! 🚨
            """.strip()
            
            if send_telegram_alert(test_message):
                st.sidebar.success("✅ Test alert sent!")
            else:
                st.sidebar.error("❌ Failed to send test alert")

enhanced_alert_controls()

# --- Fetch and Process Data ---
historical_df = fetch_historical_data(selected_id)
live_df = fetch_live_data(selected_id)
full_df = pd.concat([historical_df, live_df]).drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# Filter for current day
today = datetime.now().date()
start_time = datetime.combine(today, time(9, 0))
end_time = datetime.combine(today, time(23, 59, 59))
full_df = full_df[(full_df['timestamp'] >= pd.Timestamp(start_time)) & (full_df['timestamp'] <= pd.Timestamp(end_time))]

agg_df = aggregate_data(full_df, interval)
st.session_state['agg_df'] = agg_df  # Store for alert checking

# --- Main Display ---
if mobile_view:
    inject_mobile_css()
    stock_name = selected_option.split(' (')[0]
    st.markdown(f"# 📊 {stock_name}")
    st.caption(f"🔄 Updates every {refresh_interval}s • {interval}min intervals")
    
    if not agg_df.empty:
        st.markdown("---")
        st.markdown("### 📈 Charts")
        chart_html = create_tradingview_chart_with_sr_levels(stock_name, agg_df, interval)
        components.html(chart_html, height=650, width=0)
        
        st.markdown("---")
        st.markdown("### 📋 Recent Activity")
        create_mobile_table(agg_df)
        
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
    st.title(f"Order Flow Dashboard: {selected_option}")
    if not agg_df.empty:
        st.subheader("Candlestick Chart")
        chart_html = create_tradingview_chart_with_sr_levels(selected_option, agg_df, interval)
        components.html(chart_html, height=650, width=0)
        
        st.caption("Full history + live updates")
        agg_df_formatted = agg_df.copy()
        agg_df_formatted['close'] = agg_df_formatted['close'].round(1)
        for col in ['buy_volume', 'sell_volume', 'buy_initiated', 'sell_initiated', 'delta', 'cumulative_delta', 'tick_delta', 'cumulative_tick_delta']:
            agg_df_formatted[col] = agg_df_formatted[col].round(0).astype(int)
        
        columns_to_show = ['timestamp', 'close', 'buy_initiated', 'sell_initiated', 'tick_delta', 'cumulative_tick_delta', 'inference']
        column_abbreviations = {
            'timestamp': 'Time', 
            'close': 'Close', 
            'buy_initiated': 'Buy Initiated', 
            'sell_initiated': 'Sell Initiated', 
            'tick_delta': 'Tick Delta', 
            'cumulative_tick_delta': 'Cumulative Tick Delta', 
            'inference': 'Inference'
        }
        agg_df_table = agg_df_formatted[columns_to_show].rename(columns=column_abbreviations)
        styled_table = agg_df_table.style.background_gradient(cmap="RdYlGn", subset=['Tick Delta', 'Cumulative Tick Delta'])
        st.dataframe(styled_table, use_container_width=True, height=600)
        
        csv = agg_df_table.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", csv, "orderflow_data.csv", "text/csv")
    else:
        st.warning("No data available for this security.")
