import os
import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime, timedelta, time
import re
import streamlit.components.v1 as components

# Page configuration
st.set_page_config(layout="wide", page_title="Order Flow Dashboard")

# Auto-refresh controls
refresh_enabled = st.sidebar.toggle('🔄 Auto-refresh', value=True)
refresh_interval = st.sidebar.selectbox('Refresh Interval (seconds)', [5, 10, 15, 30, 60], index=2)
if refresh_enabled:
    st_autorefresh(interval=refresh_interval * 1000, key="data_refresh", limit=None)

# Custom CSS for styling (keep your previous style, just add full-width fixes)
def inject_full_width_chart_css():
    st.markdown("""
    <style>
        /* Remove default changes for full-width responsiveness */
        .main > div {padding-top:0;padding-bottom:0;}
        /* Full width container for the chart area */
        .element-container {width:100% !important; max-width:100% !important;}
        /* Keep your previous styling as desired for header, delta boxes, footer, etc. */
    </style>
    """, unsafe_allow_html=True)

# Load stock list, etc.
@st.cache_data
def load_stock_mapping():
    try:
        stock_df = pd.read_csv("stock_list.csv")
        return {str(k): v for k, v in zip(stock_df['security_id'], stock_df['symbol'])}
    except Exception as e:
        st.error(f"⚠️ Failed to load stock list: {e}")
        return {}

# Fetch functions for data, as previously defined (simplified here for brevity)
def fetch_live_data(security_id):
    # Your existing API call to live data
    # ...
    return pd.DataFrame()  # Replace with your actual data fetching

def aggregate_data(df, interval):
    # your aggregation logic
    # ...
    return pd.DataFrame()

# Chart creation function (full-width, responsive)
def create_tradingview_chart(stock_name, chart_data, interval):
    """Create a full-width tradingview style lightweight chart with no layout issues"""
    if chart_data.empty:
        return '<div style="text-align: center; padding: 40px; color: #6c757d;">No data available</div>'
    # Prepare data for chart (last 100 bars for performance)
    candle_data = []
    for _, row in chart_data.tail(100).iterrows():
        try:
            ts = int(pd.to_datetime(row['timestamp']).timestamp())
            candle_data.append({
                'time': ts,
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'close': float(row.get('close', 0))
            })
        except:
            continue
    chart_id = "chart_full_width"
    # Embed the lightweight charts with width=100%
    return f"""
<div class="full-width-chart-container">
  <div id="{chart_id}"></div>
</div>
<script src="https://unpkg.com/lightweight-charts@4.1.3/dist/lightweight-charts.standalone.production.js"></script>
<script>
(function() {{
    const container = document.getElementById('{chart_id}');
    if (!container || typeof LightweightCharts === 'undefined') {{
        container.innerHTML = '<div style="text-align:center; padding: 20px; color:#888;">Chart library error</div>';
        return;
    }}
    container.innerHTML = '';
    const chart = LightweightCharts.createChart(container, {{
        width: container.clientWidth,
        height: 500,
        layout: {{
            backgroundColor: '#ffffff',
            textColor: '#333'
        }},
        grid: {{
            vertLines: {{ color: '#f0f0f0' }},
            horzLines: {{ color: '#f0f0f0' }}
        }},
        crosshair: {{
            mode: LightweightCharts.CrosshairMode.Normal
        }},
        rightPriceScale: {{
            borderColor: '#cccccc',
            scaleMargins: {{ top: 0.1, bottom: 0.1 }}
        }},
        timeScale: {{
            borderColor: '#cccccc',
            timeVisible: true,
            secondsVisible: false
        }},
        autoSize: true
    }});
    const candleSeries = chart.addCandlestickSeries({{
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderVisible: false,
        wickUpColor: '#22c55e',
        wickDownColor: '#ef4444'
    }});
    candleSeries.setData({json.dumps(candle_data)});
    chart.timeScale().fitContent();

    // Responsive resize
    new ResizeObserver(entries => {{
        if (entries.length === 0) return;
        chart.applyOptions({{ width: container.clientWidth }});
    }}).observe(container);
})();
</script>
    """

# Inject CSS once
inject_full_width_chart_css()

# Sidebar controls, stock selection etc.
# (Using your previous code: load stock list, select, fetch data, etc.)

# When ready to display chart
if not agg_df.empty:
    chart_html = create_tradingview_chart(stock_name, agg_df, interval)
    components.html(chart_html, height=600, width=0)  # width=0 ensures full container width

# Rest of your previous code remains unchanged (headers, delta boxes, data table, footer)

# Keep your existing layout, headers, metrics, etc. after the chart embedding

# Example of header (unchanged)
st.markdown(f"""
<div class="trading-header">
    <div class="stock-info">
        <span style="background:#3b82f6;padding:6px 12px;border-radius:6px;margin-right:15px;font-size:12px;font-weight:bold;">NSE</span>
        <span class="stock-name">{stock_name}</span>
        <div class="price-info" style="margin-left:15px;">
            <span style="font-size:24px;font-weight:bold;">{current_price}</span>
            <span class="{price_class}" style="font-size:18px;">{price_change} {price_change_pct}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Keep your delta boxes and data table code as your previous layout

# Footer unchanged
st.markdown("---")
st.markdown(f"""<div style='text-align: center; font-size:12px; color:#888;'>Order Flow Dashboard | Updated in real-time</div>""", unsafe_allow_html=True)
