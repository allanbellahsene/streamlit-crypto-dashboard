import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from binance.client import Client
import requests
import re
import os
from dotenv import load_dotenv
import traceback
import logging
import hashlib
from functools import wraps
import time
from functools import lru_cache
import sys
import subprocess

st.write(f"Python version: {sys.version}")
st.write(f"Python executable: {sys.executable}")

if sys.version_info < (3, 9):
    st.error("This app requires Python 3.9 or newer. Please update your Python version.")
    st.stop()

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import setuptools
except ImportError:
    install('setuptools')

logging.basicConfig(filename='app.log', level=logging.ERROR)

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []

    def __call__(self, func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            now = time.time()
            self.calls = [call for call in self.calls if call > now - self.period]
            if len(self.calls) >= self.max_calls:
                raise Exception("Rate limit exceeded. Please try again later.")
            self.calls.append(now)
            return func(*args, **kwargs)
        return wrapped

def setup_binance_client():
    load_dotenv()
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_PRIVATE_KEY")
    
    if not api_key or not api_secret:
        st.error("Binance API keys are missing. Please set BINANCE_API_KEY and BINANCE_PRIVATE_KEY in your .env file.")
        st.stop()
    
    try:
        client = Client(api_key, api_secret)
        # Perform a simple API call to test the connection
        client.get_account()
        return client
    except Exception as e:
        st.error(f"Failed to initialize Binance client. Please check your API keys. Error: {str(e)}")
        st.stop()

# Use this function at the beginning of your main app logic
client = setup_binance_client()

def validate_inputs(start_date, end_date, btc_window, coin_window):
    errors = []
    
    # Validate dates
    if start_date >= end_date:
        errors.append("Start date must be before end date.")
    if end_date > datetime.now().date():
        errors.append("End date cannot be in the future.")
    if start_date < datetime(2009, 1, 3).date():  # Bitcoin's genesis block date
        errors.append("Start date cannot be before January 3, 2009.")
    
    # Validate window sizes
    if btc_window <= 0 or coin_window <= 0:
        errors.append("Window sizes must be positive integers.")
    if btc_window > 500 or coin_window > 500:
        errors.append("Window sizes cannot exceed 500.")
    
    # Check if the date range is too large
    if (end_date - start_date) > timedelta(days=365*7):
        errors.append("Date range cannot exceed 7 years.")
    
    return errors

def handle_error(error):
    # Generate a unique error ID
    error_id = hashlib.md5(str(error).encode()).hexdigest()[:8]
    
    # Log the full error for debugging
    logging.error(f"Error ID {error_id}: {str(error)}\n{traceback.format_exc()}")
    
    # Sanitize the error message
    sanitized_error = re.sub(r'File ".*?"', 'File "..."', str(error))
    sanitized_error = re.sub(r'(api_key|secret|password)=\S+', r'\1=***', sanitized_error, flags=re.IGNORECASE)
    sanitized_error = re.sub(r'\S+@\S+\.\S+', 'email@example.com', sanitized_error)
    sanitized_error = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', 'xxx.xxx.xxx.xxx', sanitized_error)
    
    # Display a user-friendly error message
    st.error(f"An unexpected error occurred. If this persists, please contact support with Error ID: {error_id}")
    
    # For debugging in development, uncomment the line below:
    # st.error(f"Debug info: {sanitized_error}")

# Set page config for dark theme
st.set_page_config(page_title="QuantiFi Momentum Crypto Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for dark theme with improved readability
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        color: #FF1493;
        border-color: #FF1493;
        background-color: #2D2D2D;
    }
    .stSelectbox>div>div>select {
        color: #FF1493;
        background-color: #2D2D2D;
    }
    .stDateInput>div>div>input {
        color: #FF1493;
        background-color: #2D2D2D;
    }
    .stNumberInput>div>div>input {
        color: #FF1493;
        background-color: #2D2D2D;
    }
    .metric-card {
        background-color: #2D2D2D;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        color: #FFFFFF;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #FF1493;
    }
    .dataframe {
        font-size: 12px;
    }
    .dataframe th {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .pink-title {
        color: #FF1493 !important;
        font-size: 3em !important;
        font-weight: bold !important;
        text-align: center !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
    }
    .dataframe td {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    h1, h2, h3, label, .stTextInput>label, .stSelectbox>label, .stDateInput>label, .stNumberInput>label {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

@RateLimiter(max_calls=10, period=60)  # 10 calls per minute, adjust as needed
@lru_cache(maxsize=1) 
def get_top_50_cryptos():
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": 100,
            "page": 1,
            "sparkline": False
        }
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad responses
        data = response.json()
        
        if not isinstance(data, list):
            raise ValueError("Unexpected data format from API")
        
        stablecoins = ['USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'USDN', 'LEOUSDT']
        filtered_data = [coin for coin in data if isinstance(coin, dict) and coin.get('symbol', '').upper() not in stablecoins]
        
        return {coin.get('name', f"Unknown {i}"): f"{coin.get('symbol', '').upper()}USDT" for i, coin in enumerate(filtered_data[:50])}
    except Exception as e:
        st.error(f"Error fetching top cryptocurrencies: {str(e)}")
        return {"Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT"}  # Fallback options

@RateLimiter(max_calls=5, period=60)  # 5 calls per minute
def import_binance_data(symbol, start_date, end_date, interval='1d', contract='spot'):
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    try:
        if contract == 'futures':
            df = client.futures_historical_klines(symbol, interval=interval, start_str=start_str, end_str=end_str)
        else:
            df = client.get_historical_klines(symbol, interval=interval, start_str=start_str, end_str=end_str)
        
        df = pd.DataFrame(df, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df = df.drop(['Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'], axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        for col in df.columns:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_signals(df, btc_window, coin_window):
    try:
        btc_data = import_binance_data("BTCUSDT", df.index[0], df.index[-1], interval=interval, contract=contract)
        if btc_data is None:
            raise ValueError("Failed to fetch BTC data")
        
        df['BTC'] = btc_data['Close']
        df['BTC_MA'] = df['BTC'].rolling(window=btc_window).mean()
        df['Coin_MA'] = df['Close'].rolling(window=coin_window).mean()
        df['Signal'] = np.where((df['BTC'] > df['BTC_MA']) & (df['Close'] > df['Coin_MA']), 1, 0)
        return df
    except Exception as e:
        st.error(f"An error occurred while calculating signals: {str(e)}")
        return None

def backtest_strategy(df):
    df['Strategy_Return'] = df['Close'].pct_change() * df['Signal'].shift(2)
    df['Buy_Hold_Return'] = df['Close'].pct_change()
    df['Strategy_Equity'] = (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold_Equity'] = (1 + df['Buy_Hold_Return']).cumprod()
    return df

def calculate_metrics(df):
    strategy_return = df['Strategy_Equity'].iloc[-1] - 1
    buy_hold_return = df['Buy_Hold_Equity'].iloc[-1] - 1
    strategy_sharpe = np.sqrt(252) * df['Strategy_Return'].mean() / df['Strategy_Return'].std()
    buy_hold_sharpe = np.sqrt(252) * df['Buy_Hold_Return'].mean() / df['Buy_Hold_Return'].std()
    strategy_max_drawdown = (df['Strategy_Equity'] / df['Strategy_Equity'].cummax() - 1).min()
    buy_hold_max_drawdown = (df['Buy_Hold_Equity'] / df['Buy_Hold_Equity'].cummax() - 1).min()
    
    return {
        'Strategy Return': f'{strategy_return:.2%}',
        'Buy & Hold Return': f'{buy_hold_return:.2%}',
        'Strategy Sharpe': f'{strategy_sharpe:.2f}',
        'Buy & Hold Sharpe': f'{buy_hold_sharpe:.2f}',
        'Strategy Max Drawdown': f'{strategy_max_drawdown:.2%}',
        'Buy & Hold Max Drawdown': f'{buy_hold_max_drawdown:.2%}'
    }

def plot_equity_curves(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Strategy_Equity'], mode='lines', name='Strategy', line=dict(color='#FF1493', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['Buy_Hold_Equity'], mode='lines', name='Buy & Hold', line=dict(color='#00FFFF', width=2)))
    fig.update_layout(
        title=dict(text='Strategy vs Buy & Hold', font=dict(color='#FFFFFF', size=24)),
        xaxis_title=dict(text='Date', font=dict(color='#FFFFFF')),
        yaxis_title=dict(text='Equity', font=dict(color='#FFFFFF')),
        plot_bgcolor='#2D2D2D',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF'),
        legend=dict(font=dict(color='#FFFFFF'), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor="#444444"),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=False, zeroline=False),
        height=600
    )
    return fig

def plot_signals_and_price(df, ticker):
    # Use the entire dataframe or last 365 days, whichever is shorter
    last_365_days = df.tail(min(365, len(df)))
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    # Price and MA plot
    fig.add_trace(go.Scatter(x=last_365_days.index, y=last_365_days['Close'], mode='lines', name=f'{ticker} Price', line=dict(color='#FFFFFF')), row=1, col=1)
    fig.add_trace(go.Scatter(x=last_365_days.index, y=last_365_days['Coin_MA'], mode='lines', name='MA(50)', line=dict(color='#FFA500')), row=1, col=1)
    
    # Find actual buy and sell signals
    signal_changes = last_365_days['Signal'].diff().fillna(0)
    buy_signals = last_365_days[signal_changes == 1]
    sell_signals = last_365_days[signal_changes == -1]
    
    # Buy signals (green triangles)
    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'],
        mode='markers', name='Buy Signal',
        marker=dict(symbol='triangle-up', size=10, color='green'),
    ), row=1, col=1)
    
    # Sell signals (red triangles)
    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'],
        mode='markers', name='Sell Signal',
        marker=dict(symbol='triangle-down', size=10, color='red'),
    ), row=1, col=1)
    
    # Bitcoin regime filter
    btc_above_ma = last_365_days['BTC'] > last_365_days['BTC_MA']
    fig.add_trace(go.Scatter(
        x=last_365_days.index, y=last_365_days['BTC'],
        fill='tozeroy', fillcolor='rgba(0, 255, 0, 0.1)', # Green for bull market
        line=dict(color='rgba(0, 0, 0, 0)'), name='Bull Market',
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=last_365_days.index, y=last_365_days['BTC'].where(~btc_above_ma),
        fill='tozeroy', fillcolor='rgba(255, 0, 0, 0.1)', # Red for bear market
        line=dict(color='rgba(0, 0, 0, 0)'), name='Bear Market',
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=last_365_days.index, y=last_365_days['BTC'], mode='lines', name='BTC Price', line=dict(color='#FFFFFF')), row=2, col=1)
    fig.add_trace(go.Scatter(x=last_365_days.index, y=last_365_days['BTC_MA'], mode='lines', name='BTC MA(100)', line=dict(color='#FFA500')), row=2, col=1)
    
    # Add rectangles for legend
    fig.add_trace(go.Scatter(
        x=[last_365_days.index[0]], y=[last_365_days['BTC'].max()],
        mode='markers', marker=dict(size=15, color='rgba(0, 255, 0, 0.1)', symbol='square'),
        name='Bull Market'
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=[last_365_days.index[0]], y=[last_365_days['BTC'].max()],
        mode='markers', marker=dict(size=15, color='rgba(255, 0, 0, 0.1)', symbol='square'),
        name='Bear Market'
    ), row=2, col=1)
    
    fig.update_layout(
        title=dict(text=f'Last {min(365, len(df))} Days: Signals, Price, and Bitcoin Regime', font=dict(color='#FFFFFF', size=24)),
        plot_bgcolor='#2D2D2D',
        paper_bgcolor='#1E1E1E',
        font=dict(color='#FFFFFF'),
        legend=dict(font=dict(color='#FFFFFF'), bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
        hoverlabel=dict(bgcolor="#444444"),
        height=800
    )
    fig.update_xaxes(title_text="Date", title_font=dict(color='#FFFFFF'), showgrid=False, zeroline=False)
    fig.update_yaxes(title_font=dict(color='#FFFFFF'), showgrid=False, zeroline=False)
    fig.update_yaxes(title_text=f"{ticker} Price", row=1, col=1)
    fig.update_yaxes(title_text="BTC Price", row=2, col=1)
    
    return fig

def display_disclaimer():
    st.markdown("""
    ## Disclaimer and Terms of Use

    **IMPORTANT: Read this disclaimer carefully before using this application.**

    1. **No Financial Advice**: This application is for educational and informational purposes only. It does not constitute financial advice, trading advice, or any other type of advice. Always conduct your own research before making any investment decisions.

    2. **Risk Warning**: Cryptocurrency trading carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade cryptocurrencies, you should carefully consider your investment objectives, level of experience, and risk appetite.

    3. **No Guarantee**: Past performance is not indicative of future results. The backtest results and any strategies presented in this application do not guarantee future performance or success.

    4. **Data Accuracy**: While we strive to ensure the accuracy of the data presented, we cannot guarantee its completeness or accuracy. The data may be delayed or incorrect.

    5. **API Usage**: This application uses third-party APIs. By using this application, you agree to comply with the terms of service of these API providers.

    6. **Limitation of Liability**: In no event shall the creators or distributors of this application be liable for any damages or losses resulting from the use of this application or the information it provides.

    7. **User Responsibility**: You are solely responsible for any decisions and actions you take based on the information provided by this application.

    By using this application, you acknowledge that you have read, understood, and agree to these terms and conditions.
    """)


# Main app logic
try:
    st.markdown("<h1 class='pink-title'>QuantiFi Momentum Crypto Dashboard</h1>", unsafe_allow_html=True)

    display_disclaimer()

    crypto_options = get_top_50_cryptos()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_crypto = st.selectbox('Select a cryptocurrency', list(crypto_options.keys()))
        ticker = crypto_options[selected_crypto]
    with col2:
        contract = st.selectbox('Contract Type', ['spot', 'futures'])
    with col3:
        interval = st.selectbox('Data Frequency', ['1d', '4h', '1h'])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        start_date = st.date_input('Start Date', datetime(2020, 1, 1))
    with col2:
        end_date = st.date_input('End Date', datetime.now())
    with col3:
        btc_window = st.number_input('Regime Filter Window (# of bars)', min_value=1, max_value=500, value=100)
    with col4:
        coin_window = st.number_input('Coin-specific Window (# of bars)', min_value=1, max_value=500, value=50)
    
    errors = validate_inputs(start_date, end_date, btc_window, coin_window)

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    if st.button('Run Backtest'):
        with st.spinner('Fetching data...'):
            df = import_binance_data(ticker, start_date, end_date, interval, contract)
        
        if df is not None and not df.empty:
            with st.spinner('Calculating signals...'):
                df = calculate_signals(df, btc_window=btc_window, coin_window=coin_window)
            if df is not None:
                with st.spinner('Running backtest...'):
                    df = backtest_strategy(df)
                
                st.plotly_chart(plot_equity_curves(df), use_container_width=True)
                
                metrics = calculate_metrics(df)
                col1, col2, col3 = st.columns(3)
                for i, (col, metrics_pair) in enumerate(zip([col1, col2, col3], [
                    ('Strategy Return', 'Buy & Hold Return'),
                    ('Strategy Sharpe', 'Buy & Hold Sharpe'),
                    ('Strategy Max Drawdown', 'Buy & Hold Max Drawdown')
                ])):
                    with col:
                        for metric in metrics_pair:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{metric}</div>
                                <div class="metric-value">{metrics[metric]}</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.subheader("Signals for Last 30 Days")
                last_30_days = df.tail(30).reset_index()
                last_30_days['Date'] = last_30_days['timestamp'].dt.date
                last_30_days['Signal'] = last_30_days['Signal'].map({0: 'Sell/Cash', 1: 'Buy/Hold'})
                last_30_days = last_30_days[['Date', 'Close', 'Signal']].sort_values('Date', ascending=False)
                last_30_days.columns = ['Date', 'Price', 'Signal']
                st.dataframe(last_30_days.style.applymap(lambda _: 'font-weight: bold', subset=['Signal']).format({'Price': '${:.2f}'}))
                
                st.plotly_chart(plot_signals_and_price(df, ticker), use_container_width=True)
            else:
                st.warning("Unable to calculate signals. Please try different parameters or a different cryptocurrency.")
        else:
            st.warning("No data available for the selected parameters. Please try a different cryptocurrency, date range, or contract type.")

except Exception as e:
    handle_error(e)

st.write("Note: This app uses a simple momentum strategy based on moving averages. The strategy parameters are measured in bars, not days. Always do your own research before making investment decisions. None of this is financial advice.")