import pandas as pd
import numpy as np
import yfinance as yf


def fetch_bitcoin_data(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetching historical Bitcoin price data from Yahoo Finance (BTC-USD).
    Returns a DataFrame with Date index and OHLCV columns."""
    df = yf.download("BTC-USD", start=start_date, end=end_date, progress=False)
    df.dropna(inplace=True) 
    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Adding technical indicators like moving average and RSI) to the DataFrame."""
    # Example: 14-day Relative Strength Index (RSI)
    window = 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    # Example: 20-day moving average
    df['MA20'] = df['Close'].rolling(window=20).mean()
    # We may forward-fill any NaN values resulting from indicators at the start
    df.fillna(method='bfill', inplace=True)
    return df

