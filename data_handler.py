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

def prepare_data(start_date: str, end_date: str, split_ratio: float = 0.8):
    '''
    Here we will fetch the Bitcoin data and prepare training and testing datasets.
    The split_ratio determines the proportion of data used for training vs testing.
    '''
    df = fetch_bitcoin_data(start_date, end_date)
    df = add_technical_indicators(df)

    ## Converting some features to percentage
    df['Close_pct'] = df['Close'].pct_change() * 100
    
    # Define features to use in the environment state later
    feature_columns = ['Close', 'MA20', 'RSI', 'Close_pct']
    data = df[feature_columns].copy()

    ## Split into training and testing datasets
    split_idx = int(len(data) * split_ratio)
    train_data = data.iloc[:split_idx].reset_index(drop=True)
    test_data = data.iloc[split_idx:].reset_index(drop=True)
    return train_data, test_data

if __name__ == "__main__":
    train, test = prepare_data("2016-01-01", "2023-01-01")
    print(f"Training Data Sample: {train.head}, Training Shape: {train.shape}\n")
    print(f"Testing Data Sample: {test.head}, Testing Shape: {test.shape}\n")
    print("Testing Data Sample:\n", test.head())
    print(train.head(3).T)