import pandas as pd
import os

def load_data(tickers):
    """Load stock data from CSV files."""
    data_frames = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/raw/{ticker}_data.csv', index_col='Date', parse_dates=True)
            data_frames[ticker] = df
            print(f"Loaded data for {ticker}")
        except FileNotFoundError:
            print(f"Data file for {ticker} not found.")
    return data_frames

def preprocess_data(data_frames):
    """Clean and preprocess the stock data."""
    processed_data = {}
    for ticker, df in data_frames.items():
        # Drop rows with missing values
        df.dropna(inplace=True)
        
        # Calculate daily returns
        df['Daily Return'] = df['Adj Close'].pct_change()
        
        # Calculate moving averages (e.g., 20-day and 50-day)
        df['20-day MA'] = df['Adj Close'].rolling(window=20).mean()
        df['50-day MA'] = df['Adj Close'].rolling(window=50).mean()
        
        processed_data[ticker] = df
        print(f"Processed data for {ticker}")
    
    return processed_data

def save_processed_data(processed_data):
    """Save the processed data to CSV files."""
    os.makedirs('data/processed', exist_ok=True)
    for ticker, df in processed_data.items():
        df.to_csv(f'data/processed/{ticker}_processed.csv')
        print(f"Saved processed data for {ticker}")

if __name__ == "__main__":
    # Define your stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Use the same tickers as before

    # Load, preprocess, and save the data
    raw_data = load_data(tickers)
    processed_data = preprocess_data(raw_data)
    save_processed_data(processed_data)
