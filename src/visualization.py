import pandas as pd
import matplotlib.pyplot as plt
import os

def load_processed_data(tickers):
    """Load processed stock data from CSV files."""
    data_frames = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'data/processed/{ticker}_processed.csv', index_col='Date', parse_dates=True)
            data_frames[ticker] = df
            print(f"Loaded processed data for {ticker}")
        except FileNotFoundError:
            print(f"Processed data file for {ticker} not found.")
    return data_frames

def plot_stock_prices(data_frames):
    """Plot historical stock prices."""
    plt.figure(figsize=(14, 7))
    for ticker, df in data_frames.items():
        plt.plot(df['Adj Close'], label=ticker)
    plt.title('Stock Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid()
    plt.show()

def plot_daily_returns(data_frames):
    """Plot daily returns."""
    plt.figure(figsize=(14, 7))
    for ticker, df in data_frames.items():
        plt.plot(df['Daily Return'], label=ticker, alpha=0.5)
    plt.title('Daily Returns')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.grid()
    plt.show()

def plot_moving_averages(data_frames):
    """Plot moving averages."""
    plt.figure(figsize=(14, 7))
    for ticker, df in data_frames.items():
        plt.plot(df['20-day MA'], label=f'{ticker} 20-day MA')
        plt.plot(df['50-day MA'], label=f'{ticker} 50-day MA', linestyle='--')
    plt.title('Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Define your stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Use the same tickers as before

    # Load and visualize the processed data
    processed_data = load_processed_data(tickers)
    plot_stock_prices(processed_data)
    plot_daily_returns(processed_data)
    plot_moving_averages(processed_data)
