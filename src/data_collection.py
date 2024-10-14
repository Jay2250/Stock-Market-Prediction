import yfinance as yf
import os

def fetch_stock_data(tickers, start_date, end_date):
    """Fetch historical stock data for a list of tickers."""
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            data.to_csv(f'data/raw/{ticker}_data.csv')  # Save to raw data directory
            print(f"Saved data for {ticker}")
        except Exception as e:
            print(f"Failed to download data for {ticker}: {e}")

def fetch_financials(tickers):
    """Fetch financial statements for a list of tickers."""
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            financials.to_csv(f'data/raw/{ticker}_financials.csv')
            print(f"Saved financials for {ticker}")
        except Exception as e:
            print(f"Failed to download financials for {ticker}: {e}")

if __name__ == "__main__":
    # Define your stock tickers and date range
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Add more tickers as needed
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    
    # Ensure the raw data directory exists
    os.makedirs('data/raw', exist_ok=True)
    
    # Fetch the data
    fetch_stock_data(tickers, start_date, end_date)
    fetch_financials(tickers)
