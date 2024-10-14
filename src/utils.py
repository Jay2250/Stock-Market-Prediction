import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(tickers, folder='data/processed/'):
    """Load processed stock data from CSV files."""
    data_frames = {}
    for ticker in tickers:
        try:
            df = pd.read_csv(f'{folder}{ticker}_processed.csv', index_col='Date', parse_dates=True)
            data_frames[ticker] = df
            print(f"Loaded processed data for {ticker}")
        except FileNotFoundError:
            print(f"Processed data file for {ticker} not found.")
    return data_frames

def prepare_data(df):
    """Prepare the data for modeling."""
    df = df.dropna()  # Drop rows with missing values
    X = df[['20-day MA', '50-day MA']]  # Features
    y = df['Adj Close']  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(predictions, y_test):
    """Evaluate model performance."""
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return mae, rmse

def save_model(model, filename):
    """Save a trained model to a file."""
    from joblib import dump
    dump(model, filename)
