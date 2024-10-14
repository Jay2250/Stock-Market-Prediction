import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import matplotlib.pyplot as plt

from joblib import dump

from sklearn.ensemble import RandomForestRegressor
from utils import load_data, prepare_data, evaluate_model, save_model



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

def prepare_data(df):
    """Prepare the data for modeling."""
    df = df.dropna()  # Drop rows with missing values
    X = df[['20-day MA', '50-day MA']]  # Features
    y = df['Adj Close']  # Target variable
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train the linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model performance."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    return predictions, y_test

def predict_future(model, df):
    """Make predictions for future stock prices."""
    last_row = df.iloc[-1]
    future_data = pd.DataFrame(columns=['20-day MA', '50-day MA'])
    
    # Use the latest moving averages for prediction
    future_data.loc[0] = [last_row['20-day MA'], last_row['50-day MA']]
    future_prediction = model.predict(future_data)

    return future_prediction[0]

def visualize_predictions(df, future_price):
    """Visualize actual prices and predicted future price."""
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['Adj Close'], label='Actual Prices', color='blue')
    
    # Add the predicted future price to the last date
    plt.scatter(df.index[-1] + pd.Timedelta(days=1), future_price, color='orange', label='Predicted Future Price', s=100, zorder=5)
    
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()


def train_random_forest(X_train, y_train):
    """Train the Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_random_forest_model(model, X_test, y_test):
    """Evaluate the Random Forest model performance."""
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Random Forest - Mean Absolute Error: {mae:.2f}")
    print(f"Random Forest - Root Mean Squared Error: {rmse:.2f}")
    return predictions


if __name__ == "__main__":
    # Define your stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL']  # Updated list of tickers

    # Load the processed data
    processed_data = load_processed_data(tickers)
    
    for ticker, df in processed_data.items():
        # Prepare the data
        X_train, X_test, y_train, y_test = prepare_data(df)

        # Train the model
        model = train_model(X_train, y_train)
        # Train the Random Forest model
        rf_model = train_random_forest(X_train, y_train)

        # Evaluate the Random Forest model
        rf_predictions = evaluate_random_forest_model(rf_model, X_test, y_test)
        
        # Evaluate the model
        predictions = model.predict(X_test)
        mae, rmse = evaluate_model(model, X_test, y_test)
        print(f"{ticker} - Mean Absolute Error: {mae}")
        print(f"{ticker} - Root Mean Squared Error: {rmse}")

        # Save the trained model
        save_model(model, f'models/{ticker}_linear_regression.joblib')
        
        # Save the trained models
        dump(model, f'models/{ticker}_linear_regression.joblib')
        dump(rf_model, f'models/{ticker}_random_forest.joblib')


        # Evaluate the model
        predictions, y_test = evaluate_model(model, X_test, y_test)

        # Make a prediction for the next day's price
        future_price = predict_future(model, df)
        print(f"Predicted future price for {ticker}: {future_price:.2f}")

        # Visualize predictions
        visualize_predictions(df, future_price)