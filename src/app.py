import streamlit as st
import pandas as pd
from joblib import load

def load_model(model_path):
    return load(model_path)

st.title("Stock Price Prediction")

# User input for stock ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if st.button("Predict"):
    # Load the trained model
    model = load_model(f'models/{ticker}_linear_regression.joblib')
    
    # Here you would prepare your input data for prediction
    # For example, using the latest available data:
    # latest_data = ... (load your latest data logic)
    # future_price = model.predict(latest_data)
    
    # Display the prediction
    # st.write(f"Predicted future price for {ticker}: {future_price}")
