from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.dropdown import DropDown
from kivy.lang import Builder
from joblib import load
import pandas as pd

Builder.load_file('stock_gui.kv')

class StockApp(App):
    def build(self):
        self.dropdown = DropDown()
        return StockPredictor(self.dropdown)
    
class StockPredictor(BoxLayout):
    def __init__(self, dropdown, **kwargs):
        super().__init__(**kwargs)
        self.model = None
        self.ticker = ""
        self.model_name = "linear_regression"  # Default model
        
        self.dropdown = dropdown  # Store the passed dropdown reference
        self.create_dropdown_buttons()

    def create_dropdown_buttons(self):
        for model in ['linear_regression', 'random_forest']:
            btn = Button(text=model, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn: self.dropdown.select(btn.text))
            self.dropdown.add_widget(btn)

    def load_model(self, ticker):
        """Load the model for the given ticker."""
        try:
            self.model = load(f'./models/{ticker}_{self.model_name}.joblib')
            return True
        except FileNotFoundError:
            return False

    def predict(self):
        """Predict the future stock price for the given ticker."""
        if self.model and self.ticker:
            # Load the latest data for prediction
            df = pd.read_csv(f'./data/processed/{self.ticker}_processed.csv', index_col='Date', parse_dates=True)
            last_row = df.iloc[-1]
            features = [[last_row['20-day MA'], last_row['50-day MA']]]
            future_price = self.model.predict(features)[0]
            return future_price
        return None

    def on_predict(self):
        """Handle prediction button click."""
        self.ticker = self.ids.ticker_input.text.strip().upper()
        if self.load_model(self.ticker):
            future_price = self.predict()
            if future_price is not None:
                self.ids.result_label.text = f"Predicted price for {self.ticker}: ${future_price:.2f}"
            else:
                self.ids.result_label.text = "Prediction failed. Check your data."
        else:
            self.ids.result_label.text = "Model not found for this ticker."

    def on_model_select(self, model_name):
        """Handle model selection."""
        self.model_name = model_name
        self.ids.model_button.text = f"Model: {model_name}"


if __name__ == '__main__':
    StockApp().run()
