from neuralprophet import load,set_log_level,set_random_seed
import pandas as pd
import matplotlib.pyplot as plt

class NeuralProphetPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.loaded_model = None
        set_log_level("ERROR")
        set_random_seed(42)

    def load_model(self):
        # Load the saved model
        self.loaded_model = load(self.model_path)
        return self.loaded_model

    def load_csv_data(self, csv_file):
        # Load CSV data from provided location
        return pd.read_csv(csv_file)

    def make_predictions(self, data, periods):
        # Make future dataframe for predictions
        future = self.loaded_model.make_future_dataframe(data, periods=periods)
        # Forecast
        forecast = self.loaded_model.predict(future)
        return forecast

    def plot_forecast(self, forecast):
        # Plot forecast
        plt.plot(forecast["ds"], forecast["yhat1"], label="Forecast")
        plt.xlabel("Date")
        plt.ylabel("Forecasted Values")
        plt.title("Forecast")
        plt.legend()
        plt.show()

# Example usage:
if __name__ == "__main__":
    model_path = "Neural_Prophet_artifacts/128b15b75a854e8189aaae862ae6bacf/artifacts/np-model.np"
    predictor = NeuralProphetPredictor(model_path)
    predictor.load_model()
    # Load CSV data
    csv_file = "../../artifacts/train_data.csv"
    data = predictor.load_csv_data(csv_file)
    # Assuming test_data length is required for forecasting periods
    future_forecast = predictor.make_predictions(data, 30)
    predictor.plot_forecast(future_forecast)
