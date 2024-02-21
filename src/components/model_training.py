import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException
import sys
from src.logger import logging

def compute_forecast_metrics(forecast, test_data):
    try:
        logging.info(r"Master precision metrics extraction")

        # Filter the forecast_df based on the condition
        filtered_forecast_df = forecast[forecast['ds'] > '2023-12-31'][["ds", "yhat1"]]

        # Joining filtered_forecast_df with test_data on "ds" and "DT" columns
        joined_df = pd.merge(filtered_forecast_df, test_data, left_on="ds", right_on="ds")

        # Resetting the index
        joined_df = joined_df[["ds", "yhat1", "y"]].reset_index(drop=True)

        # Creating a new column with the difference between "yhat1" and "NMSG"
        joined_df["difference"] = joined_df["y"] - joined_df["yhat1"]

        # Compute Mean Absolute Error (MAE)
        mae = mean_absolute_error(joined_df["y"], joined_df["yhat1"])

        # Compute Mean Squared Error (MSE)
        mse = mean_squared_error(joined_df["y"], joined_df["yhat1"])

        # Compute Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mse)

        # Compute Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((joined_df["y"] - joined_df["yhat1"]) / joined_df["y"])) * 100

        # Compute Symmetric Mean Absolute Percentage Error (SMAPE)
        smape = np.mean((np.abs(joined_df["yhat1"] - joined_df["y"])) / (
                    np.abs(joined_df["yhat1"]) + np.abs(joined_df["y"]))) * 200

        # Compute R-squared (R²) or Coefficient of Determination
        r_squared = r2_score(joined_df["y"], joined_df["yhat1"])

        # Create a dictionary
        metrics = {
            "Mean Absolute Error": mae,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Percentage Error": mape,
            "Symmetric Mean Absolute Percentage Error": smape,
            "R-squared": r_squared
        }

        logging.info(f"Metrics {metrics}")
        # for k, v in metrics.items():
        #     print(f"{k}: {v}")
        return metrics

    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)
