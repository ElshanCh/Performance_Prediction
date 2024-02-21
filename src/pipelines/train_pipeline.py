from src.logger import *
from src.exception import CustomException
import os
import pandas as pd
import sys
from neuralprophet import NeuralProphet, set_log_level, set_random_seed
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.components.model_training import *
from src.components.data_ingestion import DataLoader
from src.components.data_transformation import DataTransformer
from src.utils import *
import mlflow
from src.mlflow_utils import *
from src.utils import *
import logging

class HyperparameterTuning:
    def __init__(self,hyperparameters):
        self.hyperparameters = hyperparameters
    def run(self):
        set_log_level("ERROR")
        set_random_seed(42)
        logging.info("Loading Train & Test Datasets")

        train_data = read_csv_and_convert_date('../../artifacts/train_data.csv')
        test_data = read_csv_and_convert_date('../../artifacts/test_data.csv')



        experiment_name = "Neural_Prophet"
        artifact_location = experiment_name+"_artifacts"

        experiment_id = create_mlflow_experiment(
            experiment_name=experiment_name,
            artifact_location=artifact_location,
            tags={"env": "dev", "version": "1.0.0"}
        )

        experiment = get_mlflow_experiment(experiment_name=experiment_name)
        print("Experiment Name: {}".format(experiment.name))
        print("Experiment ID: {}".format(experiment.experiment_id))

        logging.info("HyperParameter Tuning started")
        logging.info(f"HyperParameters dict: {hyperparameters}")

        param_combinations = GridSearch(hyperparameters, verbose=False)
        print(f"Number of Possible Parameter Combinations: {len(param_combinations)}")
        logging.info(f"Number of Possible Parameter Combinations: {len(param_combinations)}")

        for model_params in param_combinations:
            with mlflow.start_run(run_name="hyperparameter_tuning_v1", experiment_id=experiment.experiment_id) as run:
                mlflow.log_params(model_params)

                model = NeuralProphet(**model_params)
                model.add_country_holidays("IT")

                model.fit(train_data, freq='D', )
                future = model.make_future_dataframe(train_data, periods=len(test_data))
                forecast = model.predict(future)

                metrics = compute_forecast_metrics(forecast, test_data)
                mlflow.log_metrics(metrics)

                artifact_location = mlflow.get_artifact_uri()
                if not os.path.exists(artifact_location):
                    os.makedirs(artifact_location)
                plt = generate_comparison_plot(forecast, test_data)
                plot_file = os.path.join(artifact_location, "comparison_plot_matplotlib.png")
                plt.savefig(plot_file)
                plt.close()
                mlflow.log_artifact(plot_file)

if __name__ == "__main__":
    hyperparameters = {
    'growth': ['linear'],
    'seasonality_mode': ['additive'],
    'learning_rate': [0.003],
    'n_changepoints': [0, 1],
    'changepoints_range': [0.95],
    # 'yearly_seasonality': ["auto", True],
    # 'weekly_seasonality': ["auto", True],
    # 'daily_seasonality': ["auto", True],
    'epochs': [150],
    'trend_reg': [5]
    }
    hyperparameter_tuning = HyperparameterTuning(hyperparameters)
    hyperparameter_tuning.run()
