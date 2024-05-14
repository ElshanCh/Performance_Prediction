from src.logger import *
from src.exception import CustomException
import os
import sys
from neuralprophet import NeuralProphet, set_log_level, set_random_seed, save
from src.components.model_training import *
from src.utils import *
import mlflow
from src.mlflow_utils import *
from src.utils import *
import logging
import time


class HyperparameterTuning:
    def __init__(self,hyperparameters,file_path, experiment_name, local =True):
        self.hyperparameters = hyperparameters
        self.experiment_name = experiment_name
        self.file_path = file_path
        # Set variable 'local' to True if you want to run this notebook locally
        # self.local = local
        # # Set our tracking server uri for logging
        # mlflow.set_tracking_uri(uri="http://127.0.0.1:8080") if self.local else None

    def run(self):
        try: 
            set_log_level("ERROR")
            set_random_seed(42)
            logging.info("Loading Train & Test Datasets")

            train_data = read_csv_and_convert_date(f'../../{self.file_path}/train_data.csv')
            test_data = read_csv_and_convert_date(f'../../{self.file_path}/test_data.csv')
            print(len(train_data))

            # train_data = train_data[["ds","y"]][train_data["SERVER"]=="simislnxnss00.si.it"]
            # test_data = test_data[["ds","y"]][test_data["SERVER"]=="simislnxnss00.si.it"]
            print(len(train_data))
            train_data.drop_duplicates(inplace=True)
            test_data.drop_duplicates(inplace=True)

            print(len(train_data))
            # print(train_data[train_data.duplicated()])


            # experiment_name = "Neural_Prophet_DT_TO500RT"
            artifact_location = self.experiment_name +"_artifacts"

            experiment_id = create_mlflow_experiment(
                experiment_name=self.experiment_name ,
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
                    
                    start = time.time()
                    model = NeuralProphet(**model_params)
                    model.add_country_holidays("IT")

                    model.fit(train_data, freq='h')
                    end = time.time()
                    mlflow.log_metric("duration", end - start)
                    
                    future = model.make_future_dataframe(train_data, periods=len(test_data))
                    forecast = model.predict(future)

                    metrics = compute_forecast_metrics(forecast, test_data)
                    mlflow.log_metrics(metrics)

                    artifact_location = mlflow.get_artifact_uri()
                    if not os.path.exists(artifact_location):
                        os.makedirs(artifact_location)
                    
                    # Saving the model in artifacts
                    model_file = "np-model.np"
                    model_path = os.path.join(artifact_location, model_file)
                    save(model, model_path)

                    plt = generate_comparison_plot(forecast, test_data)
                    # Saving the plot in artifacts
                    plt_file = "comparison_plot_matplotlib.png"
                    plot_path = os.path.join(artifact_location, plt_file)
                    plt.savefig(plot_path)
                    plt.close()
                    
                    mlflow.log_artifact(plot_path)
        except Exception as e:
            custom_exception = CustomException(e, sys)
            logging.error(custom_exception)


if __name__ == "__main__":
    hyperparameters = {
    # 'growth': ['linear'],
    'seasonality_mode': ['additive'],
    # 'seasonality_mode': ['additive','multiplicative'],
    # 'changepoints_range': [0.95,0.1],
    # 'changepoints_range': [0.1],
    'learning_rate': [0.9,0.01,0.001],
    # 'learning_rate': [0.001, 0.01],
    'yearly_seasonality': [True],
    # 'yearly_seasonality': ["auto", True], 
    'weekly_seasonality': ["auto"],
    # 'weekly_seasonality': ["auto", True],
    'daily_seasonality': [True],
    # 'daily_seasonality': ["auto", True],
    # 'n_changepoints': [0, 1],
    'epochs': [200],
    "loss_func" : ["MSE"],
    # 'epochs': [150,250],
    # 'trend_reg': [5]
    "batch_size" : [256],
    "ar_layers": [10],  # Number of hidden layers
    }

    # data_set_name = "DT_CPU_MAX"
    data_set_name = "DT_NMSG"
    file_path = f"artifacts/v2/{data_set_name}"
    experiment_name = f"Neural_Prophet_{data_set_name}_hourly"
    hyperparameter_tuning = HyperparameterTuning(hyperparameters,file_path,experiment_name)
    hyperparameter_tuning.run()
