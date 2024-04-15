from src.components.data_ingestion import DataLoader
from src.components.data_transformation import DataTransformer
from src.utils import plot_data, save_to_csv
from src.exception import CustomException
from src.logger import logging
import sys
import os


def run_etl_pipeline(data_directory='../../Data/train_data/', columns = [], split_date='2023-12-31'):
    try:
        ####################################################################################################
        # Data Loading
        ####################################################################################################
        data_loader = DataLoader(data_directory, columns =columns)
        df_full = data_loader.load_all_parquet_files()

        ####################################################################################################
        # Data Transformation
        ####################################################################################################
        transformer = DataTransformer(df_full, columns=columns)
        df_daily_sum = transformer.transform_data()
        train_data, test_data = transformer.split_data(df_daily_sum, split_date=split_date)
        plot_data(df_daily_sum)

        ####################################################################################################
        # Data Saving
        ####################################################################################################
        # Define the folder path
        folder_path = "../../artifacts/{}".format('_'.join(columns))

        # Create the directory if it does not exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save train_data to CSV
        save_to_csv(train_data, "{}/train_data.csv".format(folder_path))

        # Save test_data to CSV
        save_to_csv(test_data, "{}/test_data.csv".format(folder_path))

    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)
        raise

# Running the ETL pipeline
if __name__ == "__main__":
    from datetime import datetime

    # columns=["DT", "NMSG"]
    columns=["DT", "TO500RT","SERVER"]
    if "NMSG" in columns:
        run_etl_pipeline(columns=columns)
    else:
        date_string = '2023-12-31 23:59:59'
        date_format = '%Y-%m-%d %H:%M:%S'
        datetime_object = datetime.strptime(date_string, date_format)
        run_etl_pipeline(split_date=datetime_object, columns=columns)
