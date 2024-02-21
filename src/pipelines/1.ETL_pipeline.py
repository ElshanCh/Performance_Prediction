from src.components.data_ingestion import DataLoader
from src.components.data_transformation import DataTransformer
from src.utils import plot_data, save_to_csv
from src.exception import CustomException
from src.logger import logging
import sys

def run_etl_pipeline(data_directory='../../Data/train_data/', split_date='2023-12-31'):
    try:
        ####################################################################################################
        # Data Loading
        ####################################################################################################
        data_loader = DataLoader(data_directory)
        df_full = data_loader.load_all_parquet_files()

        ####################################################################################################
        # Data Transformation
        ####################################################################################################
        transformer = DataTransformer(df_full)
        df_daily_sum = transformer.transform_data()
        train_data, test_data = transformer.split_data(df_daily_sum, split_date=split_date)
        plot_data(df_daily_sum)

        ####################################################################################################
        # Data Saving
        ####################################################################################################
        save_to_csv(train_data, "../artifacts/train_data.csv")
        save_to_csv(test_data, "../artifacts/test_data.csv")
    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)
        raise

# Running the ETL pipeline
if __name__ == "__main__":
    run_etl_pipeline()
