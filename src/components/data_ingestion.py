from src.exception import CustomException
import os
import pandas as pd
import sys
from src.logger import logging


import os
import logging
import pandas as pd

class DataLoader:
    def __init__(self, directory, columns = []):
        self.directory = directory
        self.columns = columns

    def load_parquet_files_generator(self):
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.parquet'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_parquet(file_path, engine='fastparquet', columns=self.columns)
                        yield df
                    except Exception as e:
                        logging.info(f"Incompatible file has been detected in the following path: {file_path}. Error: {e}")

    def load_all_parquet_files(self):
        logging.info(f"Data loading from {self.directory} has started")

        df_generator = self.load_parquet_files_generator()
        df_full = pd.concat(df_generator, ignore_index=True)

        logging.info("Data loading has finished")
        logging.info(f"Loaded data length: {len(df_full)}")

        return df_full



# Example usage:
if __name__ == "__main__":
    data_directory = '../../Data/train_data/'
    data_loader = DataLoader(data_directory, columns=["DT","NMSG", "TO500RT"])
    df_full = data_loader.load_all_parquet_files()
    df_full.head()
