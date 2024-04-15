import pandas as pd
from src.exception import CustomException
from src.logger import logging
import os
import sys
from datetime import datetime


class DataTransformer:
    def __init__(self, df_full, columns):
        self.df_full = df_full
        self.columns = columns
    
    def transform_data(self):
        try:
            # Rename columns
            logging.info("Renaming the columns of the DataFrame")
            self.df_full.rename(columns={'DT': 'ds', self.columns[1]: 'y'}, inplace=True)

            # Convert "DS" column to datetime if it's not already
            logging.info("Conversion of DS column to datetime")
            self.df_full['ds'] = pd.to_datetime(self.df_full['ds'])

            if "NMSG" in self.columns:
                # Group by date and sum the "NMSG" values
                logging.info("Grouping the data by date")
                df_daily_sum = self.df_full.groupby(self.df_full['ds'].dt.date)['y'].sum().reset_index()
                return df_daily_sum

            else: 
                return self.df_full
            
        except Exception as e:
            custom_exception = CustomException(e, sys)
            logging.error(custom_exception)
            raise
    

    
    def split_data(self, df, split_date):
        try:
            if "NMSG" in self.columns:
                split_date = datetime.strptime(split_date, '%Y-%m-%d').date()

            logging.info(f"Splitting the data into training and testing sets. Defined Split Date: {split_date}")
            train_data = df[df['ds'] <= split_date]
            train_data.loc[:, 'ds'] = pd.to_datetime(train_data['ds'].copy())
            test_data = df[df['ds'] > split_date]
            test_data.loc[:, 'ds'] = pd.to_datetime(test_data['ds'].copy())

            return train_data, test_data
        
        except Exception as e:
            custom_exception = CustomException(e, sys)
            logging.error(custom_exception)
            raise

# Example usage:
if __name__ == "__main__":
    # Example DataFrame
    df_full = pd.DataFrame({
        'ds': ['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02'],
        'y': [1, 2, 3, 4]
    })

    transformer = DataTransformer(df_full)
    df_daily_sum = transformer.transform_data()
    transformer.plot_data(df_daily_sum)
    train_data, test_data = transformer.split_data(df_daily_sum, split_date = '2023-12-31')
