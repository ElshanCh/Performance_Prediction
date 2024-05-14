import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging
import sys
import os
import pandas as pd 
import itertools
import json
import plotly.graph_objs as go

####################################################################################################
# Plotting
####################################################################################################
def plot_data(df):
    try:
        logging.info("Plotting Numbers of Messages per Day")
        plt.figure(figsize=(15, 5))
        plt.plot(df['ds'], df['y'])
        plt.xlabel('Date')
        plt.ylabel('Number of Messages')
        plt.title('Numbers of Messages per Day')
        plt.show()
    
    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)
        raise


####################################################################################################
# Saving to csv
####################################################################################################
def save_to_csv(df, file_path):
    """
    Save a Pandas DataFrame to a CSV file.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be saved.
        file_path (str): The file path where the CSV file will be saved.

    Returns:
        None
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"DataFrame saved to {file_path}")

    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)

# Example usage:
# Assuming df is your DataFrame and file_path is the path where you want to save the CSV file
# save_to_csv(df, 'output.csv')



####################################################################################################
# Reading and converting from csv
####################################################################################################
def read_csv_and_convert_date(file_path):
    """
    Read a CSV file into a Pandas DataFrame and convert the 'ds' column to datetime.

    Parameters:
        file_path (str): The file path of the CSV file to be read.

    Returns:
        pandas.DataFrame: The DataFrame read from the CSV file with the 'ds' column converted to datetime.
    """
    try:
        # Read CSV into DataFrame
        df = pd.read_csv(file_path)
        
        # Convert 'ds' column to datetime if it exists
        if 'ds' in df.columns:
            df['ds'] = pd.to_datetime(df['ds'])
        return df
    
    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)

    

# Example usage:
# Assuming file_path is the path where your CSV file is located
# df = read_csv_and_convert_date('input.csv')


####################################################################################################
# Custom GridSearch
####################################################################################################
def GridSearch(hyperparameters, verbose=True):
    try:
        hyperparameter_combinations = [dict(zip(hyperparameters.keys(), values)) for values in itertools.product(*hyperparameters.values())]
        total_combinations = len(hyperparameter_combinations)
        if verbose:
            print("Total combinations: ", total_combinations)
            # print(json.dumps(hyperparameter_combinations, indent=4))
        
        return hyperparameter_combinations

    except Exception as e:
        custom_exception = CustomException(e, sys)
        logging.error(custom_exception)



####################################################################################################
# Plotting test_data['y'] vs forecast['yhat1']
####################################################################################################

import matplotlib.pyplot as plt

def generate_comparison_plot(forecast, test_data, figsize=(30, 15)):
    # Create a figure with the specified size
    plt.figure(figsize=figsize)

    # Calculate difference
    test_data['difference'] = test_data["y"] - forecast["yhat1"]
    
    # Plot yhat1 and y
    plt.plot(forecast["ds"], forecast["yhat1"], label='yhat1', color='orange')
    plt.plot(test_data["ds"], test_data["y"], label='y', color='green')

    # Plot positive and negative differences
    positive_difference = test_data[test_data['difference'] > 0]
    negative_difference = test_data[test_data['difference'] < 0]
    plt.bar(positive_difference["ds"], positive_difference["difference"], color='blue', label='Positive Difference')
    plt.bar(negative_difference["ds"], negative_difference["difference"], color='red', label='Negative Difference')

    # Set title and labels
    plt.title('Comparison between yhat1 and y with difference highlighted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()  # Adjust layout to prevent clipping of labels

    return plt

