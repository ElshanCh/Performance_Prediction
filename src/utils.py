import matplotlib.pyplot as plt
from src.exception import CustomException
from src.logger import logging
import sys
import os



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