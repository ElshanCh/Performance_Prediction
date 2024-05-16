Project Description for Handover

**Project Title:** Hourly Forecasting of Message Volume and Anomaly Classification

**Development Environment:** WSL/Ubuntu-22.04
**Programming Language:** Python

---

**1. Hourly Forecasting of Message Volume (NMSG)**

**Objective:** Develop a model using the NeuralProphet algorithm to forecast the hourly volume of incoming messages to the servers based on historical data.

**Data Source:** Sourced from the S3 production "Performance" bucket, stored in "/Data/train_data/".

**Development Process:**
- **Proof of Concept (POC):** Initiated in "notebooks/NeuralProphet_NMSG_testing.ipynb" before modularizing into the "src" folder.
- **Folder Structure:** 
  - **"src/components":** Houses defined components utilized within specific pipelines.
  - **"src/pipelines":** Contains key pipelines for the project

- **Script References:**

  - **1. ETL_pipeline.py:** This script facilitates Export-Transform-Load operations on the data. The `date_string` variable within the script can be modified to specify the date up to which you intend to train the model. All data beyond this date will be reserved for testing purposes.
      ```python
      # Running the ETL pipeline
      if __name__ == "__main__":
          from datetime import datetime

          columns=["DT", "NMSG"]
          # columns=["DT", "TO500RT","SERVER"]
          if "NMSG_1" in columns:
              run_etl_pipeline(columns=columns)
          else:
              date_string = '2023-12-31 23'  # Modify this date as needed
              date_format = '%Y-%m-%d %H'
              datetime_object = datetime.strptime(date_string, date_format)
              run_etl_pipeline(split_date=datetime_object, columns=columns)
      ```
  - **2. Hyperparam_tuning.py:** Conducts hyperparameter tuning using MLflow for model optimization.
  - **3. Predict_pipeline.py:**
  Utilized for making predictions on the selected model post hyperparameter tuning. In this script, the trained data is reused for making future predictions.
      ```python
      csv_file = "../../artifacts/v2/DT_NMSG/train_data.csv"
      data = predictor.load_csv_data(csv_file)
      # Assuming test_data length is required for forecasting periods
      future_forecast = predictor.make_predictions(data, 1024)
      ```
    The number `1024` represents the quantity of data points for future predictions and can be adjusted as needed.

**Future Development**:
  - The model should be periodically retrained in order to correspond to new trends and fresh data
  - Should be developed Endpoint  for the model to be able to call the model and make prediction
  - Should be decided where to deploy the model.

**Note:** Uncommenting "-e ." in the initial run of requirements.txt activates setup.py, facilitating the use of Python modules from different locations. Following this, it can be commented out.

---

**2. Anomaly Classification**

**Objective:** Develop an anomaly detection model that utilizes the LSTM RNN algorithm to predict anomalies based on historical patterns, using the last 24 hours of performance data.

**Development Process:**
- **Proof of Concept (POC):** Currently in the POC stage, with development stages documented in the following notebooks:
- **Folder Structure:** 
  - **notebooks/LSTM_Classification_data_preparation.ipynb:** Details the data preprocessing stage.
  - **notebooks/LSTM_Classification_model.ipynb:** Focuses on the implementation of the LSTM model itself.

- **Script References:**
  - **notebooks/LSTM_Classification_data_preparation.ipynb:** 
  Here there has been the same raw data for `NMSG` that has been used also in the upper model + some other data (`SERVER`,`TO500RT`)

    1. You may see that there are `NMSG` and `TO500RT` values for 3 different `SERVER`.
    The idea was not to separte the serverse, that is why the values of `NMSG` and `TO500RT` have been aggregated (sum) for the same `DT`.
    2. Next based on the values of `NMSG` and `TO500RT` there has been calculated new KPI `ANOMALY` (which is binary). This KPI will be used as a target column for the column (i.e we will try to predict it).
        ```python
        df_hourly['ANOMALY'] = df_hourly.apply(lambda row: 'YES' if (row['TO500RT'] / row['NMSG'] * 100 > 0.01) else 'NO', axis=1)
        ```
    3. Before proceeding with the model we need still add some more data that could be helpful for the model building, i.e adding more attributes such as `CPU_MAX`,`CPU_AVG`. These columns unlike the `NMSG` data were not obtained from S3 bucket, but were downloaded from the Performance KPIs UI by hand and loaded into `../Data/CPU` folder.

    4. Next the tables with `CPU_MAX`,`CPU_AVG` and `NMSG` get joined on the `DT` column.

    5. The final table gets sinked into `../artifacts/DATE_NMSG_CPUMAX_CPUAVG_ANOMALY/df_input_1.csv`

  - **notebooks/LSTM_Classification_model.ipynb:**

    1. Time-series data preparation
        ```python
        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            """
            Frame a time series as a supervised learning dataset.

            Arguments:
            data: Sequence of observations as a list or NumPy array.
            n_in: Number of lag observations as input (X).
            n_out: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.

            Returns:
            Pandas DataFrame of series framed for supervised learning.
            """
        ```

        This function converts a time series dataset into a supervised learning format suitable for training machine learning models. It frames the dataset by shifting the input and output sequences based on the specified number of lag observations.

        - `data`: The input time series data, either as a list or NumPy array.
        - `n_in`: The number of lag observations to use as input features (X).
        - `n_out`: The number of observations to predict as output (y).
        - `dropnan`: A boolean indicating whether to drop rows with NaN values after framing.

        The function returns a Pandas DataFrame where each row represents a time step, and each column corresponds to a lagged observation or a future time point prediction.

        In the context of time series analysis:
        - **Lag Observations (Input Sequence)**: These are past observations used as features to predict future values. For example, `var1(t-1)` represents the value of variable 1 at the previous time step.
        - **Output Sequence (Forecast Sequence)**: These are future observations to be predicted. For example, `var1(t)` represents the value of variable 1 at the current time step, while `var1(t+1)` represents the value at the next time step.

        By specifying `n_in` and `n_out`, you control the number of past observations used for prediction (`n_in`) and the number of future observations to predict (`n_out`). This technique allows the conversion of a time series problem into a supervised learning problem suitable for training machine learning models like LSTM (Long Short-Term Memory) networks.

    2. After Analysing the data it has been revieled that data in `ANOMALY` column is imbalanced. 
    `"0" with 20k+` vs `"1" with 2k data`
    So it has been decided to drop 17k of data with "0" class (this can be regulated in the futre).

    3. At the stage of the defining the Layer for the NN model it has been observed that more you add Layers the model starts to function worse,  so avoid adding extra Layers.

**Future Development**:
  - The model should be periodically retrained in order to correspond to new trends and fresh data.
  - Should be decided and defined from where the data can be streamed seamlessly both for `NMSG` and `CPU_MAX`,`CPU_AVG.
  - Should be developed Endpoint  for the model to be able to call the model and make prediction
  - Should be decided where to deploy the model.

---
