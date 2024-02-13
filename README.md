# Time Series Analysis

There are two main types of time series:

- Univariate Time Series
- Multivariate Time Series

**Univariate Time Series -** there is only one variable being observed or measured at successive points in time (weather prediction vs time). 

- Techniques commonly used for modeling and forecasting univariate time series:
  - autoregression (AR), 
  - moving average (MA), 
  - autoregressive integrated moving average (ARIMA),
  - exponential smoothing  

**Multivariate Time Series -** there are two or more variables being observed or measured at successive points in time (temperature and humidity over time). 

- Multivariate time series analysis aims to capture the relationships between different variables and understand how changes in one variable influence the behavior of others. In multivariate time series analysis, it's important to consider both the individual behavior of each variable and the collective dynamics of the system.
- Modeling multivariate time series often requires techniques that can account for dependencies between variables, such as vector autoregression (VAR), vector autoregressive moving average (VARMA), and state space models.
- In multivariate time series analysis, it's important to consider both the individual behavior of each variable and the collective dynamics of the system.

Time Series could be used for the following purposes:

- **Forecasting** - predict future behavior.
- **Imputation** - predict past behavior. Could be useful to fulfill gaps within your data or collect data for time before we had.
- **Anomaly Detection -** identifying observations or patterns in data that deviate significantly from the norm or expected behavior.
- **Analyze to spot patterns in data that determine what generated the series itself**. Example, analyze sound waves to spot words in them that could be used in NN for speech recognition.


## Steps to develop forecasting:
### Step 1: Assess Stationarity
**Stationarity**: Stationarity is an essential property of time series data. A stationary time series has constant mean, variance, and autocovariance over time. Stationary data simplifies analysis and modeling.<br>
**What You Do**: Check if the data’s mean and variance are consistent over time. <br>
**Why It Matters**: Stationarity is pivotal for certain statistical models to yield reliable forecasts.

### Step 2: Inspect for Trends
**Trend**: Trend refers to the long-term, non-periodic component of a time series. It indicates the general direction in which the series is moving. Trends can be linear or nonlinear.<br>
**What You Do**: Look for long-term increase or decrease patterns in your data. <br>
**Why It Matters**: Understanding trends helps predict future movements and adjust strategies accordingly.



### Step 3: Identify Seasonality
**Seasonality**: Seasonality represents patterns that repeat at fixed intervals within a time series. It could be daily, weekly, monthly, or any other periodicity. Identifying and modeling seasonality is crucial for accurate forecasting.<br>
**What You Do**: Detect repeating patterns or cycles over specific intervals. <br>
**Why It Matters**: Recognizing seasonality allows for more nuanced forecasts, anticipating peaks and troughs tied to time.

### Step 4: Examine Autocorrelation
**Autocorrelation**: Autocorrelation measures the relationship between observations at different time lags. It helps identify patterns and dependencies within a time series. <br>
Commonly used tools for analyzing autocorrelation:
  - Autocorrelation plots, 
  - autocorrelation function (ACF), 
  - and partial autocorrelation function (PACF).<br>

Time series that we encounter in real life probably will have a bit of each of this features: Trend + Seasonality + Autocorrelation + Noise.<br>
**What You Do**: Determine if current values in the series are influenced by past values. <br>
**Why It Matters**: Knowing the degree of autocorrelation can signal the predictability of the series and help refine models.



### Step 5: Decompose the Series
**Decomposition**: Time series decomposition breaks down a series into its constituent components, typically trend, seasonality, and residual. This technique can aid in understanding and modeling the various elements of a time series.<br>
**What You Do**: Break down the data into trend, seasonal, and irregular components. <br>
**Why It Matters**: Isolating components can simplify the series and clarify underlying patterns for better modelling.



### Step 6: Apply Differencing
**Differencing:** Differencing is a pivotal technique in time series analysis, specifically designed to tackle the challenges of non-stationarity. It focuses on removing trends and cycles by calculating the difference between consecutive data points.<br>
**What You Do**: Use differencing to stabilize the mean of the series by removing changes at a lagged interval. <br>
**Why It Matters**: This process can help achieve stationarity, crucial for various forecasting methods.

### Step 7: Seasonal Adjustment
**Seasonal Adjustment**: Seasonal adjustment involves removing the seasonal component from a time series to focus on the underlying trend and residual. This step can improve the modeling process and reveal more meaningful insights.<br>
**What You Do**: Extract and remove effects that repeat at the same period every cycle, such as quarterly fiscal reports or yearly tax changes, which affect stock performances. This is done to isolate the non-seasonal trend and irregular components that are unique and irregular, thus providing a clearer picture.<br>
**Why It Matters**: Just as a seasoned sailor knows how to read the seasonal winds for a successful voyage, adjusting for seasonality in your data allows you to chart a course that looks beyond the cyclical tides and into the undercurrents that drive long-term success.

### Step 8: Partitioning
**Partitioning:** Partitioning refers to the division of your time series data into distinct subsets for the purposes of model training, validation, and testing. In time series analysis, two common partitioning techniques are:
- **Fixed Partitioning:** Involves splitting the dataset into predefined segments, such as training, validation, and testing sets, typically without overlap.
- **Roll-Forward Partitioning**: Starts with a short training period and gradually increases it over time, continually refining the model as new data becomes available.<br>
**What You Do**: Choose between *Fixed Partitioning* or *Roll-Forward Partitioning* based on the specific requirements of your time series analysis.<br>
**Why It Matters**: In both cases, partitioning enables you to assess the efficacy of your model accurately, leading to more informed decision-making and better forecasting abilities. Just as a skilled navigator adjusts their course based on changing conditions, partitioning empowers you to adapt and refine your modeling strategies to navigate the complexities of time series data effectively.



### Step 9: Time Series Cross-Validation
Once you have partitioned your data using one of the above-mentioned techniques, you may apply time-series cross-validation. 
**Cross-Validation:** Cross-validation is a critical step in assessing the generalizability of your model. Techniques like k-fold cross-validation help in estimating the performance of the model on unseen data and mitigate issues like overfitting.
**Time Series Cross-Validation:** Time series cross-validation is a specific type of cross-validation designed for time series data. It splits the data into multiple segments (e.g., folds) while preserving the temporal order. Unlike traditional cross-validation, which assumes the data points are independent, time series cross-validation maintains the chronological order during the split. This ensures that future data is not inadvertently used to predict past outcomes. Each segment is used as a testing set once, with the rest of the data used for training. This process is repeated for each segment, and the performance metrics are averaged across all folds to provide an overall estimate of model performance.
Time series cross-validation techniques:
- **Rolling Window Cross-Validation:** This method refers to implementation of cross-validation over *Roll-Forward Partitioning*. In this approach, a fixed-size training window moves forward in time, and the model is trained and evaluated at each step. This allows the model to learn from past data and test on future data, mimicking real-world scenarios more closely.
- **Time Series Split Cross-Validation:** This method refers to implementation of cross-validation over *Fixed Partitioning*. This method splits the data into multiple folds, ensuring that each fold contains contiguous blocks of time. Each fold then serves as a testing set while the remaining data is used for training.<br>
**What You Do**:  Implement *Rolling Window Cross-Validation* or *Time Series Split Cross-Validation.*<br>
**Why It Matters**: These techniques provide a more realistic evaluation of the model's performance because they preserve the temporal order of observations. By ensuring that the training set only includes data from before the test set, you effectively emulate how the model would perform in real-world forecasting.



### Step 10: Select Forecasting Technique
**Forecasting Techniques**: This involves choosing the appropriate model based on the characteristics of the data. There are several methods for time series forecasting. Some popular approaches include:
- **Naive Forecasting**: Using the last observed value as the forecast for the next period.
  - Assumes that the future values of the time series will be the same as the most recent observed value.
  - Assumes no trend, seasonality, or other patterns in the data.
- **Moving Average (MA)**: Calculating the average of a subset of data points to smooth out short-term fluctuations.
  - Assumes that the time series data exhibit random fluctuations around a constant mean over time.
  - Assumes no trend or seasonality in the data.
  - Assumes that the observations are independent of each other.
- **Exponential Smoothing**: Assigning exponentially decreasing weights to older observations to capture trends and seasonality.
  - Assumes that the time series data exhibit random fluctuations around a changing mean over time.
  - Assumes no trend or seasonality in the data.
  - Assumes that more recent observations are more relevant than older observations.
- **Autoregressive (AR) Models**: Using past observations to predict future values, assuming a linear relationship between current and past values.
  - Assumes that the current value of the time series can be explained by its past values.
  - Assumes stationarity of the time series data.
  - Assumes no seasonality in the data.
- **Autoregressive Moving Average (ARMA) Models**: Combining autoregressive and moving average components to capture both autocorrelation and short-term fluctuations.
  - Combines autoregressive and moving average components.
  - Assumes stationarity of the time series data.
  - Assumes no seasonality in the data.
- **Autoregressive Integrated Moving Average (ARIMA) Models**: Extending ARMA models to account for non-stationarity by differencing the data.
  - Allows for differencing to achieve stationarity.
  - Combines autoregressive, differencing, and moving average components.
  - Assumes no seasonality in the data.
- **Seasonal ARIMA (SARIMA) Models**: Incorporating seasonal components into ARIMA models to handle seasonal patterns in the data.
  - Extends ARIMA to account for seasonality in the data.
  - Assumes stationarity after seasonal differencing.
  - Combines autoregressive, differencing, and moving average components along with seasonal components.
- **Linear Regression**: Modeling the relationship between the independent and dependent variables assuming linearity.
  - Assumes a linear relationship between the independent variables and the dependent variable.
  - Assumes independence of errors (residuals).
  - Assumes homoscedasticity of residuals (constant variance of errors).
  - Assumes that the residuals are normally distributed.
  - Assumes no multicollinearity among the independent variables.
- **Random Forests**: Ensemble learning method that builds multiple decision trees and combines their predictions.
  - Does not rely heavily on assumptions about the distribution or relationship between variables.
  - Tends to handle non-linear relationships well.
  - Is robust to overfitting.
- **Neural networks:** Deep learning models (e.g., LSTM), capable of capturing complex patterns in sequential data.
  - Assumes that the data have temporal dependencies.
  - Requires a sufficiently large dataset for training to capture patterns effectively.
  - Is sensitive to hyperparameters and architecture choices.
  - Assumes the data exhibit patterns that can be learned through the network's architecture.
  - Requires careful preprocessing and normalization of data.
  - May suffer from overfitting, especially with limited data.<br>
**What You Do**: Choose from methods like ARIMA, Exponential Smoothing, or Machine Learning approaches. <br>
**Why It Matters**: Picking the right method is key to capturing the characteristics of your time series effectively.

### Step 11: Tune Model Parameters
**What You Do**: Adjust the knobs of model complexity to achieve a good fit. <br>
**Why It Matters**: Well-tuned parameters balance the model's ability to fit data with its capacity to generalize forecasts.

### Step 12: Evaluate/Validate Model
**Model Evaluation**: Evaluating the performance of time series models is crucial. Common metrics for assessing forecast accuracy include mean absolute error (MAE), mean squared error (MSE), root mean squared error (RMSE), and mean absolute percentage error (MAPE). Cross-validation techniques, such as rolling origin or expanding window, can be employed to assess model robustness.<br>
**What You Do**: Use historical data to test how well your model predicts. <br>
**Why It Matters**: Validation ensures your model’s performance is robust and reliable before actual forecasting.

### Step 13: Forecast
**What You Do**: Make predictions using your tuned and validated model. <br>
**Why It Matters**: Accurate forecasts allow for informed decision-making and strategic planning.

### Step 14:Uncertainty Estimation
**Uncertainty Estimation**: In many real-world scenarios, uncertainty estimation is crucial for decision-making.  This step involves quantifying the variability and uncertainty associated with model predictions. The goal is to generate a distribution of possible outcomes or estimates of uncertainty for each forecasted value which provides **confidence intervals**. Common techniques used for uncertainty estimation are:
- **Bootstrapping**
- **Bayesian methods**
- **Monte Carlo Simulation**

## Conclusion
These steps are your compass in the landscape of time series analysis, guiding you to deeper insights and more confident forecasts. Remember, the goal isn’t just to predict the future—it’s to understand the story your data tells, so you can write the next chapter with clarity and confidence. Keep these steps handy; they are the milestones on your journey to mastering time series forecasting.

## Further improvements:
- **Trailing and centered windows**.
- **Ensemble Methods**
- **Having more than 1 label**