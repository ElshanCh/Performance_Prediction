28/02/2024
----------------------------------------------

# ***Anomaly detection for Time Series Analysis***

- **Time series** are data that record the value of one or more variables at different point in time. 

- Time series are very important because they allow us to analyse the past, understand the present, and predict the future.

- Time series are often **non-stationary**, i.e. their statistical properties (such as mean and variance) change over time. This makes it difficult to apply traditional statistical methods, which assume the stationarity of the data.

- **Seasonality of data** -> the periodic variation of data as a function of time. We need to have at least one complete cycle of observations that covers all possible seasons. If data is collected every day, we need to have at least a year’s worth of data to be able to analyze annual seasonality.

- To be able to analyze a time series, we need to meet certain **requirements**:
    1. **Sufficient number of data points** (i.e. observations of the variable over time),
    2. **Good understanding of the domain of the data** (i.e. the context in which the data is generated and the meaning of the variables),
    3. **Clear definition of the goals of the analysis** (i.e. what we want to find out from the data and how we want to use it):
        1. Describe the behavior of data over time and its main characteristics
        2. Predict future data values based on past values
        3. Detect anomalies in data and their causes
        4. Test hypotheses about data and their relationships
        5. Optimize data-driven decisions and actions

- An **anomaly** is a value or event that deviates significantly from the normal trend of the data. The anomalies can be of two types: 
    1. **Punctual** / **Point** (i.e. isolated values that are very different from other values in the time series)
    2. **Collective** (i.e. groups of values that are different from the rest of the time series)

- **Noise Anomalies** caused due to  measuring, transmission or data processing errors.
- **Signals Anomalies** caused due to structural changes, fraudulent activity, exceptional events, or other factors affecting the data.

- In order to detect anomalies in time series, we first need to have **expectations** about the normal movement of data over time. These expectations are based on the analysis of the ***main components of a time series***, which are:

    1. **Trend**, i.e. the direction and speed of data change over the long run. For example, an increasing trend indicates that the data is increasing over time, while a decreasing trend indicates that the data is decreasing over time.
    2. **Seasonality**, i.e. the periodic variation of data as a function of time. For example, annual seasonality indicates that the data has a cyclical pattern that repeats itself every year, such as toy store sales increasing in December and decreasing in January.
    3. **Cyclicality**, i.e. the irregular variation of data as a function of time. For example, economic cyclicality indicates that data has a fluctuating trend that depends on external factors, such as GDP, inflation, unemployment, etc.
    4. **Noise**, i.e. the random variation of data as a function of time. For example, noise can be caused by measuring, transmitting, or processing errors.

### **Differentiation in Time Series Analysis**
- **Differentiation** in the context of time series analysis refers to the process of transforming data to make it stationary or approximately stationary by removing trends and non-constant variations in statistical properties such as mean and variance over time. This transformation is necessary to enable the application of traditional statistical methods, which assume stationary data. For example, if we have a time series {x1, x2, x3, …}, its first difference is {x2 — x1, x3 — x2, …}.

- For example, we can see a time series that has an increasing trend and an annual seasonality. Its first difference removes the trend, but not the seasonality. Its second difference removes both trend and seasonality.

- *Differentiation is the basis of* one of the most widely used models for time series analysis and anomaly detection: the **ARIMA** model.

### **Introduction to the ARIMA model**
The **ARIMA model** *is one of the most widely used models for time series analysis and anomaly detection*. ARIMA stands for Autoregressive Integrated Moving Average. This model combines three main components:

- **The autoregressive (AR)** component, which models the correlation between time series values and previous values. For example, if the data is cyclical, the time series values will be affected by the past values.
- **The built-in (I)** component, which models the differentiation of the time series to make it stationary. For example, if the data has a trend or seasonality, differentiation removes these components from the time series.
- **The moving average (MA)** component, which models the correlation between time series errors and previous errors. For example, if the data has noise, the time series errors will be affected by past errors.

The ARIMA model has three main parameters: **p**, **d** and **q**. 
1. The **p parameter** indicates the number of autoregressive terms to use in the model.  
2. The **parameter d** indicates the number of times the time series must be differentiated to make it stationary. 
3. The **q parameter** indicates the number of moving average terms to use in the model. For example, an ARIMA(1,1,1) model uses an autoregressive term, a difference, and a moving average term.

The ARIMA model can be used to describe, predict, and detect anomalies in a time series. To do this, we need to follow a few steps:

1. First, we need to **check whether the time series is stationary or not**. We can use statistical tests, such as the augmented Dickey-Fuller test, to check whether the mean and variance of the time series are constant over time.
2. Second, we need to **differentiate the time series until it is stationary**. We can use graphs, such as the graph of autocorrelation functions and partial autocorrelation functions, to determine the number of differences needed.
3. Third, we need to **estimate the parameters of the ARIMA model using optimization methods**, such as the maximum likelihood method. We can use model selection criteria, such as the Akaike information criterion or the Bayesian information criterion, to choose the optimal values of the p, d, and q parameters.
4. Fourth, we need to **validate the ARIMA model using verification methods**, such as the Ljung-Box test or the Jarque-Bera test. We can use graphs, such as the residuals graph or the forecast graph, to check if the model fits well with the data and if there is any anomaly in the data.
5. Fifth, we need to use the ARIMA model to **describe the main features of the time series**, predict future time series values, and detect anomalies in the time series. We can use measures of accuracy, such as mean square error or mean absolute error, to evaluate the quality of predictions and anomalies.