# Time_Series_Analysis_SNAP_Stock
#Team Members: Arunachalaeswar Chandran(20250179), Jingxi Lu(20250186), Yushi Zhang(20250197), Yanan Cui

This study applies time series analysis to forecast the stock price of Snapchat (SNAP). By evaluating historical market data, the project compares multiple predictive models—including Linear Regression, ETS, and ARIMA—to determine the most accurate method for short-term (30-day) stock price forecasting.

## Datasets
Yahoo Finance API – Live acquisition of SNAP stock data (from 2018-01-01) using the quantmod package. The analysis primarily focuses on the Adjusted Closing Price to build the predictive models.

## What the R Script does
### 1. Data Acquisition & Cleaning
**I. Automated Data Retrieval & Quality Checks**

Fetches live financial data and performs missing value (NA) detection to ensure dataset integrity. The data is then structured into a strict Time Series object (frequency = 252 trading days/year).

### 2. Exploratory Data Analysis (EDA)
**I. Moving Average Visualization**

Computes the 20-Day Simple Moving Average (SMA) and visualizes it against the actual closing price using ggplot2 to identify broader market trends and volatility.

**II. Stationarity Testing**

Performs the Augmented Dickey-Fuller (ADF) Test. The analysis confirms the raw financial data is non-stationary, establishing the need for differencing in subsequent modeling steps.

### 3. Model Building & Forecasting
**I. Exponential Smoothing & ARIMA**

Splits the dataset into training and testing sets (reserving the final 30 days for validation). Trains an ETS (ZZN) model and an optimized ARIMA model (auto.arima with stepwise selection) to capture complex patterns.

**II. Autoregressive Trend Modeling**

Develops a baseline Linear Regression model and significantly improves it by engineering an optimized version that incorporates lag-1 price (autoregressive term) to better follow market shifts.

### 4. Evaluation & Final Prediction
**I. Comprehensive Model Comparison**

Visualizes the performance of all distinct models (Optimized LR vs. ETS vs. ARIMA) overlaid on the actual test set data to evaluate real-world predictive accuracy.

**II. Residual Diagnostics & 30-Day Forecast**

Validates the winning ARIMA model by analyzing its residuals (white noise check) and generates a final, plotted 30-day future price prediction.

