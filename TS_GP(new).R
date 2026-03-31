# ==============================================================================
# Time Series Analysis: SNAP Stock Price Forecasting
# ==============================================================================



############# Load Libraries & Data Acquisition and Cleaning  #############
# Load Libraries
library(quantmod)
library(forecast)
library(tseries)
library(ggplot2)
library(TTR)
library(xts)
library(lpSolve)

# Data Acquisition
getSymbols("SNAP", src="yahoo", from="2018-01-01")
snap_close <- Cl(SNAP) # Extract Adjusted Closing Price

# Data Cleaning & Preparation
# Check if any missing values exist
any_na <- any(is.na(snap_close))
cat("Any missing values:", any_na, "\n")

# Count total number of NAs
na_count <- sum(is.na(snap_close))
cat("Total number of missing values:", na_count, "\n")

# Convert to Time Series object (252 trading days per year)
snap_ts <- ts(as.numeric(snap_close), frequency = 252)

# plot the stock prices of SNAP
plot(snap_ts, 
     main = "Snapchat (SNAP) Closing Price History",
     xlab = "Time (Yearly Index)",
     ylab = "Price (USD)", 
     col = "blue", 
     lwd = 2) 

grid()


##################### Exploratory Data Analysis  #####################
# Exploratory Data Analysis
ma20 <- SMA(snap_close, 20)

cat("\n=== 20-Day Moving Average (Last 10 Values) ===\n")
print(tail(ma20, 10))  # Show last 10 values

# Build data frame
df <- data.frame(
  Date = index(snap_close),
  Price = as.numeric(snap_close),
  MA20 = as.numeric(ma20)
)

# Plot
ggplot(df, aes(x = Date)) +
  geom_line(aes(y = Price), color = "blue", linewidth = 1) +
  geom_line(aes(y = MA20), color = "red", linewidth = 1.2) +
  labs(
    title = "SNAP Stock Price & 20-Day Moving Average",
    x = "Date", y = "Price (USD)"
  ) +
  theme_minimal()



################## Augmented Dickey-Fuller (ADF) Test  ##################
# Augmented Dickey-Fuller (ADF) Test
print(adf.test(snap_ts)) 
# Conclusion: p-value = 0.8707 > 0.05, snap_ts is non-stationary.
# Note: Financial data is usually non-stationary; differencing is required (d=1)



######################## ETS model & ARIMA model ########################
# Split data into training and test sets
train_size <- length(snap_ts) - 30
train_data <- subset(snap_ts, end = train_size)
test_data <- subset(snap_ts, start = train_size + 1) #avoid overlap between the training set and the test set

# ETS Model
fit_ets <- ets(train_data, model = "ZZN")
forecast_ets <- forecast(fit_ets, h = 30)

# ETS Test Accuracy
cat("\n=== ETS Test Set Accuracy ===\n")
accuracy(forecast_ets, test_data)

# Conclusion: MAPE is approximately 7.1%: model's average forecasting error is relatively low and acceptable for short-term predictions. 
#             A Theil's U > 1: the model's performance does not outperform a simple Naive benchmark.
#             ACF1 is about 0.86: the significant autocorrelation in the prediction residuals

# ARIMA Model
# Use stepwise + approximation for optimal speed and accuracy
fit_arima <- auto.arima(train_data, 
                        stepwise = TRUE,  # improve the speed of parameter selection
                        approximation = TRUE,  #balance speed and accuracy
                        trace = FALSE) #not show the parameter search process
forecast_arima <- forecast(fit_arima, h = 30)

# ARIMA Test Accuracy
cat("\n=== ARIMA Test Set Accuracy ===\n")
accuracy(forecast_arima, test_data)

#Conclusion:  ARIMA test set RMSE=0.394, MAPE=7.07%,Theil’s U=2.87， close to ETS, showing consistent error.
#             ACF1 near 0 in training but 0.867 in test: good historical fit, poor adaptation to new market shocks.
#             Traditional linear models struggle to predict SNAP.


################# Linear Regression Model (Trend Modeling)#################

# Convert ts to vector to avoid length mismatch
train_time <- 1:length(train_data)
test_time <- (length(train_data)+1):(length(train_data)+30)
fit_lm <- lm(as.numeric(train_data) ~ train_time) #basic linear regression model
forecast_lm_ts <- ts(predict(fit_lm, newdata=data.frame(train_time=test_time)), start=end(train_data), frequency=252)
train_vec <- as.numeric(train_data)
n_train <- length(train_vec)

# Create lag 1 price (autoregressive term)
train_lag1 <- c(NA, train_vec[-n_train])

# Remove NA to ensure same length
train_clean <- train_vec[!is.na(train_lag1)]
lag1_clean <- train_lag1[!is.na(train_lag1)]
time_clean  <- 1:length(train_clean)

# Fit optimized model(with 'historical prices' in addition)
fit_lm_opt <- lm(train_clean ~ time_clean + lag1_clean)

# Iterative forecast for 30 days
opt_pred <- numeric(30)
last_p <- tail(train_vec, 1)

for (i in 1:30) {
  new_t <- length(time_clean) + i
  pred <- predict(fit_lm_opt, newdata = data.frame(
    time_clean = new_t,
    lag1_clean = last_p
  ))
  opt_pred[i] <- as.numeric(pred)
  last_p <- opt_pred[i]
}

forecast_lm_opt_ts <- ts(opt_pred, start = end(train_data), frequency = 252)


dev.new()

# Calculate dynamic Y-axis limits
all_values <- c(
  as.numeric(test_data),
  as.numeric(forecast_arima$mean),
  as.numeric(opt_pred)
)
y_min <- min(all_values) * 0.8
y_max <- max(all_values) * 1.2

# Plot actual test data
plot(test_data,
     main="SNAP Model Comparison: Original vs Optimized Linear Regression",
     col="black", lwd=3,
     ylim=c(y_min, y_max),
     ylab="Price (USD)",
     xlab="Time (Prediction Horizon)")

# Add original linear regression
lines(forecast_lm_ts, col="darkgreen", lwd=2, lty=2)

# Add optimized linear regression
lines(forecast_lm_opt_ts, col="purple", lwd=2, lty=1)

# Add ETS model
lines(forecast_ets$mean, col="red", lwd=2)

# Add ARIMA model
lines(forecast_arima$mean, col="blue", lwd=2)

# Add legend
legend("topleft",
       legend=c(
         "Actual Test Price",
         "Original LR (Baseline, Poor)",
         "Optimized LR (Improved)",
         "ETS Model",
         "ARIMA Model"
       ),
       col=c("black", "darkgreen", "purple", "red", "blue"),
       lwd=c(3,2,2,2,2),
       lty=c(1,2,1,1,1),
       bty="n", cex=0.8)



# ==============================================================================
# FINAL MODEL COMPARISON: LM vs ETS vs ARIMA
# ==============================================================================
# FINAL MODEL COMPARISON: 整体趋势图 (The Big Picture)
# ==============================================================================
fc_arima <- forecast_arima
fc_ets <- forecast_ets
fc_lm <- forecast_lm_opt_ts
time_idx <- time(snap_ts)

dev.new()

plot(fc_arima, main="SNAP Stock Forecast Comparison (Full History)", 
     col="blue", fcol="blue", shadecols=rgb(0,0,1,0.1))

# 1. Add actual test set data
lines(test_data, col="black", lwd=3)

# 2. Add Linear Regression
lines(ts(as.numeric(fc_lm), start=time_idx[length(time_idx)]/252 + 2021, frequency=252), 
      col="darkgreen", lwd=2, lty=2)

# 3. Add ETS Model
lines(fc_ets$mean, col="red", lwd=2)

# Add legend
legend("topleft", 
       legend=c("ARIMA (Blue + Shade)", "ETS (Red)", "Linear Regression (Green Dash)", "Actual Test Set"),
       col=c("blue", "red", "darkgreen", "black"), 
       lwd=c(2, 2, 2, 3), lty=c(1, 1, 2, 1), bty="n", cex=0.8)

# ==============================================================================
# FINAL MODEL COMPARISON: 放大对比版 (Zoomed In) - 核心展示图
# ==============================================================================
dev.new()

# Plot main chart (Keep include=80 for zoom-in effect)
plot(fc_arima, main="SNAP Stock Forecast Comparison (Zoomed In)", 
     col="blue", fcol="blue", shadecols=rgb(0,0,1,0.1), 
     include=80, 
     ylab="Price (USD)", xlab="Time")

# 1. Add actual test set data (Black line)
lines(test_data, col="black", lwd=3)

# 2. Add ETS forecast (Red line)
lines(fc_ets$mean, col="red", lwd=2)

# 3. Fix and add Linear Regression forecast (Green dashed line) - Align with ARIMA
fc_lm_aligned <- ts(as.numeric(fc_lm), start=start(fc_arima$mean), frequency=frequency(fc_arima$mean))
lines(fc_lm_aligned, col="darkgreen", lwd=2, lty=2)

# Add legend
legend("topleft", 
       legend=c("ARIMA (Blue + Shade)", "ETS (Red)", "Linear Regression (Green Dash)", "Actual Test Set"),
       col=c("blue", "red", "darkgreen", "black"), 
       lwd=c(2, 2, 2, 3), lty=c(1, 1, 2, 1), bty="n", cex=0.8)



######################## Residual Diagnostics ########################
# Verify if residuals behave like white noise
dev.new()
checkresiduals(fit_arima) 
summary(fit_arima)


######################## Final 30-Day Forecast ########################
dev.new()
final_forecast <- forecast(fit_arima, h=30)
autoplot(final_forecast) + 
  ggtitle("SNAP Stock 30-Day ARIMA Forecast") +
  xlab("Year") + ylab("Stock Price") +
  theme_minimal()