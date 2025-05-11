import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load dataset
df = pd.read_csv('stock_prices.csv')  # Replace with actual path if needed
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df = df.asfreq('D')  # Ensure daily frequency
df['Close'].fillna(method='ffill', inplace=True)

# EDA: Plot close prices
plt.figure(figsize=(10, 6))
df['Close'].plot(title='Stock Close Prices')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Feature engineering (optional)
df['lag1'] = df['Close'].shift(1)
df['rolling_mean'] = df['Close'].rolling(window=5).mean()

# Train-test split
train = df['Close'][:-30]
test = df['Close'][-30:]

# Train ARIMA model
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# Evaluate
mae = mean_absolute_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

# Plot
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast')
plt.title('ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.grid(True)
plt.show()

# Print metrics
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
