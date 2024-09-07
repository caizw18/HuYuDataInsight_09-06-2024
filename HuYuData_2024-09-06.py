pip install arch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
import yfinance as yf

# Download stock data (e.g., Apple)
data = yf.download('AAPL', start='2015-01-01', end='2024-01-01')

# Calculate daily returns
data['Returns'] = 100 * data['Adj Close'].pct_change().dropna()

# Plot the returns
data['Returns'].plot(figsize=(10, 5))
plt.title('Daily Returns of AAPL')
plt.show()

# Remove the first NaN value from Returns (due to pct_change())
returns = data['Returns'].dropna()

# Specify the GARCH(1,1) model
model = arch_model(returns, vol='Garch', p=1, q=1)

# Fit the model
garch_fit = model.fit()

# Print the results
print(garch_fit.summary())

# Plot volatility forecast
garch_fit.plot(annualize='D')
plt.show()

# Make predictions (out-of-sample forecast)
forecast = garch_fit.forecast(horizon=5)
print(forecast.variance[-1:])