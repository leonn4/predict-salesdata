import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data from your image
data = {
    'Year': [2012, 2013, 2014, 2015, 2016],
    'Large Mask (units)': [631782, 678927, 638691, 659849, 709362],
    'Large Mask (rupiah)': [11372076000, 12220686000, 11496438000, 11877282000, 12768516000],
    'Small Mask (units)': [223542, 278327, 262891, 265782, 292539],
    'Small Mask (rupiah)': [3800214000, 4731559000, 4469147000, 4518294000, 4973163000]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Set the 'Year' column as the index and convert it to datetime
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.set_index('Year', inplace=True)

# Check stationarity using the Augmented Dickey-Fuller test
def check_stationarity(series):
    result = adfuller(series)
    print(f'Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= 0.05:
        print("The series is stationary.")
    else:
        print("The series is non-stationary. Consider differencing.")

# Check stationarity for both mask types
print("Large Mask (units) stationarity test:")
check_stationarity(df['Large Mask (units)'])

print("Small Mask (units) stationarity test:")
check_stationarity(df['Small Mask (units)'])

# Define a function to fit ARIMA model and forecast
def forecast_mask_sales(mask_type, steps=3):
    # Select the relevant column for the mask type
    mask_units = df[mask_type]
    
    # Differencing to make the data stationary if needed
    mask_units_diff = mask_units.diff().dropna()
    
    # Create and fit the ARIMA model
    model_arima = ARIMA(mask_units_diff, order=(1, 1, 1))  # You may need to adjust the order
    arima_fit = model_arima.fit()

    # Forecast the specified number of periods into the future
    forecast_diff = arima_fit.forecast(steps=steps)

    # Reverse differencing
    last_value = mask_units.iloc[-1]
    forecast = [last_value + sum(forecast_diff[:i+1]) for i in range(steps)]
    
    return forecast

# Forecast for Large Masks for the years 2017, 2018, 2019
large_mask_forecast = forecast_mask_sales('Large Mask (units)', steps=3)
print("Large Mask Forecast for 2017, 2018, 2019:", large_mask_forecast)

# Forecast for Small Masks for the years 2017, 2018, 2019
small_mask_forecast = forecast_mask_sales('Small Mask (units)', steps=3)
print("Small Mask Forecast for 2017, 2018, 2019:", small_mask_forecast)

# Assuming 'y_true' is the actual values for evaluation
# Replace with your actual data for the relevant period
y_true_large_mask = [717150, 739785, 757137]  # Replace with actual values for Large Mask forecast
y_pred_large_mask = large_mask_forecast  # Large Mask forecast results

y_true_small_mask = [302037, 316683, 329794]  # Replace with actual values for Small Mask forecast
y_pred_small_mask = small_mask_forecast  # Small Mask forecast results

# Calculate MAE and RMSE for Large Mask
mae_large = mean_absolute_error(y_true_large_mask, y_pred_large_mask)
rmse_large = np.sqrt(mean_squared_error(y_true_large_mask, y_pred_large_mask))

# Calculating MAE and RMSE for Small Mask
mae_small = mean_absolute_error(y_true_small_mask, y_pred_small_mask)
rmse_small = np.sqrt(mean_squared_error(y_true_small_mask, y_pred_small_mask))

# Display the evaluation results
print(f"\nModel Evaluation for Large Mask:")
print(f"Mean Absolute Error (MAE): {mae_large}")
print(f"Root Mean Square Error (RMSE): {rmse_large}")

print(f"\nModel Evaluation for Small Mask:")
print(f"Mean Absolute Error (MAE): {mae_small}")
print(f"Root Mean Square Error (RMSE): {rmse_small}")
