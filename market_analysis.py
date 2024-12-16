
# Refactored Code from Session 6.ipynb
# This script analyzes market values using ARIMA modeling and related techniques.

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA

# Load dataset
df = pd.read_csv("market_data.csv")  # Replace with actual path to your dataset
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Plotting ACF (Autocorrelation Function)
def plot_acf_analysis(data, lags=40, title="ACF Analysis"):
    '''
    Plots the autocorrelation function (ACF) of the given data.
    '''
    plot_acf(data, zero=False, lags=lags)
    plt.title(title, size=20)
    plt.show()

# ARIMA modeling function
def run_arima_model(data, order=(0, 0, 1)):
    '''
    Fits an ARIMA model to the given data and returns the results.
    '''
    model = ARIMA(data, order=order)
    results = model.fit()
    return results

# Example usage
if __name__ == "__main__":
    # Plot ACF for the market value
    plot_acf_analysis(df['market_value'], title="ACF for Market Value")

    # Fit ARIMA(0, 0, 1) model
    results = run_arima_model(df['market_value'])
    print(results.summary())
