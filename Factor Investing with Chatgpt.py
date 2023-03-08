import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Ticker symbols for the S&P 500, S&P 500 Momentum index, and S&P 500 Quality index ETFs
tickers = ["SPY", "SPMO", "SPHQ"]

# Download historical data for the tickers until 2022/12/31
data = yf.download(tickers, start="2015-10-12", end="2023-02-08")

# Extract the adjusted closing prices for each ticker
close = data["Adj Close"]

# Divide each column by its first value
close = close.divide(close.iloc[0])

# Calculate the horizontal mean of SPMO and SPHQ and store the result in a new column "mean"
close["factors"] = close[["SPMO", "SPHQ"]].mean(axis=1)

# Print the first five rows of the data
  #print(close)

# Plot the "factors" column against the "SPY" column
  #plt.plot(close.index, close["SPY"], label="SPY")
  #plt.plot(close.index, close["factors"], label="factors")
  #plt.legend()
  #plt.show()

# Calculate the difference between the "factors" column and the "SPY" column and store the result in a new column "diff"
close["diff"] = close["factors"] - close["SPY"]

# Plot the "diff" column
  #plt.plot(close.index, close["diff"], label="factors - SPY")
  #plt.legend()
  #plt.show()

# Plot the "diff" column
fig, ax = plt.subplots()
ax.plot(close.index, close["diff"], label="Outperformance")
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1) # Add a line at y=0
ax.legend()

# Format the y-axis to display values in percent
fmt = '%.0f%%' # Set the format to display percentage values
yticks = mtick.PercentFormatter(xmax=1.0, decimals=0, symbol='%')
ax.yaxis.set_major_formatter(yticks)

plt.show()
