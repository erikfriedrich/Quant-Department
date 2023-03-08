import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# make a list with the tickers that we want the data for
# Ticker symbols for the S&P 500, S&P 500 Momentum index, S&P 500 Quality index ETFs, and S&P 500 High Beta ETF
tickers = ["SPY", "SPMO", "SPHQ", "SPHB"]


# Download historical data for the tickers; from October 12th 2015 till today [has to be updated if you want the most recent data]
data = yf.download(tickers, start="2015-10-12", end="2023-02-08")
#print(data) #if you want take a look at the data you've pulled, but only the last and first five (or so) will be displayed


# Extract the losing prices for each ticker from our dataframe "data" and turns it into a new dataframe called "close"
close = data["Close"]
#print(close) # if you want to take a look at the new dataframe and its values


# Divide each column by its first value -> to get the relative performance 
close = close.divide(close.iloc[0])
#print(close) # to see what happend to the data in close


# Calculate the horizontal mean of SPMO, SPHQ, SPHB and store the result in a new column "factors" -> every factor has a weighting of 1/3
# this gives us the hypothetical performance of our factor portfolio
close["factors"] = close[["SPMO", "SPHQ", "SPHB"]].mean(axis=1)
#print(close) to see changes to the dataframe or print(close["factors"]) should only give you the new fators column


# Calculate the difference between the "factors" column and the "SPY" column and store the result in a new column "diff"
# so we can better examine their relative performance as well as the "behavior" relative to each other
close["diff"] = close["factors"] - close["SPY"]
# print(close) to see changes, print(close["diff"] should only print out the column "diff"

# now we want to plot the data that we've just retrieved, transformed and manipulated
# Plot the "diff" column
fig, ax = plt.subplots()
ax.plot(close.index, close["diff"], label="Outperformance")
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1) # Add a line at y=0
ax.legend() # right now, we don't do anything to our legend; (if you wand to) look into matplot documentation or as chatgpt how to work with the legend

# Goal: Format the y-axis to display values in percent (nicer to look at)
fmt = '%.0f%%' # Set the format to display percentage values
yticks = mtick.PercentFormatter(xmax=1.0, decimals=0, symbol='%') # this will transform the y-axis to % if we input it into the next line
ax.yaxis.set_major_formatter(yticks) # use 'yticks' as the format for out y-axis

plt.show() # this shows the plot that we've created beforehand
