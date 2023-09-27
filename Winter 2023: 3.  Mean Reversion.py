import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from tabulate import tabulate

qb = QuantBook()

# define a function that 
  # simulates a SMA Mean Reversion strategy for given tickers,
  # a specific duration of the SMA,
  # a threshold on when to buy/sell,
  # a "safety net" when to exit given an unpredictable market,
  # and the total number of days that we want to have simulated

def SMAMeanReversionSafety(tickers, n_sma, threshold, safety_threshold, n_days):
    results = {}  # create a dictionary to store the results for each ticker

    # loop that goes through each ticker that is the list of tickers
    for ticker in tickers:
        
        # add our ticker to the equities
        symbol = qb.AddEquity(ticker).Symbol

        # get the data for our ticker
        df = qb.History(symbol, n_days, Resolution.Daily)["close"].unstack(level=0)

        # calculate the SMA
        df["SMA"] = df[symbol].rolling(n_sma).mean()

        # calculate the STD in the same timeframe
        df["STD"] = df[symbol].rolling(n_sma).std()

        # define the upper and lower bounds depending on our threshold
        df["Upper"] = df["SMA"] + threshold * df["STD"]
        df["Safety_Upper"] = df["SMA"] + safety_threshold * df["STD"]
      
        df["Lower"] = df["SMA"] - threshold * df["STD"]
        df["Safety_Lower"] = df["SMA"] - safety_threshold * df["STD"]

        # make a column to signal if we'll go long (1), short (2) or stay neutral (0)
        df["Signal"] = np.where((df[symbol]>df["Upper"]) & (df[symbol]<df["Safety_Upper"]), -1, 0)
        df["Signal"] = np.where((df[symbol]<df["Lower"]) & (df[symbol] > df["Safety_Lower"]), 1, df["Signal"])

        # drop first (n_sma) number of days, bc NaN values
        # get all the values after the first n_sma days
        # this is useful otherwise the first entry for the BAH Return column wouldn't be 1 (might be higher or lower) => screws our return metrics
        df = df.iloc[n_sma:]

        # calculate the log return of each day
        df["Log_Returns"] = np.log(df[symbol]/df[symbol].shift(1)).fillna(0)

        # calculate our strategies returns
        # days X return is dependend on the equities return that day and the direction we chose the day before => we have to take the signal from the day before
        df["Strategy_Returns"] = df["Log_Returns"] * df["Signal"].shift(1).fillna(0)
        df["Strategy_Cumulative_Returns"] = np.exp(df['Strategy_Returns'].cumsum()).fillna(0)

        # get high and low for our strategy
        df["Strategy_High"] = df["Strategy_Cumulative_Returns"].cummax()
        df["Strategy_Low"] = df["Strategy_Cumulative_Returns"].cummin()

        # we want to compare to buy and hold => calcualte every metric again

        # return + high and low
        df["BAH_Cumulative_Returns"] = np.exp(df['Log_Returns'].cumsum()).fillna(0)
        df["BAH_High"] = df['BAH_Cumulative_Returns'].cummax()
        df["BAH_Low"] = df['BAH_Cumulative_Returns'].cummin()

        results[ticker] = df

    return results

def SMAMeanReversion(tickers, n_sma, threshold, n_days):
    results = {}  # dictionary to store results for each ticker

    for ticker in tickers:
        # add our ticker to the equities
        symbol = qb.AddEquity(ticker).Symbol

        # get the data for our ticker
        df = qb.History(symbol, n_days, Resolution.Daily)["close"].unstack(level=0)

        # calculate the SMA
        df["SMA"] = df[symbol].rolling(n_sma).mean()

        # calculate the STD in the same timeframe
        df["STD"] = df[symbol].rolling(n_sma).std()

        # define the upper and lower bounds depending on our threshold
        df["Upper"] = df["SMA"] + threshold * df["STD"]
        df["Lower"] = df["SMA"] - threshold * df["STD"]

        # make a column to signal if we'll go long (1), short (2) or stay neutral (0)
        df["Signal"] = np.where((df[symbol]>df["Upper"]), -1, 0)
        df["Signal"] = np.where((df[symbol]<df["Lower"]), 1, df["Signal"])

        # drop first (n_sma) number of days, bc NaN values
        # get all the values after the first n_sma days
        # this is useful otherwise the first entry for the BAH Return column wouldn't be 1 (might be higher or lower) => screws our return metrics
        df = df.iloc[n_sma:]

        # calculate the log return of each day
        df["Log_Returns"] = np.log(df[symbol]/df[symbol].shift(1)).fillna(0)

        # calculate our strategies returns
        # days X return is dependend on the equities return that day and the direction we chose the day before => we have to take the signal from the day before
        df["Strategy_Returns"] = df["Log_Returns"] * df["Signal"].shift(1).fillna(0)
        df["Strategy_Cumulative_Returns"] = np.exp(df['Strategy_Returns'].cumsum()).fillna(0)

        # get high and low for our strategy
        df["Strategy_High"] = df["Strategy_Cumulative_Returns"].cummax()
        df["Strategy_Low"] = df["Strategy_Cumulative_Returns"].cummin()

        # we want to compare to buy and hold => calcualte every metric again

        # return + high and low
        df["BAH_Cumulative_Returns"] = np.exp(df['Log_Returns'].cumsum()).fillna(0)
        df["BAH_High"] = df['BAH_Cumulative_Returns'].cummax()
        df["BAH_Low"] = df['BAH_Cumulative_Returns'].cummin()

        results[ticker] = df

    return results

def getStrategyStats(data, rfr):
    stats_dict = {}

    for ticker, df in data.items():
        # create dictionaries for our strategy as well as buy and hold
        Trading_Strat, BAH_Strat = {}, {}

        # total returns
        Trading_Strat["Total_Returns"] = df["Strategy_Cumulative_Returns"][-1] - 1
        BAH_Strat["Total_Returns"] = df["BAH_Cumulative_Returns"][-1] - 1

        # compound annual growth rate
        Trading_Strat["Annual_Returns"] = (Trading_Strat['Total_Returns'] + 1) ** (1 / (len(df) / 252)) - 1
        BAH_Strat["Annual_Returns"] = (BAH_Strat['Total_Returns'] + 1) ** (1 / (len(df) / 252)) - 1

        # annual volatility
        Trading_Strat["Annual_Volatility"] = df["Strategy_Returns"].std() * np.sqrt(252)
        BAH_Strat["Annual_Volatility"] = df["Log_Returns"].std() * np.sqrt(252)

        # sharpe ratio
        Trading_Strat["Sharpe_Ratio"] = (Trading_Strat['Annual_Returns'] - rfr) / Trading_Strat['Annual_Volatility']
        BAH_Strat["Sharpe_Ratio"] = (BAH_Strat['Annual_Returns'] - rfr) / BAH_Strat['Annual_Volatility']

        # sortino ratio
        df_negative = df[df["Strategy_Returns"]<0]
        negative_std = df_negative["Strategy_Returns"].std() * np.sqrt(252)
        Trading_Strat["Sortiono_Ratio"] = (Trading_Strat["Annual_Returns"]-rfr)/negative_std

        df_negative = df[df["Log_Returns"]<0]
        negative_std = df_negative["Log_Returns"].std() * np.sqrt(252)
        BAH_Strat["Sortiono_Ratio"] = (BAH_Strat["Annual_Returns"]-rfr)/negative_std

        # max drawdown
        trading_dd = (df["Strategy_Cumulative_Returns"] - df["Strategy_High"]) / df["Strategy_High"]
        bah_dd = (df["BAH_Cumulative_Returns"] - df["BAH_High"]) / df["BAH_High"]

        Trading_Strat["Max Draw Down"] = trading_dd.min()
        BAH_Strat["Max Draw Down"] = bah_dd.min()

        stats_dict[ticker] = {"Trading_Stats": Trading_Strat, "BAH_Stats": BAH_Strat}

    return stats_dict
    
def plotData(plot_tickers):
    
    for ticker in plot_tickers:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        ax.plot(data_safety[ticker]['Strategy_Cumulative_Returns'], label=f'{ticker} - Mean Reversion Strategy with Safety')
        ax.plot(data[ticker]['Strategy_Cumulative_Returns'], label=f'{ticker} - Mean Reversion Strategy')
        ax.plot(data_safety[ticker]['BAH_Cumulative_Returns'], label=f'{ticker} - Buy and Hold')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns (%)')
        ax.set_title('Cumulative Returns for Mean Reversion and Buy and Hold Strategies')
        
        ax.legend()
        plt.show()

def displayPerformance(display_tickers):

    for ticker in display_tickers:

        ticker_df = pd.DataFrame(stats_dict[ticker]).round(3)
        ticker_df = ticker_df.drop("BAH_Stats", axis=1)
        
        ticker_safe_df = pd.DataFrame(safe_stats_dict[ticker]).round(3)
        
        merged_df = pd.merge(ticker_df, ticker_safe_df, left_index=True, right_index=True)
        merged_df.rename(columns={"Trading_Stats_x": "Trading Stats", "Trading_Stats_y": "Trading Stats with Safety Mechanism","BAH_Stats": "Buy and Hold Stats"}, inplace=True)
        
        print(f"Performance Statistics for {ticker}")
        print(tabulate(merged_df, headers='keys', tablefmt="grid"))
        print()

tickers = ["SNAP", "AAL", "A", "TSLA"]

n_sma = 200
threshold = 1 # number of std it has to deviate from the sma
safety_threshold = 3 # losses during rare events/ downward moves (e.g. Covid-Crash) => "Safety Net"

n_days = 100000 # this number of days is large enough so that we get all the data from the listing until now
                # if n_days if further back than the listing, the df will automatically start at the day of the listing
                # if you want less days, just reduce it
                # with datetime you could also get it from a specific date on
rfr = 0.02

data = SMAMeanReversion(tickers, n_sma, threshold, n_days)
data_safety = SMAMeanReversionSafety(tickers, n_sma, threshold, safety_threshold, n_days)

stats_dict = getStrategyStats(data, rfr)
safe_stats_dict = getStrategyStats(data_safety, rfr)

displayPerformance(tickers)
plotData(tickers)

# unfortunately, due to data limitations, QuantConnect doenst allow enough cells to output
# if you want to see the data for every file use: displayPerformance(["TICKER"])
