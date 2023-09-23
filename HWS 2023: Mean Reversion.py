import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

qb = QuantBook()

------------------------------------------------------------------------------------------------------------------------

ticker = "SNAP"
n_sma = 200
threshold = 1 # number of std it has to deviate from the sma
safety_threshold = 3 # losses during rare events/ downward moves (e.g. Covid-Crash) => "Safety Net"

n_days = 100000 # this number of days is large enough so that we get all the data from the listing until now
                # if n_days if further back than the listing, the df will automatically start at the day of the listing
                # if you want less days, just reduce it
                # with datetime you could also get it from a specific date on
rfr = 0.02

data = SMAMeanReversion(ticker, n_sma, threshold, n_days)
data_safety = SMAMeanReversionSafety(ticker, n_sma, threshold, safety_threshold, n_days)

stats_dict = getStrategyStats(data, rfr)
safe_stats_dict = getStrategyStats(data_safety, rfr)

df_stats = pd.DataFrame(stats_dict).round(3)
df_safe_stats = pd.DataFrame(safe_stats_dict).round(3)

df_safe_stats.columns = ["Mean Reversion with Safety", "Buy and Hold"]
df_stats.columns = ["Mean Reversion", "X"]

df_stats = pd.concat([df_stats.T, df_safe_stats.T])
df_stats.drop("X", axis=0, inplace=True)
df_stats

------------------------------------------------------------------------------------------------------------------------

data

------------------------------------------------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data_safety['Strategy_Cumulative_Returns'], label='Mean Reversion Strategy with Safety')
ax.plot(data['Strategy_Cumulative_Returns'], label='Mean Reversion Strategy')
ax.plot(data_safety['BAH_Cumulative_Returns'], label=f'{ticker}')
ax.set_xlabel('Date')
ax.set_ylabel('Returns (%)')
ax.set_title('Cumulative Returns for Mean Reversion and Buy and Hold Strategies')
ax.legend()
plt.show()

# large drop at the end could be earning (-28% percent)

------------------------------------------------------------------------------------------------------------------------

def SMAMeanReversionSafety(ticker, n_sma, threshold, safety_threshold, n_days):
    
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
    df["Signal"] = np.where((df[symbol]>df["Upper"]) & (df[symbol]<df["Safety_Upper"]), 1, 0)
    df["Signal"] = np.where((df[symbol]<df["Lower"]) & (df[symbol] > df["Safety_Lower"]), -1, df["Signal"])

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
    
    return df

------------------------------------------------------------------------------------------------------------------------

def SMAMeanReversion(ticker, n_sma, threshold, n_days):
    
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
    df["Signal"] = np.where((df[symbol]>df["Upper"]), 1, 0)
    df["Signal"] = np.where((df[symbol]<df["Lower"]), -1, df["Signal"])

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
    
    return df

------------------------------------------------------------------------------------------------------------------------

def getStrategyStats(data, rfr):

    # create dictionaries for our strategy as well as buy and hold
    Trading_Strat, BAH_Strat = {}, {}

    # Total Returns
    Trading_Strat["Total_Returns"] = data["Strategy_Cumulative_Returns"][-1] - 1
    BAH_Strat["Total_Returns"] = data["BAH_Cumulative_Returns"][-1] - 1


    # Mean Annual Returns
    Trading_Strat["Annual_Returns"] = Trading_Strat['Total_Returns'] * (1/(len(data)/252))
    BAH_Strat["Annual_Returns"] = BAH_Strat['Total_Returns'] * (1/(len(data)/252))

    # Annual Volatility
    Trading_Strat["Annual_Volatility"] = data["Strategy_Returns"].std() * np.sqrt(252)
    BAH_Strat["Annual_Volatility"] = data["Log_Returns"].std() * np.sqrt(252)

    # Sharpe Ratio
    Trading_Strat["Sharpe_Ratio"] = (Trading_Strat['Annual_Returns'] - rfr)/ Trading_Strat['Annual_Volatility']
    BAH_Strat["Sharpe_Ratio"] = (BAH_Strat['Annual_Returns'] - rfr)/ BAH_Strat['Annual_Volatility']

    # Max Drawdown
    trading_dd = (data["Strategy_Cumulative_Returns"] - data["Strategy_High"]) / data["Strategy_High"]
    bah_dd = (data["BAH_Cumulative_Returns"] - data["BAH_High"]) / data["BAH_High"]

    Trading_Strat["Max Draw Down"] = trading_dd.min()
    BAH_Strat["Max Draw Down"] = bah_dd.min()

    stats_dict = {"Trading_Stats": Trading_Strat, "BAH_Stats": BAH_Strat}

    return stats_dict
