import pandas as pd 
import numpy as np
from datetime import datetime, timedelta
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import statsmodels.api as sm
from math import floor
from sklearn import linear_model
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels

qb = QuantBook()

tickers = ['KO', 'PEP', 'ADBE', 'MSFT']

start = datetime(2010,1,1)
end = datetime(2018,1,1)

results = pd.DataFrame()

for ticker in tickers:
    symbol = qb.AddEquity(ticker).Symbol
    df = qb.History(symbol, start, end, Resolution.Daily)["close"].unstack(level=0)
    results[ticker] = df

stock_returns = results.pct_change().fillna(0)
stock_prices = np.log(results)

def stationary(data):
    results_stationary = {}

    for col in data.columns:

        test_result = adfuller(data[col])
        p_value = test_result[1]

        if p_value <= 0.05:
            results_stationary[col] = "stationary"
        else:
            results_stationary[col] = "non-stationary"

    return results_stationary

stationary(spreads)

def cointegration(data):

    results_cointegration = {}

    list_symbols  = data.columns
    possible_pairs = list(combinations(list_symbols, 2))

    for pair in possible_pairs:
        score, p_value, _ = coint(data[pair[0]], data[pair[1]])
        
        if p_value <= 0.05:
            results_cointegration[pair] = "cointegrated"

    return results_cointegration

results_cointegration = cointegration(stock_prices)
results_cointegration

def spread(ticker_pairs):

    spread = pd.DataFrame()

    for key in results_cointegration.keys():
    
        x = stock_prices[key[1]]
        y = stock_prices[key[0]]
        
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        
        x = x[key[1]]
        b = model.params[key[1]]

        spread[key] = y - b * x
        spread[key].plot(figsize=(12,6))
        plt.axhline(spread[key].mean(), color='black')
        plt.legend(['Spread between ' + key[0] + " and " + key[1]])
        plt.show()

        print(key[1])
        print(key[0])

    return spread

spreads = spread(results_cointegration)
print(spreads)

def z_score(spread_pairs):

    z_scores = pd.DataFrame()

    for pair in spread_pairs.columns:
        
        z_scores[pair] = (spread_pairs[pair] - spread_pairs[pair].mean()) / spread_pairs[pair].std() 
        z_scores[pair].plot(figsize=(12,6))
        plt.axhline(z_scores[pair].mean())
        plt.axhline(1.0, color='red')
        plt.axhline(-1.0, color='green')
        plt.show()
        print(pair)
        
    return z_scores


z_scores = z_score(spreads)
print(scores)


def trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns):

    strategy_returns = pd.DataFrame()
    
    for pair in spreads.columns:

            returns_for_pair = []

            for i in range(len(stock_prices)-1):

                if z_scores[pair][i] < -std_open:
                    returns_for_pair.append(- stock_returns[pair[1]][i+1] + stock_returns[pair[0]][i+1])

                elif z_scores[pair][i] > std_open:
                    returns_for_pair.append(+ stock_returns[pair[1]][i+1] - stock_returns[pair[0]][i+1])

                else:
                    returns_for_pair.append(0)

            strategy_returns[pair] = returns_for_pair

    return strategy_returns

std_open = 1
std_out = 0.5
returns = trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns)
print(returns)

def cumulative_returns(returns):

    cumulative_return = {}
    pair_returns = pd.DataFrame()

    for pair in returns.columns:

        pair_returns[pair] = (1 + returns[pair]).cumprod() -1
        cumulative_return[pair] = pair_returns[pair].iloc[-1]
        
    return cumulative_return

cumulative_returns(returns)
