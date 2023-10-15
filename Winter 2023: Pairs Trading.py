import pandas as pd 
import numpy as np
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import statsmodels.api as sm
from math import floor
from statsmodels.tsa.stattools import adfuller, coint

qb = QuantBook()

# firstly, we'll need to get the following information about the stocks, we want to "investigate":
        # (1) the stocks returns -> to calculate our performance in the end
        # (2) the (log) price of the stocks -> to check for cointegration
                # the linearization helps to detect cointegration

def getStockInfo(tickers, start, end):

    # create a new DataFrame to store relevant informaition in
    results = pd.DataFrame()

    # goes trough every ticker in our list of thickers
    for ticker in tickers:
        try:
            # adds them as a stock symbol to our list of equities, so QuantBook can work with it
            symbol = qb.AddEquity(ticker).Symbol

            # get daily price data for the ticker currently in, over our defined time periods (start -> end)
            df = qb.History(symbol, start, end, Resolution.Daily)["close"].unstack(level=0)

            if df.empty:
                print(f"No data available for {ticker} in the specified date range.")
                continue
            # add the df referring to ONE ticker to our results dataframe as its own column
            results[ticker] = df

        except KeyError:
            print(f"Data for {ticker} not found.")
            continue
        
    if results.isnull().values.any():
        print("WARNING, The DataFrame containts NaN value(s). The specified range is probably too long.")
    # calculate returns for each stock using pct_change() and fillna(0) [first values would be NaN]
    stock_returns = results.pct_change().fillna(0)

    # turn results dataframe with "normal" prices into a dataframe of logarithmic prices
    stock_prices = np.log(results)

    return stock_returns, stock_prices

# function that tests for cointegration between pairs of tickers
def cointegration(data, tickers):

    # we want to store the results in a dict, so we have to create one
    results_cointegration = {}

    # we define a list of symbols (list_symbols) that contains the column name of every 
    #list_symbols  = data.columns 
    possible_pairs = list(combinations(tickers, 2))

    # Cointegration:
            #  
    for pair in possible_pairs:
        score, p_value, _ = coint(data[pair[0]], data[pair[1]])
        
        if p_value <= 0.05:
            results_cointegration[pair] = "cointegrated"

    print(results_cointegration)

    return results_cointegration

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
        

        # uncomment to visualize each pairs spread
        #spread[key].plot(figsize=(12,6))
        #plt.axhline(spread[key].mean(), color='black')
        #plt.legend(['Spread between ' + key[0] + " and " + key[1]])
        #plt.show()

    return spread

def z_score(spread_pairs):

    z_scores = pd.DataFrame()

    for pair in spread_pairs.columns:
        
        z_scores[pair] = (spread_pairs[pair] - spread_pairs[pair].mean()) / spread_pairs[pair].std() 
        
        # uncomment to visualize z_scores
        #z_scores[pair].plot(figsize=(12,6))
        #plt.axhline(z_scores[pair].mean())
        #plt.axhline(1.0, color='red')
        #plt.axhline(-1.0, color='green')
        #plt.show()
        
    return z_scores

def trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns):

    strategy_returns = pd.DataFrame(index=stock_prices.index)
    
    for pair in spreads.columns:

            returns_for_pair = []

            for i in range(len(stock_prices)-1):

                if z_scores[pair][i] < -std_open:
                    returns_for_pair.append(- stock_returns[pair[1]][i+1] + stock_returns[pair[0]][i+1])

                elif z_scores[pair][i] > std_open:
                    returns_for_pair.append(+ stock_returns[pair[1]][i+1] - stock_returns[pair[0]][i+1])

                else:
                    returns_for_pair.append(0)

            returns_for_pair.append(0)

            strategy_returns[pair] = returns_for_pair

    return strategy_returns

def cumulative_returns(returns):

    cumulative_return = {}
    pair_returns = pd.DataFrame()

    for pair in returns.columns:

        pair_returns[pair] = (1 + returns[pair]).cumprod() -1
        cumulative_return[pair] = pair_returns[pair].iloc[-1]
        
    return cumulative_return

def visualize_strategy_performance(strategy_returns):
    """
    Visualize the performance of the strategy over time.
    """

    for pair in strategy_returns.columns:

        cumulative_portfolio_returns = (1 + strategy_returns[pair]).cumprod() - 1
        
        plt.figure(figsize=(15, 6))
        plt.plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns.values * 100)  # Convert to percentage
        plt.title(f"Strategy Cumulative Returns over Time for {pair}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns (%)")
        plt.legend(["Strategy Performance"])
        plt.grid(True)

def compute_sharpe_ratio(returns, rfr, cumulative_returns):

    results = {}
    """
    Calculate the Sharpe ratio of the strategy based on the returns.
    """
    for col in returns.columns:

        expected_portfolio_return = (cumulative_returns[col] + 1) ** (1/ (len(returns)/252)) - 1
        annual_volatility = returns[col].std() * np.sqrt(252)

        sharpe_ratio = (expected_portfolio_return - rfr) / annual_volatility

        results[col] = sharpe_ratio 
    
    return results

rfr = 0.02
std_open = 1
std_out = 0.5 # currently not used, but maybe in next weeks version
tickers = ['KO', 'PEP', 'ADBE', 'MSFT']
start = datetime(2010,1,1)
end = datetime(2018,1,1)

# assigns the returned dataframes from getStockInfo in the correct order to new dataframes (that we name the same, because they contain the same information and we're outside of the function now, so there won't be any interference)
stock_returns, stock_prices = getStockInfo(tickers, start, end)
results_cointegration = cointegration(stock_prices, tickers)
spreads = spread(results_cointegration)
z_scores = z_score(spreads)
returns = trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns)
cum_return = cumulative_returns(returns)    
visualize_strategy_performance(returns)
sharpe_ratios = compute_sharpe_ratio(returns, rfr, cum_return)

# Displaying cumulative returns
print("Cumulative Returns:")
for pair, ret in cum_return.items():
    print(f"{pair[0]} and {pair[1]}: {ret:.2%}")

# Display each strategies Sharpe-Ratio
print("Sharpe Rations:")
for pair, sr in sharpe_ratios.items():
    print(f"{pair}: {sr:.2}")
