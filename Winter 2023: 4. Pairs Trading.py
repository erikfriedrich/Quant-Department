import pandas as pd
import numpy as np
from datetime import datetime
from itertools import combinations
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

qb = QuantBook()

# firstly, we'll need to get the following information about the stocks, we want to "investigate":
        # (1) the stocks returns -> to calculate our performance in the end
        # (2) the (log) price of the stocks -> to check for cointegration
                # the linearization helps to detect cointegration

def getStockInfo(tickers, start, end):
    """
    Fetch daily closing prices for the specified tickers over the given date range.

    Parameters:
    - tickers (list): List of stock tickers.
    - start (datetime): Start date for the data retrieval.
    - end (datetime): End date for the data retrieval.

    Returns:
    - stock_returns (DataFrame): Daily returns for each stock.
    - stock_prices (DataFrame): Logarithmic prices for each stock.
    """
    # DataFrame to store the fetched data
    results = pd.DataFrame()

    # Loop through each ticker to retrieve data
    for ticker in tickers:
        try:
            # Add the ticker to QuantBook for data access
            symbol = qb.AddEquity(ticker).Symbol

            # Fetch daily closing prices for the ticker
            df = qb.History(symbol, start, end, Resolution.Daily)["close"].unstack(level=0)

            if df.empty:
                print(f"No data available for {ticker} in the specified date range.")
                continue
            
            results[ticker] = df

        except KeyError:
            print(f"Data for {ticker} not found.")
            continue

    # Warning for missing values
    if results.isnull().values.any():
        print("WARNING: The DataFrame contains NaN value(s). The specified range might be too long.")

    # Calculate daily stock returns and logarithmic prices
    stock_returns = results.pct_change().fillna(0)
    stock_prices = np.log(results)

    return stock_returns, stock_prices

def cointegration(data, tickers):
    """
    Check for cointegration between pairs of tickers.

    Parameters:
    - data (DataFrame): Stock price data.
    - tickers (list): List of tickers.

    Returns:
    - results_cointegration (dict): Pairs that are cointegrated.
    """
    results_cointegration = {}
    possible_pairs = list(combinations(tickers, 2))

    for pair in possible_pairs:
        _, p_value, _ = coint(data[pair[0]], data[pair[1]])
        
        # A p-value below 0.05 suggests cointegration
        if p_value <= 0.05:
            results_cointegration[pair] = "cointegrated"

    print(results_cointegration)
    return results_cointegration

def spread(ticker_pairs):
    """
    Calculate the spread between cointegrated stock pairs.

    Parameters:
    - ticker_pairs (dict): Cointegrated stock pairs.

    Returns:
    - spread (DataFrame): Spread values for each pair.
    """
    spread = pd.DataFrame()

    for key in results_cointegration.keys():
        x = stock_prices[key[1]]
        y = stock_prices[key[0]]

        # Use Ordinary Least Squares (OLS) for regression
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()

        # Calculate spread
        spread[key] = y - model.params[key[1]] * x[key[1]]
        
        # uncomment to visualize each pairs spread
        #spread[key].plot(figsize=(12,6))
        #plt.axhline(spread[key].mean(), color='black')
        #plt.legend(['Spread between ' + key[0] + " and " + key[1]])
        #plt.show()


    return spread

def z_score(spread_pairs):
    """
    Calculate the z-score for spread of stock pairs.

    Parameters:
    - spread_pairs (DataFrame): Spread values for stock pairs.

    Returns:
    - z_scores (DataFrame): Z-scores for each pair.
    """
    z_scores = pd.DataFrame()

    for pair in spread_pairs.columns:
        # Calculate z-score
        z_scores[pair] = (spread_pairs[pair] - spread_pairs[pair].mean()) / spread_pairs[pair].std()

        # uncomment to visualize z_scores
        #z_scores[pair].plot(figsize=(12,6))
        #plt.axhline(z_scores[pair].mean())
        #plt.axhline(1.0, color='red')
        #plt.axhline(-1.0, color='green')
        #plt.show()

    return z_scores

def trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns):
    """
    Determine trading signals based on z-scores.

    Parameters:
    - stock_prices (DataFrame): Logarithmic prices for each stock.
    - spreads (DataFrame): Spread values for each pair.
    - z_scores (DataFrame): Z-scores for each pair.
    - std_open (float): Z-score threshold for trading signals.
    - std_out (float): Not currently used.
    - stock_returns (DataFrame): Daily returns for each stock.

    Returns:
    - strategy_returns (DataFrame): Strategy returns based on trading signals.
    """
    strategy_returns = pd.DataFrame(index=stock_prices.index)

    for pair in spreads.columns:
        returns_for_pair = []

        # Determine buy/sell/hold signals based on z-scores
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
    """
    Calculate cumulative returns for the strategy.

    Parameters:
    - returns (DataFrame): Strategy returns based on trading signals.

    Returns:
    - cumulative_return (dict): Cumulative returns for each stock pair.
    """
    cumulative_return = {}
    pair_returns = pd.DataFrame()

    for pair in returns.columns:
        # Calculate cumulative returns
        pair_returns[pair] = (1 + returns[pair]).cumprod() -1
        cumulative_return[pair] = pair_returns[pair].iloc[-1]

    return cumulative_return

def visualize_strategy_performance(strategy_returns):
    """
    Plot cumulative returns for the strategy.

    Parameters:
    - strategy_returns (DataFrame): Strategy returns based on trading signals.
    """
    for pair in strategy_returns.columns:
        cumulative_portfolio_returns = (1 + strategy_returns[pair]).cumprod() - 1

        # Plot cumulative returns
        plt.figure(figsize=(15, 6))
        plt.plot(cumulative_portfolio_returns.index, cumulative_portfolio_returns.values * 100)
        plt.title(f"Strategy Cumulative Returns over Time for {pair}")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Returns (%)")
        plt.legend(["Strategy Performance"])
        plt.grid(True)

def compute_sharpe_ratio(returns, rfr, cumulative_returns):
    """
    Calculate Sharpe ratio for the strategy.

    Parameters:
    - returns (DataFrame): Strategy returns based on trading signals.
    - rfr (float): Risk-free rate.
    - cumulative_returns (dict): Cumulative returns for each stock pair.

    Returns:
    - results (dict): Sharpe ratio for each stock pair.
    """
    results = {}

    for col in returns.columns:
        expected_portfolio_return = (cumulative_returns[col] + 1) ** (1/ (len(returns)/252)) - 1
        annual_volatility = returns[col].std() * np.sqrt(252)

        # Compute the Sharpe ratio
        sharpe_ratio = (expected_portfolio_return - rfr) / annual_volatility

        results[col] = sharpe_ratio

    return results

# Define constants and parameters
rfr = 0.02  # Risk-free rate
std_open = 1  # Z-score threshold for opening positions
std_out = 0.5  # Z-score threshold for closing positions (currently unused)
tickers = ['KO', 'PEP', 'ADBE', 'MSFT']  # List of stock tickers to analyze
start = datetime(2010,1,1)  # Start date for data retrieval
end = datetime(2018,1,1)  # End date for data retrieval

# Execute the trading strategy pipeline
stock_returns, stock_prices = getStockInfo(tickers, start, end)
results_cointegration = cointegration(stock_prices, tickers)
spreads = spread(results_cointegration)
z_scores = z_score(spreads)
returns = trade(stock_prices, spreads, z_scores, std_open, std_out, stock_returns)
cum_return = cumulative_returns(returns)
visualize_strategy_performance(returns)
sharpe_ratios = compute_sharpe_ratio(returns, rfr, cum_return)

# Display cumulative returns for each stock pair
print("Cumulative Returns:")
for pair, ret in cum_return.items():
    print(f"{pair[0]} and {pair[1]}: {ret:.2%}")

# Display Sharpe ratio for each stock pair
print("Sharpe Ratios:")
for pair, sr in sharpe_ratios.items():
    print(f"{pair}: {sr:.2f}")
