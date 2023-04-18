import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# Goal: make a list with the tickers that we want the data for

# yfinance does not supply Index Data -> we'll need to replicate it with ETFs
# Ticker symbols for the S&P 500 ETF, S&P 500 Momentum ETF, S&P 500 Quality ETF, and S&P 500 High Beta ETF
# no ETFs for Seasonality and Carry

tickers = ["SPY", "SPMO", "SPHQ", "SPHB"]


# Goal: download historical data for the tickers

# from October 12th 2015 till yesterday [we'll use datetime to do this]
# using today is enough, because yfinance will only supply us with data till yesterday

today = datetime.today().strftime('%Y-%m-%d')
data = yf.download(tickers, start="2015-10-12", end=today)


# Goal: Extract the losing prices for each ticker from our dataframe "data"
# turn it into a new dataframe called "df"

df = data["Close"]


# Goal: find portfolio (weights) that maximises the Sharpe Ratio (remember: a important measure for risk-adjusted returns)

# we do this using pypfopt, to get the expected returns of each asset and their covariance

# Note: we'll have to drop the SPY column for this one
df = df.drop(["SPY"], axis=1)

# calculate expected returns for each asset
returns = expected_returns.mean_historical_return(df)

# calculate the covariances between the assets
covariance = risk_models.sample_cov(df)

# we'll use Efficient Frontier for our optimization, "providing" the expected returns and the covariances as the arguments
ef = EfficientFrontier(returns, covariance)

# calculates the optimal weighting for each asset, given a risk free rate of 2% [can be changed, given the macro environment]
# and that we use efficient frontier to maximize the sharpe ratio
raw_weightings= ef.max_sharpe(risk_free_rate=0.02) # weightings are returned as a dict

#"cleaning" the weights just means rounding to fifth decimal place, if a weighting is lower than 1/10000 it's rounded to 0
cleaned_weightings = ef.clean_weights() 

# safe the performance statistics: 
# expected return, volatility and sharpe ratio (using default risk free rate of 2%)
statistics = ef.portfolio_performance() 


# Goal: Get daily simple return of our portfolio given the weightings -> then turn into log returns for symmetry and time additivity

# calculate percentage change in regard to last period for every column
df = df.pct_change()

# calculate simple return of our portfolio given daily returns for each asset and their weighting
df["portfolio simple return"] = df["SPHB"] * cleaned_weightings["SPHB"] + df["SPMO"] * cleaned_weightings["SPMO"] + df["SPHQ"] * cleaned_weightings["SPHQ"] 

# turn simple returns into log returns
df["portfolio log return"] = np.log(1 + df["portfolio simple return"]) 

# turn SPY closing prices into daily returns
df["spy log return"] = np.log(data["Close"]["SPY"]).diff() 

# fills NaN values with 0 -> needed since pct_change for first period is NaN
df.fillna(0, inplace=True)

# plot our portfolio against the spy
plt.plot(np.exp(df["portfolio log return"]).cumprod(), label = "Return of Our Portfolio")
plt.plot(np.exp(df["spy log return"]).cumprod(), label = "Return  of Spy")
plt.legend(loc=2)
plt.title('Our Factor Portfolio against the SPY')
plt.grid(True, alpha = 0.5)
plt.show()






plt.show()
