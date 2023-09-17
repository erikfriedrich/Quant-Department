# import libraries that we might/will need
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

# run the following command if you get an error with the libraries: pip install ortools==9.4.0

# initiate QuantBook
qb = QuantBook()

# get data for the factors: Quality, Momentum, Value, Size, Small Cap and Emerging Markets
equities = ["QUAL", "MTUM", "VLUE", "SIZE", "IJR", "IEMG"]

for ticker in equities:
    qb.AddEquity(ticker)

df = qb.History(qb.Securities.Keys, 2524, Resolution.Daily)["close"].unstack(level=0) # 2524 might seem weird but it maximizes our timeframe

# df or print(df) <- to see the dataframe in research.ipynb


# to use EfficientFrontier we have to provide the returns of each asset as well as the covariances
returns = expected_returns.mean_historical_return(df)
covariance = risk_models.sample_cov(df)

# initiate Efficient Frontier, with weight bounds so that we don't leave out any factors or have one with our entire portfolio in it
# it will return to us the optimal weighting of our portfolio
ef = EfficientFrontier(returns, covariance, weight_bounds=(0.05, 0.50))
raw_weightings= ef.max_sharpe(risk_free_rate=0.02)
cleaned_weightings = ef.clean_weights()

# print(raw_weights)
# print(cleaned_weights)


# pct change to calculate return
df = df.pct_change()
df.fillna(0, inplace=True)

# add a new column with the return of our portfolio
df["portfolio simple return"] = sum(df[symbol] * cleaned_weightings[symbol] for symbol in cleaned_weightings.keys())

# Add Spy for comparison
SPY = qb.AddEquity("SPY").Symbol
df["Spy"]= qb.History(SPY, 2524, Resolution.Daily)["close"].unstack(level=0).pct_change().fillna(0)
df

# Cumulative Return Plot
cumulative_returns = (df[['portfolio simple return']] + 1).cumprod()
spy_returns = (df["Spy"] + 1).cumprod()

fig, ax = plt.subplots(figsize=(12, 6))

cumulative_returns.plot(ax=ax, label='Portfolio')
spy_returns.plot(ax=ax, label='SPY')

plt.legend()

# Customize the plot
ax.set_title('Cumulative Returns Over Time vs SPY')
plt.grid(True, alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()

# Show the plot
plt.show()

# a. Volatility
portfolio_volatility = df["portfolio simple return"].std()
print(f"Portfolio Volatility: {portfolio_volatility:.4f}")

# b. Maximum Drawdown
cumulative_return = (df["portfolio simple return"] + 1).cumprod()
max_drawdown = ((cumulative_return.cummax() - cumulative_return) / cumulative_return.cummax()).max()
print(f"Maximum Drawdown: {max_drawdown:.4f}")

# 4. Correlation Matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 7))
plt.title('Correlation Matrix Heatmap')
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.tight_layout()
plt.show()
