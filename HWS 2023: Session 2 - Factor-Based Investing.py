# import libraries that we might/will need
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

# get data for the factors: Quality, Momentum, Value, Size and Small Cap
QUAL = qb.AddEquity("QUAL")
MTUM = qb.AddEquity("MTUM")
VLUE = qb.AddEquity("VLUE")
SIZE = qb.AddEquity("SIZE")
SVAL = qb.AddEquity("SVAL")

history = qb.History(qb.Securities.Keys, 700, Resolution.Daily)

# history or print(history) <- to see the dataframe in research.ipynb

# unstack the df (as usual)
df = history["close"].unstack(level=0)

# to use EfficientFrontier we have to provide the returns of each asset as well as the covariances
returns = expected_returns.mean_historical_return(df)
covariance = risk_models.sample_cov(df)

# initiate Efficient Frontier, with weight bounds so that we don't leave out any factors or have one with our entire portfolio in it
# it will return to us the optimal weighting of our portfolio
ef = EfficientFrontier(returns, covariance, weight_bounds=(0.05, 0.50))
raw_weightings= ef.max_sharpe(risk_free_rate=0.02)
cleaned_weightings = ef.clean_weights()

# pct change to calculate return
df = df.pct_change()
df.fillna(0, inplace=True)

# add a new column with the return of our portfolio
df["portfolio simple return"] = df["QUAL VIBZ5HTB7N8L"] * cleaned_weightings["QUAL VIBZ5HTB7N8L"] + df["MTUM VFUDGZIY8ZMT"] * cleaned_weightings["MTUM VFUDGZIY8ZMT"] + df["VLUE VFUDGZIY8ZMT"] * cleaned_weightings["VLUE VFUDGZIY8ZMT"] + df["SIZE VFUDGZIY8ZMT"] * cleaned_weightings["SIZE VFUDGZIY8ZMT"] + df["SVAL XJ34A6UAQI5H"] * cleaned_weightings["SVAL XJ34A6UAQI5H"]

# prints the returns of every strategy
plt.plot((df["portfolio simple return"]+1).cumprod(), label = "Return of Our Portfolio")
plt.plot((df["SVAL"]+1).cumprod(), label = "SVAL")
plt.plot((df["QUAL"]+1).cumprod(), label = "QUAL")
plt.plot((df["MTUM"]+1).cumprod(), label = "MTUM")
plt.plot((df["SIZE"]+1).cumprod(), label = "SIZE")
plt.plot((df["VLUE"]+1).cumprod(), label = "VLUE")
plt.legend(loc=2)
plt.title('Our Factor Portfolio')
plt.grid(True, alpha = 0.5)
plt.show()
