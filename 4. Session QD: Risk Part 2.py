# THIS ONLY WORKS WITH PART 1 OF THE CODE

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

#df = df.truncate(before = "2020-01-01", after = "2020-06-01")

# get annualized returns -> need the simple return 
    # avg return -> 1 year
    
strategy_vix_simple_return = np.exp(df["strategy vix"]) - 1
strategy_vix_mean_simple_return = np.mean(strategy_vix_simple_return)

annualized_return_strategy_vix = strategy_vix_mean_simple_return * 252 * 100
#print(annualized_return_strategy_vix)


# volatility for 1 year, based on daily prices

vola_vix = df['strategy vix'].std()*np.sqrt(252) * 100
#print("vola", vola_vix)

# sharpe ratio
sharpe_ratio_vix = (annualized_return_strategy_vix - 3) / vola_vix


# goal: beta 

covariance_vix = np.cov(df["strategy vix"], df["spy return"])[0][1]
variance_spy = np.var(df['spy return'])

beta_vix = covariance_vix / variance_spy


# risk_free_rate = 3% -> calculate our alpha

df["simple return vix"] = (np.exp(df["strategy vix"])-1) * 100

plt.hist(df["simple return vix"], bins = 20)
plt.xlabel("Returns in %")
plt.ylabel("Frequency")
plt.title("Histogram of Returns for Strategy Vix")
plt.show()

# calculate skewness and kurtosis (that's why we imported scipy.stats)
skewness = skew(df["simple return vix"])
kurtosis = kurtosis(df["simple return vix"])
