# this is only the first part, with some additions to last week
# you can just copy and paste this


import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', None)

# read in excel file as our dataframe
df = pd.read_excel(r'C:\Users\Erik\Downloads\df_session4.xlsx')

# make date column the index
df = df.set_index('date')

# make a new column with the relative value of gold to silver
df["relative"] = df['gold']/df['silver']

# make two new columns with the log returns of each asset
df['spy return'] = np.log(df['spy']).diff()
df["gold return"] = np.log(df["gold"]).diff()
df["silver return"] = np.log(df["silver"]).diff()

df['simple long gold'] = 0.5 * df['gold'].pct_change(1) - 0.5 * df['silver'].pct_change(1)
df['simple short gold'] = (-0.5) * df['gold'].pct_change(1) + 0.5 * df['silver'].pct_change(1)

df['long both'] = 0.5*df['gold'].pct_change(1) + 0.5*df["silver"].pct_change(1)
df["log long both"] = np.log(df['long both'] + 1)

df['log long gold'] = np.log(df['simple long gold'] + 1)
df['log short gold'] = np.log(df['simple short gold'] + 1)

# calculate the average gold to silver ratio [around 69 in this timeframe]
avg = df['relative'].mean(axis=0)

# calculate the standard deviation of the gold/silver ratio [ around 13 in this timeframe ]
std = df['relative'].std()

# make a ceiling value
    # if relative > avg+std we'll go short gold, long silver
# make a floor value
    # if relative < avg-std we'll go long gold, short silver
# if the avg-std <= relative <= avg+std we'll hold the s&p 500

ceiling = avg+1.5*std
floor = avg-1.5*std

# make our trading signals based on that
df['signal'] = np.where(df["relative"] >= ceiling, "short gold long silver", "long spy")
df['signal'] = np.where(df["relative"] <= floor, "long gold short silver", df['signal'])

# get the returns of our trading strategy depending on the signal
df['strategy spy'] = np.where(df['signal'] == "short gold long silver", df['log short gold'], df['spy return'])
df['strategy spy'] = np.where(df['signal'] == "long gold short silver", df['log long gold'], df['strategy spy'])

df['strategy gs'] = np.where(df['signal'] == "short gold long silver", df['log short gold'], df['log long both'])
df['strategy gs'] = np.where(df['signal'] == "long gold short silver", df['log long gold'], df['strategy gs'])

# make vix strategy, if above 30, we go long gold and long silver instead of long spy

df["strategy vix"] = np.where(df["vix"] > 30, df["log long both"], df["strategy spy"])

# fill in NaN values with 0
df.fillna(0, inplace = True)

# this cuts off all the data before 2000
#df = df.truncate(before = '2000-01-01')

#plotting the different strategies
plt.plot(np.exp(df['strategy spy']).cumprod(), label = "return of our strategy spy")
plt.plot(np.exp(df['spy return']).cumprod(), label = "return of spy")
plt.plot(np.exp(df['strategy gs']).cumprod(), label = 'gold and silver strategy')
plt.plot(np.exp(df['strategy vix']).cumprod(), label = 'vix strategy')

plt.legend(loc=2)
plt.title("Our trading strategy against the S&P 500")
plt.grid(True, alpha = .5)


import seaborn as sns; sns.set()
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
