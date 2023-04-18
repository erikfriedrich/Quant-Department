# the first part is from last weeks session
# the second part focuses on computing various measurements and visualizing them
    # participants will have to work on their own strategies during the exam, followed by a presentation
    # during this presentation there should also be an emphasis on these measurements

# import the libraries we'll need
import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# display numbers to the third decimal place
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# print out all rows of a dataframe
pd.set_option('display.max_rows', None)

# read in excel file as our dataframe
df = pd.read_excel(r"C:\Users\Erik\Downloads\df_session4.xlsx")

# make date column the index
df = df.set_index('date')

# this cuts off all the data before 2000
df = df.truncate(before = '2000-01-01')

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

# this is where the New Part begins
# make vix strategy, if above 30, we go long gold and long silver instead of long spy
    # thesis: market is to volatile/ unpredictable -> our investors want "safety"

df["strategy vix"] = np.where(df["vix"] > 30, df["log long both"], df["strategy spy"])

# fill in NaN values with 0
df.fillna(0, inplace = True)

#plotting the different strategies
plt.plot(np.exp(df['strategy spy']).cumprod(), label = "return of our strategy spy")
plt.plot(np.exp(df['spy return']).cumprod(), label = "return of spy")
plt.plot(np.exp(df['strategy gs']).cumprod(), label = 'gold and silver strategy')
plt.plot(np.exp(df['strategy vix']).cumprod(), label = 'vix strategy')

plt.legend(loc=2)
plt.title("Our trading strategy against the S&P 500")
plt.grid(True, alpha = .5)
plt.show()

# get annualized returns -> need the simple return 
    # avg return -> 1 year
    
strategy_vix_simple_return = np.exp(df["strategy vix"]) - 1
strategy_vix_mean_simple_return = np.mean(strategy_vix_simple_return)

annualized_return_strategy_vix = strategy_vix_mean_simple_return * 252 * 100 # tumes 100 for displaying purposes, it's obviously not mathematically correct

# volatility for 1 year, based on daily prices
    # 252 is approx. the number of trading days in a year
    # if you need a quick estimate u can use 256, because sqrt(256)=16 but mind you, it's a little bit overestimated then

vola_vix = df['strategy vix'].std()*np.sqrt(252) * 100 # times 100 - again, for displaying purposes


# goal: calculate sharpe ratio
risk_free_rate = 3 # should be 0.03 or 3%, but because we've multplied everything before with 100, we need to it again 
sharpe_ratio_vix = (annualized_return_strategy_vix - risk_free_rate) / vola_vix
# how do we view the sharpe ratio in our case [think as a portfolio manager, but also as an investor]


# goal: calculate beta 

# calculate covariance between our strategy and the spy
# the beta is defined as the covariance between the market returns and the individual returns of a stock (in our case the strategy) divided by the variance of the market returns
covariance_vix = np.cov(df["strategy vix"], df["spy return"])[0][1]

# calculate variance of spy 
variance_spy = np.var(df['spy return'])

# final computation
beta_vix = covariance_vix / variance_spy


# goal: calculate alpha
    # remember the CAPM from our first session?
    
#  Alpha = Portfolio Return - (Risk-Free Rate) - beta * (Market Return - (Risk-Free Rate))

portfolio_return =  np.exp(df['strategy vix']).cumprod()[-1] - 1
market_return = np.exp(df['spy return'])[-1] - 1
risk_free_rate = 0.03

alpha_vix = portfolio_return - risk_free_rate - beta_vix *  (market_return - risk_free_rate)

# name macro economic reason, that could explain this outperformance ?
    # is our assumption of a risk-free rate of 3% fair, given a timeframe of 23 years?
        # think back


# Goal: look how the returns of our strategy are disributed

# calculate simple returns of our strategy
    # easier to intepret than log returns
    
df["simple return vix"] = (np.exp(df["strategy vix"])-1) * 100 # times 100 for displaying purposes

plt.hist(df["simple return vix"], bins = np.arange(-10, 10, 0.25))
plt.xlabel("Returns in %")
plt.ylabel("Frequency")
plt.title("Histogram of Returns for Strategy Vix")
plt.show()

# calculate skewness and kurtosis (that's why we imported scipy.stats)

# if they're more tilted to the right or left (more positive or negative returns)
skewness = skew(df["simple return vix"])

# how "pointy" the curve is, compared to normal distribution
kurtosis = kurtosis(df["simple return vix"])




