import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

# display decimals to third place
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# print ALL rows
# (don't do it if your PC is slow)
pd.set_option('display.max_rows', None)

# read in excel file as our dataframe
    # if you've forgotten how to do it, look into last sessions code
df = pd.read_excel(r"/Users/erikfriedrich/Downloads/df_session3.xlsx")

# make date column the index
df = df.set_index('date')


# Strategy:
# compare relative value of gold and silver to historic average

    # make a ceiling value
        # if relative > avg+1,5*std we'll go short gold, long silver
    # make a floor value
        # if relative < avg-1,5*std we'll go long gold, short silver
    # if the avg-std <= relative <= avg+std we'll (1) hold the s&p 500 or (2) go long silver and gold
        # make a new column with the relative value of gold to silver
        
df["relative"] = df['gold']/df['silver']

# make two new columns with the log returns of each asset
df['spy return'] = np.log(df['spy']).diff()
df["gold return"] = np.log(df["gold"]).diff()
df["silver return"] = np.log(df["silver"]).diff()

# simple return if we go long gold with half our portfolio and short silver with the other half
    # the one in the brackets indicates how many periods back we go
        # try to experiment with different values
        
df['simple long gold'] = 0.5 * df['gold'].pct_change(1) - 0.5 * df['silver'].pct_change(1)

# simple return if we go short gold with half our portfolio and long silver with the other
df['simple short gold'] = (-0.5) * df['gold'].pct_change(1) + 0.5 * df['silver'].pct_change(1)

# get the return if we're long both -> if the relative value of gold and silver is within our acceptible bandwith
df['long both'] = 0.5*df['gold'].pct_change(1) + 0.5*df["silver"].pct_change(1)

# turn all the simple returns into log returns
df["log long both"] = np.log(df['long both'] + 1)
df['log long gold'] = np.log(df['simple long gold'] + 1)
df['log short gold'] = np.log(df['simple short gold'] + 1)

# out of interest
# calculate correlation between gold and silver ( around 0.9 )
corr_gs = df['gold'].corr(df['silver'])

# to get floor and ceiling

# calculate the average gold to silver ratio [around 57.5]
avg = df['relative'].mean(axis=0)

# calculate the standard deviation of the gold/silver ratio [ around 19.9 ]
std = df['relative'].std()

# calculate floor and ceiling based on the avg relative value and the std of the relative values
ceiling = avg+1.5*std
floor = avg-1.5*std

# make our trading signals based on that
df['signal'] = np.where(df["relative"] > ceiling, "short gold long silver", "long spy")
df['signal'] = np.where(df["relative"] < floor, "long gold short silver", df['signal'])

# get the returns of our trading strategy depending on the signal

df['strategy 1'] = np.where(df['signal'] == "short gold long silver", (-1)*df['gold return'], df['spy return'])
df['strategy 1'] = np.where(df['signal'] == "long gold short silver", (-1)*df['silver return'], df['strategy'])

df['strategy 2'] = np.where(df['signal'] == "short gold long silver", df['log short gold'], df['log long both'])
df['strategy 2'] = np.where(df['signal'] == "long gold short silver", df['log long gold'], df['strategy3'])

# fill in NaN values with 0
df.fillna(0, inplace = True)

# this cuts off all the data before 2000
df = df.truncate(before = '2000-01-01')

# plotting the different strategies
plt.plot(np.exp(df['strategy 1']).cumprod(), label = "return of our strategy 1 (with spy)")
plt.plot(np.exp(df['spy return']).cumprod(), label = "return of spy")
plt.plot(np.exp(df['strategy2']).cumprod(), label = 'return of our strategy 2 (without spy)')
plt.legend(loc=2)
plt.title("Our trading strategy against the S&P 500")
plt.grid(True, alpha = .5)

# calculate (daily) volatilities and plot them against eachother    
    # what implications could the volatility of a strategy have for a Portfolio Manager?
    
df['Strategy Volatility'] = df['strategy'].rolling(window=252).std() * np.sqrt(252)
df['S&P Volatility'] = df['spy return'].rolling(window=252).std() * np.sqrt(252)
df[['Strategy Volatility', 'S&P Volatility']].plot(figsize=(8,6))

# print correlation between our strategy and s&p
    # give me an interpretation of this value
corr_strategy_spy = df['strategy'].corr(df['spy return'])






print("correlation of our strategy to the spy:", corr_strategy_spy)
