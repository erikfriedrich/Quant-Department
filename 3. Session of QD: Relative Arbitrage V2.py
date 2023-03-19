import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 10)

# read in excel file as our dataframe
df = pd.read_excel(r"C:\Users\Erik\Desktop\df_session3.xlsx")

# make date column the index
df = df.set_index('date')

# make a new column with the relative value of gold to silver
df["relative"] = df['gold']/df['silver']

# make two new columns with the log returns of each asset
df['spy return'] = np.log(df['spy']).diff()
df["gold return"] = np.log(df["gold"]).diff()
df["silver return"] = np.log(df["silver"]).diff()

# columns for each part of the strategy (long gold, short silver and short gold, long silver)
df['simple long gold'] = 0.5 * df['gold'].pct_change(1) - 0.5 * df['silver'].pct_change(1)
df['simple short gold'] = (-0.5) * df['gold'].pct_change(1) + 0.5 * df['silver'].pct_change(1)

# second strategy
df['long both'] = 0.5*df['gold'].pct_change(1) + 0.5*df["silver"].pct_change(1)
df["log long both"] = np.log(df['long both'] + 1)

# turn the simple returns into log returns
df['log long gold'] = np.log(df['simple long gold'] + 1)
df['log short gold'] = np.log(df['simple short gold'] + 1)

# calculate correlation between gold and silver ( around 0.9 )
corr_gs = df['gold'].corr(df['silver'])

# calculate the average gold to silver ratio [around 57.5]
avg = df['relative'].mean(axis=0)

# calculate the standard deviation of the gold/silver ratio [ around 19.9 ]
std = df['relative'].std()

# make a ceiling value
    # if relative > avg+std we'll go short gold, long silver
# make a floor value
    # if relative < avg-std we'll go long gold, short silver
# if the avg-std <= relative <= avg+std we'll hold the s&p 500

ceiling = avg+1.5*std
floor = avg-1.5*std

# make our trading signals based on that
df['signal'] = np.where(df["relative"] > ceiling, "short gold", "long spy")
df['signal'] = np.where(df["relative"] < floor, "long gold", df['signal'])

# get the returns of our trading strategy depending on the signal
df['strategy'] = np.where(df['signal'] == "short gold", df['log short gold'], df['spy return'])
df['strategy'] = np.where(df['signal'] == "long gold", df['log long gold'], df['strategy'])

# second strategy

df['2 strategy'] = np.where(df['signal'] == "short gold", df['log short gold'], df['log long both'])
df['2 strategy'] = np.where(df['signal'] == 'long gold', df['log long gold'], df['2 strategy'])

# fill in NaN values with 0
df.fillna(0, inplace = True)

# cut off all data before 2000 
df = df.truncate(before = '2000-01-01')

# plot our strategy against the spy
plt.plot(np.exp(df['strategy']).cumprod(), label = "return of our strategy")
plt.plot(np.exp(df['spy return']).cumprod(), label = "return of spy")
plt.plot(np.exp(df['2 strategy']).cumprod(), label = 'return of second strategy')
plt.legend(loc=2)
plt.title("Our trading strategy against the S&P 500")
plt.grid(True, alpha = .5)

# calculate the volatilities and plot them against eachother
df['Strategy Volatility'] = df['strategy'].rolling(window=252).std() * np.sqrt(252)
df['S&P Volatility'] = df['spy return'].rolling(window=252).std() * np.sqrt(252)
df["2 Strategy Volatility"] = df['2 strategy'].rolling(window=252).std() * np.sqrt(252)

df[['Strategy Volatility', 'S&P Volatility', '2 Strategy Volatility']].plot(figsize=(8,6))

# print correlation between our strategies and s&p 
corr_strategy_spy = df['strategy'].corr(df['spy return'])
print("correlation of our strategy to the spy:", corr_strategy_spy) # 0.9 (pracitcally perfect)

corr_2strategy_spy = df['2 strategy'].corr(df['spy return'])
print("correlation of our second strategy to the spy:", corr_2strategy_spy) # 0.1 (almost no correlation)
