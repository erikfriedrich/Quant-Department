import pandas as pd
import numpy as np 
import datetime
import matplotlib.pyplot as plt

pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', None)

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
df['signal'] = np.where(df["relative"] > ceiling, "short gold long silver", "long spy")
df['signal'] = np.where(df["relative"] < floor, "long gold short silver", df['signal'])

# get the returns of our trading strategy depending on the signal
df['strategy'] = np.where(df['signal'] == "short gold long silver", (-1)*df['gold return'], df['spy return'])
df['strategy'] = np.where(df['signal'] == "long gold short silver", (-1)*df['silver return'], df['strategy'])

# fill in NaN values with 0
df.fillna(0, inplace = True)

# this cuts off all the data before 2000
df = df.truncate(before = '2000-01-01')

# plotting the different strategies
plt.plot(np.exp(df['strategy']).cumprod(), label = "return of our strategy")
plt.plot(np.exp(df['spy return']).cumprod(), label = "return of spy")
plt.legend(loc=2)
plt.title("Our trading strategy against the S&P 500")
plt.grid(True, alpha = .5)

# calculate volatilities and plot them against eachother
df['Strategy Volatility'] = df['strategy'].rolling(window=252).std() * np.sqrt(252)
df['S&P Volatility'] = df['spy return'].rolling(window=252).std() * np.sqrt(252)

df[['Strategy Volatility', 'S&P Volatility']].plot(figsize=(8,6))

# print correlation between our strategy and s&p 
corr_strategy_spy = df['strategy'].corr(df['spy return'])
print("correlation of our strategy to the spy:", corr_strategy_spy)
