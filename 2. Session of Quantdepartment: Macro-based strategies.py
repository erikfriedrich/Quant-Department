import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# this creates a dataframe from the excel file that we've stored our table in
        # download the df_2 excel file from this repository and save it on your pc
            # then copy the path of the file into the quotes
                
# if you don't know how to obtain the path of a file here are some sites that can help you:
        # Mac: https://setapp.com/how-to/how-to-find-the-path-of-a-file-in-mac
        # Windows: https://www.howtogeek.com/670447/how-to-copy-the-full-path-of-a-file-on-windows-10/
        
df = pd.read_excel(r"[insert file path here]") # use the commodities excel if you want oil, else use the regular one

# now we set the date as our index; this will help us later (when we plot it)
df = df.set_index('date')

# strategy: long gold if 10-year lower, short gold if 10-year higher (than last week, can be adapted to monthly etc.)

# new column with the change of the 10-yr compared to last week -> will be used to create trading signals
    # one variation here could be to say that the change has to be of a specific amplitude
        # periods = 1 could be left out since it's the standard value, but it makes it easier to change them in the future
        
    
df['10-yr change'] = df['10-yr'].diff(periods = 1)

# this just changes the display format for floats; otherwise we'd have the 10-yr change column displayed in scientific notation with e

pd.set_option('display.float_format', lambda x: '%.3f' % x)

# make new column 'signal': 1 if 10-yr is lower than week before (we go or stay long), -1 if 10-yr is higher than week before (we go or stay short)

df['signal'] = np.where(df['10-yr change'] <= 0, 1, -1)

# make new column with logarithmic returns of simple buy-and-hold strategy
        # we use logarithmic returns since they're symmetric, meaning that logarithmic returns of the same magnitude but opposite signs will cancel eachother out
                # if u use normal returns -50% and +50% will not get you back to 100% 

df['buy and hold'] = np.log(df['gold']).diff()

# make new column with logarithmic returns of our trading strategy
    # we have to shift all values back by one
        # this is because if we get a trading singal, say "buy"
            # we buy gold and then get the return of the following week

df['strategy'] = df['signal'] * df['buy and hold'].shift(periods = -1)

# make new column that just shows if we would have made any trades

df['trades'] = df['signal'].diff()
df['trades'].value_counts()

# now we want to know how it compares to the s&p 500, for that we first calculate it's return, before plotting it
        # and oil

df['s&p 500 return'] = np.log(df['s&p 500']).diff()
df['oil return'] = np.log(df['commodities']).diff()

# this code will calculate the correlation between the different asstes and our strategy
corr_gold_strategy = df['strategy'].corr(df['buy and hold'])
corr_treasury_strategy = df['strategy'].corr(df['10-yr'])
corr_gold_treasury = df['gold'].corr(df['10-yr'])

# this fills in all 'NaN' values with 0
df.fillna(0, inplace = True)

# 'uncomment' the following line if you want to look at the performance starting from 2010
    # I advise you to try it, the result is interesting
        # df = df.truncate(before='2010-01-01')

# plot buy and hold against our strategy

plt.plot(np.exp(df['buy and hold']).cumprod(), label = 'Standard Buy and Hold')
plt.plot(np.exp(df['strategy']).cumprod(), label = 'Our Trading Strategy')
plt.plot(np.exp(df['s&p 500 return']).cumprod(), label = 'Benchmark: S&P 500')
plt.plot(np.exp(df['oil return']).cumprod(), label = 'Benchmark: Oil')
plt.legend(loc=2)
plt.title('Trading Strategy based on 10-yr treasury and gold')
plt.grid(True, alpha = 0.5)

# Calculate absolute return of Buy n Hold, our Strategy and the S&P 500
print('Returns:')
print(np.exp(df['buy and hold']).cumprod()[-2] - 1, 'Buy and Hold')
print(np.exp(df['strategy']).cumprod()[-2] - 1, 'Our Strategy')
print(np.exp(df['s&p 500 return']).cumprod()[-2] - 1, 'S&P 500')
print(np.exp(df['oil return']).cumprod()[-1] - 1, 'Oil')

# we see that our strategy underperformed very very badly
    # what changes to our strategy could we make to improve it?
    
# calculate the volatility of everything; we'll use 36 weeks instead of 252 days
print('Volatilities:')
print(df['buy and hold'].std()*36**0.5, 'Buy and Hold')
print(df['strategy'].std()*36**0.5, 'Our Strategy')
print(df['s&p 500 return'].std()*36**0.5, 'S&P 500')
print(df['oil return'].std()*36**0.5, 'Oil')

# print out the correlations that we've calculated before
print('Correlation:')
print('Correlation between Gold and our Strategy:', corr_gold_strategy)
print('Correlation between Treasury and our Strategy:',corr_treasury_strategy)
print('Correlation between Gold and Treasury:',corr_gold_treasury)
