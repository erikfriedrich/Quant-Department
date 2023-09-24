# Initiate QuantBook, so we can get Data provided by QuantConnect
qb = QuantBook()

# define the equities, we want to get the data for
spy = qb.AddEquity("SPY", Resolution.Daily).Symbol
tsla = qb.AddEquity("TSLA", Resolution.Daily).Symbol

# download trading data for spy and tesla for the last 3 000 days
history = qb.History([spy, tsla], 3000)

# we define a new dataframe that only contains the closing price and unstack it

# UNSTACK

# [---SPY---]
# [---TSLA---]

# =>

# [---SPY---][---TSLA---]

df = history["close"].unstack(level=0)
df.head()

# make new columns containing daily returns
df["Spy Returns"] = (df["Spy"].pct_change()).fillna(0)
df["Tsla Returns"] = (df["Tsla"].pct_change()).fillna(0)
df.head()

# get some statistics on the daily returns
df["Spy Returns"].describe()
df["Tsla Returns"].describe()

# get the covariance matrix for our entire dataframe
# but we only need the COV between the Daily Returns and the VAR of Spy Returns
cov_matrix = df.cov()
df.cov()

covariance = cov_matrix.iloc[2,3]
print(f"covariance = {covariance}")

variance = cov_matrix.iloc[2,2]
print(f"variance = {variance}")

# calculating beta by definition
beta = covariance / variance
print(f"beta = {beta}")

# to calculate the alpha we'll need some additional data
# alpha = portfolio return - (risk-free rate + beta*(expected market return - risk-free rate)

# Portfolio Return
preturn = (df["TSLA"][-1]/df["Tsla"][0])**(1/(3000/365.25))-1
print(f"preturn = {preturn}")

# Market Return
mreturn = (df["SPY"][-1]/df["spy"][0])**(1/(3000/365.25))-1
print(f"mreturn = {mreturn}")

# Risk-Free Rate
rfr = 0.02 # more ore less average from last 10-years
print(f"rfr = {rfr}")

# Final-Step: Calculate the Alpha
alpha = preturn-rfr-beta*(mreturn-rfr)
print(f"alpha = {alpha}")

import matplotlib.pyplot as plt
import seaborn as sns

# Line chart of closing prices 
plt.plot(df['Spy'], label='SPY')
plt.plot(df['Tsla'], label='TSLA')
plt.legend()
plt.xlabel('Year')
plt.ylabel('Closing Price')
plt.title('Closing Prices Over Time')
plt.show()

# Scatter plot of daily returns
plt.scatter(df['Spy Returns'], df['Tsla Returns'])
plt.xlabel('SPY Returns')  
plt.ylabel('TSLA Returns')
plt.title('Relationship Between Daily Returns')
plt.show() 

# Histogram of daily returns
plt.hist(df['Spy Returns'], alpha=0.5, label='SPY')
plt.hist(df['Tsla Returns'], alpha=0.5, label='TSLA')
plt.legend()
plt.xlabel('Daily Returns')
plt.ylabel('Frequency')
plt.title('Distribution of Daily Returns')
plt.show()

# Bar chart of risk metrics
metrics = [alpha, beta, preturn, mreturn, rfr]
labels = ['Alpha', 'Beta', 'Portfolio Return', 'Market Return', 'Risk Free Rate']
plt.bar(labels, metrics)
plt.ylabel('Values')
plt.title('Risk Metrics')
plt.show()

# Normalize stock prices
df['Spy_norm'] = df['Spy'] / df['Spy'].iloc[0] 
df['Tsla_norm'] = df['Tsla'] / df['Tsla'].iloc[0]

# Plot normalized prices
plt.plot(df['Spy_norm'], label='SPY Normalized')
plt.plot(df['Tsla_norm'], label='TSLA Normalized')
plt.xlabel('Days')
plt.ylabel('Normalized Price')
plt.title('Normalized Closing Prices Over Time')
plt.legend()
plt.show()

rolling_correlation = df['Spy Returns'].rolling(window=90).corr(df['Tsla Returns'])
plt.figure(figsize=(12,6))
rolling_correlation.plot()
plt.title('90-Day Rolling Correlation between SPY and TSLA Returns')
plt.xlabel('Date')
plt.ylabel('Correlation')
plt.grid(True)
plt.tight_layout()
plt.show()
