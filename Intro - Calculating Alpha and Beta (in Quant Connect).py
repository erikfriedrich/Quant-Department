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
covariance

variance = cov_matrix.iloc[2,2]
variance

# calculating beta by definition
beta = covariance / variance
beta

# to calculate the alpha we'll need some additional data
# alpha = portfolio return - (risk-free rate + beta*(expected market return - risk-free rate)

# Portfolio Return
preturn = (df["TSLA"][-1]/df["Tsla"][0])**(1/(3000/365.25))-1
print(preturn)

# Market Return
mreturn = (df["SPY"][-1]/df["spy"][0])**(1/(3000/365.25))-1
print(mreturn)

# Risk-Free Rate
rfr = 0.02 # more ore less average from last 10-years
print(rfr)

# Final-Step: Calculate the Alpha
alpha = preturn-rfr-beta*(mreturn-rfr)
alpha
