import yfinance as yf
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# only using MSCI and iShares to have the best comparability

momentum = yf.Ticker('IS3R.DE') # iShares MSCI World Momentum
quality = yf.Ticker('IS3Q.DE') # iShares MSCI World Quality
world = yf.Ticker('EUNL.DE') # iShares MSCI World

# get historic stock data
# start date for momentum and quality is 2014-10-06 => adjust for world

momentum_history = momentum.history(period='max')
quality_history = quality.history(period='max')
world_history = world.history(start='2014-10-06')

# get first price from each series

momentum_base = momentum_history['Close'].iloc[0]
quality_base = quality_history['Close'].iloc[0]
world_base = world_history['Close'].iloc[0]

# divide each series by first price to get relative prices

momentum_df = momentum_history[['Close']].div(momentum_base)
quality_df = quality_history[['Close']].div(quality_base)
world_df = world_history[['Close']].div(world_base)

# rename columns so we can merge them later

momentum_df = momentum_df.rename(columns={'Close': 'Close_Momentum'})
quality_df = quality_df.rename(columns={'Close': 'Close_Quality'})
world_df = world_df.rename(columns={'Close': 'World'})

# merge momentum and quality

m_q_df = momentum_df.join(quality_df)

# add column with mean (implies that every factor has a weighting if 50%)

m_q_df['Factors'] = m_q_df.mean(axis=1)

# creating backtesting dataframe to visualize, dropping Momentum and Quality because we only need the combination

backtest_df = m_q_df.join(world_df)
backtest_df = backtest_df.drop(['Close_Momentum', 'Close_Quality'], axis=1)

# plotting our backtesting data

plt.plot(backtest_df)
plt.xlabel('Date')
plt.ylabel('Performance')
plt.title('Backtest: Quality and Momentum vs. MSCI World')
plt.legend(['Factors', 'World'], loc='upper left')
plt.show()
