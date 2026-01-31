"""
Trend Following Trading Strategy
Based on FX Futures
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

#===================================================================
#FUNCTIONS
#===================================================================

def download(ticker, start="2020-01-01", end="2025-01-01"):
    """
    Download data from Yahoo Finance
	INPUT:
	ticker:	str; ticker symbol for Yahoo Finance
	start: 	str; date with default
	end: 	str; date with default

    RETURN:
    df:  	Pandas DataFrame; daily closing prices
    """
    try:
        df = pd.read_csv('data.csv')
        print('Using stored data ...')
    except:
        print('Downloading ...')
        df = yf.download(ticker, start=start, end=end)
        df.to_csv('data.csv', header = False)
    #Focus on closing prices
    df = df.iloc[:,:2]
    df.columns = ['Date', 'Close']
    df = df.set_index('Date')
    return df

def compute_signals(df, short_sma=30, long_sma=200):
	"""INPUT:
	df:			DataFrame, indexed with closing prices labelled 'Close'
	short_ma:	int; short moving average; default 30 days
	long_ma:	int; long moving average; default 200 days
	RETURN:
	df:			DataFrame, indexed with added trading signals
	"""
	df["sma_short"] = df["Close"].rolling(short_sma).mean()
	df["sma_long"] = df["Close"].rolling(long_sma).mean()
	df["signal"] = 0
	df.loc[df["sma_short"] > df["sma_long"], "signal"] = 1
	df.loc[df["sma_short"] < df["sma_long"], "signal"] = -1
	df["position"] = df["signal"].shift(1).fillna(0)
	df = df.dropna()
	return df


def backtest(df_signal, initial_capital=100, tx_cost=0.0, leverage=1.0):
	"""
	Backtest using daily returns and signals/positions
	INPUT:
	df_signal: 			DataFrame; closing prices (Close) and positions
	initial_capital:	int; intitial capital; default settings
	tx_cost:			float; transaction costs; default settings
	leverage:			float; leverage due to margin; default settings 

	RETURN:
	df:					DataFrame; trading returns, cumulative equity 
	
	"""
	df = df_signal.copy()
	df['Return'] = np.log(df.Close) - np.log(df.Close.shift(1))
	df = df.dropna()
	#Identify trades if position changes
	df['pos_change'] = df['position'].diff().fillna(0).abs()
	#Transaction cost applied on entry and exit 
	df['tcost'] = df['pos_change'] * tx_cost
	#Strategy returns: position * returns - transaction costs 
	#Leverage applied
	df['strat_ret'] = (df['position'] * df['Return'] - df['tcost'])*leverage
	#Equity position
	df['equity'] = np.exp(df['strat_ret'].cumsum()) * initial_capital
	return df

#===================================================================
#PARAMETERS
#===================================================================
START_DATE = "2015-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 1000
SHORT_SMA = 50
LONG_SMA = 200
TRANSACTION_COST = 0.00005
TICKER = 'GBPUSD=X'
LEVERAGE = 30

df = download(TICKER, START_DATE, END_DATE)
print(df)

df_signal = compute_signals(df, SHORT_SMA, LONG_SMA)
print(df_signal)

df_result = backtest(df_signal, INITIAL_CAPITAL, TRANSACTION_COST, LEVERAGE)
print(df_result)

#Remove index for plotting
df_result = df_result.reset_index(drop=True)

#Plotting
y = df_result['Close']
y1 = df_result['sma_short']
y2 = df_result['sma_long']
y3 = df_result['equity']

fig, (ax1, ax2) = plt.subplots(
    2, 1, 
    sharex=True, 
    figsize=(8, 6),
    gridspec_kw={'height_ratios': [2, 1]}  # top subplot a bit larger
)

#Trading signals
ax1.plot(y, label='FX', linewidth=2)
ax1.plot(y1, label='Short MA', color='tab:red', linewidth=2)
ax1.plot(y2, label='Long MA', color='tab:orange', linewidth=2, linestyle='--')
ax1.set_ylabel('Trading signals')
ax1.set_title('Short and long-term MA')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

#Equity position
ax2.plot(y3, label='Equity', color='tab:green', linewidth=2)
ax2.set_ylabel('Value')
ax2.set_xlabel('X-axis')
ax2.set_title('Trading position')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

#Improve layout spacing
fig.tight_layout()
plt.subplots_adjust(hspace=0.3)
plt.savefig('Fig_trade.png')
plt.show()
