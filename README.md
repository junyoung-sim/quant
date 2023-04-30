# Generalized Deep Reinforcement Learning for Trading

## Description

The trading model open-sourced in this repository observes a PAA-discretized multivariate state space consisting of the historical data of a stock of interest and major market-indicating securities (SPY, IEF, EUR=X, GSG), selects a trading action (short, idle, long), and observes a discrete reward based on the correctness of the trading action and independent of volatility. The trading model's reward-maximizing behavior is optimized with a standard deep q-network (DQN) with adaptive synchronization that (1) effectively test its Markov Decision Process and (2) enables to stabilize and track learning performance on generalizing new experiences in trading each stock.

Research involved in the implementation of this trading model is published in the Journal of Student Research as *Generalized Deep Reinforcement Learning for Trading* (https://doi.org/10.47611/jsrhs.v12i1.4316). Empirical results shown in this article is based on training the trading model on the top 50 holdings of the S&P 500 and testing it on the top 100 holdings of the S&P 500 during the time frame between 2006 and 2022 with no PAA-discretization applied to the multivariate state space. Up-to-date code in this repository may produce different (better) results that reported in the original research.

## Repository Information

./data (historical data storage; included in .gitignore)
./lib (C++ header files; data structures and processing, deep neural network, trading model agent)
./models (trading model paramter checkpoint; raw text file)
./python (python modules used for data processing and analysis)
./res (all build/test/run outputs are saved here)
./src (C++ source files; main, data structures and processing, deep neural network, trading model agent, checkpoint management)

## Usage

**Prerequisites**: basic C++ tools, numpy, matplotlib, pandas, access of the FMP API (a key must be saved as ./apikey)

**Build**:
~~~
./exec build <list of tickers separated by spaces> ./models/<checkpoint name>
./python/summary.py (optional; will output ./res/summary.png that shows change in cumulative mean return-on-investment)
~~~

**Test**:
~~~
./exec test <list of tickers separated by spaces> ./models/<checkpoint name>
~~~

**Run**:
~~~
./exec run <list of tickers separated by spaces> ./models/<checkpoint name>
~~~

*** All content in the trading model may be used for individual research purposes at his or her own risk, and the author and contributor(s) to the work related to this repository do not hold any responsibility for such risks. ***