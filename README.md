# Generalized Deep Reinforcement Learning for Trading

## Description

The trading model open-sourced in this repository observes a PAA-discretized multivariate state space consisting of the historical data of a stock of interest and major market-indicating securities (SPY, IEF, EUR=X, GSG), selects a trading action (short, idle, long), and observes a discrete reward based on the correctness of the trading action and independent of volatility. The trading model's reward-maximizing behavior is optimized with a standard deep q-network (DQN) with adaptive synchronization that (1) effectively test its Markov Decision Process and (2) enables to stabilize and track learning performance on generalizing new experiences in trading each stock.

Research involved in the implementation of this trading model is published in the Journal of Student Research as *Generalized Deep Reinforcement Learning for Trading* (https://doi.org/10.47611/jsrhs.v12i1.4316). Empirical results shown in this article is based on training the trading model on the top 50 holdings of the S&P 500 and testing it on the top 100 holdings of the S&P 500 during the time frame between 2006 and 2022 with no PAA-discretization applied to the multivariate state space. Up-to-date code, model checkpoint(s), and outputs in this repository may show results different from (better than) those reported in the original research.

## Performance Report

**Build**: Jul 21, 2006 - May 7, 2023 (S&P 500 Top 100)

**Test**: Jul 21, 2006 - May 9, 2023 (S&P 500 Top 100)

| Metric | Benchmark | Model  |
|--------|-----------|--------|
| E(R)   | 0.1263    | 0.4639 |
| SD(R)  | 0.3097    | 0.4947 |
| SR     | 0.4078    | 0.9377 |
| MDD    | 0.5903    | 0.4036 |

E(R) = annualized return, SD(R) = return standard deviation, SR = sharpe ratio, MDD = maximum drawdown

***Refer to ./res for full build and test results along with up-to-date model outputs.***

## Usage

**Prerequisites**: basic C++ tools, numpy, matplotlib, pandas, access to the FMP API (a key must be saved in apikey as text file)

**Build**:
~~~
./exec build <list of tickers separated by spaces> ./models/checkpoint
~~~

**Test**:
~~~
./exec test <list of tickers separated by spaces> ./models/checkpoint
~~~

Test Result Output (AAPL)

![alt text](https://github.com/junyoung-sim/quant/blob/master/res/test/AAPL.png?raw=true)

**Run**:
~~~
./exec run <list of tickers separated by spaces> ./models/checkpoint
~~~

Up-To-Date Output (AAPL)

![alt text](https://github.com/junyoung-sim/quant/blob/master/res/AAPL.png?raw=true)
