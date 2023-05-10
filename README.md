# Generalized Deep Reinforcement Learning for Trading

## Description

The trading model open-sourced in this repository observes a PAA-discretized multivariate state space consisting of the historical data of a stock of interest and major market-indicating securities (SPY, IEF, EUR=X, GSG), selects a trading action (short, idle, long), and observes a discrete reward based on the correctness of the trading action and independent of volatility. The trading model's reward-maximizing behavior is optimized with a standard deep q-network (DQN) with adaptive synchronization that (1) effectively test its Markov Decision Process and (2) enables to stabilize and track learning performance on generalizing new experiences in trading each stock.

Research involved in the implementation of this trading model is published in the Journal of Student Research as *Generalized Deep Reinforcement Learning for Trading* (https://doi.org/10.47611/jsrhs.v12i1.4316). Empirical results shown in this article is based on training the trading model on the top 50 holdings of the S&P 500 and testing it on the top 100 holdings of the S&P 500 during the time frame between 2006 and 2022 with no PAA-discretization applied to the multivariate state space. Up-to-date code, model checkpoint(s), and outputs in this repository may show results different from (better than) those reported in the original research.

## Performance 

**Build: May 7, 2023 (S&P 500 Top 100)**
**Test: May 9, 2023 (S&P Top 100)**

------------------------------------------------------------
METRIC                               BENCHMARK    MODEL
------------------------------------------------------------
E(R) (Annualized Return)             0.1263       0.4639
SD(R) (Return Standard Deviation)    0.3097       0.4947 
SR (Sharpe Ratio)                    0.4078       0.9377 
MDD (Maximum Drawdown)               0.5903       0.4036 

***Refer to ./res for full build and test results along with up-to-date model outputs.***

## Usage

**Prerequisites**: basic C++ tools, numpy, matplotlib, pandas, access to the FMP API (a key must be saved in apikey as text file)

**Build**:
~~~
./exec build <list of tickers separated by spaces> ./models/<checkpoint name>
./python/summary.py (optional; will output ./res/summary.png that shows change in cumulative mean return-on-investment)
~~~

**Test**:
~~~
./exec test <list of tickers separated by spaces> ./models/<checkpoint name>
~~~

Test Result Example (AAPL; tested on May 9, 2023)
![alt text](https://github.com/junyoung-sim/quant/res/test/AAPL.png?raw=true)

**Run**:
~~~
./exec run <list of tickers separated by spaces> ./models/<checkpoint name>
~~~

Up-to-date Output Example (AAPL)
![alt text](https://github.com/junyoung-sim/quant/res/AAPL.png?raw=true)

***All content in this repository may be used for individual purposes at his or her own risk. The author and contributor(s) to the work related to this repository do not hold any responsibility for such risks.***