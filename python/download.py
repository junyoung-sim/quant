#!/usr/bin/env python3

import sys
import yfinance as yf

for line in open("./data/tickers", "r").readlines():
    ticker = line
    if ticker[-1] == "\n":
        ticker = ticker[:-1]
    yf.download(ticker).to_csv("./data/{}.csv" .format(ticker))
