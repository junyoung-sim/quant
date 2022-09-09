#!/usr/bin/env python3

import sys
import yfinance as yf

for ticker in open("./data/tickers", "r").readline().split():
    yf.download(ticker).to_csv("./data/{}.csv" .format(ticker))