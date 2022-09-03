#!/usr/bin/env python3

import yfinance as yf

tickers = [ticker[:-1] for ticker in open("tickers").readlines()]

for ticker in tickers:
    yf.download(ticker).to_csv("{}.csv" .format(ticker))
