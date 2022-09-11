#!/usr/bin/env python3

import sys
import yfinance as yf

for ticker in sys.argv[1:]:
    yf.download(ticker).to_csv("./data/{}.csv" .format(ticker))