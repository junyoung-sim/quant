#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    tickers = [sys.argv[i] for i in range(1, len(sys.argv))]

    df = pd.read_csv("./data/{}.csv" .format(tickers[0]))[["Date", "Adj Close"]]
    df = df.rename(columns={"Adj Close": tickers[0]})

    for i in range(1, len(tickers)):
        asset = pd.read_csv("./data/{}.csv" .format(tickers[i]))[["Date", "Adj Close"]]
        asset = asset.rename(columns={"Adj Close": tickers[i]})

        df = df.merge(asset, on="Date")

    df.to_csv("./data/cleaned.csv")

if __name__ == "__main__":
    main()
