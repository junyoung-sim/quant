#!/usr/bin/env python3

import sys
import pandas as pd

def main():
    tickers = sys.argv[1:]

    df = None
    for i in range(len(tickers)):
        asset = pd.read_csv("./data/{}.csv" .format(tickers[i]))[["Date", "Adj Close"]]
        asset = asset[asset['Adj Close'].notna()]
        asset = asset.rename(columns={"Adj Close": tickers[i]})

        if i == 0:
            df = asset
        else:
            df = df.merge(asset, on="Date")

    df.to_csv("./data/cleaned.csv")

if __name__ == "__main__":
    main()
