#!/usr/bin/env python3

import sys
import pandas as pd

def main():
    ticker = sys.argv[1]
    asset = pd.read_csv("./data/{}.csv" .format(ticker))
    vix = pd.read_csv("./data/^VIX.csv")

    asset = asset.loc[asset["Date"].isin(vix["Date"])]
    vix = vix.loc[vix["Date"].isin(asset["Date"])]

    asset.to_csv("./data/{}_adjusted.csv" .format(ticker), index=False)
    vix.to_csv("./data/^VIX_adjusted.csv", index=False)

if __name__ == "__main__":
    main()
