#!/usr/bin/env python3

import sys
import pandas as pd

df = None
for i in range(1, len(sys.argv)):
    ticker = sys.argv[i]
    asset = pd.read_csv("./data/{}.csv" .format(ticker))[["Date", "Adj Close"]]
    asset = asset[asset["Adj Close"].notna()]
    asset = asset.rename(columns={"Adj Close": ticker})

    if i == 1:
        df = asset
    else:
        df = df.merge(asset, on="Date")

print("{} ~ {}\n" .format(df["Date"][0], df["Date"][df["Date"].shape[0] - 1]))

df = df.drop(columns=["Date"])
df.to_csv("./data/cleaned.csv", index=False)