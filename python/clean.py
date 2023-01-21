#!/usr/bin/env python3

import sys
import pandas as pd

df = None
for i in range(1, len(sys.argv)):
    ticker = sys.argv[i]
    asset = pd.read_csv("./data/{}.csv" .format(ticker))[["date", "adjClose"]]

    asset = asset[asset["adjClose"].notna()]
    asset = asset.rename(columns={"adjClose": ticker})
    asset = asset.loc[::-1]

    if i == 1:
        df = asset
    else:
        df = df.merge(asset, on="date")

print("{} ~ {}\n" .format(df["date"][0], df["date"][df["date"].shape[0] - 1]))

df = df.drop(columns=["date"])
df.to_csv("./data/cleaned.csv", index=False)