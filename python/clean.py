#!/usr/bin/env python3

import sys
import pandas as pd

df = None
for i in range(1, len(sys.argv)):
    ticker = sys.argv[i]
    dat = pd.read_csv("./data/{}.csv" .format(ticker))[["date", "adjClose"]]

    dat = dat[dat["adjClose"].notna()]
    dat = dat.rename(columns={"adjClose": ticker})
    dat = dat.loc[::-1]

    if i == 1:
        df = dat
    else:
        df = df.merge(dat, on="date")

print("{}: {} ~ {}" .format(sys.argv[1], df["date"][0], df["date"][df["date"].shape[0] - 1]))

df = df.drop(columns=["date"])
df.to_csv("./data/cleaned.csv", index=False)