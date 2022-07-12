#!/usr/bin/env python3

import pandas as pd
import datetime as datetime

def main():
    asset = pd.read_csv("./data/SPXL_5min.csv")
    vix = pd.read_csv("./data/VIX_5min.csv")

    market_time = pd.to_timedelta(['09:30:00', '16:00:00'])
    timestamp = pd.to_timedelta(asset["Date-Time"].str.split().str[1])
    asset = asset.loc[(timestamp >= market_time[0]) & (timestamp <= market_time[1])]

    vix = vix.loc[vix["Date-Time"].isin(asset["Date-Time"])]
    asset = asset.loc[asset["Date-Time"].isin(vix["Date-Time"])]

    asset.to_csv("./data/SPXL_5min_adjusted.csv", index=False)
    vix.to_csv("./data/VIX_5min_adjusted.csv", index=False)

if __name__ == "__main__":
    main()
