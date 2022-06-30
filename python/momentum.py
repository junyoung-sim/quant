#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def stochastic_oscillator(series):
    N = 14
    osc = []
    for i in range(N, series.shape[0] + 1):
        window = series[i-N:i]
        MAX = np.max(window)
        MIN = np.min(window)

        osc.append((window[-1] - MIN) * 100 / (MAX - MIN))

    osc = np.array(osc)
    return osc

def momentum_backtest():
    x = pd.read_csv("./data/JPM.csv")
    close = np.array(x["Close"])

    position = 0 # -1: short, 0: hold, 1: long
    position_t = -1
    roi = 1.00

    short_signal = 90.0
    long_signal = 10.0
    loss_cut = -0.01

    N = 50
    for t in range(N, close.shape[0]):
        look_back = close[t-N:t] # [t-N, t) = [t-N, t-1]
        osc = stochastic_oscillator(look_back)

        plt.subplot(1,2,1)
        plt.plot(look_back[-osc.shape[0]:])
        plt.subplot(1,2,2)
        plt.plot(osc)
        plt.show()

        if position == 0:
            if osc[-1] >= short_signal:
                position = -1
                position_t = t - 1
                print("\nPOSITION ({}) IN @t={}" .format(position, position_t))
            elif osc[-1] <= long_signal:
                position = 1
                position_t = t - 1
                print("\nPOSITION ({}) IN @t={}" .format(position, position_t))
            else:
                pass
        else:
            r = 1.00 + position * (close[t] - close[position_t]) / close[position_t]
            print("{}, {}" .format(r, osc[-1]))

            if (r - 1.00 <= loss_cut) | ((position == -1) & (osc[-1] <= 50.0)) | ((position == 1) & (osc[-1] >= 50.0)):
                roi *= r
                print("LIQUIDATED! ROI = {}" .format(roi))

                position = 0

if __name__ == "__main__":
    momentum_backtest()

