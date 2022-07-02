#!/usr/bin/env python3

import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def standardize(series):
    return (series - series.mean()) / np.std(series)

def exponential_moving_average(series, periods):
    smoothing = 2.00 / (periods + 1)
    weights = np.array([math.pow(1.00 - smoothing, periods - 1 - t) for t in range(periods)])

    ema = []
    for t in range(series.shape[0] - periods + 1):
        ema.append((np.sum(series[t:t+periods] * weights) / np.sum(weights)))
    ema = np.array(ema)
    return ema

def moving_average_convergence_divergence(series):
    ema12 = exponential_moving_average(series, 12)
    ema26 = exponential_moving_average(series, 26)
    ema12 = ema12[-ema26.shape[0]:]

    macd = ema12 - ema26
    return macd

def stochastic_oscillator(series, periods):
    osc = []
    for t in range(series.shape[0] - periods + 1):
        osc.append((series[t+periods-1] - np.min(series[t:t+periods])) / (np.max(series[t:t+periods]) - np.min(series[t:t+periods])))
    osc = np.array(osc)
    return osc

def relative_strength_index(series, periods):
    rsi = []
    for t in range(series.shape[0] - periods + 1):
        mean_gain = 0.00
        mean_loss = 0.00
        for i in range(periods - 1):
            difference = (series[t:t+periods][i+1] - series[t:t+periods][i]) / series[t:t+periods][i]
            if difference > 0.00:
                mean_gain += difference
            else:
                mean_loss += abs(difference)
        rsi.append(1 - 1 / (1 + mean_gain / mean_loss))
    rsi = np.array(rsi)
    return rsi

def main():
    path_to_data = sys.argv[1]
    series = np.array(pd.read_csv(path_to_data)["Close"])

    look_back = 60
    test = series[10000:10000+look_back]

    macd = moving_average_convergence_divergence(test)
    osc = stochastic_oscillator(test, 14)
    rsi = relative_strength_index(test, 14)

    min_size = min([test.shape[0], macd.shape[0], osc.shape[0], rsi.shape[0]])

    plt.subplot(4,1,1)
    plt.plot(test[-min_size:])
    plt.subplot(4,1,2)
    plt.plot(macd[-min_size:])
    plt.subplot(4,1,3)
    plt.plot(osc[-min_size:])
    plt.subplot(4,1,4)
    plt.plot(rsi[-min_size:])
    plt.show()



if __name__ == "__main__":
    main()

