#!/usr/bin/env python3

import sys
import math
import numpy as np

def annualized_return(hist):
    years = 0
    expectation = 1.00
    returns = []
    for t in range(0, len(hist), 251):
        t_prime = min(t + 251, len(hist) - 1)
        ret = 1.00 + (hist[t_prime] - hist[t]) / hist[t]
        expectation *= ret
        years += 1

        returns.append(ret)

    annualized = math.pow(expectation, 1 / years) - 1.00
    returns = np.array(returns)
    return annualized, returns

def maximum_drawdown(hist):
    mdd = 0.00
    t1, t2 = 0, 1
    while t1 < len(hist):
        while (t2 < len(hist) - 1) & (hist[t1] == max(hist[t1:t2+1])):
            t2 += 1
        dd = (min(hist[t1:t2+1]) - hist[t1]) / hist[t1]
        mdd = min(dd, mdd)
        t1 = t2
        t2 = t1 + 1
    return mdd

def main():
    ticker = sys.argv[1]
    benchmark, model, action = [1.00], [1.00], []

    for line in open("./res/log", "r").readlines():
        if line != "\n":
            benchmark.append(float(line.split(" ")[0]))
            model.append(float(line.split(" ")[1]))
            action.append(int(line.split(" ")[2]))

    model_short_pos = action.count(0) / len(action)
    model_idle_pos = action.count(1) / len(action)
    model_long_pos = action.count(2) / len(action)

    benchmark_annualized, benchmark_returns = annualized_return(benchmark)
    model_annualized, model_returns = annualized_return(model)

    benchmark_stdev = np.std(benchmark_returns)
    model_stdev = np.std(model_returns)

    benchmark_sharpe = benchmark_annualized / benchmark_stdev
    model_sharpe = model_annualized / model_stdev

    benchmark_mdd = maximum_drawdown(benchmark)
    model_mdd = maximum_drawdown(model)

    print("Benchmark: (annualized, stdev, sharpe, mdd) = ({}, {}, {}, {})" .format(benchmark_annualized, benchmark_stdev, benchmark_sharpe, benchmark_mdd))
    print("Model: (annualized, stdev, sharpe, mdd) = ({}, {}, {}, {})" .format(model_annualized, model_stdev, model_sharpe, model_mdd))
    print("Model Action Frequency: (long, idle, short) = ({}, {}, {})" .format(model_long_pos, model_idle_pos, model_short_pos))

if __name__ == "__main__":
    main()