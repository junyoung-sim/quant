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
    mdd = 0
    dp = hist.copy()
    for t in range(1, len(hist)):
        dp[t] = max(dp[t-1], dp[t])
        mdd = min((hist[t] - dp[t]) / dp[t], mdd)
    return abs(mdd)

if __name__ == "__main__":
    mode = sys.argv[1]

    if mode == "push":
        ticker = sys.argv[2]
        benchmark, model, action = [], [], []
        for line in open("./res/log", "r").readlines():
            if line != "\n":
                benchmark.append(float(line.split(" ")[0]))
                model.append(float(line.split(" ")[1]))
                action.append(int(line.split(" ")[2]))

        benchmark_annualized, benchmark_returns = annualized_return(benchmark)
        model_annualized, model_returns = annualized_return(model)

        benchmark_stdev = np.std(benchmark_returns)
        model_stdev = np.std(model_returns)

        benchmark_sharpe = benchmark_annualized / benchmark_stdev
        model_sharpe = model_annualized / model_stdev

        benchmark_mdd = maximum_drawdown(benchmark)
        model_mdd = maximum_drawdown(model)

        stats = open("./res/stats", "a")
        stats.write("{},{},{},{},{},{},{},{},{}\n"
                    .format(ticker, benchmark_annualized, benchmark_stdev, benchmark_sharpe, benchmark_mdd,
                            model_annualized, model_stdev, model_sharpe, model_mdd))
        stats.close()
    
    elif mode == "summary":
        stats = [line.replace("\n", "").split(",")[1:] for line in open("./res/stats", "r").readlines()]
        stats = np.array([[float(val) for val in line] for line in stats]).T

        benchmark_annualized = stats[0].mean()
        benchmark_stdev = stats[1].mean()
        benchmark_sharpe = benchmark_annualized / benchmark_stdev
        benchmark_mdd = stats[3].mean()

        model_annualized = stats[4].mean()
        model_stdev = stats[5].mean()
        model_sharpe = model_annualized / model_stdev
        model_mdd = stats[7].mean()

        print()
        print("------------------------------")
        print("METRIC    BENCHMARK    MODEL")
        print("------------------------------")
        print("E(R)      {:.4f}       {:.4f}" .format(benchmark_annualized, model_annualized))
        print("SD(R)     {:.4f}       {:.4f}" .format(benchmark_stdev, model_stdev))
        print("SR        {:.4f}       {:.4f}" .format(benchmark_sharpe, model_sharpe))
        print("MDD       {:.4f}       {:.4f}" .format(benchmark_mdd, model_mdd))
    
    else:
        print("Invalid command given.")