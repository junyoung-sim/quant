#!/usr/bin/env python3

import sys
import math

def annualized_return(history):
    hist = history.copy()
    hist.insert(0, 1.00)

    years = 0
    expectation = 1.00
    returns = []

    for t in range(0, len(hist), 251):
        t_prime = t + 251
        if t_prime >= len(hist):
            t_prime = len(hist) - 1

        ret = 1.00 + (hist[t_prime] - hist[t]) / hist[t]
        expectation *= ret
        years += 1

        returns.append(ret)

    annualized = math.pow(expectation, 1 / years) - 1.00

    return annualized, returns

def main():
    ticker = sys.argv[1]
    benchmark, model, action = [], [], []

    for line in open("./res/log", "r").readlines():
        if line != "\n":
            benchmark.append(float(line.split(" ")[0]))
            model.append(float(line.split(" ")[1]))
            action.append(int(line.split(" ")[2]))

    short_pos = action.count(0) / len(action)
    idle_pos = action.count(1) / len(action)
    long_pos = action.count(2) / len(action)

    benchmark_annualized, benchmark_returns = annualized_return(benchmark)
    model_annualized, model_returns = annualized_return(model)

if __name__ == "__main__":
    main()