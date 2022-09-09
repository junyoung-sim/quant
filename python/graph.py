#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

ticker = sys.argv[1]
benchmark, model = [], []

for line in open("./res/log", "r").readlines():
    if line != "\n":
        benchmark.append(float(line.split(" ")[0]))
        model.append(float(line.split(" ")[1]))

plt.figure(figsize=(30,8))
plt.plot(benchmark, color="green", label="Benchmark (100% long)")
plt.plot(model, color="orange", label="Model")
plt.legend()
plt.title("Cumulative Return")
plt.savefig("./res/{}.png" .format(ticker))