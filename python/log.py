#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

ticker = sys.argv[1]
benchmark, model, action = [], [], []

for line in open("./res/log", "r").readlines():
    if line != "\n":
        benchmark.append(float(line.split(" ")[0]))
        model.append(float(line.split(" ")[1]))
        action.append(int(line.split(" ")[2]))

plt.figure(figsize=(30,8))

plt.subplot(2, 1, 1)
plt.plot(benchmark, color="green", label="Benchmark (100% long)")
plt.plot(model, color="orange", label="Model")
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(action, color="blue", label="action (0:short, 1:idle, 2:long)")
plt.legend()
plt.savefig("./res/{}.png" .format(ticker))