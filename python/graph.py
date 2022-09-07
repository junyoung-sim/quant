#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

ticker = sys.argv[1]
log = open("./res/log", "r").readlines()

benchmark, model = [], []

for line in log:
    if line != "\n":
        benchmark.append(float(line.split(" ")[0]))
        model.append(float(line.split(" ")[1]))

plt.figure(figsize=(30,8))
plt.plot(benchmark, color="green")
plt.plot(model, color="orange")
plt.title("Cumulative Return")
plt.savefig("./res/{}.png" .format(ticker))
