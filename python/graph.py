#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

log = open("./res/log", "r").readlines()

loss, benchmark, model, eps, alpha = [], [], [], [], []
relative_performance = []

for line in log:
    if line != "\n":
        loss.append(float(line.split(" ")[0]))
        benchmark.append(float(line.split(" ")[1]))
        model.append(float(line.split(" ")[2]))
        eps.append(float(line.split(" ")[3]))
        alpha.append(float(line.split(" ")[4]))
        relative_performance.append(model[-1] > benchmark[-1])

plt.figure(figsize=(20,5))
plt.subplot(1, 3, 1)
plt.plot(loss)
plt.title("Mean Loss")
plt.subplot(1, 3, 2)
plt.plot(eps)
plt.title("Epsilon")
plt.subplot(1, 3, 3)
plt.plot(alpha)
plt.title("Alpha")
plt.savefig("./res/param.png")

plt.figure(figsize=(30,8))
plt.plot(benchmark, color="green")
plt.plot(model, color="orange")
plt.title("Cumulative Return")
plt.savefig("./res/performance.png")

plt.figure(figsize=(30,8))
plt.plot(relative_performance)
plt.savefig("./res/relative_performance.png")