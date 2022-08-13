#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

log = open("./res/log", "r").readlines()

loss, benchmark, model, eps, alpha = [], [], [], [], []

for line in log:
    if line != "\n":
        loss.append(float(line.split(" ")[0]))
        benchmark.append(float(line.split(" ")[1]))
        model.append(float(line.split(" ")[2]))
        eps.append(float(line.split(" ")[3]))
        alpha.append(float(line.split(" ")[4]))

t = [i for i in range(len(loss))]

plt.figure(figsize=(20,5))
plt.subplot(1, 3, 1)
plt.plot(t, loss)
plt.title("Mean Loss")
plt.subplot(1, 3, 2)
plt.plot(t, eps)
plt.title("Epsilon")
plt.subplot(1, 3, 3)
plt.plot(t, alpha)
plt.title("Alpha")
plt.savefig("./res/param.png")

plt.figure(figsize=(30,8))
plt.plot(benchmark, color="green")
plt.plot(model, color="orange")
plt.title("Cumulative Return")
plt.savefig("./res/performance.png")
