#!/usr/bin/env python3

import matplotlib.pyplot as plt

log = open("./res/log", "r").readlines()

loss = []
benchmark = []
model = []

for line in log:
    if line != "\n":
        loss.append(float(line.split(" ")[0]))
        benchmark.append(float(line.split(" ")[1]))
        model.append(float(line.split(" ")[2]))

t = [i for i in range(len(loss))]

plt.figure(figsize=(15,5))

plt.subplot(1, 2, 1)
plt.plot(t, loss, color="blue")
plt.title("Mean Loss")

plt.subplot(1, 2, 2)
plt.plot(t, benchmark, color="green")
plt.plot(t, model, color="orange")
plt.title("Cumulative Return")

plt.savefig("./res/log.png")

