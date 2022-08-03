#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt

log = open("./res/log", "r").readlines()

ticker = sys.argv[1]
loss = []
model = []

for line in log:
    if line != "\n":
        loss.append(float(line.split(" ")[0]))
        model.append(float(line.split(" ")[1]))

t = [i for i in range(len(loss))]

plt.figure(figsize=(15,8))

plt.subplot(1, 2, 1)
plt.plot(t, loss, color="blue")
plt.title("Mean Loss")

plt.subplot(1, 2, 2)
plt.plot(t, model, color="green")
plt.title("Cumulative Return")

plt.savefig("./res/{}.png" .format(ticker))

