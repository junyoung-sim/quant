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

state = [float(line) for line in open("./res/state", "r").readlines()]
current_action = int(state[500])

plt.figure(figsize=(10,10))

plt.subplot(3, 1, 1)
plt.plot(benchmark, color="green", label=ticker)
plt.plot(model, color="orange", label="Model")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(action, color="blue", label="(0, 1, 2) = (short, idle, long)")
plt.legend()

plt.subplot(3, 1, 3)
plt.title("action={}" .format(current_action))
plt.plot(state[0:100], label=ticker)
plt.plot(state[100:200], label="SPY")
plt.plot(state[200:300], label="IEF")
plt.plot(state[300:400], label="EUR=X")
plt.plot(state[400:500], label="GSG")
plt.legend()

plt.savefig("./res/{}.png" .format(ticker))