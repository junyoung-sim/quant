#!/usr/bin/env python3

import matplotlib.pyplot as plt

state = [float(line.replace("\n", '')) for line in open("./res/state", "r").readlines()]

plt.plot(state[0:100], label="X")
plt.plot(state[100:200], label="SPY")
plt.plot(state[200:300], label="IEF")
plt.plot(state[300:400], label="EUR=X")
plt.plot(state[400:500], label="GSG")

plt.legend()
plt.savefig("./res/state.png")