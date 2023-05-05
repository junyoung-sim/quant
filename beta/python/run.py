#!/usr/bin/env python3

import matplotlib.pyplot as plt

state = [float(val) for val in open("./res/state", "r").readline().split(" ")[:-1]]

plt.figure(figsize=(8,5))
plt.plot(state[0:100], label="X")
plt.plot(state[100:200], label="SPY")
plt.plot(state[200:300], label="IEF")
plt.plot(state[300:400], label="EUR=X")
plt.plot(state[400:500], label="GSG")

plt.legend()
plt.savefig("./res/state.png")