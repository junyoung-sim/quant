#!/usr/bin/env python3

import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def main():
    ticker = sys.argv[1]
    mean_loss = []
    buy_and_hold = []
    model_return = []

    for line in open("./data/log", "r"):
        if line != '\n':
            mean_loss.append(float(line.split(" ")[0]))
            buy_and_hold.append(float(line.split(" ")[1]))
            model_return.append(float(line.split(" ")[2]))

    t = [i for i in range(len(mean_loss))]

    plt.figure(figsize=(15,5))

    plt.subplot(1, 3, 1)
    plt.plot(t, mean_loss, color="blue")
    plt.title("Mean Q-Value Loss")

    plt.subplot(1, 3, 2)
    plt.plot(t, buy_and_hold, color="green")
    plt.title("Cumulative Return (Benchmark)")

    plt.subplot(1, 3, 3)
    plt.plot(t, model_return, color="orange")
    plt.title("Cumulative Return (Model)")

    plt.savefig("./res/{}.png" .format(ticker))

if __name__ == "__main__":
    main()
