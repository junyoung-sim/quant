#!/usr/bin/env python3

import matplotlib.pyplot as plt

def main():
    loss = []
    benchmark = []
    model = []
    eps = []
    alpha = []

    for line in open("./data/log", "r").readlines():
        if line != "\n":
            loss.append(float(line.split(" ")[0]))
            benchmark.append(float(line.split(" ")[1]))
            model.append(float(line.split(" ")[2]))
            eps.append(float(line.split(" ")[3]))
            alpha.append(float(line.split(" ")[4]))

    t = [i for i in range(len(loss))]

    plt.figure(figsize=(15,8))

    plt.subplot(2, 2, 1)
    plt.plot(t, loss, color="blue")
    plt.title("Mean Q-Value Loss")

    plt.subplot(2, 2, 2)
    plt.plot(t, eps, color="blue")
    plt.title("Epsilon")

    plt.subplot(2, 2, 3)
    plt.plot(t, alpha, color="blue")
    plt.title("Alpha")

    plt.subplot(2, 2, 4)
    plt.plot(t, benchmark, color="green")
    plt.plot(t, model, color="orange")
    plt.title("Cumulative Return")

    plt.savefig("./res/results.png")

if __name__ == "__main__":
    main()

