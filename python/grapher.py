#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

def graph(i):
    mean_loss = []
    buy_and_hold = []
    model_return = []
    for line in open("./data/log", "r"):
        if line != '\n':
            mean_loss.append(float(line.split(" ")[0]))
            buy_and_hold.append(float(line.split(" ")[1]))
            model_return.append(float(line.split(" ")[2]))

    t = [i for i in range(len(mean_loss))]

    plt.subplot(1, 2, 1)
    plt.plot(t, mean_loss, color="blue")
    plt.title("Mean Q-Value Loss")

    plt.subplot(1, 2, 2)
    plt.plot(t, buy_and_hold, color="green")
    plt.plot(t, model_return, color="orange")
    plt.title("Cumulative Return")

def main():
    ani = animation.FuncAnimation(fig, graph, save_count=0, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
