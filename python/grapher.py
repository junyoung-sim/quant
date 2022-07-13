#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()

def graph(i):
    mean_loss = []
    buy_and_hold = []
    model_return = []
    performance_residual = []
    for line in open("./data/log", "r"):
        if line != '\n':
            mean_loss.append(float(line.split(" ")[0]))
            buy_and_hold.append(float(line.split(" ")[1]))
            model_return.append(float(line.split(" ")[2]))
            performance_residual.append(model_return[-1] - buy_and_hold[-1])

    t = [i for i in range(len(mean_loss))]

    plt.subplot(1, 3, 1)
    plt.plot(t, mean_loss, color="blue")
    plt.title("Mean Q-Value Loss")

    plt.subplot(1, 3, 2)
    plt.plot(t, buy_and_hold, color="green")
    plt.plot(t, model_return, color="orange")
    plt.title("Cumulative Return")

    plt.subplot(1, 3, 3)
    plt.plot(t, performance_residual, color="red")
    plt.title("Performance Residual")

def main():
    ani = animation.FuncAnimation(fig, graph, save_count=0, cache_frame_data=False)
    plt.show()

if __name__ == "__main__":
    main()
