#!/usr/bin/env python3

import matplotlib.pyplot as plt

benchmark_roi = []
benchmark_avg = []

model_roi = []
model_avg = []

for line in open("./res/summary", "r").readlines():
    if line != "\n":
        benchmark = float(line.split(" ")[0])
        model = float(line.split(" ")[1])

        benchmark_roi.append(benchmark)
        benchmark_avg.append(sum(benchmark_roi) / len(benchmark_roi))
        
        model_roi.append(model)
        model_avg.append(sum(model_roi) / len(model_roi))

plt.plot(benchmark_avg, label="Benchmark", color="c")
plt.plot(model_avg, label="Model", color="b")
plt.xlabel("Experience")
plt.ylabel("Mean ROI")
plt.legend()
plt.savefig("./res/summary.png")