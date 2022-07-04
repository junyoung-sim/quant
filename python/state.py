#!/usr/bin/env python3

import matplotlib.pyplot as plt

def main():
   price = [float(val) for val in open("./data/price", "r").readline().split(" ")]
   macd = [float(val) for val in open("./data/macd", "r").readline().split(" ")]
   sosc = [float(val) for val in open("./data/sosc", "r").readline().split(" ")]
   rsi = [float(val) for val in open("./data/rsi", "r").readline().split(" ")]

   t = [k for k in range(len(price))]

   plt.subplot(4, 1, 1)
   plt.plot(t, price, color="blue")
   plt.title("Price")

   plt.subplot(4, 1, 2)
   plt.plot(t, macd, color="blue")
   plt.title("MACD")

   plt.subplot(4, 1, 3)
   plt.plot(t, sosc, color="blue")
   plt.title("SOSC")

   plt.subplot(4, 1, 4)
   plt.plot(t, rsi, color="blue")
   plt.title("RSI")

   plt.show()

if __name__ == "__main__":
    main()
