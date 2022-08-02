
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<std::string> tickers = {"AUD=X", "EUR=X", "GBP=X", "JPY=X", "KRW=X"};

    double eps_init = 0.50;
    double eps_min = 0.01;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.50;
    unsigned int memory_capacity = 1000;
    unsigned int batch_size = 64;
    unsigned int sync_interval = 10;
    unsigned int epoch = 10;

    std::default_random_engine seed;
    Quant quant(seed);

    for(unsigned int e = 0; e < epoch; e++) {
        eps_init = (0.10 - 0.50) / epoch * e + 0.50;

        std::shuffle(tickers.begin(), tickers.end(), seed);
        for(std::string ticker: tickers) {
            std::vector<double> series = read_csv("./data/fx/" + ticker + ".csv", "Adj Close");

            quant.optimize(series, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, "./models/checkpoint", seed);

            std::system(("./python/log.py " + ticker).c_str());
            std::system("rm -rf ./res/log && touch ./res/log");

            std::vector<double>().swap(series);
        }
    }

    return 0;
}
