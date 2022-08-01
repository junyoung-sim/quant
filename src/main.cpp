
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<std::string> tickers = {"AAPL", "AMGN", "AMZN", "AXP", "BA", "BRK-B", "CAT",
                                        "CRM", "CSCO", "CVX", "DIA", "DIS", "GOOG", "GS", "HD",
                                        "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MA", "MCD",
                                        "MMM", "MRK", "MSFT", "NKE", "NVDA", "PFE", "PG", "TRV",
                                        "UNH", "V", "VZ", "WBA", "WMT", "XOM"};

    double eps_init = 1.00;
    double eps_min = 0.01;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.90;
    unsigned int memory_capacity = 1000;
    unsigned int batch_size = 512;
    unsigned int sync_interval = 100;
    unsigned int look_back = 252;
    unsigned int epoch = 10;

    Quant quant;
    std::default_random_engine seed;

    for(unsigned int e = 1; e <= epoch; e++) {
        std::shuffle(tickers.begin(), tickers.end(), seed);
        for(std::string ticker: tickers) {
            std::vector<double> series = read_csv("./data/" + ticker + ".csv", "Adj Close");
            quant.optimize(series, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, look_back, "./models/checkpoint", seed);

            std::system(("./python/log.py " + ticker).c_str());
            std::system("rm -rf ./res/log && touch ./res/log");

            std::vector<double>().swap(series);
        }

        eps_init -= 0.10;
    }
    return 0;
}
