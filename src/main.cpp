
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/trader.hpp"

int main(int argc, char *argv[])
{
    Trader trader;
/*    std::vector<std::string> tickers = {"AAPL", "AMZN", "BRK-B", "GOOG", "HD", "JNJ", "JPM", "MA", "MSFT",
                                        "NVDA", "PFE", "PG", "UNH", "UNH", "V", "XOM", "SPY"};
*/

    std::vector<std::string> tickers = {"AAPL"};

    double EPSILON_INIT = 0.50;
    double EPSILON_MIN = 0.01;
    double ALPHA_INIT = 0.0001;
    double ALPHA_MIN = 0.00001;
    double GAMMA = 0.30;
    unsigned int MEMORY_CAPACITY = 1000;
    unsigned int BATCH_SIZE = 64;
    unsigned int SYNC_INTERVAL = 100;
    unsigned int LOOK_BACK = 50;

    unsigned int EPOCH = 1;
    for(unsigned int epoch = 1; epoch <= EPOCH; epoch++) {
        for(std::string ticker: tickers) {
            // download data via broker api

            std::system(("./python/clean.py " + ticker).c_str());
            std::vector<double> asset = read_csv("./data/" + ticker + "_adjusted.csv", "Adj Close");
            std::vector<double> vix = read_csv("./data/" + ticker + "_adjusted.csv", "Adj Close");

            trader.optimize(asset, vix, EPSILON_INIT, EPSILON_MIN, ALPHA_INIT, ALPHA_MIN,
                            GAMMA, MEMORY_CAPACITY, BATCH_SIZE, SYNC_INTERVAL, LOOK_BACK, "./models/checkpoint");

//            std::system(("./python/graph.py " + ticker).c_str());
            std::system("rm ./data/log && touch ./data/log");
        }

        EPSILON_INIT = (0.10 - EPSILON_INIT) / EPOCH * epoch + EPSILON_INIT;
    }

    return 0;
}
