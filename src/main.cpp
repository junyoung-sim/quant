
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/trader.hpp"

int main(int argc, char *argv[])
{
    std::vector<double> asset = read_csv("./data/SPXL_5min_adjusted.csv", "Close");
    std::vector<double> vix = read_csv("./data/VIX_5min_adjusted.csv", "Close");

    Trader trader;
    trader.optimize(asset, vix, "./model/checkpoint");

    return 0;
}
