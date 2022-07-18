
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../lib/data.hpp"
#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<double> asset = read_csv("./data/SPXL_5min_adjusted.csv", "Close");
    std::vector<double> vix = read_csv("./data/VIX_5min_adjusted.csv", "Close");

    double eps_init = 0.50;
    double eps_min = 0.10;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.30;
    unsigned int memory_capacity = 10000;
    unsigned int batch_size = 64;
    unsigned int sync_interval = 5000;
    unsigned int look_back = 50;

    Quant quant;
    quant.optimize(asset, vix, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, look_back, "./models/checkpoint");

    return 0;
}
