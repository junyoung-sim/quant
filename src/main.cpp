
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/data.hpp"
#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<double> asset = read_csv("./data/SPXL_5min_adjusted.csv", "Close");

    double eps_init = 0.50;
    double eps_min = 0.01;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.30;
    unsigned int memory_capacity = 10000;
    unsigned int batch_size = 64;
    unsigned int sync_interval = 1000;

    Quant quant;
    quant.optimize(asset, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, "./models/checkpoint");

    std::vector<double>().swap(asset);

    return 0;
}
