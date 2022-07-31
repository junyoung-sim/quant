
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <iostream>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    double eps_init = 0.50;
    double eps_min = 0.01;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.50;
    unsigned int memory_capacity = 1000;
    unsigned int batch_size = 512;
    unsigned int sync_interval = 100;
    unsigned int look_back = 10000;
    unsigned int epoch = 10;

    Quant quant;
    std::default_random_engine seed;

    std::vector<double> test = read_csv("./data/SPXL_5min_adjusted.csv", "Close");
    quant.optimize(test, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, look_back, "./models/checkpoint", seed);

    return 0;
}
