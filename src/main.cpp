
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <chrono>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    double eps_init = 0.50;
    double eps_min = 0.01;
    double alpha_init = 0.0001;
    double alpha_min = 0.00001;
    double gamma = 0.90;
    unsigned int memory_capacity = 10000;
    unsigned int batch_size = 64;
    unsigned int sync_interval = 5000;

    std::default_random_engine seed;
    seed.seed(std::chrono::system_clock::now().time_since_epoch().count());

    Quant quant(seed);

    std::vector<double> series = read_csv("./data/stock/SPXL_5min_adjusted.csv", "Close");
    quant.optimize(series, eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval, "./models/checkpoint", seed);

    std::vector<double>().swap(series);

    return 0;
}
