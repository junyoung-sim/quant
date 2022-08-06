
#include <cstdlib>
#include <vector>
#include <string>

#include <iostream>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<std::string> tickers;
    unsigned int arg = 0;
    while(true) {
        if(strcmp(argv[++arg], "/") != 0)
            tickers.push_back(argv[arg]);
        else
            break;
    }

    std::string checkpoint = argv[++arg];

    // --- //

    Market market(tickers);

    double eps_init = 1.00;
    double eps_min = 0.01;
    double alpha_init = 0.001;
    double alpha_min = 0.0001;
    double gamma = 0.50;
    unsigned int memory_capacity = 100;
    unsigned int batch_size = 32;
    unsigned int sync_interval = 100;

    Quant quant(market, checkpoint);
    quant.optimize(eps_init, eps_min, alpha_init, alpha_min, gamma, memory_capacity, batch_size, sync_interval);

    return 0;
}