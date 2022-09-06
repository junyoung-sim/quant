#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    unsigned int arg = 0;

    std::string mode = argv[++arg];

    std::vector<std::string> tickers;
    while(strcmp(argv[++arg], "/") != 0)
        tickers.push_back(argv[arg]);

    std::string checkpoint = argv[++arg];

    // --- //

    std::vector<Market> dataset;
    for(std::string &ticker: tickers) {
        std::cout << "Loading market data for " << ticker << "\n";
        dataset.push_back(Market({ticker, "SPY", "^TNX", "IEF", "GSG"}));
    }

    std::vector<std::string>().swap(tickers);

    // --- //

    Quant quant(dataset, checkpoint);

    if(mode == "build")
        quant.build();
    else if(mode == "run")
        quant.run();
    else
        std::cout << "Invalid mode given!\n";

    return 0;
}