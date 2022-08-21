
#include <cstdlib>
#include <vector>
#include <string>

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

    std::vector<Market> market_dataset;
    for(std::string ticker: tickers) {
        std::cout << "Loading market data for " << ticker << "\n";
        market_dataset.push_back(Market({ticker, "TLT", "GOLD", "^VIX"}));
    }

    std::vector<std::string>().swap(tickers);

    // --- //

    Quant quant(market_dataset, checkpoint);
    quant.optimize();

    return 0;
}