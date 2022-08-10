
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
    for(std::string ticker: tickers)
        market_dataset.push_back(Market({ticker, "TLT", "GOLD", "SLV", "^VIX"}));

    // --- //

    Quant quant(market_dataset, checkpoint);

    return 0;
}