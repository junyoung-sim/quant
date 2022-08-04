
#include <cstdlib>
#include <vector>
#include <string>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::vector<std::string> tickers = {"SPY", "TLT", "GSG", "^VIX"};
    Market market(tickers);

    Quant quant(market);

    return 0;
}